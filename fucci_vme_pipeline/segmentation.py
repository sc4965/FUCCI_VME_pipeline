"""Stage 2: segmentation.

Two independent pieces, deliberately separable so each can be tested/debugged
on its own:

1. `run_cellpose_sam` -- the actual mask generation. Lazy-imports `cellpose`
   (and its PyTorch dependency) since that's a heavy, GPU-oriented install
   that only needs to exist on the imaging workstation, never on a machine
   just editing/testing this code.
2. `regionprops_from_labels` -- pure-numpy per-object feature extraction
   (area, centroid, eccentricity, condensation score) from any label image,
   real or synthetic. This is fully testable without cellpose installed.

`condensation_score` is this pipeline's concrete implementation of the design
doc's qualitative "how tightly the signal is gathered" idea: the
intensity-weighted mean-squared radius from the intensity centroid,
normalized against the value a uniform disk of the same area would give.
Score -> 1 means intensity is concentrated near the centroid (condensed
chromatin); score -> 0 means intensity is spread uniformly across the
object's footprint (diffuse interphase chromatin or a whole-cell blob).
This is a starting definition, meant to be validated/tuned against annotated
mitotic vs. interphase nuclei, not treated as biologically exact from day one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig


@dataclass
class ObjectProps:
    label: int
    area_px: int
    centroid_y: float
    centroid_x: float
    eccentricity: float
    condensation_score: float
    mean_intensity: float
    population: str  # "nuclear_candidate" | "large_candidate"


def run_cellpose_sam(image_stack: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Segment a (T, Y, X) intensity stack into per-frame instance labels.

    Returns a (T, Y, X) int label array. Requires `cellpose` to be installed;
    only called on the imaging workstation, never in --demo mode.

    Fails loudly if no GPU is detected and `config.require_gpu` is True
    (the default), matching Cellpose's own recommended pattern -- on CPU,
    Cellpose-SAM runs at roughly a minute per single small tile, so a silent
    fallback would make a real run appear to hang rather than fail fast.

    Prints per-frame progress: large frames (Cellpose-SAM tiles images into
    fixed 256x256 patches internally, so a 2048x2048 frame means dozens of
    tiles through a fairly heavy transformer backbone) can each take real
    time, and with no progress output there'd be no way to tell "still
    working" from "actually hung" during a long real-data run.
    """
    from cellpose import core, models  # lazy: heavy PyTorch dependency

    if config.require_gpu and not core.use_gpu():
        raise RuntimeError(
            "No GPU detected (checked via cellpose.core.use_gpu()). "
            "If running in Colab: Runtime > Change runtime type > GPU, then "
            "restart. Set config.require_gpu=False to run on CPU anyway "
            "(expect roughly a minute per small tile -- impractical at full "
            "dataset scale, only useful for a tiny sanity check)."
        )

    model = models.CellposeModel(gpu=True, pretrained_model=config.cellpose_pretrained_model)
    labels = np.zeros_like(image_stack, dtype=np.int32)
    n_frames = image_stack.shape[0]
    for t in range(n_frames):
        print(f"Segmenting frame {t + 1}/{n_frames}...", flush=True)
        mask, _, _ = model.eval(
            image_stack[t],
            diameter=config.cellpose_diameter,
            flow_threshold=config.cellpose_flow_threshold,
            cellprob_threshold=config.cellpose_cellprob_threshold,
        )
        labels[t] = mask
    return labels


def _second_moments(ys: np.ndarray, xs: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, float]:
    """Weighted covariance of pixel positions: returns (var_y, var_x, cov_yx)."""
    w = weights if weights is not None else np.ones_like(ys, dtype=np.float64)
    w_sum = w.sum()
    cy = (w * ys).sum() / w_sum
    cx = (w * xs).sum() / w_sum
    dy = ys - cy
    dx = xs - cx
    var_y = (w * dy * dy).sum() / w_sum
    var_x = (w * dx * dx).sum() / w_sum
    cov_yx = (w * dy * dx).sum() / w_sum
    return var_y, var_x, cov_yx


def _eccentricity_from_moments(var_y: float, var_x: float, cov_yx: float) -> float:
    """Standard second-moment eccentricity: sqrt(1 - lambda_min/lambda_max)
    of the covariance matrix [[var_y, cov_yx], [cov_yx, var_x]].
    """
    trace = var_y + var_x
    det = var_y * var_x - cov_yx**2
    disc = max(trace**2 - 4 * det, 0.0)
    sqrt_disc = np.sqrt(disc)
    lam_max = (trace + sqrt_disc) / 2
    lam_min = (trace - sqrt_disc) / 2
    if lam_max <= 1e-12:
        return 0.0
    ratio = max(lam_min / lam_max, 0.0)
    return float(np.sqrt(max(1.0 - ratio, 0.0)))


def _condensation_score(ys: np.ndarray, xs: np.ndarray, intensity: np.ndarray) -> float:
    """Intensity-weighted mean-squared radius, normalized against this
    SAME object's own footprint filled uniformly -- not an idealized disk.

    An earlier version normalized against a uniform disk of the same area,
    which conflates shape elongation with intensity concentration: an
    elongated mitotic chromatin mass (correctly high eccentricity) scored
    condensation_score == 0 regardless of how concentrated its actual
    fluorescence was, simply because its spatial extent along the long axis
    exceeds what a disk of equal area would have. Found by running the full
    pipeline end to end on a synthetic forced-mitosis frame, not by the
    per-function unit tests (which only tested condensation on round shapes
    and eccentricity on uniformly-lit shapes -- never the conjunction).
    Using the object's own actual pixel footprint as the "uniform" reference
    isolates "is intensity concentrated within this shape" from "what shape
    is this" (already captured separately by eccentricity).
    """
    total_intensity = intensity.sum()
    if total_intensity <= 0:
        return 0.0
    wcy = (intensity * ys).sum() / total_intensity
    wcx = (intensity * xs).sum() / total_intensity
    r2_weighted = (intensity * ((ys - wcy) ** 2 + (xs - wcx) ** 2)).sum() / total_intensity

    ucy = ys.mean()
    ucx = xs.mean()
    r2_uniform = ((ys - ucy) ** 2 + (xs - ucx) ** 2).mean()
    if r2_uniform <= 0:
        return 0.0

    score = 1.0 - (r2_weighted / r2_uniform)
    return float(np.clip(score, 0.0, 1.0))


def mean_intensity_by_label(labels: np.ndarray, intensity_image: np.ndarray) -> dict[int, float]:
    """Per-label mean intensity on an arbitrary channel, reusing the labels
    from the segmentation (green) channel -- e.g. reading out cdt1/slbp/
    geminin at the nuclei the green channel already located, per Stage 2's
    "green locates, other channels characterize" design.
    """
    ys, xs = np.nonzero(labels)
    if ys.size == 0:
        return {}
    label_ids = labels[ys, xs]
    values = intensity_image[ys, xs].astype(np.float64)

    result: dict[int, float] = {}
    for lbl in np.unique(label_ids):
        mask = label_ids == lbl
        result[int(lbl)] = float(values[mask].mean())
    return result


def regionprops_from_labels(
    labels: np.ndarray,
    intensity_image: np.ndarray,
    config: PipelineConfig,
) -> list[ObjectProps]:
    """Per-object features for one frame's label image, pure numpy.

    Applies `min_object_area_px` as a hard floor (drops debris/noise) and
    tags each surviving object's coarse population by
    `max_nuclear_candidate_area_px` -- a deliberately generous pre-filter,
    NOT the authoritative infected/uninfected call (see Stage 5).
    """
    props: list[ObjectProps] = []
    ys_all, xs_all = np.nonzero(labels)
    if ys_all.size == 0:
        return props
    label_ids = labels[ys_all, xs_all]

    for lbl in np.unique(label_ids):
        mask = label_ids == lbl
        ys = ys_all[mask]
        xs = xs_all[mask]
        area_px = ys.size
        if area_px < config.min_object_area_px:
            continue

        intensity = intensity_image[ys, xs].astype(np.float64)
        var_y, var_x, cov_yx = _second_moments(ys.astype(np.float64), xs.astype(np.float64))
        eccentricity = _eccentricity_from_moments(var_y, var_x, cov_yx)
        condensation = _condensation_score(ys.astype(np.float64), xs.astype(np.float64), intensity)
        population = (
            "nuclear_candidate" if area_px <= config.max_nuclear_candidate_area_px else "large_candidate"
        )

        props.append(
            ObjectProps(
                label=int(lbl),  # the true mask integer ID -- do not rely on btrack's own "label" property, which silently renumbers
                area_px=int(area_px),
                centroid_y=float(ys.mean()),
                centroid_x=float(xs.mean()),
                eccentricity=eccentricity,
                condensation_score=condensation,
                mean_intensity=float(intensity.mean()),
                population=population,
            )
        )
    return props
