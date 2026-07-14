"""Synthetic data generation for --demo mode.

Produces ChannelStack objects that look like the output of `ingestion.py`
(same shape/calibration contract) without touching the `nd2` package or any
real files. This is the fast sanity check: confirms pipeline logic and
dependencies work before ever pointing at real ND2s on the lab workstation.
"""
from __future__ import annotations

import numpy as np

from .config import PipelineConfig
from .ingestion import ChannelStack


def make_synthetic_channel_stack(
    role: str,
    wavelength_nm: int,
    n_frames: int = 20,
    size: int = 256,
    pixel_size_um: float = 0.65,
    frame_interval_min: float = 6.0,
    seed: int = 0,
) -> ChannelStack:
    rng = np.random.default_rng(seed)
    data = rng.integers(50, 150, size=(n_frames, size, size), dtype=np.uint16)
    return ChannelStack(
        role=role,
        wavelength_nm=wavelength_nm,
        data=data,
        pixel_size_um=pixel_size_um,
        frame_interval_min=frame_interval_min,
        source_path=None,
    )


def make_synthetic_dataset(config: PipelineConfig, n_frames: int = 20, size: int = 256) -> dict[str, ChannelStack]:
    """One synthetic ChannelStack per role in the config's channel map."""
    stacks = {}
    for wavelength_nm, channel_role in config.channel_map.items():
        stacks[channel_role.role] = make_synthetic_channel_stack(
            role=channel_role.role,
            wavelength_nm=wavelength_nm,
            n_frames=n_frames,
            size=size,
            seed=wavelength_nm,
        )
    return stacks


def make_synthetic_movie(
    n_frames: int = 20,
    size: int = 200,
    pixel_size_um: float = 0.65,
    frame_interval_min: float = 6.0,
):
    """A small, fully-synthetic scene exercising the whole pipeline's glue:
    4 non-moving FUCCI-4 nuclei cycling through phases (sine waves per
    channel, phase-offset per nucleus), one of which goes through a forced
    high-condensation "mitosis" at frames 10-11, plus one fixed infected
    cell (large, diffuse, strong green signal only).

    Positions are fixed (no drift) deliberately -- tracking correctness
    itself is already covered in test_tracking.py; this scene is for
    testing that segmentation -> tracking -> classification -> neighbors
    -> dimensionality reduction glue together correctly end to end.

    Returns (labels, images, pixel_size_um, frame_interval_min) where
    `labels` is (T, Y, X) int and `images` is a dict of role -> (T, Y, X)
    float arrays for "nuclear_infection", "cdt1", "slbp", "geminin".
    """
    nucleus_positions = [(40, 40), (40, 160), (160, 40), (160, 160)]
    infected_position = (100, 100)
    half_nucleus = 4  # 8x8 nuclei
    half_infected = 30  # 60x60 = 3600px infected blob -- deliberately well above the default
    # max_nuclear_candidate_area_px (2000), so the demo actually exercises
    # the coarse population split into a separate "large_candidate" bucket
    # rather than everything landing in "nuclear_candidate" by accident.

    labels = np.zeros((n_frames, size, size), dtype=np.int32)
    images = {
        role: np.full((n_frames, size, size), 20.0, dtype=np.float64)  # background level
        for role in ("nuclear_infection", "cdt1", "slbp", "geminin")
    }

    next_label = 1
    nucleus_labels = []
    for _ in nucleus_positions:
        nucleus_labels.append(next_label)
        next_label += 1
    infected_label = next_label

    phase_offsets = np.linspace(0, 2 * np.pi, len(nucleus_positions), endpoint=False)

    for t in range(n_frames):
        # infected cell: large, diffuse, strong -- only in the green/nuclear_infection channel
        iy, ix = infected_position
        labels[t, iy - half_infected : iy + half_infected, ix - half_infected : ix + half_infected] = infected_label
        images["nuclear_infection"][
            t, iy - half_infected : iy + half_infected, ix - half_infected : ix + half_infected
        ] = 500.0

        for i, (cy, cx) in enumerate(nucleus_positions):
            lbl = nucleus_labels[i]
            forced_mitosis = i == 0 and t in (10, 11)

            phase = phase_offsets[i] + 2 * np.pi * t / n_frames
            cdt1_val = 20 + 200 * max(0.0, np.cos(phase))
            slbp_val = 20 + 200 * max(0.0, np.cos(phase - 2 * np.pi / 3))
            geminin_val = 20 + 200 * max(0.0, np.cos(phase - 4 * np.pi / 3))

            if forced_mitosis:
                # elongated, concentrated -- condensed chromatin. Geminin is
                # forced high too: real M-phase biology has Geminin still
                # elevated (it's degraded only at mitotic exit), which is
                # exactly the confirmatory signal classify_cell_cycle
                # requires alongside condensation -- an earlier version of
                # this demo left Geminin on its ordinary sinusoidal value
                # here, which happened to sit right at the classification
                # threshold and produced "unknown" instead of "M" despite
                # condensation being correctly detected.
                geminin_val = 220.0
                sl = (slice(cy - 2, cy + 2), slice(cx - 10, cx + 10))
                labels[t][sl] = lbl
                images["nuclear_infection"][t, cy - 1 : cy + 1, cx - 2 : cx + 2] = 800.0
                images["nuclear_infection"][t][sl] = np.maximum(images["nuclear_infection"][t][sl], 30.0)
            else:
                sl = (slice(cy - half_nucleus, cy + half_nucleus), slice(cx - half_nucleus, cx + half_nucleus))
                labels[t][sl] = lbl
                images["nuclear_infection"][t][sl] = 150.0

            images["cdt1"][t][sl] = cdt1_val
            images["slbp"][t][sl] = slbp_val
            images["geminin"][t][sl] = geminin_val

    return labels, images, pixel_size_um, frame_interval_min
