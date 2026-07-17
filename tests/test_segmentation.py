"""Pure-numpy tests for regionprops_from_labels: no cellpose/PyTorch needed."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.config import PipelineConfig  # noqa: E402
from fucci_vme_pipeline.segmentation import regionprops_from_labels  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def test_circle_is_low_eccentricity():
    size = 41
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size // 2
    r = 10
    circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    labels = circle.astype(np.int32)
    intensity = np.where(circle, 100, 0).astype(np.float64)

    config = PipelineConfig(min_object_area_px=5)
    props = regionprops_from_labels(labels, intensity, config)
    assert len(props) == 1, len(props)
    p = props[0]
    assert p.eccentricity < 0.2, f"expected round object, got eccentricity={p.eccentricity}"


def test_elongated_rectangle_is_high_eccentricity():
    size = 61
    labels = np.zeros((size, size), dtype=np.int32)
    labels[25:35, 5:55] = 1  # 10 x 50: long, thin -- like a condensed mitotic plate
    intensity = np.where(labels == 1, 100, 0).astype(np.float64)

    config = PipelineConfig(min_object_area_px=5)
    props = regionprops_from_labels(labels, intensity, config)
    assert len(props) == 1
    p = props[0]
    assert p.eccentricity > 0.9, f"expected elongated object, got eccentricity={p.eccentricity}"


def test_concentrated_intensity_scores_higher_than_uniform():
    size = 41
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size // 2
    r = 15
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    labels = disk.astype(np.int32)

    uniform_intensity = np.where(disk, 100, 0).astype(np.float64)

    concentrated_intensity = np.zeros((size, size), dtype=np.float64)
    inner = (yy - cy) ** 2 + (xx - cx) ** 2 <= 3 * 3
    concentrated_intensity[disk] = 10.0
    concentrated_intensity[inner] = 500.0

    config = PipelineConfig(min_object_area_px=5)
    uniform_props = regionprops_from_labels(labels, uniform_intensity, config)[0]
    concentrated_props = regionprops_from_labels(labels, concentrated_intensity, config)[0]

    assert concentrated_props.condensation_score > uniform_props.condensation_score, (
        concentrated_props.condensation_score,
        uniform_props.condensation_score,
    )
    # a uniformly-filled disk should score near the low end of the scale
    assert uniform_props.condensation_score < 0.3, uniform_props.condensation_score


def test_condensation_score_on_elongated_shape_with_concentrated_intensity():
    # the real mitotic case: an elongated (high-eccentricity) chromatin mass
    # whose intensity is genuinely concentrated within its own footprint --
    # must NOT score 0 just because it isn't disk-shaped (regression test
    # for the disk-normalization bug found via full-pipeline testing).
    size = 41
    labels = np.zeros((size, size), dtype=np.int32)
    labels[19:21, 5:35] = 1  # thin, long: 2 x 30
    uniform_intensity = np.where(labels == 1, 100.0, 0.0)

    concentrated_intensity = np.full((size, size), 0.0)
    concentrated_intensity[labels == 1] = 20.0
    concentrated_intensity[19:21, 18:22] = 500.0  # bright core near the center of the long axis

    config = PipelineConfig(min_object_area_px=5)
    uniform_props = regionprops_from_labels(labels, uniform_intensity, config)[0]
    concentrated_props = regionprops_from_labels(labels, concentrated_intensity, config)[0]

    assert uniform_props.eccentricity > 0.9  # confirm this shape is genuinely elongated
    assert uniform_props.condensation_score < 0.1, (
        "a uniformly-lit elongated object must not be called condensed just for being elongated",
        uniform_props.condensation_score,
    )
    assert concentrated_props.condensation_score > 0.3, (
        "intensity genuinely concentrated within an elongated footprint must score meaningfully above 0",
        concentrated_props.condensation_score,
    )


def test_condensation_score_on_ring_pattern():
    # a real mitotic pattern: condensed chromatin forming a ring partway
    # between the object's center and edge (e.g. spindle viewed end-on),
    # with a dim center and dim outer band -- a center-relative spatial
    # metric would score this as diffuse (bright pixels sit away from the
    # centroid), but the material genuinely is concentrated, just not in
    # a central blob. Regression test for exactly this failure mode.
    size = 41
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size // 2
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= 18**2
    labels = disk.astype(np.int32)

    uniform_intensity = np.where(disk, 80.0, 0.0)

    ring_intensity = np.full((size, size), 20.0)
    ring = ((yy - cy) ** 2 + (xx - cx) ** 2 >= 9**2) & ((yy - cy) ** 2 + (xx - cx) ** 2 <= 11**2)
    ring_intensity[ring] = 400.0
    ring_intensity = np.where(disk, ring_intensity, 0.0)

    config = PipelineConfig(min_object_area_px=5)
    uniform_props = regionprops_from_labels(labels, uniform_intensity, config)[0]
    ring_props = regionprops_from_labels(labels, ring_intensity, config)[0]

    assert uniform_props.condensation_score < 0.1, uniform_props.condensation_score
    assert ring_props.condensation_score > 0.3, (
        "a ring of concentrated intensity must score as condensed, not diffuse, "
        "even though its bright pixels sit away from the object's center",
        ring_props.condensation_score,
    )


def test_condensation_score_on_bilobed_pattern():
    # the other real pattern described: condensed chromatin appearing as
    # two separate bright regions (e.g. "two rectangles") within one
    # connected segmented object, with a dim gap between them at the
    # object's center -- same failure mode as the ring, opposite geometry.
    size = 41
    labels = np.zeros((size, size), dtype=np.int32)
    labels[15:26, 2:39] = 1  # one wide connected object

    uniform_intensity = np.where(labels == 1, 80.0, 0.0)

    bilobed_intensity = np.full((size, size), 20.0)
    bilobed_intensity[16:25, 4:14] = 400.0  # left rectangle
    bilobed_intensity[16:25, 27:37] = 400.0  # right rectangle
    bilobed_intensity = np.where(labels == 1, bilobed_intensity, 0.0)

    config = PipelineConfig(min_object_area_px=5)
    uniform_props = regionprops_from_labels(labels, uniform_intensity, config)[0]
    bilobed_props = regionprops_from_labels(labels, bilobed_intensity, config)[0]

    assert uniform_props.condensation_score < 0.1, uniform_props.condensation_score
    assert bilobed_props.condensation_score > 0.3, (
        "two separated bright regions flanking a dim center must score as "
        "condensed, not diffuse",
        bilobed_props.condensation_score,
    )


def test_min_area_filter_drops_debris():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[5:7, 5:7] = 1  # 4px speck
    labels[15:25, 15:25] = 2  # 100px real object
    intensity = np.where(labels > 0, 100, 0).astype(np.float64)

    config = PipelineConfig(min_object_area_px=20)
    props = regionprops_from_labels(labels, intensity, config)
    assert len(props) == 1, [p.label for p in props]
    assert props[0].label == 2


def test_population_split_by_area():
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[5:15, 5:15] = 1  # 100px: small, nuclear-scale
    labels[20:80, 20:80] = 2  # 3600px: large, whole-cell-scale
    intensity = np.where(labels > 0, 100, 0).astype(np.float64)

    config = PipelineConfig(min_object_area_px=5, max_nuclear_candidate_area_px=2000)
    props = {p.label: p for p in regionprops_from_labels(labels, intensity, config)}
    assert props[1].population == "nuclear_candidate", props[1].population
    assert props[2].population == "large_candidate", props[2].population


if __name__ == "__main__":
    _run("round object has low eccentricity", test_circle_is_low_eccentricity)
    _run("elongated rectangle has high eccentricity", test_elongated_rectangle_is_high_eccentricity)
    _run("concentrated intensity scores higher than uniform", test_concentrated_intensity_scores_higher_than_uniform)
    _run("condensation score on elongated shape with concentrated intensity", test_condensation_score_on_elongated_shape_with_concentrated_intensity)
    _run("condensation score on ring pattern", test_condensation_score_on_ring_pattern)
    _run("condensation score on bilobed pattern", test_condensation_score_on_bilobed_pattern)
    _run("min_object_area_px filters out debris", test_min_area_filter_drops_debris)
    _run("population split by coarse area pre-filter", test_population_split_by_area)
    print("\nAll segmentation tests passed.")
