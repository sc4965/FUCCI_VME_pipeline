"""Tests for crop_around (pure numpy, no display backend needed)."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.annotation_review import crop_around  # noqa: E402


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def test_crop_centered_well_within_bounds():
    image = np.arange(400 * 400).reshape(400, 400)
    crop, x0, y0 = crop_around(image, x=200, y=200, half_size=50)
    assert crop.shape == (100, 100)
    assert x0 == 150
    assert y0 == 150
    assert crop[0, 0] == image[150, 150]


def test_crop_clips_at_image_edge_without_error():
    image = np.arange(400 * 400).reshape(400, 400)
    crop, x0, y0 = crop_around(image, x=5, y=5, half_size=50)
    assert x0 == 0
    assert y0 == 0
    assert crop.shape == (55, 55)  # clipped, not the full 100x100


def test_crop_clips_at_bottom_right_edge():
    image = np.arange(400 * 400).reshape(400, 400)
    crop, x0, y0 = crop_around(image, x=395, y=395, half_size=50)
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100
    assert x0 <= 345
    assert y0 <= 345


if __name__ == "__main__":
    _run("crop centered well within bounds", test_crop_centered_well_within_bounds)
    _run("crop clips at top-left image edge without error", test_crop_clips_at_image_edge_without_error)
    _run("crop clips at bottom-right image edge", test_crop_clips_at_bottom_right_edge)
    print("\nAll annotation-review tests passed.")
