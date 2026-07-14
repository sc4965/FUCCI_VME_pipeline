"""Tests for read_tiff_channel against real tifffile-written files (not
mocked) -- writes actual ImageJ-hyperstack and plain-grayscale TIFFs to a
temp dir and reads them back, the same way test_ingestion.py exercises the
real nd2 API via a mock built from its introspected structure.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.config import PipelineConfig  # noqa: E402
from fucci_vme_pipeline.ingestion import read_tiff_channel  # noqa: E402

TMP_DIR = Path(tempfile.mkdtemp(prefix="fucci_tiff_test_"))


def _run(label, fn):
    fn()
    print(f"[ OK ] {label}")


def _write_imagej_tiff(filename: str, pixel_size_um: float = 0.65, frame_interval_min: float = 6.0, unit: str = "um"):
    import tifffile

    arr = np.random.default_rng(0).integers(50, 150, size=(10, 32, 32), dtype=np.uint16)
    path = TMP_DIR / filename
    tifffile.imwrite(
        path,
        arr,
        imagej=True,
        resolution=(1 / pixel_size_um, 1 / pixel_size_um),
        metadata={"axes": "TYX", "unit": unit, "finterval": frame_interval_min * 60},
    )
    return path


def _write_plain_tiff(filename: str):
    import tifffile

    arr = np.random.default_rng(1).integers(50, 150, size=(5, 32, 32), dtype=np.uint16)
    path = TMP_DIR / filename
    tifffile.imwrite(path, arr)  # no ImageJ metadata at all
    return path


def test_autodetects_pixel_size_and_frame_interval_from_imagej_metadata():
    path = _write_imagej_tiff("sample_488.tif", pixel_size_um=0.65, frame_interval_min=6.0)
    config = PipelineConfig()
    stack = read_tiff_channel(path, config)
    assert stack.data.shape == (10, 32, 32)
    assert abs(stack.pixel_size_um - 0.65) < 1e-6
    assert abs(stack.frame_interval_min - 6.0) < 1e-6
    assert stack.role == "nuclear_infection"


def test_channel_identity_resolved_from_filename():
    path = _write_imagej_tiff("experiment1_561.tif")
    config = PipelineConfig()
    stack = read_tiff_channel(path, config)
    assert stack.role == "cdt1"


def test_plain_tiff_requires_explicit_calibration():
    path = _write_plain_tiff("plain_640.tif")
    config = PipelineConfig()
    try:
        read_tiff_channel(path, config)
        raise AssertionError("expected ValueError for missing calibration metadata")
    except ValueError as e:
        assert "pixel_size_um" in str(e) or "XResolution" in str(e), str(e)


def test_plain_tiff_works_with_explicit_overrides():
    path = _write_plain_tiff("plain_640_b.tif")
    config = PipelineConfig()
    stack = read_tiff_channel(path, config, pixel_size_um=0.5, frame_interval_min=5.0)
    assert stack.data.shape == (5, 32, 32)
    assert stack.pixel_size_um == 0.5
    assert stack.frame_interval_min == 5.0
    assert stack.role == "geminin"


def test_unresolvable_filename_requires_wavelength_override():
    path = _write_plain_tiff("no_wavelength_here.tif")
    config = PipelineConfig()
    try:
        read_tiff_channel(path, config, pixel_size_um=0.5, frame_interval_min=5.0)
        raise AssertionError("expected ValueError for unresolvable channel identity")
    except ValueError as e:
        assert "could not resolve channel identity" in str(e), str(e)

    stack = read_tiff_channel(path, config, pixel_size_um=0.5, frame_interval_min=5.0, wavelength_override=405)
    assert stack.role == "slbp"


def test_non_micron_unit_requires_explicit_pixel_size():
    path = _write_imagej_tiff("sample_405_inches.tif", unit="inch")
    config = PipelineConfig()
    try:
        read_tiff_channel(path, config)
        raise AssertionError("expected ValueError for non-micron unit")
    except ValueError as e:
        assert "pixel_size_um" in str(e), str(e)


if __name__ == "__main__":
    try:
        _run("auto-detects pixel size and frame interval from ImageJ metadata", test_autodetects_pixel_size_and_frame_interval_from_imagej_metadata)
        _run("channel identity resolved from filename", test_channel_identity_resolved_from_filename)
        _run("plain TIFF with no metadata requires explicit calibration", test_plain_tiff_requires_explicit_calibration)
        _run("plain TIFF works when given explicit overrides", test_plain_tiff_works_with_explicit_overrides)
        _run("unresolvable filename requires wavelength_override", test_unresolvable_filename_requires_wavelength_override)
        _run("non-micron unit requires explicit pixel_size_um", test_non_micron_unit_requires_explicit_pixel_size)
        print("\nAll TIFF ingestion tests passed.")
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
