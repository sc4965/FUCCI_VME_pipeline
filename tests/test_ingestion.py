"""Exercises read_nd2_channel's parsing/validation logic against a fake
nd2.ND2File that mimics the real package's verified API surface (checked by
introspecting nd2==0.11.3), without needing an actual .nd2 file on disk.
"""
from __future__ import annotations

import sys
import types
from collections import namedtuple
from unittest import mock

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/tests/", 1)[0]))

from fucci_vme_pipeline.config import PipelineConfig  # noqa: E402
from fucci_vme_pipeline.ingestion import read_nd2_channel  # noqa: E402

VoxelSize = namedtuple("VoxelSize", "x y z")


def _make_fake_nd2_module(*, sizes, voxel, period_ms, arr_shape, channel_name=None, excitation_nm=None, has_channel_meta=True):
    ChannelMeta = types.SimpleNamespace(name=channel_name, excitationLambdaNm=excitation_nm)
    Channel = types.SimpleNamespace(channel=ChannelMeta)
    Metadata = types.SimpleNamespace(channels=[Channel] if has_channel_meta else None)
    Loop = types.SimpleNamespace(
        type="TimeLoop",
        parameters=types.SimpleNamespace(periodMs=period_ms),
    )

    class FakeND2File:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @property
        def sizes(self):
            return sizes

        def asarray(self):
            return np.zeros(arr_shape, dtype=np.uint16)

        def voxel_size(self, channel=0):
            return voxel

        @property
        def experiment(self):
            return [Loop] if period_ms is not None else []

        @property
        def metadata(self):
            return Metadata

    fake_mod = types.SimpleNamespace(ND2File=FakeND2File)
    return fake_mod


def _run(label, fn):
    try:
        fn()
        print(f"[ OK ] {label}")
    except AssertionError as e:
        print(f"[FAIL] {label}: {e}")
        raise


def test_normal_case():
    fake = _make_fake_nd2_module(
        sizes={"T": 10, "Y": 64, "X": 64},
        voxel=VoxelSize(0.65, 0.65, 1.0),
        channel_name="Laser 488",  # name-based match, the common NIS-Elements case
        excitation_nm=488.0,
        period_ms=360_000.0,  # 6 min
        arr_shape=(10, 64, 64),
    )
    with mock.patch.dict(sys.modules, {"nd2": fake}):
        config = PipelineConfig()
        stack = read_nd2_channel("fake_488.nd2", config)
    assert stack.data.shape == (10, 64, 64), stack.data.shape
    assert stack.role == "nuclear_infection", stack.role
    assert abs(stack.pixel_size_um - 0.65) < 1e-9
    assert abs(stack.frame_interval_min - 6.0) < 1e-9


def test_rejects_z_axis():
    fake = _make_fake_nd2_module(
        sizes={"T": 10, "Z": 3, "Y": 64, "X": 64},
        voxel=VoxelSize(0.65, 0.65, 1.0),
        channel_name="488",
        period_ms=360_000.0,
        arr_shape=(10, 3, 64, 64),
    )
    with mock.patch.dict(sys.modules, {"nd2": fake}):
        config = PipelineConfig()
        try:
            read_nd2_channel("fake_z.nd2", config)
            raise AssertionError("expected ValueError for Z axis")
        except ValueError as e:
            assert "Z axis" in str(e), str(e)


def test_rejects_multichannel():
    fake = _make_fake_nd2_module(
        sizes={"T": 10, "C": 4, "Y": 64, "X": 64},
        voxel=VoxelSize(0.65, 0.65, 1.0),
        channel_name="488",
        period_ms=360_000.0,
        arr_shape=(10, 4, 64, 64),
    )
    with mock.patch.dict(sys.modules, {"nd2": fake}):
        config = PipelineConfig()
        try:
            read_nd2_channel("fake_c.nd2", config)
            raise AssertionError("expected ValueError for multichannel")
        except ValueError as e:
            assert "channels" in str(e), str(e)


def test_rejects_nonsquare_pixels():
    fake = _make_fake_nd2_module(
        sizes={"T": 10, "Y": 64, "X": 64},
        voxel=VoxelSize(0.65, 0.90, 1.0),
        channel_name="488",
        period_ms=360_000.0,
        arr_shape=(10, 64, 64),
    )
    with mock.patch.dict(sys.modules, {"nd2": fake}):
        config = PipelineConfig()
        try:
            read_nd2_channel("fake_px.nd2", config)
            raise AssertionError("expected ValueError for non-square pixels")
        except ValueError as e:
            assert "square pixels" in str(e), str(e)


def test_unknown_wavelength_raises():
    fake = _make_fake_nd2_module(
        sizes={"T": 10, "Y": 64, "X": 64},
        voxel=VoxelSize(0.65, 0.65, 1.0),
        channel_name="Cy5.5",  # no digits matching a known laser line
        excitation_nm=700.0,  # doesn't match any configured role either
        period_ms=360_000.0,
        arr_shape=(10, 64, 64),
    )
    with mock.patch.dict(sys.modules, {"nd2": fake}):
        config = PipelineConfig()
        try:
            read_nd2_channel("fake_unknown.nd2", config)
            raise AssertionError("expected ValueError for unmatched wavelength")
        except ValueError as e:
            assert "could not resolve channel identity" in str(e), str(e)


def test_wavelength_override_bypasses_metadata():
    fake = _make_fake_nd2_module(
        sizes={"T": 5, "Y": 32, "X": 32},
        voxel=VoxelSize(0.11, 0.11, 1.0),
        has_channel_meta=False,  # no channel metadata at all
        period_ms=360_000.0,
        arr_shape=(5, 32, 32),
    )
    with mock.patch.dict(sys.modules, {"nd2": fake}):
        config = PipelineConfig()
        stack = read_nd2_channel("fake_override.nd2", config, wavelength_override=561)
    assert stack.role == "cdt1", stack.role


if __name__ == "__main__":
    _run("normal 3D (T,Y,X) case with metadata match", test_normal_case)
    _run("rejects file with leftover Z axis", test_rejects_z_axis)
    _run("rejects file with multiple channels", test_rejects_multichannel)
    _run("rejects non-square pixels", test_rejects_nonsquare_pixels)
    _run("rejects unmatched emission wavelength", test_unknown_wavelength_raises)
    _run("wavelength_override bypasses missing metadata", test_wavelength_override_bypasses_metadata)
    print("\nAll ingestion tests passed.")
