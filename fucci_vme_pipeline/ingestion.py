"""Stage 1: ingestion (ND2 or channel-split grayscale TIFF).

Reads a single-channel, already max-intensity-projected (over Z) stack per
acquisition channel, i.e. one file per fluorophore, each `(T, Y, X)`. Pixel
size and frame interval are read from the file's own metadata where
possible rather than hardcoded, since these differ across experiments.

TIFF has a real limitation ND2 doesn't: plain grayscale TIFF exports often
carry NO calibration metadata at all (confirmed by direct testing --
`tifffile.TiffFile(...).imagej_metadata` is simply `None` for a bare
grayscale export). Only ImageJ-saved hyperstacks reliably embed pixel size
(via the XResolution tag + an `imagej_metadata['unit']` string) and frame
interval (`imagej_metadata['finterval']`, in seconds). `read_tiff_channel`
auto-detects these when present, but if they're missing, it fails loudly
and requires them as explicit arguments rather than guessing -- a wrong
silent pixel size or frame interval would corrupt every downstream stage
(exposure radii, phase-transition timing, m_phase_duration).

The `nd2`/`tifffile` packages are imported lazily so that `--demo` mode
(see `demo.py`) never needs either installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig


@dataclass
class ChannelStack:
    """One channel's imaging data plus its calibration and identity."""

    role: str
    wavelength_nm: int | None
    data: np.ndarray  # (T, Y, X)
    pixel_size_um: float
    frame_interval_min: float
    source_path: str | None = None

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]


def _find_frame_interval_ms(experiment) -> float | None:
    """Search ND2 experiment loops for the time-loop period."""
    for loop in experiment:
        loop_type = getattr(loop, "type", None)
        if loop_type in ("TimeLoop", "NETimeLoop"):
            params = getattr(loop, "parameters", None)
            period_ms = getattr(params, "periodMs", None)
            if period_ms is not None:
                return float(period_ms)
    return None


def _nearest_wavelength(wavelength_nm: float, known: list[int], tol_nm: float = 10.0) -> int | None:
    for candidate in known:
        if abs(candidate - wavelength_nm) <= tol_nm:
            return candidate
    return None


def _wavelength_from_channel_name(name: str, known: list[int]) -> int | None:
    """Laser-line-style channel names (e.g. "488", "Laser 488", "GFP 488")
    often encode the excitation wavelength directly -- try that before
    falling back to the excitationLambdaNm metadata field (ND2) or
    requiring an explicit override (TIFF, which has no such field at all).
    """
    found = [int(n) for n in re.findall(r"\d{3}", name) if int(n) in known]
    if len(found) == 1:
        return found[0]
    return None


def _resolve_wavelength(channel_meta, config: PipelineConfig, path_name: str) -> int:
    """Identify which configured laser line this channel corresponds to.

    Config keys (405/488/561/640) are excitation laser lines, matching this
    lab's file-naming convention -- NOT emission peaks, which for e.g.
    mNeonGreen (~517nm) or mScarlet3 (~594nm) would not match those numbers
    at all. Tries channel name first (usually encodes the laser line
    directly in NIS-Elements exports), then falls back to excitationLambdaNm.
    """
    known = list(config.channel_map)
    name = getattr(channel_meta, "name", None)
    if name:
        matched = _wavelength_from_channel_name(name, known)
        if matched is not None:
            return matched

    excitation = getattr(channel_meta, "excitationLambdaNm", None)
    if excitation is not None:
        matched = _nearest_wavelength(excitation, known)
        if matched is not None:
            return matched

    raise ValueError(
        f"{path_name}: could not resolve channel identity from name="
        f"{name!r} or excitationLambdaNm={excitation!r} against known "
        f"laser lines {sorted(known)}. Pass wavelength_override explicitly."
    )


def read_nd2_channel(
    path: str | Path,
    config: PipelineConfig,
    wavelength_override: int | None = None,
) -> ChannelStack:
    """Load one channel-split ND2 file as a calibrated `(T, Y, X)` stack.

    Raises `ValueError` with a diagnostic message (rather than silently
    misinterpreting axes) if the file still contains a Z or multi-channel
    axis, since this pipeline expects projection/splitting to happen before
    the file reaches it.
    """
    import nd2  # lazy: only needed for real data, not --demo

    path = Path(path)
    with nd2.ND2File(str(path)) as f:
        sizes = dict(f.sizes)
        if sizes.get("Z", 1) > 1:
            raise ValueError(
                f"{path.name}: file still has a Z axis (sizes={sizes}). "
                "This pipeline expects an already max-intensity-projected "
                "stack with no Z dimension -- project across Z before export."
            )
        if sizes.get("C", 1) > 1:
            raise ValueError(
                f"{path.name}: file has {sizes.get('C')} channels (sizes={sizes}); "
                "this pipeline expects one channel per file. Split channels "
                "before export (e.g. Fiji Image > Color > Split Channels)."
            )

        arr = np.asarray(f.asarray())
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(
                f"{path.name}: expected (T, Y, X) after squeezing singleton "
                f"axes, got shape {arr.shape} from sizes={sizes}. Inspect "
                "this file's axis layout manually before proceeding."
            )

        voxel = f.voxel_size(channel=0)
        if voxel.x <= 0 or voxel.y <= 0:
            raise ValueError(f"{path.name}: invalid pixel size from metadata: {voxel}")
        if abs(voxel.x - voxel.y) > 0.01 * max(voxel.x, voxel.y):
            raise ValueError(
                f"{path.name}: non-square pixels (x={voxel.x}, y={voxel.y} um) "
                "-- this pipeline assumes square pixels; adjust before proceeding."
            )
        pixel_size_um = float(voxel.x)

        period_ms = _find_frame_interval_ms(f.experiment)
        if period_ms is None:
            raise ValueError(
                f"{path.name}: could not find a TimeLoop/NETimeLoop period in "
                "this file's experiment metadata. Pass the frame interval "
                "manually if this file genuinely lacks it."
            )
        frame_interval_min = period_ms / 1000.0 / 60.0

        wavelength_nm = wavelength_override
        if wavelength_nm is None:
            channels = f.metadata.channels if f.metadata is not None else None
            if not channels:
                raise ValueError(
                    f"{path.name}: no channel metadata found. Pass "
                    "wavelength_override explicitly for this file."
                )
            wavelength_nm = _resolve_wavelength(channels[0].channel, config, path.name)

        role = config.channel_role(wavelength_nm).role

    return ChannelStack(
        role=role,
        wavelength_nm=wavelength_nm,
        data=arr,
        pixel_size_um=pixel_size_um,
        frame_interval_min=frame_interval_min,
        source_path=str(path),
    )


_LENGTH_UNIT_STRINGS = {"um", "micron", "microns", "µm", "u"}


def _pixel_size_from_imagej_tags(first_page, imagej_meta: dict, path_name: str) -> float:
    unit = imagej_meta.get("unit")
    if unit not in _LENGTH_UNIT_STRINGS:
        raise ValueError(
            f"{path_name}: no usable pixel-size unit in ImageJ metadata "
            f"(found unit={unit!r}). Pass pixel_size_um explicitly."
        )

    x_tag = first_page.tags.get("XResolution")
    y_tag = first_page.tags.get("YResolution")
    if x_tag is None:
        raise ValueError(f"{path_name}: no XResolution tag found. Pass pixel_size_um explicitly.")

    x_num, x_den = x_tag.value
    if x_num == 0:
        raise ValueError(f"{path_name}: invalid XResolution tag value {x_tag.value}.")
    pixel_size_x_um = x_den / x_num

    if y_tag is not None:
        y_num, y_den = y_tag.value
        if y_num > 0:
            pixel_size_y_um = y_den / y_num
            if abs(pixel_size_x_um - pixel_size_y_um) > 0.01 * max(pixel_size_x_um, pixel_size_y_um):
                raise ValueError(
                    f"{path_name}: non-square pixels (x={pixel_size_x_um}, "
                    f"y={pixel_size_y_um} {unit}) -- this pipeline assumes "
                    "square pixels; adjust before proceeding."
                )

    return float(pixel_size_x_um)


def read_tiff_channel(
    path: str | Path,
    config: PipelineConfig,
    pixel_size_um: float | None = None,
    frame_interval_min: float | None = None,
    wavelength_override: int | None = None,
) -> ChannelStack:
    """Load one channel-split grayscale TIFF stack as a calibrated
    `(T, Y, X)` stack.

    Auto-detects pixel size and frame interval from ImageJ hyperstack
    metadata when present; otherwise `pixel_size_um`/`frame_interval_min`
    must be supplied explicitly (see module docstring for why this isn't
    guessed). Channel identity has no metadata equivalent to ND2's
    per-channel record at all, so it's always resolved from the filename
    (e.g. `..._488.tif`) unless `wavelength_override` is given.
    """
    import tifffile  # lazy: only needed for real data, not --demo

    path = Path(path)
    with tifffile.TiffFile(str(path)) as tf:
        arr = np.asarray(tf.asarray())
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(
                f"{path.name}: expected (T, Y, X) after squeezing singleton "
                f"axes, got shape {arr.shape}. If this is a multi-channel or "
                "un-projected z-stack TIFF, split/project it before ingestion."
            )

        resolved_pixel_size_um = pixel_size_um
        resolved_frame_interval_min = frame_interval_min
        imagej_meta = tf.imagej_metadata or {}

        if resolved_pixel_size_um is None:
            resolved_pixel_size_um = _pixel_size_from_imagej_tags(tf.pages[0], imagej_meta, path.name)
        if resolved_frame_interval_min is None:
            finterval_s = imagej_meta.get("finterval")
            if finterval_s is None:
                raise ValueError(
                    f"{path.name}: no ImageJ 'finterval' metadata found "
                    "(this file may not be an ImageJ hyperstack). Pass "
                    "frame_interval_min explicitly."
                )
            resolved_frame_interval_min = float(finterval_s) / 60.0

    wavelength_nm = wavelength_override
    if wavelength_nm is None:
        wavelength_nm = _wavelength_from_channel_name(path.stem, list(config.channel_map))
        if wavelength_nm is None:
            raise ValueError(
                f"{path.name}: could not resolve channel identity from the "
                f"filename against known laser lines {sorted(config.channel_map)}. "
                "Pass wavelength_override explicitly."
            )

    role = config.channel_role(wavelength_nm).role
    return ChannelStack(
        role=role,
        wavelength_nm=wavelength_nm,
        data=arr,
        pixel_size_um=resolved_pixel_size_um,
        frame_interval_min=resolved_frame_interval_min,
        source_path=str(path),
    )
