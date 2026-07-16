"""Visual review of annotation-to-object matches, directly against the raw
image data -- for exactly the situation where match-distance statistics
alone can't tell a genuine match from a wrong-neighbor mismatch in a dense
field, and there's no way to just go re-check the original acquisition
software.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def crop_around(image: np.ndarray, x: float, y: float, half_size: int = 100) -> tuple[np.ndarray, int, int]:
    """Crops a `2*half_size` square around (x, y) from a single 2D frame.

    Returns (crop, x_offset, y_offset) -- the offsets are needed to convert
    original-image coordinates into the crop's own local coordinate frame
    for plotting markers on it.
    """
    height, width = image.shape
    x0 = max(0, int(x) - half_size)
    x1 = min(width, int(x) + half_size)
    y0 = max(0, int(y) - half_size)
    y1 = min(height, int(y) + half_size)
    return image[y0:y1, x0:x1], x0, y0


def show_annotation_match(
    image: np.ndarray,
    ann_x: float,
    ann_y: float,
    match_x: float | None = None,
    match_y: float | None = None,
    half_size: int = 100,
    title: str = "",
):
    """Plots a cropped region around an annotated point, marking the
    original click (red X) and, if given, the pipeline's matched object
    centroid (cyan circle) -- so a genuine match (markers on the same
    cell) is visually distinguishable from a wrong-neighbor mismatch
    (markers on two different cells) without needing the original
    acquisition software.
    """
    crop, x0, y0 = crop_around(image, ann_x, ann_y, half_size=half_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    vmin, vmax = np.percentile(crop, [1, 99.5])
    ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax)
    ax.plot(ann_x - x0, ann_y - y0, "rx", markersize=14, markeredgewidth=3, label="your annotation")
    if match_x is not None and match_y is not None:
        ax.plot(
            match_x - x0,
            match_y - y0,
            "o",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor="cyan",
            markeredgewidth=2,
            label="nearest matched object",
        )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    plt.show()
