"""Pure helpers for converting MuJoCo geom segmentations to YOLO labels."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np


BBox = tuple[int, int, int, int]
YoloRow = tuple[int, float, float, float, float]


def bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """Return intersection over union for two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def split_for_index(index: int, total: int, val_fraction: float) -> str:
    """Split complete scene seeds, reserving the final seeds for validation."""
    if total < 2 or not 0 <= index < total:
        raise ValueError("index must address a collection containing at least two items")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    val_count = max(1, min(total - 1, round(total * val_fraction)))
    return "val" if index >= total - val_count else "train"


def mask_from_geom_ids(segmentation: np.ndarray, geom_ids: Collection[int]) -> np.ndarray:
    """Return pixels whose MuJoCo geom ID belongs to ``geom_ids``."""
    array = np.asarray(segmentation)
    if array.ndim != 3 or array.shape[-1] < 2:
        raise ValueError("segmentation must have at least two channels")
    return np.isin(array[..., 1], tuple(geom_ids))


def bbox_from_mask(mask: np.ndarray, min_visible_pixels: int = 1) -> BBox | None:
    """Return an exclusive ``(x1, y1, x2, y2)`` box for a visible mask."""
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if min_visible_pixels < 1:
        raise ValueError("min_visible_pixels must be positive")
    ys, xs = np.nonzero(array)
    if xs.size < min_visible_pixels:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def bbox_to_yolo(
    class_id: int,
    bbox: BBox,
    image_width: int,
    image_height: int,
) -> YoloRow:
    """Convert an exclusive pixel box to a normalized YOLO row."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError("bbox must lie inside the image")
    return (
        int(class_id),
        ((x1 + x2) / 2.0) / image_width,
        ((y1 + y2) / 2.0) / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )
