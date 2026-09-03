"""
src/preprocessing/mouth_roi.py
───────────────────────────────
Stable, padded mouth-region crop suitable for feeding into a lip-sync model.

Coordinate scheme
─────────────────
MediaPipe FaceMesh (478 landmarks) is used by default.
The mouth/lip region is defined by the union of MOUTH_OUTER_INDICES and
MOUTH_INNER_INDICES below.  These correspond roughly to dlib 68-point
landmarks 48–67 (the inner and outer lip contours).

  dlib 68-pt  │  MediaPipe 478-pt (approx mapping)
  ────────────┼──────────────────────────────────────
  48  (L corner) │  61
  54  (R corner) │  291
  51  (top mid)  │  0
  57  (bot mid)  │  17
  62  (inner L)  │  78
  66  (inner R)  │  308

Padding
───────
A 30% pad is applied to each side of the landmark bounding box.  This gives
the generation model:
  - Chin context (helps with chin-shadow and jaw articulation)
  - Cheek context on both sides (needed for natural blending seams)
  - Philtrum / upper-lip context above the lip line

The padded bbox is clamped to image bounds before slicing.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Landmark index sets ───────────────────────────────────────────────────────

# Outer lip boundary (20 vertices)
MOUTH_OUTER_INDICES: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
)
# Inner lip boundary (20 vertices)
MOUTH_INNER_INDICES: tuple[int, ...] = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191,
)
# All mouth vertices (sorted, unique) — used for bounding-box computation
MOUTH_ALL_INDICES: tuple[int, ...] = tuple(
    sorted(set(MOUTH_OUTER_INDICES) | set(MOUTH_INNER_INDICES))
)

# dlib 68-point mouth indices (landmarks 48–67 are the outer + inner lip contour)
DLIB68_MOUTH_INDICES: tuple[int, ...] = tuple(range(48, 68))


# ── Public API ────────────────────────────────────────────────────────────────


def extract_mouth_roi(
    frame: np.ndarray,
    landmarks: list[tuple[float, float]],
    padding: float = 0.30,
    index_scheme: str = "mediapipe",
    target_size: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop a padded mouth-region image from *frame* using face *landmarks*.

    Args:
        frame:        BGR image array (H, W, 3) — the source video frame.
        landmarks:    List of ``(x, y)`` pixel-space coordinates, one per
                      landmark.  Length must be ≥ 478 for ``"mediapipe"`` scheme
                      or ≥ 68 for ``"dlib68"`` scheme.
        padding:      Fractional padding applied to each side of the tight
                      landmark bounding box.  0.30 → 30% on every side.
        index_scheme: ``"mediapipe"`` (478-point) or ``"dlib68"`` (68-point).
        target_size:  If given as ``(width, height)``, the crop is resized to
                      this size before returning.  Useful for fixed-size model
                      inputs.

    Returns:
        ``(crop, bbox)`` where:
        - ``crop`` is a BGR numpy array of the padded mouth region.
        - ``bbox`` is the padded ``(x, y, w, h)`` rectangle *in the original
          frame coordinate space* (clamped to image bounds).

    Raises:
        ValueError:  If *landmarks* is empty, *index_scheme* is unknown, or
                     the computed bounding box is degenerate (zero area after
                     clamping).
    """
    if not landmarks:
        raise ValueError("landmarks list is empty — cannot extract mouth ROI")

    h_img, w_img = frame.shape[:2]

    # ── Select mouth landmark indices ──────────────────────────────────────────
    if index_scheme == "mediapipe":
        mouth_indices = MOUTH_ALL_INDICES
    elif index_scheme == "dlib68":
        mouth_indices = DLIB68_MOUTH_INDICES
    else:
        raise ValueError(
            f"Unknown index_scheme '{index_scheme}'. Choose 'mediapipe' or 'dlib68'."
        )

    # Guard against out-of-range indices
    valid_indices = [i for i in mouth_indices if i < len(landmarks)]
    if not valid_indices:
        raise ValueError(
            f"No valid mouth landmark indices found. "
            f"landmarks has {len(landmarks)} entries; "
            f"mouth scheme '{index_scheme}' expects indices up to "
            f"{max(mouth_indices)}."
        )

    # ── Tight bounding box around mouth landmarks ──────────────────────────────
    mouth_pts = [landmarks[i] for i in valid_indices]
    xs = [p[0] for p in mouth_pts]
    ys = [p[1] for p in mouth_pts]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    mouth_w = x_max - x_min
    mouth_h = y_max - y_min

    if mouth_w <= 0 or mouth_h <= 0:
        raise ValueError(
            f"Degenerate mouth bounding box ({mouth_w:.1f}×{mouth_h:.1f}). "
            "Check that landmarks are in pixel-space (not normalised [0,1])."
        )

    # ── Apply padding (30% on each side) ──────────────────────────────────────
    pad_x = padding * mouth_w
    pad_y = padding * mouth_h

    x1 = int(x_min - pad_x)
    y1 = int(y_min - pad_y)
    x2 = int(x_max + pad_x)
    y2 = int(y_max + pad_y)

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(
            f"Padded mouth bbox is entirely outside the image ({x1},{y1},{x2},{y2}) "
            f"for image size {w_img}×{h_img}."
        )

    # ── Crop ──────────────────────────────────────────────────────────────────
    crop = frame[y1:y2, x1:x2].copy()

    if target_size is not None:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    bbox = (x1, y1, crop_w, crop_h)

    logger.debug(
        "Mouth ROI: tight (%.0f,%.0f)→(%.0f,%.0f) | padded %s",
        x_min, y_min, x_max, y_max, bbox,
    )

    return crop, bbox


def draw_mouth_landmarks(
    frame: np.ndarray,
    landmarks: list[tuple[float, float]],
    index_scheme: str = "mediapipe",
    color_outer: tuple[int, int, int] = (0, 255, 0),
    color_inner: tuple[int, int, int] = (0, 200, 255),
    radius: int = 2,
) -> np.ndarray:
    """
    Draw mouth landmarks on a copy of *frame* for debug visualisation.

    Returns the annotated frame (does not modify in-place).
    """
    out = frame.copy()

    if index_scheme == "mediapipe":
        for i in MOUTH_OUTER_INDICES:
            if i < len(landmarks):
                pt = (int(landmarks[i][0]), int(landmarks[i][1]))
                cv2.circle(out, pt, radius, color_outer, -1)
        for i in MOUTH_INNER_INDICES:
            if i < len(landmarks):
                pt = (int(landmarks[i][0]), int(landmarks[i][1]))
                cv2.circle(out, pt, radius, color_inner, -1)
    elif index_scheme == "dlib68":
        for i in DLIB68_MOUTH_INDICES:
            if i < len(landmarks):
                pt = (int(landmarks[i][0]), int(landmarks[i][1]))
                cv2.circle(out, pt, radius, color_outer, -1)

    return out
