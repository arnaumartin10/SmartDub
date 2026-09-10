"""
src/postprocessing/compositing.py
──────────────────────────────────
Composite MuseTalk's 256×256 generated face crop back into the original
full-resolution video frame with soft, seamless blending.

Key steps:
  1. **Geometric inversion** — undo the square-pad-and-resize that was applied
     in `coarse_lipsync._square_musetalk_frame` before feeding the crop to
     MuseTalk.  This is the most critical step: the 256×256 output is
     "un-squared" back to the original rectangular bbox dimensions.
  2. **Colour matching** — adjust the crop's colour distribution to the
     surrounding original pixels (delegated to `color_matching.match_color`).
  3. **Soft alpha mask** — build a feathered mask from the jaw/mouth landmarks
     (convex hull + Gaussian blur) so the blend follows the face contour, not
     the bounding-box edge.
  4. **Alpha blend** — merge the colour-matched crop into the original frame.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.postprocessing.color_matching import match_color

logger = logging.getLogger(__name__)

# ── Jaw contour landmark indices (MediaPipe 478-point mesh) ──────────────────
# These trace the jawline from ear to ear, used together with mouth landmarks
# to form the convex hull for the blending mask.
JAW_CONTOUR_INDICES: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454,  # right jawline
    323, 361, 288, 397, 365, 379, 378, 400, 377, 152,  # chin
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234,  # left jawline
)

# Mouth outer contour — same indices as face_tracking.MOUTH_OUTER_INDICES
MOUTH_OUTER_INDICES: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
)

# Combined indices for the blending mask hull
_BLEND_MASK_INDICES: tuple[int, ...] = tuple(
    sorted(set(JAW_CONTOUR_INDICES) | set(MOUTH_OUTER_INDICES))
)


# ── Geometric inversion ─────────────────────────────────────────────────────


def _unsquare_crop(
    generated_crop: np.ndarray,
    original_bbox_w: int,
    original_bbox_h: int,
) -> np.ndarray:
    """Undo the square-pad-and-resize applied by ``_square_musetalk_frame``.

    Geometric inversion walkthrough
    ────────────────────────────────
    The forward path in ``coarse_lipsync._square_musetalk_frame`` does:

      1. Take a rectangular face crop of shape ``(orig_h, orig_w)``
      2. Compute ``side = max(orig_h, orig_w)``
      3. Pad symmetrically to ``(side, side)``:
         - ``pad_top  = (side - orig_h) // 2``
         - ``pad_left = (side - orig_w) // 2``
      4. Resize the padded square to ``(256, 256)``

    To invert this:

      1. Resize the 256×256 generated output back to ``(side, side)``
      2. Compute the same padding offsets
      3. Slice out the ``(orig_h, orig_w)`` region, discarding the padding

    This ensures the generated content is mapped back to exactly the same
    rectangular region it was originally extracted from, without introducing
    additional stretching or aspect-ratio distortion.

    Args:
        generated_crop: The 256×256 BGR crop from MuseTalk.
        original_bbox_w: Width of the original bounding box (before squaring).
        original_bbox_h: Height of the original bounding box (before squaring).

    Returns:
        BGR image resized to ``(original_bbox_h, original_bbox_w)``.
    """
    side = max(original_bbox_h, original_bbox_w)

    # Step 1: Resize 256×256 → (side, side) — inverse of the final resize
    upscaled = cv2.resize(
        generated_crop, (side, side), interpolation=cv2.INTER_LANCZOS4
    )

    # Step 2: Compute the same padding offsets used in the forward path
    pad_top = (side - original_bbox_h) // 2
    pad_left = (side - original_bbox_w) // 2

    # Step 3: Slice out the original rectangular region
    rect = upscaled[
        pad_top : pad_top + original_bbox_h,
        pad_left : pad_left + original_bbox_w,
    ]

    return rect.copy()


# ── Soft alpha mask ──────────────────────────────────────────────────────────


def _build_alpha_mask(
    frame_shape: tuple[int, int],
    landmarks: list[tuple[float, float]],
    bbox: tuple[int, int, int, int],
    blur_kernel: int = 51,
) -> np.ndarray:
    """Build a feathered [0, 1] alpha mask from jaw/mouth landmarks.

    The mask is a convex hull of jaw + mouth-contour landmarks, filled as a
    binary mask and then Gaussian-blurred to produce soft, feathered edges.
    The mask is computed in full-frame coordinates.

    Args:
        frame_shape: ``(height, width)`` of the full-resolution frame.
        landmarks:   478-point pixel-space landmarks from ``track_face``.
        bbox:        ``(x, y, w, h)`` bounding box (used as fallback if
                     landmark indices are out of range).
        blur_kernel: Gaussian blur kernel size (must be odd).  Larger values
                     produce wider feathering.  51 works well for 1080p.

    Returns:
        Float32 array of shape ``(height, width)`` with values in [0, 1].
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Ensure blur kernel is odd
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    # ── Collect landmark points for the convex hull ──────────────────────────
    hull_points = []
    for idx in _BLEND_MASK_INDICES:
        if idx < len(landmarks):
            px, py = landmarks[idx]
            # Clamp to frame bounds
            px = max(0.0, min(float(w - 1), float(px)))
            py = max(0.0, min(float(h - 1), float(py)))
            hull_points.append([int(px), int(py)])

    if len(hull_points) < 3:
        # Fallback: use the bounding box as the mask region
        logger.warning(
            "Only %d hull points found (need ≥3); falling back to bbox mask",
            len(hull_points),
        )
        bx, by, bw, bh = bbox
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w, bx + bw)
        y2 = min(h, by + bh)
        mask[y1:y2, x1:x2] = 255
    else:
        points = np.array(hull_points, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

    # ── Feather the edges ────────────────────────────────────────────────────
    mask_float = cv2.GaussianBlur(
        mask.astype(np.float32), (blur_kernel, blur_kernel), 0
    )
    # Normalize to [0, 1]
    max_val = mask_float.max()
    if max_val > 0:
        mask_float /= max_val

    return mask_float


# ── Public API ───────────────────────────────────────────────────────────────


def composite_frame(
    original_frame: np.ndarray,
    generated_crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarks: list[tuple[float, float]],
    apply_color_match: bool = True,
    blur_kernel: int = 51,
) -> np.ndarray:
    """Composite a MuseTalk 256×256 crop back into the original full frame.

    Pipeline:
      1. Geometric inversion: 256×256 → original bbox rectangle
      2. Colour matching (optional): align to surrounding pixels
      3. Soft alpha mask: convex hull of jaw/mouth landmarks + Gaussian blur
      4. Alpha blend into original frame

    Args:
        original_frame:   Full-resolution BGR frame (e.g. 1080×1920).
        generated_crop:   256×256 BGR crop from MuseTalk.
        bbox:             ``(x, y, w, h)`` bounding box from ``track_face``.
        landmarks:        478-point pixel-space landmarks from ``track_face``.
        apply_color_match: If True, run LAB colour matching before blending.
        blur_kernel:      Gaussian kernel size for mask feathering.

    Returns:
        Full-resolution BGR frame with the generated region composited in.
    """
    frame_h, frame_w = original_frame.shape[:2]
    bx, by, bw, bh = bbox

    # ── 1. Geometric inversion: 256×256 → original rectangle ────────────────
    # See _unsquare_crop docstring for the full geometric walkthrough.
    rect_crop = _unsquare_crop(generated_crop, bw, bh)

    # ── 2. Clamp bbox to frame bounds ────────────────────────────────────────
    x1 = max(0, bx)
    y1 = max(0, by)
    x2 = min(frame_w, bx + bw)
    y2 = min(frame_h, by + bh)
    paste_w = x2 - x1
    paste_h = y2 - y1

    if paste_w <= 0 or paste_h <= 0:
        logger.warning("Bbox %s is entirely outside frame %dx%d", bbox, frame_w, frame_h)
        return original_frame.copy()

    # Adjust the crop if bbox was clamped (i.e. face near frame edge)
    crop_x_offset = x1 - bx
    crop_y_offset = y1 - by
    rect_crop = rect_crop[
        crop_y_offset : crop_y_offset + paste_h,
        crop_x_offset : crop_x_offset + paste_w,
    ]

    # ── 3. Colour matching ───────────────────────────────────────────────────
    if apply_color_match:
        # Use a slightly expanded region around the bbox as the colour reference
        margin = max(20, int(0.15 * max(bw, bh)))
        ref_x1 = max(0, x1 - margin)
        ref_y1 = max(0, y1 - margin)
        ref_x2 = min(frame_w, x2 + margin)
        ref_y2 = min(frame_h, y2 + margin)
        surrounding = original_frame[ref_y1:ref_y2, ref_x1:ref_x2]
        if surrounding.size > 0 and rect_crop.size > 0:
            rect_crop = match_color(rect_crop, surrounding)

    # ── 4. Build soft alpha mask ─────────────────────────────────────────────
    alpha_mask = _build_alpha_mask(
        (frame_h, frame_w), landmarks, bbox, blur_kernel=blur_kernel
    )
    # Extract the mask region corresponding to the paste area
    mask_region = alpha_mask[y1:y2, x1:x2]

    # ── 5. Alpha blend ───────────────────────────────────────────────────────
    result = original_frame.copy()
    orig_region = result[y1:y2, x1:x2].astype(np.float32)
    gen_region = rect_crop.astype(np.float32)
    alpha_3ch = mask_region[:, :, np.newaxis]  # broadcast to 3 channels

    blended = orig_region * (1.0 - alpha_3ch) + gen_region * alpha_3ch
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return result
