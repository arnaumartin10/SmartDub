"""
src/postprocessing/temporal_smoothing.py
─────────────────────────────────────────
Frame-to-frame temporal smoothing for the mouth/face region.

>>> PLACEHOLDER IMPLEMENTATION <<<
This module applies a weighted moving-average filter to the mouth region of
consecutive composited frames to reduce flicker caused by per-frame MuseTalk
inference inconsistencies.

Phase 2 work (not implemented here):
  - Optical-flow-based warping to align neighbouring frames before averaging,
    which avoids ghosting on large inter-frame motions.
  - Per-pixel adaptive weighting based on flow confidence.
  - Separate handling of lip interior vs. surrounding skin.

The current approach is intentionally simple: Gaussian-weighted temporal
averaging over a sliding window, applied ONLY to the mouth region (the rest
of the frame is left untouched).  This is effective for small jitter but
will ghost on fast head motion — hence the Phase 2 TODO.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Re-use the same landmark indices as compositing for consistency
JAW_CONTOUR_INDICES: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
)
MOUTH_OUTER_INDICES: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
)
_SMOOTH_MASK_INDICES: tuple[int, ...] = tuple(
    sorted(set(JAW_CONTOUR_INDICES) | set(MOUTH_OUTER_INDICES))
)


def _mouth_mask_from_landmarks(
    frame_shape: tuple[int, int],
    landmarks: list[tuple[float, float]],
    dilation: int = 15,
) -> np.ndarray:
    """Build a binary mouth-region mask from landmarks.

    Returns a float32 mask in [0, 1] with the mouth region set to 1.0.
    A small dilation is applied so the smoothing region slightly exceeds
    the strict landmark hull (avoids edge artefacts).
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    points = []
    for idx in _SMOOTH_MASK_INDICES:
        if idx < len(landmarks):
            px = max(0, min(w - 1, int(landmarks[idx][0])))
            py = max(0, min(h - 1, int(landmarks[idx][1])))
            points.append([px, py])

    if len(points) < 3:
        return mask.astype(np.float32)

    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)

    return (mask / 255.0).astype(np.float32)


def _gaussian_weights(window: int) -> np.ndarray:
    """Compute normalised Gaussian weights for a temporal window.

    The centre frame gets the highest weight; frames further away contribute
    less.  For window=3 this gives roughly [0.25, 0.50, 0.25].
    """
    # σ chosen so that the window edges are ≈2σ away from centre
    sigma = window / 4.0
    centre = window // 2
    weights = np.array(
        [np.exp(-0.5 * ((i - centre) / max(sigma, 1e-6)) ** 2) for i in range(window)],
        dtype=np.float64,
    )
    return weights / weights.sum()


def smooth_sequence(
    frames: list[np.ndarray],
    window: int = 3,
    landmarks_per_frame: Optional[list[Optional[list[tuple[float, float]]]]] = None,
) -> list[np.ndarray]:
    """Apply temporal smoothing to the mouth region of a frame sequence.

    >>> PLACEHOLDER — Phase 2 will replace this with optical-flow-based
    >>> temporal consistency enforcement.  The current weighted moving
    >>> average is effective for small jitter but will ghost on fast
    >>> head motion.

    Args:
        frames:              List of full-resolution BGR frames (composited).
        window:              Sliding window size (must be odd, ≥1).
                             ``window=1`` is a no-op.  ``window=3`` is default.
        landmarks_per_frame: Optional per-frame landmarks.  If provided, only
                             the mouth region (convex hull of jaw+mouth
                             landmarks) is smoothed; the rest of the frame
                             is untouched.  If ``None``, the entire frame is
                             smoothed (not recommended for production).

    Returns:
        List of temporally smoothed frames (same length as input).
    """
    if window < 1:
        raise ValueError(f"window must be ≥1, got {window}")
    if window % 2 == 0:
        logger.warning("window=%d is even; rounding up to %d", window, window + 1)
        window += 1
    if not frames:
        return []
    if window == 1 or len(frames) == 1:
        return [f.copy() for f in frames]

    weights = _gaussian_weights(window)
    half = window // 2
    n = len(frames)
    result: list[np.ndarray] = []

    for i in range(n):
        # Determine the window slice (clamped to sequence bounds)
        start = max(0, i - half)
        end = min(n, i + half + 1)
        actual_indices = list(range(start, end))

        # Re-normalise weights for the actual window (handles edges)
        w_start = start - (i - half)
        w_end = w_start + len(actual_indices)
        local_weights = weights[w_start:w_end]
        local_weights = local_weights / local_weights.sum()

        # ── Compute weighted average ─────────────────────────────────────────
        accum = np.zeros_like(frames[i], dtype=np.float64)
        for j, w in zip(actual_indices, local_weights):
            accum += frames[j].astype(np.float64) * w
        smoothed = np.clip(accum, 0, 255).astype(np.uint8)

        # ── Apply only to mouth region if landmarks are available ────────────
        if landmarks_per_frame is not None and i < len(landmarks_per_frame):
            lms = landmarks_per_frame[i]
            if lms is not None and len(lms) > 0:
                frame_h, frame_w = frames[i].shape[:2]
                mask = _mouth_mask_from_landmarks((frame_h, frame_w), lms)
                mask_3ch = mask[:, :, np.newaxis]
                # Blend: smoothed in mouth region, original elsewhere
                combined = (
                    frames[i].astype(np.float64) * (1.0 - mask_3ch)
                    + smoothed.astype(np.float64) * mask_3ch
                )
                smoothed = np.clip(combined, 0, 255).astype(np.uint8)
            else:
                # No landmarks for this frame — pass through unmodified
                smoothed = frames[i].copy()

        result.append(smoothed)

    logger.info(
        "Temporal smoothing: %d frames, window=%d%s",
        n,
        window,
        " (mouth-only)" if landmarks_per_frame is not None else " (full-frame)",
    )
    return result
