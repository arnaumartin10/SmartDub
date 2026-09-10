"""
src/qc/boundary_score.py
─────────────────────────
Blend-seam visibility scoring using Sobel edge detection.

Measures edge intensity in a thin ring around the composited region's boundary
compared to edge intensity elsewhere on the face.  If the blending seam is
visible, the boundary ring will have abnormally high edge response compared
to the face interior.

The score is the ratio of (mean edge magnitude in boundary ring) /
(mean edge magnitude in face interior).  A ratio close to 1.0 means no
visible seam; ratios significantly > 1.0 indicate a visible blend boundary.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Ring width around the bbox boundary (in pixels)
_RING_WIDTH = 8


def _boundary_ring_mask(
    frame_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    ring_width: int = _RING_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Create binary masks for the boundary ring and interior of a bbox.

    Returns:
        (ring_mask, interior_mask) — both are uint8 (0 or 255).
    """
    h, w = frame_shape
    bx, by, bw, bh = bbox

    # Outer rectangle (expanded by ring_width)
    outer = np.zeros((h, w), dtype=np.uint8)
    ox1 = max(0, bx - ring_width)
    oy1 = max(0, by - ring_width)
    ox2 = min(w, bx + bw + ring_width)
    oy2 = min(h, by + bh + ring_width)
    outer[oy1:oy2, ox1:ox2] = 255

    # Inner rectangle (shrunk by ring_width)
    inner = np.zeros((h, w), dtype=np.uint8)
    ix1 = max(0, bx + ring_width)
    iy1 = max(0, by + ring_width)
    ix2 = min(w, bx + bw - ring_width)
    iy2 = min(h, by + bh - ring_width)
    if ix2 > ix1 and iy2 > iy1:
        inner[iy1:iy2, ix1:ix2] = 255

    # Ring = outer minus inner
    ring_mask = cv2.subtract(outer, inner)

    return ring_mask, inner


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel edge magnitude (float64)."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobel_x**2 + sobel_y**2)


def compute_boundary_score(
    video_path: str,
    roi_bboxes: list[tuple[int, int, int, int]],
    sample_every: int = 1,
    ring_width: int = _RING_WIDTH,
) -> float:
    """Compute blend-seam visibility score.

    Args:
        video_path:    Path to the output video.
        roi_bboxes:    Per-frame ``(x, y, w, h)`` face bounding boxes.
        sample_every:  Score every N-th frame.
        ring_width:    Width of the boundary ring in pixels.

    Returns:
        Boundary score (float, ≥ 0).  Ratio of edge intensity at the boundary
        ring vs. face interior.  ~1.0 = no visible seam, >1.5 = visible seam.
        Returns 0.0 if no frames could be scored.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    ratios: list[float] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= len(roi_bboxes):
            break

        if idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            edges = _sobel_magnitude(gray)

            ring_mask, interior_mask = _boundary_ring_mask(
                gray.shape[:2], roi_bboxes[idx], ring_width=ring_width
            )

            ring_pixels = edges[ring_mask > 0]
            interior_pixels = edges[interior_mask > 0]

            if ring_pixels.size > 0 and interior_pixels.size > 0:
                ring_mean = float(np.mean(ring_pixels))
                interior_mean = float(np.mean(interior_pixels))
                if interior_mean > 0:
                    ratios.append(ring_mean / interior_mean)

        idx += 1

    cap.release()

    if not ratios:
        return 0.0

    score = float(np.mean(ratios))
    logger.info(
        "Boundary score: %.4f over %d frames (%.4f = no seam, >1.5 = visible)",
        score,
        len(ratios),
        1.0,
    )
    return score
