"""
src/qc/sharpness_score.py
──────────────────────────
Sharpness measurement for the generated mouth region vs. a non-generated
reference region (forehead/eye area) on the same face.

Uses variance of the Laplacian — a standard blur-detection technique where
higher values indicate sharper images and lower values indicate blur.

The key output is the **sharpness ratio**: (mouth sharpness) / (reference
sharpness).  A ratio close to 1.0 means the generated region is as sharp as
the untouched face.  A ratio < 1.0 (our expected baseline) means the generated
region is softer — this is the number we want to improve once we add a
super-resolution / sharpening stage.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# The reference region is taken from the upper portion of the face bbox
# (forehead / eye area), which is NOT modified by the lip-sync pipeline.
# These fractions are relative to the face bounding box.
_REF_TOP_FRAC = 0.0    # start at top of bbox
_REF_BOTTOM_FRAC = 0.35  # use top 35% as reference (forehead + eyes)


def _variance_of_laplacian(gray: np.ndarray) -> float:
    """Compute the variance of the Laplacian (blur detection metric).

    Higher values indicate sharper images.  This is a well-established
    technique from Pech-Pacheco et al. (2000).
    """
    if gray.size == 0:
        return 0.0
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def _extract_mouth_region(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract the lower portion of the face bbox (mouth area)."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = bbox
    # Lower 50% of the bbox = mouth/chin area
    mouth_top = by + int(bh * 0.5)
    x1 = max(0, bx)
    y1 = max(0, mouth_top)
    x2 = min(w, bx + bw)
    y2 = min(h, by + bh)
    return frame[y1:y2, x1:x2]


def _extract_reference_region(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract the upper portion of the face bbox (forehead/eye area)."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = bbox
    ref_top = by + int(bh * _REF_TOP_FRAC)
    ref_bottom = by + int(bh * _REF_BOTTOM_FRAC)
    x1 = max(0, bx)
    y1 = max(0, ref_top)
    x2 = min(w, bx + bw)
    y2 = min(h, ref_bottom)
    return frame[y1:y2, x1:x2]


def compute_sharpness_score(
    video_path: str,
    roi_bboxes: list[tuple[int, int, int, int]],
    sample_every: int = 1,
) -> dict:
    """Compute sharpness of the generated mouth region vs. a reference region.

    Args:
        video_path:    Path to the output video.
        roi_bboxes:    Per-frame ``(x, y, w, h)`` face bounding boxes.
        sample_every:  Score every N-th frame (1 = all frames, 5 = every 5th).

    Returns:
        dict with keys:
          - ``mouth_sharpness``    (float): Mean variance-of-Laplacian for
                                            the mouth region.
          - ``reference_sharpness`` (float): Mean variance-of-Laplacian for
                                            the forehead/eye region.
          - ``sharpness_ratio``    (float): mouth / reference.  Values < 1.0
                                            indicate the generated region is
                                            softer than the untouched face.
          - ``n_frames_scored``    (int):   Number of frames scored.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    mouth_scores: list[float] = []
    ref_scores: list[float] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= len(roi_bboxes):
            break

        if idx % sample_every == 0:
            bbox = roi_bboxes[idx]

            mouth = _extract_mouth_region(frame, bbox)
            ref = _extract_reference_region(frame, bbox)

            if mouth.size > 0:
                mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
                mouth_scores.append(_variance_of_laplacian(mouth_gray))

            if ref.size > 0:
                ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
                ref_scores.append(_variance_of_laplacian(ref_gray))

        idx += 1

    cap.release()

    mouth_mean = float(np.mean(mouth_scores)) if mouth_scores else 0.0
    ref_mean = float(np.mean(ref_scores)) if ref_scores else 0.0
    ratio = mouth_mean / ref_mean if ref_mean > 0 else 0.0

    logger.info(
        "Sharpness: mouth=%.2f, reference=%.2f, ratio=%.4f (%d frames)",
        mouth_mean,
        ref_mean,
        ratio,
        len(mouth_scores),
    )

    return {
        "mouth_sharpness": mouth_mean,
        "reference_sharpness": ref_mean,
        "sharpness_ratio": ratio,
        "n_frames_scored": len(mouth_scores),
    }
