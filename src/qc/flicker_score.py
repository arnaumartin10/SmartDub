"""
src/qc/flicker_score.py
────────────────────────
Frame-to-frame temporal flicker measurement within the mouth ROI.

Flicker is defined as the mean squared difference (or 1 - SSIM) between
consecutive frames' mouth regions.  Higher values indicate more temporal
inconsistency / flicker.

This is a simple v1 metric — it doesn't account for legitimate motion (e.g.,
fast speech causes high pixel differences that aren't "flicker").  A more
sophisticated version would use optical-flow-compensated differences.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _extract_roi(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: tuple[int, int] = (96, 96),
) -> np.ndarray:
    """Extract and resize a bounding-box region from a frame."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = bbox
    x1 = max(0, bx)
    y1 = max(0, by)
    x2 = min(w, bx + bw)
    y2 = min(h, by + bh)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)


def compute_flicker_score(
    video_path: str,
    roi_bboxes: list[tuple[int, int, int, int]],
    metric: str = "mse",
) -> float:
    """Compute frame-to-frame flicker within the mouth ROI.

    Args:
        video_path:  Path to the output video.
        roi_bboxes:  Per-frame ``(x, y, w, h)`` bounding boxes defining the
                     mouth/face region to measure.
        metric:      ``"mse"`` (mean squared error) or ``"ssim"`` (1 - SSIM).
                     MSE is faster; SSIM is more perceptually meaningful.

    Returns:
        Flicker score (float, ≥ 0).  Higher = more flicker.
        Returns 0.0 if the video has fewer than 2 frames.
    """
    if metric not in ("mse", "ssim"):
        raise ValueError(f"metric must be 'mse' or 'ssim', got {metric!r}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    prev_roi: Optional[np.ndarray] = None
    diffs: list[float] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= len(roi_bboxes):
            break

        roi = _extract_roi(frame, roi_bboxes[idx])
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_roi is not None:
            if metric == "mse":
                diff = np.mean((roi_gray - prev_roi) ** 2)
                diffs.append(float(diff))
            elif metric == "ssim":
                ssim_val = _compute_ssim(prev_roi, roi_gray)
                diffs.append(1.0 - ssim_val)  # flicker = 1 - SSIM

        prev_roi = roi_gray
        idx += 1

    cap.release()

    if not diffs:
        return 0.0

    score = float(np.mean(diffs))
    logger.info(
        "Flicker score (%s): %.4f over %d frame pairs", metric, score, len(diffs)
    )
    return score


def _compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    C1: float = 6.5025,
    C2: float = 58.5225,
) -> float:
    """Compute SSIM between two grayscale float32 images.

    Simplified SSIM (no windowing) — sufficient for a flicker proxy metric.
    """
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(np.mean(ssim_map))
