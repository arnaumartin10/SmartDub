"""
src/postprocessing/color_matching.py
─────────────────────────────────────
Histogram matching in LAB colour space.

The MuseTalk-generated face crop often has slightly different brightness and
colour balance compared to the surrounding original frame (the generation
model was trained on its own data distribution).  Matching statistics in LAB
rather than RGB is more perceptually correct because the L channel isolates
luminance from chrominance, avoiding cross-channel colour shifts.

The algorithm (Reinhard et al. 2001 simplified):
  For each LAB channel independently:
    generated_matched = (generated - μ_gen) * (σ_surr / σ_gen) + μ_surr

This is a lightweight first pass; a more robust approach (e.g. optimal
transport / colour-transfer networks) can replace it later if needed.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum standard deviation to avoid division by zero in flat regions
_EPS = 1e-6


def match_color(
    generated_region: np.ndarray,
    surrounding_region: np.ndarray,
) -> np.ndarray:
    """Match the colour distribution of *generated_region* to *surrounding_region*.

    Both inputs are BGR uint8 images.  They do **not** need to be the same size
    — the function computes per-channel statistics independently.

    Args:
        generated_region:   BGR uint8 image (H₁, W₁, 3) — the MuseTalk output
                            that needs colour correction.
        surrounding_region: BGR uint8 image (H₂, W₂, 3) — a patch of the
                            original frame surrounding the generation area,
                            used as the colour reference.

    Returns:
        Colour-matched BGR uint8 image with the same shape as *generated_region*.

    Raises:
        ValueError: If either input is empty or not a 3-channel image.
    """
    if generated_region.size == 0:
        raise ValueError("generated_region is empty")
    if surrounding_region.size == 0:
        raise ValueError("surrounding_region is empty")
    if generated_region.ndim != 3 or generated_region.shape[2] != 3:
        raise ValueError(
            f"generated_region must be HxWx3, got shape {generated_region.shape}"
        )
    if surrounding_region.ndim != 3 or surrounding_region.shape[2] != 3:
        raise ValueError(
            f"surrounding_region must be HxWx3, got shape {surrounding_region.shape}"
        )

    # ── Convert BGR → LAB (float32 for sub-pixel precision) ──────────────────
    gen_lab = cv2.cvtColor(generated_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(surrounding_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    # ── Per-channel Reinhard transfer ────────────────────────────────────────
    for ch in range(3):
        gen_mean, gen_std = gen_lab[:, :, ch].mean(), gen_lab[:, :, ch].std()
        ref_mean, ref_std = ref_lab[:, :, ch].mean(), ref_lab[:, :, ch].std()

        # Scale generated channel to match reference statistics
        scale = ref_std / max(gen_std, _EPS)
        gen_lab[:, :, ch] = (gen_lab[:, :, ch] - gen_mean) * scale + ref_mean

    # ── Clip to valid LAB range and convert back ─────────────────────────────
    # OpenCV LAB for uint8: L ∈ [0, 255], A ∈ [0, 255], B ∈ [0, 255]
    # (internally 0-255 mapping; L=0..100 → 0..255, a,b = -128..127 → 0..255)
    gen_lab = np.clip(gen_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(gen_lab, cv2.COLOR_LAB2BGR)

    logger.debug(
        "Colour matched: gen %s → ref %s",
        generated_region.shape[:2],
        surrounding_region.shape[:2],
    )
    return result
