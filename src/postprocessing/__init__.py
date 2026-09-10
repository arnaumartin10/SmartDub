"""
src/postprocessing/__init__.py
───────────────────────────────
Postprocessing sub-package.

Modules:
  compositing.py        — Composite MuseTalk crops back into full-res frames
  color_matching.py     — LAB colour-space histogram matching
  temporal_smoothing.py — Weighted moving-average temporal consistency (Phase 1)

Planned (Phase 2+):
  super_res.py          — Super-resolution upscaling (Real-ESRGAN / HAT)
"""

from src.postprocessing.color_matching import match_color
from src.postprocessing.compositing import composite_frame
from src.postprocessing.temporal_smoothing import smooth_sequence

__all__ = ["composite_frame", "match_color", "smooth_sequence"]
