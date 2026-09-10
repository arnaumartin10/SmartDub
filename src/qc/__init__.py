"""
src/qc/__init__.py
───────────────────
Quality Control sub-package.

Modules:
  lip_sync_score.py  — SyncNet-based LSE-D / LSE-C lip-sync accuracy
  flicker_score.py   — Frame-to-frame temporal flicker in mouth ROI
  sharpness_score.py — Variance-of-Laplacian sharpness (mouth vs. reference)
  boundary_score.py  — Sobel edge-based blend-seam visibility
  qc_report.py       — Unified HTML/JSON report generator
"""

from src.qc.qc_report import generate_qc_report

__all__ = ["generate_qc_report"]
