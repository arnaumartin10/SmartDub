"""
src/generation/__init__.py
───────────────────────────
Generation sub-package.

Modules:
  coarse_lipsync.py   — MuseTalk coarse audio-driven lip-sync generation
  refinement.py       — GFPGAN-based high-frequency detail refinement
"""

from src.generation.coarse_lipsync import CoarseLipSyncGenerator
from src.generation.refinement import FaceRefiner

__all__ = ["CoarseLipSyncGenerator", "FaceRefiner"]
