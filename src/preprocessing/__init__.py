"""
src/preprocessing/__init__.py
──────────────────────────────
Preprocessing sub-package.

Planned modules:
  scene_detector.py   — Shot/scene boundary detection (PySceneDetect or custom)
  face_tracker.py     — Per-frame face detection + temporal tracking (MediaPipe/InsightFace)
  forced_alignment.py — WhisperX audio/text alignment to video-frame tokens
"""

# Status: scene detection, face tracking, and G2P-backed alignment implemented and validated
