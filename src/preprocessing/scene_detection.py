"""
src/preprocessing/scene_detection.py
──────────────────────────────────────
Scene boundary detection using PySceneDetect's content-aware detector.

Returns (start_frame, end_frame) inclusive tuples so that downstream stages
never attempt to blend generated frames across a hard cut.

Library: scenedetect 0.6.x — ContentDetector performs HSV histogram comparison
between adjacent frames, which is significantly more robust than a single-channel
pixel-difference threshold (AdaptiveDetector) for professionally graded video.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
) -> list[tuple[int, int]]:
    """
    Detect scene boundaries in a video using content-aware analysis.

    Uses PySceneDetect's ``ContentDetector``, which computes a weighted HSV
    histogram difference between adjacent frames and triggers a cut when the
    score exceeds *threshold*.  This is more robust than a simple pixel-diff
    approach for colour-graded professional footage.

    Args:
        video_path:     Path to the source video file.
        threshold:      ContentDetector sensitivity (default 27.0).  Lower
                        values catch subtler cuts; higher values ignore them.
                        Tune per-project — 27 is a good starting point for
                        professionally edited talking-head content.
        min_scene_len:  Minimum scene duration in frames.  Scenes shorter than
                        this are merged into the preceding scene to avoid
                        micro-scenes from flash frames / hard cross-dissolves.

    Returns:
        List of ``(start_frame, end_frame)`` tuples (both inclusive, 0-indexed).
        Guaranteed to cover the entire video without gaps or overlaps.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        IOError:           If the video cannot be opened by OpenCV.
    """
    video_path = str(video_path)
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ── Probe total frame count as a fallback ─────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    logger.info("Probed video: %d frames @ %.2f fps", total_frames, fps)

    # ── PySceneDetect ─────────────────────────────────────────────────────────
    from scenedetect import ContentDetector, detect  # lazy import — large package

    scene_list = detect(
        video_path,
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len),
    )

    if not scene_list:
        # No cuts detected → treat entire video as a single scene
        logger.warning(
            "ContentDetector found no cuts in %s (threshold=%.1f). "
            "Returning entire video as a single scene.",
            path.name,
            threshold,
        )
        return [(0, max(total_frames - 1, 0))]

    # Convert FrameTimecode pairs to (start, end) int tuples.
    # PySceneDetect's scene list:  [(start_tc, end_tc), ...]
    #   start_tc  = first frame of scene  (inclusive, 0-indexed)
    #   end_tc    = first frame of the NEXT scene / EOF  (exclusive)
    # We store end as (end_tc.get_frames() - 1) to make it inclusive.
    result: list[tuple[int, int]] = []
    for start_tc, end_tc in scene_list:
        start = start_tc.get_frames()
        end = max(start, end_tc.get_frames() - 1)
        result.append((start, end))

    logger.info(
        "Detected %d scene(s) in %s",
        len(result),
        path.name,
    )
    for i, (s, e) in enumerate(result):
        logger.debug("  Scene %02d: frames %d–%d (%d frames)", i, s, e, e - s + 1)

    return result
