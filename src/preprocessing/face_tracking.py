"""
src/preprocessing/face_tracking.py
────────────────────────────────────
Per-frame face detection, landmark extraction, and temporal smoothing.

Library choice: MediaPipe FaceMesh (mediapipe>=0.10.9)
───────────────────────────────────────────────────────
Two main candidates were evaluated for 2026 pip packaging:

  1. face_alignment (1adrianb/face-alignment)
     ✗ Requires dlib or pytorch-based SFD backend
     ✗ dlib compilation from source on Ubuntu 22.04 with Python 3.10 is
       fragile and fails with LLVM 14; pre-built wheels lag significantly
     ✗ Active development has slowed; last meaningful release was 2022
     ✓ Returns standard 68-point dlib landmark scheme (easy mapping to mouth_roi)

  2. MediaPipe FaceMesh (Google)
     ✓ Pure pip install: `pip install mediapipe` — zero compilation
     ✓ Actively maintained; 0.10.x released in 2024–2026 with Python 3.10/3.11 support
     ✓ Returns 478 face mesh landmarks (much richer than 68-point)
     ✓ Built-in temporal tracking mode (static_image_mode=False) — avoids re-detection
       every frame, which is critical for smooth tracking on talking-head video
     ✓ Bundles its own inference runtime — no CUDA required for preprocessing
     ✗ 478-point index scheme differs from classic dlib 68-point; mouth ROI
       landmark indices are documented in MOUTH_* constants below

MediaPipe was chosen for its clean packaging, active maintenance, and superior
tracking stability on video input.  The 68-point → 478-point mapping for the
mouth region is documented in mouth_roi.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── MediaPipe mouth landmark indices (478-point mesh) ─────────────────────────
# Outer lip contour vertices used for bounding-box derivation
MOUTH_OUTER_INDICES: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
)
# Inner lip contour
MOUTH_INNER_INDICES: tuple[int, ...] = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191,
)
# Combined (deduplicated, sorted) for bbox computation
MOUTH_ALL_INDICES: tuple[int, ...] = tuple(
    sorted(set(MOUTH_OUTER_INDICES) | set(MOUTH_INNER_INDICES))
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _landmarks_to_pixels(
    face_lms, width: int, height: int
) -> list[tuple[float, float]]:
    """Convert normalised MediaPipe landmarks to pixel-space (x, y) tuples."""
    return [(lm.x * width, lm.y * height) for lm in face_lms.landmark]


def _bbox_from_pixels(
    pixels: list[tuple[float, float]],
) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) bounding box encompassing all landmark pixels."""
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return bbox[2] * bbox[3]


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Return intersection-over-union for two ``(x, y, w, h)`` boxes."""
    first_x1, first_y1 = first[:2]
    first_x2, first_y2 = first_x1 + first[2], first_y1 + first[3]
    second_x1, second_y1 = second[:2]
    second_x2, second_y2 = second_x1 + second[2], second_y1 + second[3]

    intersection_w = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_h = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_w * intersection_h
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union else 0.0


def _detection_bbox(detection, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a MediaPipe Face Detection relative box to pixel coordinates."""
    box = detection.location_data.relative_bounding_box
    return (
        int(box.xmin * width),
        int(box.ymin * height),
        max(1, int(box.width * width)),
        max(1, int(box.height * height)),
    )


def _landmark_quality_proxy(face_lms_obj) -> float:
    """Return a geometry-validity proxy when Face Detection has no score.

    MediaPipe Face Mesh 0.10.21 exposes ``presence`` and ``visibility`` fields,
    but both are always 0.0 for this solution. This proxy is therefore not a
    MediaPipe confidence: it only measures the fraction of finite landmark
    coordinates and is deliberately used only when the separate detector fails.
    """
    points = [(landmark.x, landmark.y) for landmark in face_lms_obj.landmark]
    valid = [
        np.isfinite(x) and np.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        for x, y in points
    ]
    return float(np.mean(valid)) if valid else 0.0


def _detection_confidence(
    face_lms_obj,
    detections,
    mesh_bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> float:
    """Get the native detector score for the mesh face, or the documented proxy."""
    best_detection = None
    best_iou = 0.0
    for detection in detections or []:
        detection_bbox = _detection_bbox(detection, width, height)
        iou = _bbox_iou(mesh_bbox, detection_bbox)
        if iou > best_iou:
            best_iou = iou
            best_detection = detection

    if best_detection is not None:
        return float(np.clip(best_detection.score[0], 0.0, 1.0))
    return _landmark_quality_proxy(face_lms_obj)


class _BBoxEMA:
    """
    Exponential Moving Average smoother for face bounding boxes.

    Reduces frame-to-frame jitter caused by small detection fluctuations.
    When no face is detected in a frame, the last known smoothed state is
    propagated forward (rather than jumping to (0,0,0,0) and back), which
    keeps downstream crops stable during momentary occlusions.

    Args:
        alpha: Weight applied to the *current* frame's bbox (0 < alpha ≤ 1).
               Higher values track faster motion; lower values remove more jitter.
               0.7 is a good default for 25–30 fps talking-head footage.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def update(self, bbox: Optional[tuple]) -> Optional[tuple[int, int, int, int]]:
        if bbox is None:
            # No detection — propagate last known state
            if self._state is None:
                return None
            return tuple(self._state.astype(int).tolist())  # type: ignore[return-value]

        current = np.array(bbox, dtype=float)
        if self._state is None:
            self._state = current
        else:
            self._state = self.alpha * current + (1.0 - self.alpha) * self._state
        return tuple(self._state.astype(int).tolist())  # type: ignore[return-value]


# ── Public API ────────────────────────────────────────────────────────────────


def track_face(
    video_path: str,
    scene_bounds: tuple[int, int],
    ema_alpha: float = 0.7,
    max_faces: int = 10,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> list[Optional[dict]]:
    """
    Track the primary speaking face across every frame of a scene.

    Uses MediaPipe FaceMesh in video-tracking mode (``static_image_mode=False``)
    so landmark positions are temporally consistent across frames.  When multiple
    faces are present, the face with the largest bounding-box area is treated as
    the primary speaker.

    TODO: Replace the "largest bbox" heuristic with active-speaker detection
    using audio-visual sync scoring (e.g., SyncNet confidence or lip-motion
    energy correlated with the dubbed audio waveform).  This is necessary for
    multi-speaker scenes such as dialogue or panel discussions.

    Args:
        video_path:                Path to the source video file.
        scene_bounds:              ``(start_frame, end_frame)`` inclusive tuple
                                   from :func:`scene_detection.detect_scenes`.
        ema_alpha:                 EMA coefficient for bbox smoothing (0 < α ≤ 1).
                                   0.7 is recommended for 25–30 fps content.
        max_faces:                 Maximum number of faces MediaPipe will track
                                   simultaneously.  Higher values slow down inference.
        min_detection_confidence:  MediaPipe face detection confidence gate.
        min_tracking_confidence:   MediaPipe inter-frame tracking confidence gate.

    Returns:
        A list of length ``(end_frame - start_frame + 1)``, one entry per frame.
        Each entry is either:
        - ``None`` — no face was detected in this frame.
        - A ``dict`` with keys:
            - ``frame_number``  (int)  — absolute frame index in the source video.
            - ``bounding_box``  (tuple[int, int, int, int])  — EMA-smoothed
              ``(x, y, w, h)`` in pixel coordinates.
            - ``landmarks``     (list[tuple[float, float]])  — 478 pixel-space
              ``(x, y)`` points from MediaPipe FaceMesh.
                        - ``confidence``    (float in [0, 1])  — native Face Detection score;
                            falls back to a documented landmark geometry proxy.

    Raises:
        FileNotFoundError:  If *video_path* does not exist.
        IOError:            If the video cannot be opened by OpenCV.
    """
    import mediapipe as mp  # lazy import — avoids slow load when unused

    from pathlib import Path

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    start_frame, end_frame = scene_bounds
    n_frames = end_frame - start_frame + 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Seek to scene start
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    smoother = _BBoxEMA(alpha=ema_alpha)
    results: list[Optional[dict]] = []

    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,      # tracking mode: faster, temporally stable
        max_num_faces=max_faces,
        refine_landmarks=True,         # adds iris + lip refinement landmarks
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh, mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=min_detection_confidence,
    ) as face_detector:

        for frame_idx in range(start_frame, end_frame + 1):
            ret, bgr = cap.read()
            if not ret:
                logger.warning("Frame %d: read() failed — video may be shorter than expected", frame_idx)
                smoother.update(None)
                results.append(None)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detection_result = face_detector.process(rgb)
            mp_result = face_mesh.process(rgb)

            if not mp_result.multi_face_landmarks:
                logger.debug("Frame %d: no face detected", frame_idx)
                smoothed_bbox = smoother.update(None)
                results.append(None)
                continue

            # ── Primary face selection ────────────────────────────────────────
            # TODO: Replace with active-speaker detection (audio-visual sync).
            best_lms_obj = None
            best_pixels: list[tuple[float, float]] = []
            best_bbox: Optional[tuple[int, int, int, int]] = None
            best_area = -1

            for face_lms_obj in mp_result.multi_face_landmarks:
                pixels = _landmarks_to_pixels(face_lms_obj, width, height)
                bbox = _bbox_from_pixels(pixels)
                area = _bbox_area(bbox)
                if area > best_area:
                    best_area = area
                    best_lms_obj = face_lms_obj
                    best_pixels = pixels
                    best_bbox = bbox

            if best_lms_obj is None or best_bbox is None:
                results.append(None)
                continue

            confidence = _detection_confidence(
                best_lms_obj,
                detection_result.detections,
                best_bbox,
                width,
                height,
            )
            smoothed_bbox = smoother.update(best_bbox)

            results.append(
                {
                    "frame_number": frame_idx,
                    "bounding_box": smoothed_bbox,
                    "landmarks": best_pixels,
                    "confidence": confidence,
                }
            )

    cap.release()

    n_detected = sum(1 for r in results if r is not None)
    n_missed = n_frames - n_detected
    logger.info(
        "Scene frames %d–%d: %d/%d faces detected, %d missed",
        start_frame,
        end_frame,
        n_detected,
        n_frames,
        n_missed,
    )

    return results
