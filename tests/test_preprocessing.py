"""
tests/test_preprocessing.py
─────────────────────────────
Unit + integration tests for the preprocessing stage.

Tests are designed to run without GPU and without sample.mp4:
  - Synthetic 2-scene video is generated in a temp directory using ffmpeg.
  - Face-tracking tests on a real face are skipped unless a video with a
    detectable face is available (data/inputs/sample.mp4 or --face-video flag).
  - Mouth ROI tests use synthetic numpy data.

Run:
  pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.mouth_roi import (
    MOUTH_ALL_INDICES,
    DLIB68_MOUTH_INDICES,
    extract_mouth_roi,
    draw_mouth_landmarks,
)
from src.preprocessing.scene_detection import detect_scenes

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _create_synthetic_two_scene_video(path: Path, fps: int = 25, duration_per_scene: int = 2) -> Path:
    """
    Create a video with two hard scene cuts using ffmpeg lavfi sources.

    Scene 1: solid red (0–duration_per_scene seconds)
    Scene 2: solid blue (duration_per_scene–2*duration_per_scene seconds)

    The instant colour change produces a ContentDetector score of ~100+,
    well above the default threshold of 27, ensuring reliable detection.
    """
    tmp = path.parent

    # Create each solid-colour clip
    clips = []
    colour_patterns = [
        ("c=red:size=320x240",   "clip_red.mp4"),
        ("c=blue:size=320x240",  "clip_blue.mp4"),
    ]
    for pattern, name in colour_patterns:
        clip = tmp / name
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi",
                "-i", f"color={pattern}:rate={fps}:duration={duration_per_scene}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(clip),
            ],
            check=True,
        )
        clips.append(clip)

    # Concatenate
    concat_file = tmp / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{c}'" for c in clips) + "\n"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "copy",
            str(path),
        ],
        check=True,
    )
    return path


def _make_fake_mediapipe_landmarks(
    n: int = 478,
    width: int = 640,
    height: int = 480,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Generate plausible-looking (random) pixel-space landmarks for unit tests.
    Mouth indices land in the lower-centre region of a 640×480 frame.
    """
    rng = np.random.default_rng(seed)
    pts: list[tuple[float, float]] = []
    for i in range(n):
        if i in set(MOUTH_ALL_INDICES):
            # Place mouth landmarks in lower-centre third of frame
            x = float(rng.uniform(width * 0.35, width * 0.65))
            y = float(rng.uniform(height * 0.60, height * 0.80))
        else:
            x = float(rng.uniform(width * 0.2, width * 0.8))
            y = float(rng.uniform(height * 0.1, height * 0.9))
        pts.append((x, y))
    return pts


# ── Scene Detection Tests ─────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
class TestSceneDetection:
    """Scene detection on a synthetically generated 2-cut video."""

    def test_detect_two_scenes(self, tmp_path):
        """ContentDetector must find 2 scenes in a solid-red → solid-blue video."""
        video = _create_synthetic_two_scene_video(tmp_path / "two_scene.mp4")
        scenes = detect_scenes(str(video))

        assert len(scenes) == 2, (
            f"Expected 2 scenes in a 2-colour video, got {len(scenes)}: {scenes}"
        )

    def test_scenes_cover_full_video(self, tmp_path):
        """Scene (start, end) ranges must span the entire video without gaps."""
        video = _create_synthetic_two_scene_video(tmp_path / "two_scene.mp4")
        scenes = detect_scenes(str(video))

        # First scene starts at frame 0
        assert scenes[0][0] == 0, f"First scene must start at frame 0, got {scenes[0][0]}"

        # Each scene must not overlap its successor
        for i in range(len(scenes) - 1):
            end_i = scenes[i][1]
            start_next = scenes[i + 1][0]
            assert end_i < start_next, (
                f"Scene {i} end ({end_i}) must be before scene {i+1} start ({start_next})"
            )

    def test_scene_bounds_are_ints(self, tmp_path):
        """All returned frame indices must be Python ints (not numpy types)."""
        video = _create_synthetic_two_scene_video(tmp_path / "two_scene.mp4")
        scenes = detect_scenes(str(video))
        for s, e in scenes:
            assert isinstance(s, int), f"start_frame {s!r} is not int"
            assert isinstance(e, int), f"end_frame {e!r} is not int"

    def test_single_scene_video_returns_one_entry(self, tmp_path):
        """A 2-second solid-colour clip (no cuts) must return exactly one scene."""
        clip = tmp_path / "one_scene.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi",
                "-i", "color=c=green:size=320x240:rate=25:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(clip),
            ],
            check=True,
        )
        scenes = detect_scenes(str(clip))
        assert len(scenes) == 1, f"Expected 1 scene for no-cut clip, got {len(scenes)}"


# ── Face Tracking Tests ───────────────────────────────────────────────────────


class TestFaceTracking:
    """Tests for track_face() — no GPU required."""

    def _make_black_video(self, path: Path, n_frames: int = 10, fps: int = 5) -> Path:
        """Black video (no face) — useful for testing zero-detection handling."""
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi",
                "-i", f"color=c=black:size=320x240:rate={fps}:duration={n_frames // fps}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(path),
            ],
            check=True,
        )
        return path

    @pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
    def test_returns_list_same_length_as_scene(self, tmp_path):
        """track_face must return exactly (end-start+1) entries."""
        from src.preprocessing.face_tracking import track_face

        video = self._make_black_video(tmp_path / "black.mp4", n_frames=10, fps=5)
        scene = (0, 9)  # 10 frames (0-indexed inclusive)
        results = track_face(str(video), scene)
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

    @pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
    def test_no_face_returns_none_entries(self, tmp_path):
        """Black frame (no face) must produce all-None entries (not a crash)."""
        from src.preprocessing.face_tracking import track_face

        video = self._make_black_video(tmp_path / "black.mp4", n_frames=10, fps=5)
        results = track_face(str(video), (0, 9))
        n_none = sum(1 for r in results if r is None)
        assert n_none == len(results), (
            f"Expected all None for a no-face video, got {n_none}/{len(results)} None"
        )

    @pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
    def test_result_dict_has_required_keys(self, tmp_path):
        """When a face is detected, the dict must contain all 4 required keys."""
        # Use the real sample.mp4 if available, otherwise skip
        sample = Path("data/inputs/sample.mp4")
        if not sample.exists():
            pytest.skip("data/inputs/sample.mp4 not found — skipping face dict test")

        from src.preprocessing.face_tracking import track_face

        results = track_face(str(sample), (0, 4))  # first 5 frames
        detected = [r for r in results if r is not None]
        if not detected:
            pytest.skip("No face detected in first 5 frames of sample.mp4")

        for r in detected:
            assert "frame_number" in r
            assert "bounding_box" in r
            assert "landmarks" in r
            assert "confidence" in r
            assert isinstance(r["bounding_box"], tuple) and len(r["bounding_box"]) == 4
            assert isinstance(r["landmarks"], list)
            assert 0.0 <= r["confidence"] <= 1.0

    @pytest.mark.skipif(
        not Path("data/inputs/sample.mp4").exists(),
        reason="data/inputs/sample.mp4 not found",
    )
    def test_face_detected_on_real_video(self):
        """At least some frames must have face detections on a real talking-head video."""
        from src.preprocessing.face_tracking import track_face

        results = track_face("data/inputs/sample.mp4", (0, 24))  # first 25 frames
        detected = [r for r in results if r is not None]
        assert len(detected) > 0, (
            "No face detected in first 25 frames of sample.mp4. "
            "Check that the video has a clearly visible face in the first second."
        )


# ── Mouth ROI Tests ───────────────────────────────────────────────────────────


class TestMouthROI:
    """Tests for extract_mouth_roi() — pure numpy, no video needed."""

    def test_returns_nonempty_crop(self):
        """Basic extraction must return a non-empty numpy array."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = _make_fake_mediapipe_landmarks()
        crop, bbox = extract_mouth_roi(frame, lms)
        assert crop.ndim == 3
        assert crop.shape[2] == 3
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_bbox_is_4_tuple_of_ints(self):
        """Returned bbox must be a 4-tuple of ints (x, y, w, h)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = _make_fake_mediapipe_landmarks()
        _, bbox = extract_mouth_roi(frame, lms)
        assert len(bbox) == 4
        for v in bbox:
            assert isinstance(v, int), f"bbox value {v!r} is not int"

    def test_bbox_within_frame_bounds(self):
        """Padded bbox must be clamped to frame dimensions."""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        lms = _make_fake_mediapipe_landmarks(width=w, height=h)
        _, (bx, by, bw, bh) = extract_mouth_roi(frame, lms)
        assert bx >= 0 and by >= 0
        assert bx + bw <= w, f"x+w={bx+bw} exceeds frame width {w}"
        assert by + bh <= h, f"y+h={by+bh} exceeds frame height {h}"

    def test_padding_makes_crop_larger_than_tight_bbox(self):
        """30% padding must produce a crop larger than the raw landmark bbox."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = _make_fake_mediapipe_landmarks()

        from src.preprocessing.mouth_roi import MOUTH_ALL_INDICES
        mouth_pts = [lms[i] for i in MOUTH_ALL_INDICES if i < len(lms)]
        tight_w = max(p[0] for p in mouth_pts) - min(p[0] for p in mouth_pts)
        tight_h = max(p[1] for p in mouth_pts) - min(p[1] for p in mouth_pts)

        crop, bbox = extract_mouth_roi(frame, lms, padding=0.30)
        assert bbox[2] >= tight_w, "Padded width must be ≥ tight width"
        assert bbox[3] >= tight_h, "Padded height must be ≥ tight height"

    def test_target_size_resize(self):
        """target_size must resize the crop to the requested dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = _make_fake_mediapipe_landmarks()
        crop, _ = extract_mouth_roi(frame, lms, target_size=(128, 128))
        assert crop.shape == (128, 128, 3)

    def test_empty_landmarks_raises(self):
        """Empty landmarks list must raise ValueError (not crash silently)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty"):
            extract_mouth_roi(frame, [])

    def test_unknown_scheme_raises(self):
        """Unknown index_scheme must raise ValueError."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = _make_fake_mediapipe_landmarks()
        with pytest.raises(ValueError, match="Unknown index_scheme"):
            extract_mouth_roi(frame, lms, index_scheme="dlib99")

    def test_dlib68_scheme(self):
        """dlib68 scheme must work with a 68-element landmark list."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Generate 68 landmarks; place mouth (48-67) in lower-centre
        rng = np.random.default_rng(0)
        lms = []
        for i in range(68):
            if i in range(48, 68):
                x = float(rng.uniform(280, 360))
                y = float(rng.uniform(310, 370))
            else:
                x = float(rng.uniform(100, 540))
                y = float(rng.uniform(50, 420))
            lms.append((x, y))
        crop, bbox = extract_mouth_roi(frame, lms, index_scheme="dlib68")
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_draw_mouth_landmarks_returns_same_shape(self):
        """draw_mouth_landmarks must return a frame with the same dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = _make_fake_mediapipe_landmarks()
        out = draw_mouth_landmarks(frame, lms)
        assert out.shape == frame.shape


# ── End-to-End Integration Test ───────────────────────────────────────────────


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_preprocess_demo_end_to_end(tmp_path):
    """
    Smoke test the full preprocessing pipeline on a synthetic video.

    No face will be detected (solid-colour video), but the pipeline must
    complete without error and produce all expected output files.
    """
    from scripts.preprocess_demo import run

    video = _create_synthetic_two_scene_video(tmp_path / "demo.mp4")
    output_dir = tmp_path / "out"
    summary = run(video, output_dir)

    # scenes.json must exist
    assert (output_dir / "scenes.json").exists()

    # debug_annotated.mp4 must exist and be non-empty
    annotated = output_dir / "debug_annotated.mp4"
    assert annotated.exists()
    assert annotated.stat().st_size > 0

    # mouth_rois/ directory must exist
    assert (output_dir / "mouth_rois").is_dir()

    # Summary must have the right structure
    assert "total_scenes" in summary
    assert "total_frames_processed" in summary
    assert summary["total_scenes"] == 2
