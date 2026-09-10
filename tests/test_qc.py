"""
tests/test_qc.py
─────────────────
CPU-only unit tests for the QC scoring modules.

These tests use synthetic test clips (basic shapes/noise — no real faces)
to verify that each scoring function:
  - Handles short synthetic video without crashing
  - Returns values in expected ranges
  - Produces a non-empty HTML report
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qc.flicker_score import compute_flicker_score
from src.qc.sharpness_score import compute_sharpness_score
from src.qc.boundary_score import compute_boundary_score
from src.qc.qc_report import generate_qc_report, _status


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_video_path() -> str:
    """Create a short synthetic video with moving shapes (no real face)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    n_frames = 30
    fps = 25.0
    h, w = 480, 640

    writer = cv2.VideoWriter(
        tmp.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    assert writer.isOpened(), f"Cannot create test video: {tmp.name}"

    rng = np.random.RandomState(42)
    for i in range(n_frames):
        # Base frame with gradient + noise
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(50, 200, w, dtype=np.uint8)  # blue gradient
        frame[:, :, 1] = 100
        frame[:, :, 2] = 150
        # Add noise to make frames slightly different (simulates flicker)
        noise = rng.randint(-10, 10, (h, w, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Draw a rectangle that moves (simulates face bbox area)
        cx = 300 + i * 2
        cy = 200
        cv2.rectangle(frame, (cx - 50, cy - 60), (cx + 50, cy + 60), (200, 180, 160), -1)
        # Draw smaller rectangle inside (simulates mouth region)
        cv2.rectangle(frame, (cx - 30, cy + 10), (cx + 30, cy + 50), (180, 140, 130), -1)
        # Add some edges/detail
        cv2.circle(frame, (cx - 20, cy - 20), 8, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 20, cy - 20), 8, (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def synthetic_audio_path() -> str:
    """Create a short synthetic WAV file (silence)."""
    import struct
    import wave

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    sr = 16000
    duration = 1.2  # seconds
    n_samples = int(sr * duration)

    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # Write silence with tiny noise to avoid all-zero edge cases
        rng = np.random.RandomState(99)
        samples = rng.randint(-100, 100, n_samples, dtype=np.int16)
        wf.writeframes(samples.tobytes())

    yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def roi_bboxes() -> list[tuple[int, int, int, int]]:
    """Per-frame bounding boxes matching the synthetic video's rectangles."""
    return [(250 + i * 2, 140, 100, 120) for i in range(30)]


# ── Test: Flicker Score ──────────────────────────────────────────────────────


class TestFlickerScore:
    def test_returns_nonnegative_float(self, synthetic_video_path, roi_bboxes):
        score = compute_flicker_score(synthetic_video_path, roi_bboxes)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_mse_metric(self, synthetic_video_path, roi_bboxes):
        score = compute_flicker_score(synthetic_video_path, roi_bboxes, metric="mse")
        assert score >= 0.0

    def test_ssim_metric(self, synthetic_video_path, roi_bboxes):
        score = compute_flicker_score(synthetic_video_path, roi_bboxes, metric="ssim")
        assert score >= 0.0

    def test_invalid_metric_raises(self, synthetic_video_path, roi_bboxes):
        with pytest.raises(ValueError, match="metric must be"):
            compute_flicker_score(synthetic_video_path, roi_bboxes, metric="invalid")

    def test_identical_frames_low_flicker(self):
        """A video of identical frames should have near-zero flicker."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        writer = cv2.VideoWriter(
            tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 25, (100, 100)
        )
        for _ in range(10):
            writer.write(frame)
        writer.release()

        bboxes = [(10, 10, 80, 80)] * 10
        score = compute_flicker_score(tmp.name, bboxes)
        assert score < 1.0  # Near zero for identical frames
        Path(tmp.name).unlink(missing_ok=True)


# ── Test: Sharpness Score ────────────────────────────────────────────────────


class TestSharpnessScore:
    def test_returns_dict(self, synthetic_video_path, roi_bboxes):
        result = compute_sharpness_score(synthetic_video_path, roi_bboxes)
        assert isinstance(result, dict)
        assert "mouth_sharpness" in result
        assert "reference_sharpness" in result
        assert "sharpness_ratio" in result
        assert "n_frames_scored" in result

    def test_nonnegative_values(self, synthetic_video_path, roi_bboxes):
        result = compute_sharpness_score(synthetic_video_path, roi_bboxes)
        assert result["mouth_sharpness"] >= 0.0
        assert result["reference_sharpness"] >= 0.0
        assert result["sharpness_ratio"] >= 0.0

    def test_frames_scored_count(self, synthetic_video_path, roi_bboxes):
        result = compute_sharpness_score(synthetic_video_path, roi_bboxes)
        assert result["n_frames_scored"] > 0

    def test_sample_every(self, synthetic_video_path, roi_bboxes):
        result_all = compute_sharpness_score(
            synthetic_video_path, roi_bboxes, sample_every=1
        )
        result_sampled = compute_sharpness_score(
            synthetic_video_path, roi_bboxes, sample_every=5
        )
        assert result_sampled["n_frames_scored"] < result_all["n_frames_scored"]


# ── Test: Boundary Score ─────────────────────────────────────────────────────


class TestBoundaryScore:
    def test_returns_nonnegative_float(self, synthetic_video_path, roi_bboxes):
        score = compute_boundary_score(synthetic_video_path, roi_bboxes)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_sample_every(self, synthetic_video_path, roi_bboxes):
        score1 = compute_boundary_score(synthetic_video_path, roi_bboxes, sample_every=1)
        score5 = compute_boundary_score(synthetic_video_path, roi_bboxes, sample_every=5)
        # Both should be valid non-negative floats
        assert score1 >= 0.0
        assert score5 >= 0.0


# ── Test: Lip Sync Score (CPU-only — should gracefully return sentinel) ──────


class TestLipSyncScore:
    def test_no_gpu_returns_sentinel(self, synthetic_video_path, synthetic_audio_path):
        """On CPU-only machines without device override, LSE score should return sentinel values."""
        from src.qc.lip_sync_score import compute_lse_score

        result = compute_lse_score(synthetic_video_path, synthetic_audio_path)
        assert isinstance(result, dict)
        assert "lse_distance" in result
        assert "lse_confidence" in result
        assert "gpu_available" in result
        assert "n_windows" in result

        if not result["gpu_available"]:
            assert result["lse_distance"] == -1.0
            assert result["lse_confidence"] == -1.0

    def test_forced_cpu_scoring_evaluates_windows(self, synthetic_video_path, synthetic_audio_path, roi_bboxes):
        """When device='cpu' is forced, SyncNet should successfully evaluate windows and produce real scores."""
        from src.qc.lip_sync_score import compute_lse_score

        result = compute_lse_score(
            synthetic_video_path,
            synthetic_audio_path,
            roi_bboxes=roi_bboxes,
            device="cpu",
        )
        assert isinstance(result, dict)
        assert result["n_windows"] > 0
        assert isinstance(result["lse_distance"], float)
        assert result["lse_distance"] >= 0.0
        assert isinstance(result["lse_confidence"], float)
        assert -1.0 <= result["lse_confidence"] <= 1.0



# ── Test: Status computation ─────────────────────────────────────────────────


class TestStatusComputation:
    def test_pass(self):
        assert _status(5.0, {"pass": 8.0, "warn": 10.0}, higher_is_better=False) == "pass"

    def test_warn(self):
        assert _status(9.0, {"pass": 8.0, "warn": 10.0}, higher_is_better=False) == "warn"

    def test_fail(self):
        assert _status(12.0, {"pass": 8.0, "warn": 10.0}, higher_is_better=False) == "fail"

    def test_skip_negative(self):
        assert _status(-1.0, {"pass": 8.0, "warn": 10.0}, higher_is_better=False) == "skip"

    def test_higher_is_better_pass(self):
        assert _status(6.0, {"pass": 5.0, "warn": 3.0}, higher_is_better=True) == "pass"

    def test_higher_is_better_warn(self):
        assert _status(4.0, {"pass": 5.0, "warn": 3.0}, higher_is_better=True) == "warn"

    def test_higher_is_better_fail(self):
        assert _status(2.0, {"pass": 5.0, "warn": 3.0}, higher_is_better=True) == "fail"


# ── Test: QC Report ──────────────────────────────────────────────────────────


class TestQCReport:
    def test_generates_html_and_json(
        self, synthetic_video_path, synthetic_audio_path, roi_bboxes
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_qc_report(
                synthetic_video_path,
                synthetic_audio_path,
                roi_bboxes,
                output_dir=tmpdir,
            )

            # Check JSON was created and is non-empty
            json_path = result.get("report_json_path")
            assert json_path is not None
            assert Path(json_path).exists()
            assert Path(json_path).stat().st_size > 0
            with open(json_path) as f:
                data = json.load(f)
            assert "lip_sync" in data
            assert "flicker" in data
            assert "sharpness" in data
            assert "boundary" in data

            # Check HTML was created and is non-empty
            html_path = result.get("report_html_path")
            assert html_path is not None
            assert Path(html_path).exists()
            assert Path(html_path).stat().st_size > 100  # Should be substantial
            html_content = Path(html_path).read_text()
            assert "QC Report" in html_content
            assert "Flicker Score" in html_content

    def test_result_dict_structure(
        self, synthetic_video_path, synthetic_audio_path, roi_bboxes
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_qc_report(
                synthetic_video_path,
                synthetic_audio_path,
                roi_bboxes,
                output_dir=tmpdir,
            )
            assert "lip_sync" in result
            assert "flicker" in result
            assert "sharpness" in result
            assert "boundary" in result
            assert "timestamp" in result

            # Flicker and boundary should be non-negative
            assert result["flicker"]["score"] >= 0.0
            assert result["boundary"]["score"] >= 0.0
            assert result["sharpness"]["sharpness_ratio"] >= 0.0
