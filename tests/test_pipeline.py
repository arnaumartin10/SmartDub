"""
tests/test_pipeline.py
───────────────────────
Unit tests for the pipeline orchestrator.

All tests here use mocking — no GPU or model checkpoints required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make src importable when running pytest from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import main, parse_args, run_preprocessing, run_generation, run_postprocessing, run_qc


# ── Argument parsing ──────────────────────────────────────────────────────────


def test_parse_args_required_flags(tmp_path):
    """--input, --audio, --output are all required."""
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    out = tmp_path / "out.mp4"
    video.touch()
    audio.touch()

    args = parse_args(["--input", str(video), "--audio", str(audio), "--output", str(out)])
    assert args.input == video
    assert args.audio == audio
    assert args.output == out
    assert args.device == "cuda"  # default


def test_parse_args_missing_input():
    """Missing --input should raise SystemExit."""
    with pytest.raises(SystemExit):
        parse_args(["--audio", "a.wav", "--output", "o.mp4"])


# ── Stage stub return shapes ──────────────────────────────────────────────────


def test_run_preprocessing_returns_dict(tmp_path):
    result = run_preprocessing(tmp_path / "v.mp4", tmp_path / "a.wav")
    assert isinstance(result, dict)
    assert "status" in result


def test_run_generation_returns_dict():
    result = run_generation({"status": "placeholder"})
    assert isinstance(result, dict)
    assert "status" in result


def test_run_postprocessing_returns_dict(tmp_path):
    result = run_postprocessing({"status": "placeholder"}, tmp_path / "v.mp4")
    assert isinstance(result, dict)
    assert "status" in result


def test_run_qc_returns_dict():
    result = run_qc({"status": "placeholder"})
    assert isinstance(result, dict)
    assert "scores" in result


# ── Full pipeline main() ──────────────────────────────────────────────────────


def test_main_missing_input_returns_1(tmp_path):
    """main() should return 1 when the input video file does not exist."""
    audio = tmp_path / "audio.wav"
    audio.touch()
    rc = main(
        [
            "--input", str(tmp_path / "nonexistent.mp4"),
            "--audio", str(audio),
            "--output", str(tmp_path / "out.mp4"),
        ]
    )
    assert rc == 1


def test_main_missing_audio_returns_1(tmp_path):
    """main() should return 1 when the dubbed audio file does not exist."""
    video = tmp_path / "video.mp4"
    video.touch()
    rc = main(
        [
            "--input", str(video),
            "--audio", str(tmp_path / "nonexistent.wav"),
            "--output", str(tmp_path / "out.mp4"),
        ]
    )
    assert rc == 1


def test_main_happy_path_returns_0(tmp_path):
    """main() should return 0 when both input files exist (placeholder run)."""
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    video.touch()
    audio.touch()
    rc = main(
        [
            "--input", str(video),
            "--audio", str(audio),
            "--output", str(tmp_path / "outputs" / "out.mp4"),
        ]
    )
    assert rc == 0
