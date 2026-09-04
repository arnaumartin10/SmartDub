"""Fast unit tests for alignment normalization and viseme timeline creation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import wave

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import forced_alignment
from src.preprocessing.viseme_mapping import (
    STANDARD_ARPABET_PHONEMES,
    build_viseme_timeline,
    phonemes_to_visemes,
)


def _write_tone(path: Path, seconds: float = 2.0) -> Path:
    sample_rate = 16_000
    samples = (32767 * np.sin(
        2 * np.pi * 440 * np.arange(int(seconds * sample_rate)) / sample_rate
    )).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return path


def _mock_alignment(*_args, **_kwargs):
    return {
        "segments": [
            {
                "words": [
                    {"word": "me", "start": 0.10, "end": 0.30},
                    {"word": "fee", "start": 0.30, "end": 0.60},
                    {"word": "up", "start": 0.60, "end": 0.90},
                ]
            }
        ]
    }


def _mock_g2p(word):
    return {
        "me": ["M", "IY1"],
        "fee": ["F", "IY1"],
        "up": ["AH1", "P"],
        "creating": ["K", "R", "IY1", "T", "IH0", "NG"],
    }[word]


def test_align_audio_signature_and_frame_conversion(tmp_path, monkeypatch):
    audio = _write_tone(tmp_path / "tone.wav")
    monkeypatch.setattr(forced_alignment, "_run_whisperx", _mock_alignment)
    monkeypatch.setattr(forced_alignment, "_load_g2p", lambda: _mock_g2p)

    result = forced_alignment.align_audio(str(audio), "me", 25.0)

    assert isinstance(result, list)
    assert [item["phoneme"] for item in result] == ["M", "IY1", "F", "IY1", "AH1", "P"]
    assert result[0]["start_frame"] == 2
    assert result[0]["end_frame"] == 4
    assert result[1]["start_time_sec"] == 0.2
    assert all(result[i]["end_time_sec"] <= result[i + 1]["start_time_sec"] for i in range(len(result) - 1))


def test_empty_transcript_uses_asr_fallback(tmp_path, monkeypatch, caplog):
    audio = _write_tone(tmp_path / "tone.wav")
    calls = []
    caplog.set_level(logging.INFO)

    def mocked_asr(audio_path, transcript, duration):
        calls.append((audio_path, transcript, duration))
        return _mock_alignment()

    monkeypatch.setattr(forced_alignment, "_run_whisperx", mocked_asr)
    monkeypatch.setattr(forced_alignment, "_load_g2p", lambda: _mock_g2p)
    result = forced_alignment.align_audio(str(audio), "", 30.0)

    assert result
    assert calls[0][1] == ""
    assert "ASR" in caplog.text


def test_missing_audio_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        forced_alignment.align_audio("missing.wav", "hello", 25.0)


def test_silent_audio_is_rejected(tmp_path):
    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(np.zeros(16_000, dtype=np.int16).tobytes())
    with pytest.raises(ValueError, match="silent"):
        forced_alignment.align_audio(str(silent), "hello", 25.0)


def test_viseme_mapping_and_timeline_length():
    aligned = [
        {"phoneme": "M", "start_frame": 2, "end_frame": 3},
        {"phoneme": "F", "start_frame": 6, "end_frame": 7},
    ]
    visemes = phonemes_to_visemes(aligned)
    timeline = build_viseme_timeline(visemes, 10)

    assert [item["viseme"] for item in visemes] == ["MBP", "FV"]
    assert len(timeline) == 10
    assert timeline[:2] == ["REST", "REST"]
    assert timeline[2:4] == ["MBP", "MBP"]
    assert timeline[4:6] == ["REST", "REST"]
    assert timeline[6:8] == ["FV", "FV"]


def test_g2p_converts_word_to_real_phonemes_with_uniform_timing(monkeypatch):
    monkeypatch.setattr(forced_alignment, "_load_g2p", lambda: _mock_g2p)

    result = forced_alignment._words_to_phonemes(
        [{"word": "creating", "start_time_sec": 1.0, "end_time_sec": 2.2}],
        30.0,
    )

    assert [item["phoneme"] for item in result] == ["K", "R", "IY1", "T", "IH0", "NG"]
    assert result[0]["start_time_sec"] == 1.0
    assert result[-1]["end_time_sec"] == 2.2
    assert result[0]["start_frame"] == 30


def test_all_standard_arpabet_phonemes_have_explicit_viseme_mapping():
    aligned = [{"phoneme": phoneme} for phoneme in STANDARD_ARPABET_PHONEMES]

    mapped = phonemes_to_visemes(aligned)

    assert len(mapped) == len(STANDARD_ARPABET_PHONEMES)
    assert all(item["viseme"] != "REST" for item in mapped)


def test_unexpected_phoneme_does_not_silently_map_to_rest():
    with pytest.raises(ValueError, match="No viseme mapping"):
        phonemes_to_visemes([{"phoneme": "NOT_A_PHONEME"}])


def test_empty_alignment_produces_closed_mouth_timeline():
    assert build_viseme_timeline([], 4) == ["REST"] * 4


def test_asr_failure_is_reported(tmp_path, monkeypatch):
    audio = _write_tone(tmp_path / "tone.wav")

    def failed_asr(*_args, **_kwargs):
        raise ValueError("WhisperX ASR produced no transcript for this audio")

    monkeypatch.setattr(forced_alignment, "_run_whisperx", failed_asr)
    with pytest.raises(ValueError, match="no transcript"):
        forced_alignment.align_audio(str(audio), "", 25.0)
