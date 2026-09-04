"""WhisperX word alignment converted to G2P phoneme frame events."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any
import wave

import numpy as np

logger = logging.getLogger(__name__)


def _audio_duration_and_rms(audio_path: Path) -> tuple[float, float]:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            raw_audio = wav_file.readframes(frame_count)
    except Exception as exc:
        raise ValueError(f"Could not read audio file {audio_path}: {exc}") from exc
    if not raw_audio or sample_rate <= 0 or channels <= 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    if sample_width == 1:
        samples = (np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sample_width == 2:
        samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768
    elif sample_width == 4:
        samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2147483648
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")
    samples = samples.reshape(-1, channels).mean(axis=1)
    return float(samples.size / sample_rate), float(np.sqrt(np.mean(samples**2)))


def _extract_aligned_words(result: dict[str, Any]) -> list[dict]:
    """Extract WhisperX word intervals without treating characters as phonemes."""
    words: list[dict] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            label = str(word.get("word", word.get("text", ""))).strip()
            start = word.get("start")
            end = word.get("end")
            if label and start is not None and end is not None and float(end) > float(start):
                words.append(
                    {
                        "word": label,
                        "start_time_sec": float(start),
                        "end_time_sec": float(end),
                    }
                )
    return sorted(words, key=lambda item: (item["start_time_sec"], item["end_time_sec"]))


_ARPABET_TOKEN = re.compile(r"^[A-Z]+[0-2]?$")


def _load_g2p():
    try:
        from g2p_en import G2p
    except ImportError as exc:
        raise RuntimeError(
            "g2p_en is required to convert aligned words into phonemes. "
            "Install the pinned dependency from requirements.txt."
        ) from exc
    return G2p()


def _word_to_phonemes(word: str, g2p) -> list[str]:
    """Convert one English word to ARPAbet tokens, excluding punctuation."""
    return [token for token in g2p(word) if _ARPABET_TOKEN.fullmatch(str(token))]


def _words_to_phonemes(words: list[dict], video_fps: float) -> list[dict]:
    """Split each WhisperX word interval uniformly across its G2P phonemes."""
    g2p = _load_g2p()
    aligned: list[dict] = []
    for word_item in words:
        phonemes = _word_to_phonemes(word_item["word"], g2p)
        if not phonemes:
            logger.warning("G2P returned no phonemes for word %r", word_item["word"])
            continue
        word_start = word_item["start_time_sec"]
        word_end = word_item["end_time_sec"]
        phoneme_duration = (word_end - word_start) / len(phonemes)
        for index, phoneme in enumerate(phonemes):
            start_sec = word_start + index * phoneme_duration
            end_sec = word_end if index == len(phonemes) - 1 else start_sec + phoneme_duration
            start_frame = int(np.floor(start_sec * video_fps))
            aligned.append(
                {
                    "phoneme": phoneme,
                    "start_time_sec": start_sec,
                    "end_time_sec": end_sec,
                    "start_frame": start_frame,
                    "end_frame": max(int(np.ceil(end_sec * video_fps)) - 1, start_frame),
                }
            )
    return aligned


def _run_whisperx(audio_path: Path, transcript: str, duration: float) -> dict[str, Any]:
    try:
        import torch
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX is required for alignment. Install the pinned dependency "
            "from requirements.txt without changing numpy==1.26.4."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    audio = whisperx.load_audio(str(audio_path))
    model = whisperx.load_model("small", device, compute_type=compute_type)

    if transcript.strip():
        segments = [{"start": 0.0, "end": duration, "text": transcript.strip()}]
        language = None
        logger.info("Forced alignment mode: supplied transcript")
    else:
        logger.info("Forced alignment mode: WhisperX ASR transcript fallback")
        transcription = model.transcribe(audio, batch_size=16)
        segments = transcription.get("segments", [])
        language = transcription.get("language")
        if not segments or not any(str(item.get("text", "")).strip() for item in segments):
            raise ValueError("WhisperX ASR produced no transcript for this audio")

    if language is None:
        language = "en"
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    return whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )


def align_audio(audio_path: str, transcript: str, video_fps: float) -> list[dict]:
    """Align audio text, then convert WhisperX word intervals to G2P phonemes.

    WhisperX character alignments are deliberately not exposed as phonemes.
    Instead, each aligned word is converted with English G2P and its time span is
    divided uniformly across the resulting ARPAbet phonemes.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")

    duration, rms = _audio_duration_and_rms(path)
    if duration < 0.1:
        raise ValueError(f"Audio is too short to align meaningfully: {duration:.3f}s")
    if rms < 1e-5:
        raise ValueError("Audio is silent; no meaningful alignment is possible")
    if not transcript.strip():
        logger.info("No transcript supplied; requesting transcript from WhisperX ASR")

    result = _run_whisperx(path, transcript, duration)
    words = _extract_aligned_words(result)
    if not words:
        raise ValueError(
            "WhisperX returned no usable word timestamps for G2P phoneme alignment"
        )
    aligned = _words_to_phonemes(words, video_fps)
    if not aligned:
        raise ValueError("G2P returned no usable aligned phonemes")
    return aligned
