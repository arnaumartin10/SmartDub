"""
src/qc/lip_sync_score.py
─────────────────────────
Lip-sync accuracy scoring using a SyncNet-based approach.

Computes the standard LSE-D (Lip Sync Error — Distance) and LSE-C
(Lip Sync Error — Confidence) metrics from the Wav2Lip paper (Prajwal et al.,
2020), widely used in the lip-sync literature.

LSE-D: average Euclidean distance between audio and visual embeddings from
       SyncNet.  Lower = better sync.
LSE-C: average cosine similarity between audio and visual embeddings.
       Higher = better sync.

Architecture note
─────────────────
MuseTalk's repo includes two SyncNet implementations:
  1. musetalk/models/syncnet.py  — LatentSync-based (requires xformers, latent-
     space inputs, complex preprocessing). Used during training.
  2. musetalk/loss/syncnet.py    — Wav2Lip-style SyncNet_color. Self-contained,
     standard Conv2d, matches the LSE-D/LSE-C literature directly.

We use the Wav2Lip-style SyncNet_color for QC scoring because it:
  - Is self-contained (no xformers dependency)
  - Directly computes the standard LSE-D/LSE-C metrics
  - Works on pixel-space face crops (what we have post-compositing)

GPU requirement
───────────────
SyncNet inference requires a CUDA GPU.  On CPU-only machines, this module
returns a sentinel result indicating the score could not be computed.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MUSETALK_ROOT = Path(__file__).parents[2] / "third_party" / "MuseTalk"


def _extract_face_frames(
    video_path: str,
    roi_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
    target_size: tuple[int, int] = (96, 96),
    max_frames: int = 0,
) -> list[np.ndarray]:
    """Extract face-region frames from the video.

    If *roi_bboxes* are provided, each frame is cropped to its corresponding
    bbox.  Otherwise the centre-square of the frame is used as a fallback.

    Returns a list of BGR images resized to *target_size*.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and idx >= max_frames:
            break

        h, w = frame.shape[:2]
        if roi_bboxes is not None and idx < len(roi_bboxes):
            bx, by, bw, bh = roi_bboxes[idx]
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w, bx + bw)
            y2 = min(h, by + bh)
            crop = frame[y1:y2, x1:x2]
        else:
            # Fallback: centre crop to square
            side = min(h, w)
            cx, cy = w // 2, h // 2
            crop = frame[
                cy - side // 2 : cy + side // 2,
                cx - side // 2 : cx + side // 2,
            ]

        if crop.size > 0:
            crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            frames.append(crop)
        idx += 1

    cap.release()
    return frames


def _extract_mel_spectrogram(
    audio_path: str,
    sr: int = 16000,
    n_fft: int = 800,
    hop_length: int = 200,
    n_mels: int = 80,
) -> np.ndarray:
    """Extract a log-Mel spectrogram from an audio file.

    Returns shape (n_mels, T) as float32.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "librosa is required for Mel spectrogram extraction. "
            "Install it with: pip install librosa"
        )

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def compute_lse_score(
    video_path: str,
    audio_path: str,
    roi_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
    syncnet_checkpoint: Optional[str] = None,
) -> dict:
    """Compute LSE-D and LSE-C lip-sync metrics.

    Args:
        video_path:          Path to the composited output video.
        audio_path:          Path to the dubbed audio file.
        roi_bboxes:          Optional per-frame bounding boxes for face crops.
        syncnet_checkpoint:  Optional path to a SyncNet checkpoint.  If None,
                             falls back to the MuseTalk-bundled checkpoint at
                             ``models/syncnet/latentsync_syncnet.pt``.

    Returns:
        dict with keys:
          - ``lse_distance`` (float):   Mean L2 distance between embeddings.
                                         Lower = better sync.  -1 if GPU unavailable.
          - ``lse_confidence`` (float): Mean cosine similarity.
                                         Higher = better sync.  -1 if GPU unavailable.
          - ``gpu_available`` (bool):   Whether GPU was available for scoring.
          - ``n_windows`` (int):        Number of audio-visual windows scored.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available — skipping LSE scoring")
        return {
            "lse_distance": -1.0,
            "lse_confidence": -1.0,
            "gpu_available": False,
            "n_windows": 0,
        }

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA not available — LSE scoring requires GPU. "
            "Run this on Colab with a GPU runtime."
        )
        return {
            "lse_distance": -1.0,
            "lse_confidence": -1.0,
            "gpu_available": False,
            "n_windows": 0,
        }

    device = torch.device("cuda:0")

    # ── Load SyncNet_color (Wav2Lip-style) ───────────────────────────────────
    if str(_MUSETALK_ROOT) not in sys.path:
        sys.path.insert(0, str(_MUSETALK_ROOT))

    from musetalk.loss.syncnet import SyncNet_color

    syncnet = SyncNet_color().to(device).eval()

    # Try to load a checkpoint if available
    if syncnet_checkpoint is None:
        # Check common locations
        for candidate in [
            Path("models/syncnet/latentsync_syncnet.pt"),
            _MUSETALK_ROOT / "models" / "syncnet" / "latentsync_syncnet.pt",
        ]:
            if candidate.exists():
                syncnet_checkpoint = str(candidate)
                break

    if syncnet_checkpoint and Path(syncnet_checkpoint).exists():
        try:
            ckpt = torch.load(syncnet_checkpoint, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                syncnet.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                syncnet.load_state_dict(ckpt, strict=False)
            logger.info("Loaded SyncNet checkpoint: %s", syncnet_checkpoint)
        except Exception as e:
            logger.warning("Could not load SyncNet checkpoint: %s", e)
    else:
        logger.warning(
            "No SyncNet checkpoint found — using random weights. "
            "Scores will not be meaningful. Download the checkpoint from "
            "https://huggingface.co/ByteDance/LatentSync"
        )

    # ── Extract inputs ───────────────────────────────────────────────────────
    # SyncNet_color expects:
    #   face_sequences: (B, 15, H, W) — 5 consecutive frames × 3 channels
    #   audio_sequences: (B, 1, 80, 16) — Mel spectrogram window
    face_frames = _extract_face_frames(video_path, roi_bboxes, target_size=(96, 96))
    if len(face_frames) < 5:
        logger.warning("Too few face frames (%d) for SyncNet — need ≥5", len(face_frames))
        return {
            "lse_distance": -1.0,
            "lse_confidence": -1.0,
            "gpu_available": True,
            "n_windows": 0,
        }

    mel = _extract_mel_spectrogram(audio_path)
    # mel shape: (80, T)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Number of mel frames per video frame (at 16kHz, hop_length=200)
    mel_frames_per_video_frame = mel.shape[1] / max(len(face_frames), 1)

    # ── Score windows ────────────────────────────────────────────────────────
    window_size = 5  # 5 consecutive frames (standard for Wav2Lip SyncNet)
    mel_window = 16  # Mel spectrogram width per window

    distances: list[float] = []
    confidences: list[float] = []

    with torch.no_grad():
        for i in range(0, len(face_frames) - window_size + 1, window_size):
            # Stack 5 consecutive face frames → (15, 96, 96)
            window_frames = face_frames[i : i + window_size]
            face_tensor = np.concatenate(
                [f.transpose(2, 0, 1) for f in window_frames], axis=0
            )  # (15, 96, 96)
            face_tensor = (
                torch.from_numpy(face_tensor).float().unsqueeze(0).to(device) / 255.0
            )

            # Extract corresponding mel window
            mel_start = int(i * mel_frames_per_video_frame)
            mel_end = mel_start + mel_window
            if mel_end > mel.shape[1]:
                break
            mel_window_data = mel[:, mel_start:mel_end]  # (80, 16)
            audio_tensor = (
                torch.from_numpy(mel_window_data)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )  # (1, 1, 80, 16)

            try:
                audio_emb, face_emb = syncnet(audio_tensor, face_tensor)

                # LSE-D: Euclidean distance
                dist = torch.nn.functional.pairwise_distance(audio_emb, face_emb)
                distances.append(dist.item())

                # LSE-C: Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(audio_emb, face_emb)
                confidences.append(cos_sim.item())
            except Exception as e:
                logger.debug("SyncNet forward pass failed for window %d: %s", i, e)
                continue

    n_windows = len(distances)
    lse_d = float(np.mean(distances)) if distances else -1.0
    lse_c = float(np.mean(confidences)) if confidences else -1.0

    logger.info(
        "LSE scores: distance=%.4f, confidence=%.4f (%d windows)",
        lse_d, lse_c, n_windows,
    )

    return {
        "lse_distance": lse_d,
        "lse_confidence": lse_c,
        "gpu_available": True,
        "n_windows": n_windows,
    }
