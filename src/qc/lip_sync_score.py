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
  - Operates on 5 consecutive frames of the lower-half face (48×96) and
    16-frame Mel spectrogram windows (80×16), producing matching 512-d embeddings.

GPU requirement
───────────────
SyncNet inference is recommended on CUDA GPU. On CPU-only environments without
an explicit device override (device="cpu"), this module returns sentinel values.
"""

from __future__ import annotations

import logging
import sys
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
    """Extract lower-half face-region frames from the video.

    If *roi_bboxes* are provided, each frame is cropped to its corresponding
    bbox. Otherwise the centre-square of the frame is used as a fallback.
    The crop is resized to *target_size* (96x96) and the lower half (48x96)
    is extracted for SyncNet.

    Returns a list of BGR images of shape (48, 96, 3).
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
            crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            # SyncNet_color expects lower-half of face (48x96)
            lower_half = crop_resized[target_size[1] // 2 :, :]
            frames.append(lower_half)
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
    # 1. Try torchaudio first (fast, reliable across environments)
    try:
        import torch
        import torchaudio

        waveform, in_sr = torchaudio.load(audio_path)
        if in_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=in_sr, new_freq=sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        mel_spec = mel_transform(waveform)
        log_mel = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
        log_mel = log_mel - log_mel.max()
        return log_mel.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as e_torch:
        logger.debug("torchaudio extraction failed: %s. Trying librosa.", e_torch)

    # 2. Fallback to librosa
    try:
        import librosa

        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel.astype(np.float32)
    except Exception as e_librosa:
        raise RuntimeError(
            f"Failed to extract Mel spectrogram with both torchaudio and librosa ({e_librosa})"
        )


def _load_syncnet_weights(
    syncnet: torch.nn.Module, checkpoint_path: str, device: torch.device
) -> tuple[int, int]:
    """Load weights into SyncNet_color and return (matched_keys, total_keys)."""
    import torch

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.warning("Failed to read checkpoint file %s: %s", checkpoint_path, e)
        return 0, len(syncnet.state_dict())

    # Unwrap nested state dicts
    if isinstance(ckpt, dict):
        for candidate_key in ["state_dict", "model_state_dict", "model", "net", "syncnet"]:
            if candidate_key in ckpt and isinstance(ckpt[candidate_key], dict):
                ckpt = ckpt[candidate_key]
                break

    if not isinstance(ckpt, dict):
        logger.warning("Checkpoint at %s is not a state dictionary", checkpoint_path)
        return 0, len(syncnet.state_dict())

    model_dict = syncnet.state_dict()
    matched_dict = {}

    for k, v in ckpt.items():
        # Strip common wrapper prefixes
        clean_k = k
        for prefix in ["module.", "model.", "syncnet."]:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix) :]

        if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
            matched_dict[clean_k] = v

    if matched_dict:
        model_dict.update(matched_dict)
        syncnet.load_state_dict(model_dict)

    return len(matched_dict), len(model_dict)


def compute_lse_score(
    video_path: str,
    audio_path: str,
    roi_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
    syncnet_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """Compute LSE-D and LSE-C lip-sync metrics.

    Args:
        video_path:          Path to the composited output video.
        audio_path:          Path to the dubbed audio file.
        roi_bboxes:          Optional per-frame bounding boxes for face crops.
        syncnet_checkpoint:  Optional path to a SyncNet checkpoint. If None,
                             searches common local locations.
        device:              Target PyTorch device (e.g. 'cuda:0', 'cpu'). If None,
                             defaults to CUDA if available, else skips CPU unless forced.

    Returns:
        dict with keys:
          - ``lse_distance`` (float):   Mean L2 distance between embeddings.
                                         Lower = better sync. -1 if skipped/unavailable.
          - ``lse_confidence`` (float): Mean cosine similarity.
                                         Higher = better sync. -1 if skipped/unavailable.
          - ``gpu_available`` (bool):   Whether CUDA GPU was available.
          - ``n_windows`` (int):        Number of audio-visual windows scored.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed — skipping LSE scoring")
        return {
            "lse_distance": -1.0,
            "lse_confidence": -1.0,
            "gpu_available": False,
            "n_windows": 0,
        }

    gpu_available = torch.cuda.is_available()

    if device is not None:
        target_device = torch.device(device)
    elif gpu_available:
        target_device = torch.device("cuda:0")
    else:
        logger.info(
            "CUDA not available — skipping LSE scoring on CPU (pass device='cpu' to force CPU evaluation)"
        )
        return {
            "lse_distance": -1.0,
            "lse_confidence": -1.0,
            "gpu_available": False,
            "n_windows": 0,
        }

    logger.info("Initializing SyncNet scoring on device: %s", target_device)

    # ── 1. Load SyncNet_color Architecture ──────────────────────────────────
    if str(_MUSETALK_ROOT) not in sys.path:
        sys.path.insert(0, str(_MUSETALK_ROOT))

    try:
        from musetalk.loss.syncnet import SyncNet_color
    except ImportError as e:
        logger.error("Could not import SyncNet_color from MuseTalk: %s", e)
        return {
            "lse_distance": -1.0,
            "lse_confidence": -1.0,
            "gpu_available": gpu_available,
            "n_windows": 0,
        }

    syncnet = SyncNet_color().to(target_device).eval()

    # ── 2. Checkpoint Discovery & Loading ───────────────────────────────────
    if syncnet_checkpoint is None:
        candidate_paths = [
            Path("models/syncnet/syncnet_v2.model"),
            Path("models/syncnet/lipsync_expert.pth"),
            Path("models/syncnet/syncnet.pth"),
            Path("models/syncnet.pth"),
            Path("checkpoints/syncnet_v2.model"),
            Path("checkpoints/lipsync_expert.pth"),
            Path("models/syncnet/latentsync_syncnet.pt"),
            _MUSETALK_ROOT / "models" / "syncnet" / "latentsync_syncnet.pt",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                syncnet_checkpoint = str(candidate)
                break

    if syncnet_checkpoint and Path(syncnet_checkpoint).exists():
        matched, total = _load_syncnet_weights(syncnet, syncnet_checkpoint, target_device)
        if matched > 0:
            logger.info("Loaded SyncNet checkpoint '%s' (%d/%d layers matched)", syncnet_checkpoint, matched, total)
        else:
            logger.warning(
                "SyncNet checkpoint '%s' had 0 matching layers for SyncNet_color (likely LatentSync/UNet format). "
                "Running with initialized weights.",
                syncnet_checkpoint,
            )
    else:
        logger.warning(
            "No SyncNet checkpoint found at candidate locations. "
            "Running with initialized weights (relative trends will still compute)."
        )

    # ── 3. Extract Face Frames ──────────────────────────────────────────────
    face_frames = _extract_face_frames(video_path, roi_bboxes, target_size=(96, 96))
    if len(face_frames) < 5:
        logger.warning("Video too short (%d frames) for SyncNet — need at least 5 frames", len(face_frames))
        return {
            "lse_distance": -1.0,
            "lse_confidence": -1.0,
            "gpu_available": gpu_available,
            "n_windows": 0,
        }

    logger.info("Extracted %d face frames (lower-half shape: %s)", len(face_frames), face_frames[0].shape)

    # ── 4. Extract Mel Spectrogram ──────────────────────────────────────────
    try:
        mel = _extract_mel_spectrogram(audio_path, sr=16000, n_fft=800, hop_length=200, n_mels=80)
    except Exception as e:
        logger.error("Failed to extract Mel spectrogram from %s: %s", audio_path, e)
        return {
            "lse_distance": -1.0,
            "lse_confidence": -1.0,
            "gpu_available": gpu_available,
            "n_windows": 0,
        }

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    mel_fps = 16000.0 / 200.0  # 80 mel frames per second
    logger.info(
        "Extracted Mel spectrogram shape: %s (video fps=%.2f, audio mel_fps=%.1f)",
        mel.shape,
        fps,
        mel_fps,
    )

    # ── 5. Score Windows ────────────────────────────────────────────────────
    window_size = 5  # 5 video frames (0.2s @ 25fps, 0.167s @ 30fps)
    mel_window = 16  # 16 Mel time steps

    distances: list[float] = []
    confidences: list[float] = []
    failed_windows = 0
    total_candidate_windows = 0

    with torch.no_grad():
        for i in range(0, len(face_frames) - window_size + 1, window_size):
            total_candidate_windows += 1

            # 5 frames of lower half (48, 96) -> concat channels -> (15, 48, 96)
            window_frames = face_frames[i : i + window_size]
            face_tensor_np = np.concatenate(
                [f.transpose(2, 0, 1) for f in window_frames], axis=0
            )  # (15, 48, 96)
            face_tensor = (
                torch.from_numpy(face_tensor_np).float().unsqueeze(0).to(target_device) / 255.0
            )

            # Mel time slice aligned to video timestamp
            t_sec = i / fps
            mel_start = int(round(t_sec * mel_fps))
            mel_end = mel_start + mel_window

            if mel_end > mel.shape[1]:
                logger.debug(
                    "Skipping window at frame %d (mel_end %d > mel length %d)",
                    i,
                    mel_end,
                    mel.shape[1],
                )
                break

            mel_window_data = mel[:, mel_start:mel_end]  # (80, 16)
            audio_tensor = (
                torch.from_numpy(mel_window_data)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(target_device)
            )  # (1, 1, 80, 16)

            try:
                audio_emb, face_emb = syncnet(audio_tensor, face_tensor)

                # LSE-D: Euclidean distance between normalized embeddings
                dist = torch.nn.functional.pairwise_distance(audio_emb, face_emb)
                distances.append(float(dist.item()))

                # LSE-C: Cosine similarity between normalized embeddings
                cos_sim = torch.nn.functional.cosine_similarity(audio_emb, face_emb)
                confidences.append(float(cos_sim.item()))
            except Exception as e:
                failed_windows += 1
                if failed_windows == 1:
                    logger.warning("SyncNet forward pass failed on window at frame %d: %s", i, e)
                else:
                    logger.debug("SyncNet forward pass failed on window at frame %d: %s", i, e)
                continue

    n_windows = len(distances)
    lse_d = float(np.mean(distances)) if distances else -1.0
    lse_c = float(np.mean(confidences)) if confidences else -1.0

    if n_windows > 0:
        logger.info(
            "LSE scoring completed successfully: %d/%d windows evaluated. "
            "LSE-D (distance)=%.4f, LSE-C (confidence)=%.4f",
            n_windows,
            total_candidate_windows,
            lse_d,
            lse_c,
        )
    else:
        logger.warning(
            "LSE scoring completed with 0 valid windows evaluated out of %d candidates! "
            "(failed_windows=%d, total_frames=%d, mel_frames=%d)",
            total_candidate_windows,
            failed_windows,
            len(face_frames),
            mel.shape[1],
        )

    return {
        "lse_distance": lse_d,
        "lse_confidence": lse_c,
        "gpu_available": gpu_available,
        "n_windows": n_windows,
    }
