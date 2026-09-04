"""MuseTalk coarse audio-driven lip-sync generation.

The wrapper intentionally returns generated square crops. Full-frame placement and
blending belong to a later post-processing stage.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
import wave

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MUSETALK_ROOT = Path(__file__).parents[2] / "third_party" / "MuseTalk"


def _audio_duration(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_file:
        if wav_file.getframerate() <= 0:
            raise ValueError(f"Audio has invalid sample rate: {audio_path}")
        return wav_file.getnframes() / wav_file.getframerate()


def _square_musetalk_frame(frame: np.ndarray, size: int = 256) -> np.ndarray:
    """Convert a BGR crop to a padded square MuseTalk input."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("face_frames must contain HxWx3 numpy arrays")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("face frame dimensions must be positive")

    height, width = frame.shape[:2]
    if height != width:
        logger.warning(
            "MuseTalk expects square face crops; padding non-square crop %dx%d "
            "to square before resizing",
            width,
            height,
        )
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left
    padded = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REPLICATE,
    )
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)


def _resolve_model_paths(checkpoint_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Resolve MuseTalk root, VAE, Whisper, and v1.5 UNet paths."""
    root = checkpoint_dir
    if root.name in {"musetalk", "musetalkV15"} and (root.parent / "sd-vae").exists():
        root = root.parent
    version_dir = root / "musetalkV15"
    if not version_dir.exists():
        version_dir = root / "musetalk"
    unet_path = version_dir / ("unet.pth" if version_dir.name == "musetalkV15" else "pytorch_model.bin")
    config_path = version_dir / "musetalk.json"
    vae_dir = root / "sd-vae"
    whisper_dir = root / "whisper"
    missing = [
        str(path)
        for path in (unet_path, config_path, vae_dir, whisper_dir)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "MuseTalk checkpoints are incomplete; missing: " + ", ".join(missing)
        )
    return unet_path, config_path, vae_dir, whisper_dir


class CoarseLipSyncGenerator:
    """Callable MuseTalk 1.5 wrapper for preprocessed face crops."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda") -> None:
        import torch

        requested_device = device
        if device == "cuda":
            device = "cuda:0"
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "MuseTalk coarse generation requires an available CUDA GPU; "
                f"requested {requested_device!r} but CUDA is unavailable. "
                "Run this wrapper on a Colab T4 or another CUDA machine."
            )

        checkpoint_root = Path(checkpoint_dir).expanduser()
        if not checkpoint_root.exists():
            raise FileNotFoundError(f"MuseTalk checkpoint directory not found: {checkpoint_root}")
        unet_path, config_path, vae_dir, whisper_dir = _resolve_model_paths(checkpoint_root)

        if str(_MUSETALK_ROOT) not in sys.path:
            sys.path.insert(0, str(_MUSETALK_ROOT))

        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.utils import load_all_model
        from transformers import WhisperModel

        self.device = torch.device(device)
        self._torch = torch
        self._vae, self._unet, self._pe = load_all_model(
            unet_model_path=str(unet_path),
            vae_type=str(vae_dir),
            unet_config=str(config_path),
            device=self.device,
        )
        self._audio_processor = AudioProcessor(feature_extractor_path=str(whisper_dir))
        self._whisper = WhisperModel.from_pretrained(str(whisper_dir)).to(self.device).eval()
        self._dtype = torch.float16
        self._pe = self._pe.half().to(self.device)
        self._vae.vae = self._vae.vae.half().to(self.device)
        self._unet.model = self._unet.model.half().to(self.device).eval()
        self._whisper = self._whisper.half()
        logger.info("Loaded MuseTalk 1.5 coarse generator on %s", self.device)

    @staticmethod
    def prepare_frame(frame: np.ndarray) -> np.ndarray:
        """Public conversion helper used by the wrapper and CPU unit tests."""
        return _square_musetalk_frame(frame)

    def generate(
        self,
        face_frames: list[np.ndarray],
        audio_path: str,
        viseme_timeline: Optional[list] = None,
    ) -> list[np.ndarray]:
        """Generate square BGR crops driven directly by *audio_path*.

        ``viseme_timeline`` is accepted for pipeline compatibility but intentionally
        ignored: MuseTalk is audio-driven. Viseme conditioning is a later TODO.
        """
        if not face_frames:
            raise ValueError("face_frames must contain at least one crop")
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if viseme_timeline is not None:
            logger.debug("Ignoring viseme_timeline; MuseTalk is audio-driven")

        import torch
        from musetalk.utils.utils import datagen

        prepared_frames = [self.prepare_frame(frame) for frame in face_frames]
        duration = _audio_duration(audio_path)
        if duration <= 0:
            raise ValueError("Audio duration must be positive")
        fps = len(prepared_frames) / duration
        logger.info("MuseTalk input: %d crops, inferred %.3f fps", len(prepared_frames), fps)
        if abs(prepared_frames[0].shape[0] - prepared_frames[0].shape[1]) > 0:
            logger.warning("MuseTalk input remained non-square after preparation")

        audio_features, audio_length = self._audio_processor.get_audio_feature(audio_path)
        whisper_chunks = self._audio_processor.get_whisper_chunk(
            audio_features,
            self.device,
            self._dtype,
            self._whisper,
            audio_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        if not len(whisper_chunks):
            raise ValueError("MuseTalk produced no audio feature chunks")

        latents = [self._vae.get_latents_for_unet(frame) for frame in prepared_frames]
        self._pe.eval()
        outputs: list[np.ndarray] = []
        timesteps = torch.tensor([0], device=self.device)
        batches = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=latents,
            batch_size=8,
            delay_frame=0,
            device=self.device,
        )
        with torch.no_grad():
            for whisper_batch, latent_batch in batches:
                audio_prompt = self._pe(whisper_batch.to(self.device))
                latent_batch = latent_batch.to(self.device, dtype=self._dtype)
                predicted = self._unet.model(
                    latent_batch,
                    timesteps,
                    encoder_hidden_states=audio_prompt,
                ).sample
                outputs.extend(self._vae.decode_latents(predicted))

        if len(outputs) != len(prepared_frames):
            logger.warning(
                "MuseTalk generated %d crops for %d inputs; verify audio/video duration and fps",
                len(outputs),
                len(prepared_frames),
            )
        return [np.asarray(frame) for frame in outputs]
