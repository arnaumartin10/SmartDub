"""
src/generation/refinement.py
─────────────────────────────
Face refinement using GFPGAN.

Applies Tencent's GFPGAN (Generative Facial Prior GAN) to the coarse 256×256
generated face crop before it gets composited and scaled back into the full-res
frame.  This recovers crisp facial features (teeth, lips, skin pores) lost during
the lower-resolution generation step and establishes high-frequency detail.

Architecture & Pipeline Integration:
  MuseTalk crop (256×256)
        │
        ▼
  FaceRefiner.refine() ──► GFPGAN restoration (512×512) ──► resize to 256×256
        │
        ▼
  Geometric inversion (_unsquare_crop) ──► original bbox dimensions
        │
        ▼
  Colour matching (LAB) ──► Alpha blending (feathered jaw/chin mask)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_GFPGAN_ROOT = Path(__file__).parents[2] / "third_party" / "GFPGAN"


class FaceRefiner:
    """Wrapper for GFPGAN face restoration model."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        upscale: int = 1,
        arch: str = "clean",
        channel_multiplier: int = 2,
        _restorer: Optional[object] = None,
    ) -> None:
        """Initialize the GFPGAN FaceRefiner.

        Args:
            checkpoint_path: Path to the GFPGAN model checkpoint (e.g. GFPGANv1.4.pth).
            device: Target torch device ('cuda' or 'cpu').
            upscale: Upscaling factor for GFPGANer (default: 1).
            arch: GFPGAN architecture version (default: 'clean' for v1.4).
            channel_multiplier: Channel multiplier for StyleGAN2 backbone (default: 2).
            _restorer: Optional pre-configured restorer instance (used for testing).

        Raises:
            RuntimeError: If device is 'cuda' but CUDA is not available.
            FileNotFoundError: If checkpoint_path does not exist.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for GFPGAN FaceRefiner.")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for GFPGAN refinement on GPU, but torch.cuda.is_available() is False. "
                "Pass device='cpu' to force CPU evaluation or use --skip-refinement."
            )

        self.device = device
        self.checkpoint_path = checkpoint_path

        if _restorer is not None:
            self.restorer = _restorer
            logger.info("FaceRefiner initialized with custom/mock restorer.")
            return

        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be specified when _restorer is not provided.")

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"GFPGAN checkpoint not found: {checkpoint_path}")

        if str(_GFPGAN_ROOT) not in sys.path:
            sys.path.insert(0, str(_GFPGAN_ROOT))

        try:
            from gfpgan import GFPGANer

            self.restorer = GFPGANer(
                model_path=str(ckpt),
                upscale=upscale,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=None,
                device=torch.device(device),
            )
            logger.info("Loaded GFPGAN face refiner from: %s (device: %s)", checkpoint_path, device)
        except Exception as e:
            logger.error("Failed to initialize GFPGANer: %s", e)
            raise

    def refine(
        self,
        generated_crop: np.ndarray,
        target_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Run GFPGAN restoration on a generated face crop.

        Args:
            generated_crop: BGR uint8 image of the face crop (e.g. 256×256).
            target_size: Optional (width, height) to resize the restored face to.
                         Defaults to the input crop dimensions (e.g. (256, 256)).

        Returns:
            Refined BGR uint8 image with enhanced high-frequency detail.
            If restoration fails, gracefully returns a copy of the input crop.
        """
        if generated_crop is None or generated_crop.size == 0:
            raise ValueError("generated_crop is empty")
        if generated_crop.ndim != 3 or generated_crop.shape[2] != 3:
            raise ValueError(f"generated_crop must be HxWx3, got shape {generated_crop.shape}")

        in_h, in_w = generated_crop.shape[:2]
        out_w, out_h = target_size if target_size is not None else (in_w, in_h)

        try:
            # has_aligned=True indicates the crop is already a centered face
            # paste_back=False returns the restored face directly
            _, restored_faces, _ = self.restorer.enhance(
                generated_crop,
                has_aligned=True,
                only_center_face=True,
                paste_back=False,
                weight=0.5,
            )

            if restored_faces and len(restored_faces) > 0 and restored_faces[0] is not None:
                restored = restored_faces[0]
                # GFPGAN typically produces 512x512 output.
                # Resize back to target_size to prevent conflicting resizes in compositing.
                if (restored.shape[1], restored.shape[0]) != (out_w, out_h):
                    restored = cv2.resize(
                        restored, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4
                    )
                return restored.astype(np.uint8)
            else:
                logger.warning("GFPGAN returned no restored faces; returning unrefined crop.")
                return generated_crop.copy()
        except Exception as e:
            logger.warning(
                "GFPGAN enhancement failed for frame (%s); falling back to unrefined crop.", e
            )
            return generated_crop.copy()

    def refine_sequence(
        self,
        crops: list[np.ndarray],
        target_size: Optional[tuple[int, int]] = None,
    ) -> list[np.ndarray]:
        """Refine a sequence of face crops frame-by-frame.

        Args:
            crops: List of BGR uint8 face crops.
            target_size: Optional (width, height) for each output frame.

        Returns:
            List of refined BGR uint8 face crops.
        """
        refined = []
        for i, crop in enumerate(crops):
            refined_crop = self.refine(crop, target_size=target_size)
            refined.append(refined_crop)
            if (i + 1) % 50 == 0 or (i + 1) == len(crops):
                logger.info("Refined %d/%d face crops with GFPGAN", i + 1, len(crops))
        return refined
