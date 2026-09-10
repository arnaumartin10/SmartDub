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
import types
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_GFPGAN_ROOT = Path(__file__).parents[2] / "third_party" / "GFPGAN"


def _ensure_basicsr_shim() -> None:
    """Ensure minimal basicsr modules exist so GFPGAN architecture can import without pip basicsr."""
    if "basicsr" not in sys.modules:
        basicsr = types.ModuleType("basicsr")
        basicsr_archs = types.ModuleType("basicsr.archs")
        basicsr_arch_util = types.ModuleType("basicsr.archs.arch_util")
        basicsr_utils = types.ModuleType("basicsr.utils")
        basicsr_registry = types.ModuleType("basicsr.utils.registry")

        class DummyRegistry:
            def register(self):
                def decorator(cls):
                    return cls
                return decorator

        basicsr_registry.ARCH_REGISTRY = DummyRegistry()
        basicsr_arch_util.default_init_weights = lambda module, scale=1: None

        basicsr.archs = basicsr_archs
        basicsr.archs.arch_util = basicsr_arch_util
        basicsr.utils = basicsr_utils
        basicsr.utils.registry = basicsr_registry

        sys.modules["basicsr"] = basicsr
        sys.modules["basicsr.archs"] = basicsr_archs
        sys.modules["basicsr.archs.arch_util"] = basicsr_arch_util
        sys.modules["basicsr.utils"] = basicsr_utils
        sys.modules["basicsr.utils.registry"] = basicsr_registry


class _DirectGFPGANRestorer:
    """Self-contained GFPGAN restorer that runs GFPGANv1Clean directly via PyTorch.

    Bypasses facexlib/basicsr build dependencies and executes on pre-aligned crops.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        channel_multiplier: int = 2,
    ) -> None:
        import torch

        _ensure_basicsr_shim()

        if str(_GFPGAN_ROOT) not in sys.path:
            sys.path.insert(0, str(_GFPGAN_ROOT))

        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

        self.device = device
        self.model = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=channel_multiplier,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True,
        )

        loadnet = torch.load(model_path, map_location=device, weights_only=False)
        key = "params_ema" if "params_ema" in loadnet else "params"
        state_dict = loadnet[key] if key in loadnet else loadnet
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(device)

    def enhance(
        self,
        img: np.ndarray,
        has_aligned: bool = True,
        only_center_face: bool = True,
        paste_back: bool = False,
        weight: float = 0.5,
    ) -> tuple[list[np.ndarray], list[np.ndarray], Optional[np.ndarray]]:
        import torch

        # 1. Resize input crop to 512x512
        crop_512 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        # 2. Convert BGR uint8 -> RGB float in [-1, 1]
        rgb = cv2.cvtColor(crop_512, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        )
        tensor = (tensor - 0.5) / 0.5

        # 3. Model inference
        with torch.no_grad():
            output = self.model(tensor, return_rgb=False, weight=weight)[0]

        # 4. Convert back to BGR uint8
        out_img = (output.squeeze(0).permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5) * 255.0
        out_bgr = cv2.cvtColor(out_img.byte().cpu().numpy(), cv2.COLOR_RGB2BGR)

        return [crop_512], [out_bgr], None


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

        target_device = torch.device(device)

        # Initialize the self-contained direct restorer
        try:
            self.restorer = _DirectGFPGANRestorer(
                model_path=str(ckpt),
                device=target_device,
                channel_multiplier=channel_multiplier,
            )
            logger.info("Loaded GFPGAN face refiner from: %s (device: %s)", checkpoint_path, device)
        except Exception as e:
            logger.error("Failed to initialize GFPGAN restorer: %s", e)
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
                # GFPGAN produces 512x512 output.
                # Resize back to target_size to prevent conflicting resizes in downstream compositing.
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
