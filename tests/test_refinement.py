"""
tests/test_refinement.py
─────────────────────────
Unit tests for GFPGAN FaceRefiner wrapper and pipeline refinement integration.
CPU-only tests with mocked restorers (no GPU or model weights required).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from src.generation.refinement import FaceRefiner
from src.postprocessing.compositing import composite_frame


@pytest.fixture
def sample_crop_256() -> np.ndarray:
    """A synthetic 256x256 BGR face crop."""
    crop = np.full((256, 256, 3), 120, dtype=np.uint8)
    cv2.circle(crop, (128, 128), 50, (40, 80, 200), -1)
    cv2.ellipse(crop, (128, 160), (30, 15), 0, 0, 360, (0, 0, 180), -1)
    return crop


class TestFaceRefinerInit:
    def test_cuda_missing_raises_runtime_error(self, tmp_path):
        """When device='cuda' is requested but CUDA is unavailable, raise RuntimeError."""
        dummy_ckpt = tmp_path / "dummy.pth"
        dummy_ckpt.write_bytes(b"dummy")

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is required for GFPGAN refinement"):
                FaceRefiner(checkpoint_path=str(dummy_ckpt), device="cuda")

    def test_missing_checkpoint_raises_file_not_found(self):
        """Non-existent checkpoint path raises FileNotFoundError on CPU."""
        with pytest.raises(FileNotFoundError, match="GFPGAN checkpoint not found"):
            FaceRefiner(checkpoint_path="non_existent_gfpgan_checkpoint.pth", device="cpu")

    def test_custom_restorer_injection(self):
        """Injecting _restorer bypasses checkpoint loading for testing."""
        mock_restorer = MagicMock()
        refiner = FaceRefiner(_restorer=mock_restorer, device="cpu")
        assert refiner.restorer is mock_restorer


class TestFaceRefinerInference:
    def test_fallback_on_restoration_exception(self, sample_crop_256):
        """If GFPGANer.enhance raises an exception, return unrefined crop without crashing."""
        mock_restorer = MagicMock()
        mock_restorer.enhance.side_effect = RuntimeError("CUDA out of memory in StyleGAN2")

        refiner = FaceRefiner(_restorer=mock_restorer, device="cpu")
        result = refiner.refine(sample_crop_256)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_crop_256.shape
        assert np.array_equal(result, sample_crop_256)

    def test_fallback_on_empty_restored_faces(self, sample_crop_256):
        """If GFPGANer returns empty list of faces, return unrefined crop."""
        mock_restorer = MagicMock()
        mock_restorer.enhance.return_value = ([], [], None)

        refiner = FaceRefiner(_restorer=mock_restorer, device="cpu")
        result = refiner.refine(sample_crop_256)

        assert np.array_equal(result, sample_crop_256)

    def test_successful_refine_and_resize(self, sample_crop_256):
        """GFPGAN output (512x512) is correctly resized back to target_size (256x256)."""
        restored_512 = np.full((512, 512, 3), 200, dtype=np.uint8)
        mock_restorer = MagicMock()
        mock_restorer.enhance.return_value = (None, [restored_512], None)

        refiner = FaceRefiner(_restorer=mock_restorer, device="cpu")
        result = refiner.refine(sample_crop_256, target_size=(256, 256))

        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
        assert np.all(result == 200)

    def test_refine_sequence(self, sample_crop_256):
        """refine_sequence processes a list of crops sequentially."""
        restored_512 = np.full((512, 512, 3), 180, dtype=np.uint8)
        mock_restorer = MagicMock()
        mock_restorer.enhance.return_value = (None, [restored_512], None)

        refiner = FaceRefiner(_restorer=mock_restorer, device="cpu")
        crops = [sample_crop_256.copy() for _ in range(4)]
        refined_list = refiner.refine_sequence(crops, target_size=(256, 256))

        assert len(refined_list) == 4
        assert all(r.shape == (256, 256, 3) for r in refined_list)

    def test_invalid_crop_inputs_raise_value_error(self):
        """Invalid or empty inputs raise ValueError."""
        mock_restorer = MagicMock()
        refiner = FaceRefiner(_restorer=mock_restorer, device="cpu")

        with pytest.raises(ValueError, match="empty"):
            refiner.refine(np.array([]))

        with pytest.raises(ValueError, match="HxWx3"):
            refiner.refine(np.zeros((100, 100), dtype=np.uint8))


class TestCompositingWithRefiner:
    def test_composite_frame_with_refiner(self, sample_crop_256):
        """composite_frame successfully invokes refiner before geometric inversion."""
        orig_frame = np.full((500, 400, 3), 100, dtype=np.uint8)
        bbox = (100, 150, 120, 140)
        landmarks = [(100 + i * 2, 150 + (i % 10) * 5) for i in range(478)]

        restored_512 = np.full((512, 512, 3), 220, dtype=np.uint8)
        mock_restorer = MagicMock()
        mock_restorer.enhance.return_value = (None, [restored_512], None)
        refiner = FaceRefiner(_restorer=mock_restorer, device="cpu")

        composited = composite_frame(
            original_frame=orig_frame,
            generated_crop=sample_crop_256,
            bbox=bbox,
            landmarks=landmarks,
            refiner=refiner,
            blur_kernel=31,
            mask_dilation=10,
        )

        assert composited.shape == orig_frame.shape
        assert composited.dtype == np.uint8
        mock_restorer.enhance.assert_called_once()
