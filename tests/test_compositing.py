"""
tests/test_compositing.py
──────────────────────────
CPU-only unit tests for compositing, colour matching, and temporal smoothing.

These tests use synthetic data (no GPU, no real video) to verify:
  1. Geometric inversion (square→rectangle) produces correct dimensions
  2. Alpha mask has feathered edges and values in [0, 1]
  3. Colour matching handles different-sized regions without crashing
  4. Full composite round-trip produces correct output shape
  5. Temporal smoothing basic functionality
"""

from __future__ import annotations

import numpy as np
import pytest

import cv2
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.postprocessing.compositing import (
    _unsquare_crop,
    _build_alpha_mask,
    composite_frame,
)
from src.postprocessing.color_matching import match_color
from src.postprocessing.temporal_smoothing import smooth_sequence


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_fake_landmarks(
    n_points: int = 478,
    frame_w: int = 1080,
    frame_h: int = 1920,
    face_center_x: float = 0.5,
    face_center_y: float = 0.4,
    face_radius: float = 0.15,
) -> list[tuple[float, float]]:
    """Generate synthetic 478-point landmarks distributed around a face center."""
    rng = np.random.RandomState(42)
    landmarks = []
    for _ in range(n_points):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, face_radius)
        x = face_center_x * frame_w + r * frame_w * np.cos(angle)
        y = face_center_y * frame_h + r * frame_h * np.sin(angle)
        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        landmarks.append((float(x), float(y)))
    return landmarks


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """1080×1920 BGR frame filled with a gradient."""
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    # Horizontal gradient in blue channel
    for x in range(1080):
        frame[:, x, 0] = int(255 * x / 1080)
    # Vertical gradient in green channel
    for y in range(1920):
        frame[y, :, 1] = int(255 * y / 1920)
    frame[:, :, 2] = 128  # constant red
    return frame


@pytest.fixture
def synthetic_crop_256() -> np.ndarray:
    """256×256 BGR crop (simulates MuseTalk output)."""
    crop = np.random.RandomState(123).randint(
        80, 200, (256, 256, 3), dtype=np.uint8
    )
    return crop


@pytest.fixture
def fake_bbox() -> tuple[int, int, int, int]:
    """Bounding box (x, y, w, h) for a face in a 1080×1920 frame."""
    return (400, 600, 200, 280)  # rectangular: w=200, h=280


@pytest.fixture
def fake_landmarks() -> list[tuple[float, float]]:
    """478-point fake landmarks."""
    return _make_fake_landmarks(478, 1080, 1920)


# ── Test: Geometric inversion ───────────────────────────────────────────────


class TestGeometricInversion:
    """The square-to-rectangle inversion must produce exact original dims."""

    def test_rectangular_bbox_dimensions(self):
        """Generate a rectangular crop, square it, then unsquare it.
        Output must match original w×h."""
        orig_w, orig_h = 200, 280

        # Forward path: simulate _square_musetalk_frame
        side = max(orig_h, orig_w)  # 280
        pad_top = (side - orig_h) // 2  # 0
        pad_left = (side - orig_w) // 2  # 40
        # Create a fake rectangular image
        rect = np.random.RandomState(99).randint(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)
        padded = cv2.copyMakeBorder(
            rect, pad_top, side - orig_h - pad_top,
            pad_left, side - orig_w - pad_left,
            cv2.BORDER_REPLICATE,
        )
        assert padded.shape[:2] == (side, side)
        squared = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_AREA)

        # Inverse path
        recovered = _unsquare_crop(squared, orig_w, orig_h)

        assert recovered.shape == (orig_h, orig_w, 3), (
            f"Expected ({orig_h}, {orig_w}, 3), got {recovered.shape}"
        )

    def test_square_bbox_noop(self):
        """When the original bbox is already square, inversion is just a resize."""
        crop_256 = np.ones((256, 256, 3), dtype=np.uint8) * 100
        recovered = _unsquare_crop(crop_256, 300, 300)
        assert recovered.shape == (300, 300, 3)

    def test_wide_bbox(self):
        """Wide bbox (w > h) — inversion must handle horizontal padding."""
        crop_256 = np.ones((256, 256, 3), dtype=np.uint8) * 150
        recovered = _unsquare_crop(crop_256, 400, 200)
        assert recovered.shape == (200, 400, 3)

    def test_tall_bbox(self):
        """Tall bbox (h > w) — inversion must handle vertical padding."""
        crop_256 = np.ones((256, 256, 3), dtype=np.uint8) * 150
        recovered = _unsquare_crop(crop_256, 150, 350)
        assert recovered.shape == (350, 150, 3)

    def test_small_bbox(self):
        """Very small bbox — inversion still works."""
        crop_256 = np.ones((256, 256, 3), dtype=np.uint8) * 50
        recovered = _unsquare_crop(crop_256, 30, 50)
        assert recovered.shape == (50, 30, 3)


# ── Test: Alpha mask ────────────────────────────────────────────────────────


class TestAlphaMask:
    """Alpha mask must have feathered edges and values in [0, 1]."""

    def test_mask_range(self, fake_landmarks):
        """All mask values must be in [0, 1]."""
        mask = _build_alpha_mask(
            (1920, 1080), fake_landmarks, (400, 600, 200, 280)
        )
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_mask_shape(self, fake_landmarks):
        """Mask shape must match frame dimensions."""
        mask = _build_alpha_mask(
            (1920, 1080), fake_landmarks, (400, 600, 200, 280)
        )
        assert mask.shape == (1920, 1080)

    def test_mask_dtype(self, fake_landmarks):
        """Mask must be float32."""
        mask = _build_alpha_mask(
            (1920, 1080), fake_landmarks, (400, 600, 200, 280)
        )
        assert mask.dtype == np.float32

    def test_feathered_edges(self, fake_landmarks):
        """Mask edges must be feathered (gradient), not binary step.

        We check that the mask contains intermediate values (not just 0 and 1).
        A hard binary mask would have only two unique values; a feathered one
        has many.
        """
        mask = _build_alpha_mask(
            (1920, 1080), fake_landmarks, (400, 600, 200, 280),
            blur_kernel=51,
        )
        unique_values = np.unique(np.round(mask, decimals=2))
        # A properly feathered mask should have many distinct levels
        assert len(unique_values) > 10, (
            f"Mask has only {len(unique_values)} distinct values — "
            "edges are not feathered"
        )

    def test_mask_has_nonzero_interior(self, fake_landmarks):
        """Mask must have some non-zero region (face area)."""
        mask = _build_alpha_mask(
            (1920, 1080), fake_landmarks, (400, 600, 200, 280)
        )
        assert mask.max() > 0, "Mask is entirely zero — hull computation failed"

    def test_fallback_with_few_landmarks(self):
        """With < 3 valid landmarks, mask falls back to bbox rectangle."""
        tiny_landmarks = [(100.0, 100.0), (200.0, 200.0)]
        mask = _build_alpha_mask(
            (500, 500), tiny_landmarks, (50, 50, 200, 200)
        )
        assert mask.shape == (500, 500)
        assert mask.max() > 0  # bbox fallback should still produce a mask


# ── Test: Colour matching ────────────────────────────────────────────────────


class TestColorMatching:
    """Colour matching must handle different sizes and preserve dtype/shape."""

    def test_same_size(self):
        """Same-sized inputs should work without errors."""
        gen = np.random.RandomState(1).randint(50, 200, (100, 100, 3), dtype=np.uint8)
        ref = np.random.RandomState(2).randint(80, 220, (100, 100, 3), dtype=np.uint8)
        result = match_color(gen, ref)
        assert result.shape == gen.shape
        assert result.dtype == np.uint8

    def test_different_sizes(self):
        """Different-sized inputs must not crash."""
        gen = np.random.RandomState(3).randint(50, 200, (80, 120, 3), dtype=np.uint8)
        ref = np.random.RandomState(4).randint(80, 220, (200, 300, 3), dtype=np.uint8)
        result = match_color(gen, ref)
        assert result.shape == gen.shape
        assert result.dtype == np.uint8

    def test_output_dtype(self):
        """Output must be uint8 BGR."""
        gen = np.ones((50, 50, 3), dtype=np.uint8) * 100
        ref = np.ones((50, 50, 3), dtype=np.uint8) * 200
        result = match_color(gen, ref)
        assert result.dtype == np.uint8

    def test_empty_generated_raises(self):
        """Empty generated region must raise ValueError."""
        gen = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        ref = np.ones((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="generated_region is empty"):
            match_color(gen, ref)

    def test_empty_surrounding_raises(self):
        """Empty surrounding region must raise ValueError."""
        gen = np.ones((50, 50, 3), dtype=np.uint8)
        ref = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="surrounding_region is empty"):
            match_color(gen, ref)

    def test_uniform_input(self):
        """Uniform (flat) input should not crash (std=0 edge case)."""
        gen = np.ones((50, 50, 3), dtype=np.uint8) * 128
        ref = np.ones((50, 50, 3), dtype=np.uint8) * 128
        result = match_color(gen, ref)
        assert result.shape == (50, 50, 3)


# ── Test: Full composite round-trip ──────────────────────────────────────────


class TestCompositeFrame:
    """Full composite must produce correct output dimensions."""

    def test_output_shape(
        self, synthetic_frame, synthetic_crop_256, fake_bbox, fake_landmarks
    ):
        """Output frame must match original frame dimensions."""
        result = composite_frame(
            synthetic_frame,
            synthetic_crop_256,
            fake_bbox,
            fake_landmarks,
        )
        assert result.shape == synthetic_frame.shape, (
            f"Expected {synthetic_frame.shape}, got {result.shape}"
        )

    def test_output_dtype(
        self, synthetic_frame, synthetic_crop_256, fake_bbox, fake_landmarks
    ):
        """Output must be uint8."""
        result = composite_frame(
            synthetic_frame,
            synthetic_crop_256,
            fake_bbox,
            fake_landmarks,
        )
        assert result.dtype == np.uint8

    def test_unmodified_outside_mask(
        self, synthetic_frame, synthetic_crop_256, fake_bbox, fake_landmarks
    ):
        """Pixels far from the face should be unmodified."""
        result = composite_frame(
            synthetic_frame,
            synthetic_crop_256,
            fake_bbox,
            fake_landmarks,
        )
        # Check a corner far from the face region (face is roughly at y=600..880)
        corner = synthetic_frame[0:50, 0:50]
        result_corner = result[0:50, 0:50]
        np.testing.assert_array_equal(corner, result_corner)

    def test_without_color_matching(
        self, synthetic_frame, synthetic_crop_256, fake_bbox, fake_landmarks
    ):
        """Composite without colour matching should still work."""
        result = composite_frame(
            synthetic_frame,
            synthetic_crop_256,
            fake_bbox,
            fake_landmarks,
            apply_color_match=False,
        )
        assert result.shape == synthetic_frame.shape


# ── Test: Temporal smoothing ─────────────────────────────────────────────────


class TestTemporalSmoothing:
    """Basic temporal smoothing tests."""

    def test_empty_input(self):
        """Empty input returns empty output."""
        assert smooth_sequence([]) == []

    def test_single_frame(self):
        """Single frame should be returned as-is."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = smooth_sequence([frame], window=3)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], frame)

    def test_window_one_noop(self):
        """Window=1 should be a no-op."""
        frames = [
            np.ones((100, 100, 3), dtype=np.uint8) * i
            for i in [50, 100, 150]
        ]
        result = smooth_sequence(frames, window=1)
        assert len(result) == 3
        for orig, res in zip(frames, result):
            np.testing.assert_array_equal(orig, res)

    def test_output_length(self):
        """Output length must match input length."""
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(10)]
        result = smooth_sequence(frames, window=5)
        assert len(result) == 10

    def test_output_dtype(self):
        """Output frames must be uint8."""
        frames = [
            np.random.RandomState(i).randint(0, 255, (50, 50, 3), dtype=np.uint8)
            for i in range(5)
        ]
        result = smooth_sequence(frames, window=3)
        for f in result:
            assert f.dtype == np.uint8

    def test_invalid_window_raises(self):
        """Window < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            smooth_sequence([np.zeros((10, 10, 3), dtype=np.uint8)], window=0)
