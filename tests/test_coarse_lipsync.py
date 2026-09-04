"""CPU-only contract tests for the MuseTalk wrapper.

Actual MuseTalk model loading and inference remain unvalidated until Colab/T4
execution because this test suite must not fake model inference.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(__file__).split("/tests/")[0])

from src.generation.coarse_lipsync import CoarseLipSyncGenerator


def test_prepare_frame_pads_non_square_bgr_crop():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[:, :40] = (10, 20, 30)

    prepared = CoarseLipSyncGenerator.prepare_frame(frame)

    assert prepared.shape == (256, 256, 3)
    assert prepared.dtype == np.uint8
    assert np.array_equal(prepared[0, 0], frame[0, 0])


def test_prepare_frame_rejects_invalid_shape():
    with pytest.raises(ValueError, match="HxWx3"):
        CoarseLipSyncGenerator.prepare_frame(np.zeros((32, 32), dtype=np.uint8))


def test_init_requires_cuda(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires an available CUDA GPU"):
        CoarseLipSyncGenerator(str(tmp_path), device="cuda")


def test_prepare_frame_keeps_square_dimensions():
    frame = np.full((64, 64, 3), 127, dtype=np.uint8)
    prepared = CoarseLipSyncGenerator.prepare_frame(frame)

    assert prepared.shape == (256, 256, 3)
    assert int(prepared.mean()) == 127
