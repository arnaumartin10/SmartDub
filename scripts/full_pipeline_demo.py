#!/usr/bin/env python3
"""
scripts/full_pipeline_demo.py
──────────────────────────────
End-to-end lipsync pipeline: preprocessing → forced alignment → MuseTalk
generation → colour matching → compositing → temporal smoothing → ffmpeg remux.

Runnable from a Colab cell with the same CLI interface as generate_demo.py:

    python scripts/full_pipeline_demo.py \
        --video data/inputs/sample.mp4 \
        --audio data/inputs/sample_dub.wav \
        --transcript data/inputs/sample_dub.txt \
        --checkpoint-dir models \
        --output data/outputs/final_lipsync.mp4
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── PyTorch / accelerate compatibility patches (same as generate_demo.py) ────
try:
    import torch
    import omegaconf

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            omegaconf.listconfig.ListConfig,
            omegaconf.dictconfig.DictConfig,
        ])

    _orig_torch_load = getattr(torch, "_orig_torch_load", torch.load)
    torch._orig_torch_load = _orig_torch_load

    def _safe_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _safe_torch_load
except Exception:
    pass

try:
    import accelerate.utils.memory

    if not hasattr(accelerate.utils.memory, "clear_device_cache"):
        def clear_device_cache(*args, **kwargs):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        accelerate.utils.memory.clear_device_cache = clear_device_cache
except Exception:
    pass

# ── Imports ──────────────────────────────────────────────────────────────────

from src.generation.coarse_lipsync import CoarseLipSyncGenerator
from src.postprocessing.compositing import composite_frame
from src.postprocessing.temporal_smoothing import smooth_sequence
from src.preprocessing.face_tracking import track_face
from src.preprocessing.forced_alignment import align_audio
from src.preprocessing.scene_detection import detect_scenes
from src.preprocessing.viseme_mapping import build_viseme_timeline, phonemes_to_visemes

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("full_pipeline_demo")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_tracking_and_crops(
    video_path: Path,
    scenes: list[tuple[int, int]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Extract face crops AND preserve tracking metadata for compositing.

    Returns:
        (original_frames, face_crops, tracking_results)
        - original_frames: full-resolution BGR frames in scene order
        - face_crops: rectangular face crops (one per frame)
        - tracking_results: list of dicts with bounding_box + landmarks
    """
    original_frames: list[np.ndarray] = []
    face_crops: list[np.ndarray] = []
    tracking_data: list[dict] = []

    for scene_start, scene_end in scenes:
        tracking = track_face(str(video_path), (scene_start, scene_end))

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(scene_start))

        for frame_index, result in enumerate(tracking, start=scene_start):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
            if result is None:
                raise RuntimeError(
                    f"No face detection at frame {frame_index}; compositing "
                    "requires tracking data for every frame"
                )

            x, y, width, height = result["bounding_box"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + width)
            y2 = min(frame.shape[0], y + height)
            face_crop = frame[y1:y2, x1:x2].copy()

            if face_crop.size == 0:
                raise RuntimeError(f"Empty face crop at frame {frame_index}")

            original_frames.append(frame)
            face_crops.append(face_crop)
            tracking_data.append(result)

        cap.release()

    return original_frames, face_crops, tracking_data


def _remux_with_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> None:
    """Replace the video's audio track with the dubbed audio using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it: apt-get install ffmpeg"
        )

    # Write to a temp file first, then move (avoids partial output on failure)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(tmp_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg stderr:\n%s", result.stderr)
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg remux failed (exit {result.returncode})")

    tmp_path.rename(output_path)
    logger.info("Remuxed with audio: %s", output_path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full lipsync pipeline: preprocess → generate → composite → output"
    )
    parser.add_argument("--video", type=Path, required=True, help="Source video path")
    parser.add_argument("--audio", type=Path, required=True, help="Dubbed WAV path")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--transcript", type=Path, help="Optional transcript for alignment logging"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Final composited output video"
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Temporal smoothing window size (odd integer, 1=disabled)",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=51,
        help="Gaussian blur kernel for alpha mask feathering",
    )
    args = parser.parse_args()

    for input_path in (args.video, args.audio):
        if not input_path.exists():
            parser.error(f"Input not found: {input_path}")

    transcript = ""
    if args.transcript and args.transcript.exists():
        transcript = args.transcript.read_text(encoding="utf-8")

    # ── 1. Preprocessing ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[1/7] Scene detection")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    scenes = detect_scenes(str(args.video))

    logger.info("[2/7] Face tracking + crop extraction")
    original_frames, face_crops, tracking_data = _extract_tracking_and_crops(
        args.video, scenes
    )

    # ── 2. Forced alignment ──────────────────────────────────────────────────
    logger.info("[3/7] Forced alignment")
    aligned = align_audio(str(args.audio), transcript, fps)
    visemes = phonemes_to_visemes(aligned)
    viseme_timeline = build_viseme_timeline(visemes, len(face_crops))

    logger.info(
        "Preprocessing complete: %d scenes, %d frames, %d phonemes",
        len(scenes),
        len(face_crops),
        len(aligned),
    )

    # ── 3. MuseTalk generation ───────────────────────────────────────────────
    logger.info("[4/7] MuseTalk coarse generation")
    generator = CoarseLipSyncGenerator(str(args.checkpoint_dir))
    generated_crops = generator.generate(face_crops, str(args.audio), viseme_timeline)

    if not generated_crops:
        raise RuntimeError("MuseTalk returned no generated frames")
    if len(generated_crops) != len(original_frames):
        logger.warning(
            "MuseTalk generated %d crops for %d original frames; truncating to min",
            len(generated_crops),
            len(original_frames),
        )
    n_frames = min(len(generated_crops), len(original_frames))

    # ── 4 & 5. Colour matching + compositing ─────────────────────────────────
    logger.info("[5/7] Colour matching + compositing (%d frames)", n_frames)
    composited_frames: list[np.ndarray] = []
    for i in range(n_frames):
        composited = composite_frame(
            original_frame=original_frames[i],
            generated_crop=generated_crops[i],
            bbox=tracking_data[i]["bounding_box"],
            landmarks=tracking_data[i]["landmarks"],
            apply_color_match=True,
            blur_kernel=args.blur_kernel,
        )
        composited_frames.append(composited)
        if (i + 1) % 100 == 0:
            logger.info("  Composited %d/%d frames", i + 1, n_frames)

    # ── 6. Temporal smoothing ────────────────────────────────────────────────
    logger.info("[6/7] Temporal smoothing (window=%d)", args.smooth_window)
    landmarks_list = [
        tracking_data[i]["landmarks"] if i < len(tracking_data) else None
        for i in range(n_frames)
    ]
    smoothed_frames = smooth_sequence(
        composited_frames,
        window=args.smooth_window,
        landmarks_per_frame=landmarks_list,
    )

    # ── 7. Write video + remux audio ─────────────────────────────────────────
    logger.info("[7/7] Writing output video + audio remux")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write the video-only track first
    video_only = args.output.with_suffix(".video_only.mp4")
    frame_h, frame_w = smoothed_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_only),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise IOError(f"Could not open output video: {video_only}")
    try:
        for frame in smoothed_frames:
            writer.write(frame)
    finally:
        writer.release()

    # Remux with the dubbed audio
    _remux_with_audio(video_only, args.audio, args.output)

    # Clean up intermediate video-only file
    video_only.unlink(missing_ok=True)

    logger.info("=" * 60)
    logger.info("DONE — Final output: %s", args.output)
    logger.info("  %d frames @ %.2f fps", n_frames, fps)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
