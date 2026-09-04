#!/usr/bin/env python3
"""Run preprocessing, alignment, and MuseTalk coarse generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.coarse_lipsync import CoarseLipSyncGenerator
from src.preprocessing.face_tracking import track_face
from src.preprocessing.forced_alignment import align_audio
from src.preprocessing.scene_detection import detect_scenes
from src.preprocessing.viseme_mapping import build_viseme_timeline, phonemes_to_visemes

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("generate_demo")


def _extract_face_crops(video_path: Path, scenes: list[tuple[int, int]]) -> list:
    crops = []
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
                    f"No face detection at frame {frame_index}; coarse generation "
                    "requires a crop for every audio-driven frame"
                )
            x, y, width, height = result["bounding_box"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + width)
            y2 = min(frame.shape[0], y + height)
            face_crop = frame[y1:y2, x1:x2].copy()
            if face_crop.size == 0:
                raise RuntimeError(f"Empty face crop at frame {frame_index}")
            crops.append(face_crop)
        cap.release()
    return crops


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MuseTalk coarse lip-sync generation")
    parser.add_argument("--video", type=Path, required=True, help="Source video path")
    parser.add_argument("--audio", type=Path, required=True, help="Dubbed WAV path")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"))
    parser.add_argument("--transcript", type=Path, help="Optional transcript for alignment logging")
    parser.add_argument("--output", type=Path, required=True, help="Raw generated crop video")
    args = parser.parse_args()

    for input_path in (args.video, args.audio):
        if not input_path.exists():
            parser.error(f"Input not found: {input_path}")
    transcript = ""
    if args.transcript:
        transcript = args.transcript.read_text(encoding="utf-8")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    scenes = detect_scenes(str(args.video))
    face_frames = _extract_face_crops(args.video, scenes)
    aligned = align_audio(str(args.audio), transcript, fps)
    visemes = phonemes_to_visemes(aligned)
    viseme_timeline = build_viseme_timeline(visemes, len(face_frames))
    logger.info(
        "Preprocessing complete: %d scenes, %d crops, %d phonemes",
        len(scenes),
        len(face_frames),
        len(aligned),
    )

    generator = CoarseLipSyncGenerator(str(args.checkpoint_dir))
    generated = generator.generate(face_frames, str(args.audio), viseme_timeline)
    if not generated:
        raise RuntimeError("MuseTalk returned no generated frames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    height, width = generated[0].shape[:2]
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise IOError(f"Could not open output video: {args.output}")
    try:
        for frame in generated:
            if frame.shape[:2] != (height, width):
                raise ValueError("MuseTalk returned frames with inconsistent shapes")
            writer.write(frame)
    finally:
        writer.release()
    logger.info("Wrote %d raw MuseTalk crops to %s", len(generated), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
