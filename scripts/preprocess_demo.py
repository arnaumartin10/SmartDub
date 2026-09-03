#!/usr/bin/env python3
"""
scripts/preprocess_demo.py
───────────────────────────
End-to-end preprocessing demo: scene detection → face tracking → mouth ROI.

Outputs (in --output-dir, default data/outputs/preprocess_demo/):
  scenes.json           — detected scene boundaries
  debug_annotated.mp4   — original video with overlaid bboxes + mouth landmarks
  mouth_rois/           — one PNG per frame, cropped + padded mouth region

Usage:
  python scripts/preprocess_demo.py data/inputs/sample.mp4
  python scripts/preprocess_demo.py data/inputs/sample.mp4 --output-dir /tmp/out
  python scripts/preprocess_demo.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.face_tracking import track_face
from src.preprocessing.mouth_roi import draw_mouth_landmarks, extract_mouth_roi
from src.preprocessing.scene_detection import detect_scenes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_demo")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _draw_frame_overlay(
    frame: np.ndarray,
    result: dict | None,
    frame_idx: int,
    scene_idx: int,
) -> np.ndarray:
    """Annotate a single frame with tracking results and scene info."""
    out = frame.copy()

    # Scene label (top-left corner)
    cv2.putText(
        out,
        f"Scene {scene_idx:02d}  Frame {frame_idx}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if result is None:
        # No face — red warning text
        cv2.putText(
            out,
            "NO FACE DETECTED",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 230),
            2,
            cv2.LINE_AA,
        )
        return out

    # Bounding box (green)
    x, y, w, h = result["bounding_box"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)

    # Confidence label inside bbox top-right
    conf_text = f"conf {result['confidence']:.2f}"
    cv2.putText(
        out,
        conf_text,
        (x + w - 130, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 0),
        1,
        cv2.LINE_AA,
    )

    # Mouth landmarks overlay (inner: cyan, outer: green)
    if result.get("landmarks"):
        out = draw_mouth_landmarks(
            out,
            result["landmarks"],
            index_scheme="mediapipe",
            color_outer=(0, 220, 0),
            color_inner=(255, 200, 0),
            radius=2,
        )

    return out


def _open_writer(
    output_path: Path,
    cap: cv2.VideoCapture,
) -> cv2.VideoWriter:
    """Open a VideoWriter matching the source video's resolution and FPS."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"Cannot open VideoWriter for {output_path}")
    return writer


# ── Main ──────────────────────────────────────────────────────────────────────


def run(video_path: Path, output_dir: Path) -> dict:
    """
    Full preprocessing pipeline on *video_path*.

    Returns a summary dict with statistics for the calling script or tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = output_dir / "mouth_rois"
    roi_dir.mkdir(exist_ok=True)

    # ── 1. Scene detection ────────────────────────────────────────────────────
    logger.info("Step 1/3 — Scene detection ...")
    scenes = detect_scenes(str(video_path))
    logger.info("  → %d scene(s) detected", len(scenes))

    scenes_out = output_dir / "scenes.json"
    with open(scenes_out, "w") as f:
        json.dump(
            [{"scene_index": i, "start_frame": s, "end_frame": e}
             for i, (s, e) in enumerate(scenes)],
            f,
            indent=2,
        )
    logger.info("  → Saved scene boundaries: %s", scenes_out)

    # ── 2 + 3. Face tracking + mouth ROI per scene ───────────────────────────
    logger.info("Step 2–3/3 — Face tracking & mouth ROI extraction ...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    annotated_path = output_dir / "debug_annotated.mp4"
    writer = _open_writer(annotated_path, cap)
    cap.release()

    # Aggregate statistics
    total_frames = 0
    total_detected = 0
    confidence_sum = 0.0
    failed_frames: list[int] = []
    all_tracking: list[dict] = []

    for scene_idx, (s_start, s_end) in enumerate(scenes):
        logger.info(
            "  Scene %02d: frames %d–%d (%d frames)",
            scene_idx, s_start, s_end, s_end - s_start + 1,
        )
        scene_results = track_face(str(video_path), (s_start, s_end))

        # ── Read frames for annotation ────────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(s_start))

        for local_idx, result in enumerate(scene_results):
            frame_idx = s_start + local_idx
            ret, frame_bgr = cap.read()
            if not ret:
                logger.warning("  Frame %d: cap.read() failed", frame_idx)
                continue

            total_frames += 1

            if result is None:
                failed_frames.append(frame_idx)
                annotated = _draw_frame_overlay(frame_bgr, None, frame_idx, scene_idx)
                writer.write(annotated)
                continue

            total_detected += 1
            confidence_sum += result["confidence"]

            # Annotate debug video frame
            annotated = _draw_frame_overlay(frame_bgr, result, frame_idx, scene_idx)
            writer.write(annotated)

            # Extract and save mouth ROI
            try:
                crop, crop_bbox = extract_mouth_roi(
                    frame_bgr,
                    result["landmarks"],
                    padding=0.30,
                    index_scheme="mediapipe",
                )
                roi_path = roi_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(roi_path), crop)
            except ValueError as exc:
                logger.debug("  Frame %d: mouth ROI failed — %s", frame_idx, exc)

            # Record for JSON export
            all_tracking.append(
                {
                    "scene": scene_idx,
                    "frame": frame_idx,
                    "bbox": list(result["bounding_box"]),
                    "confidence": round(result["confidence"], 4),
                }
            )

        cap.release()

    writer.release()
    logger.info("  → Annotated video saved: %s", annotated_path)
    logger.info("  → Mouth ROI images saved in: %s", roi_dir)

    # Save full tracking data
    tracking_path = output_dir / "tracking.json"
    with open(tracking_path, "w") as f:
        json.dump(all_tracking, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_conf = confidence_sum / total_detected if total_detected > 0 else 0.0
    summary = {
        "video": str(video_path),
        "total_scenes": len(scenes),
        "total_frames_processed": total_frames,
        "faces_detected": total_detected,
        "faces_missed": len(failed_frames),
        "detection_rate_pct": round(100 * total_detected / max(total_frames, 1), 2),
        "avg_confidence": round(avg_conf, 4),
        "failed_frame_numbers": failed_frames,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Video              : {summary['video']}")
    print(f"  Scenes detected    : {summary['total_scenes']}")
    print(f"  Frames processed   : {summary['total_frames_processed']}")
    print(f"  Faces detected     : {summary['faces_detected']}")
    print(f"  Faces missed       : {summary['faces_missed']}")
    print(f"  Detection rate     : {summary['detection_rate_pct']:.1f}%")
    print(f"  Avg confidence     : {summary['avg_confidence']:.4f}")
    if summary["failed_frame_numbers"]:
        print(f"  Failed frames      : {summary['failed_frame_numbers'][:20]}", end="")
        if len(summary["failed_frame_numbers"]) > 20:
            print(f" ... ({len(summary['failed_frame_numbers'])} total)", end="")
        print()
    else:
        print("  Failed frames      : none")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="lipsync-pipeline preprocessing demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Path to source video file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/preprocess_demo"),
        help="Directory for all outputs",
    )
    args = parser.parse_args()

    if not args.video.exists():
        logger.error("Video file not found: %s", args.video)
        logger.error(
            "Place your talking-head video at data/inputs/sample.mp4 "
            "or pass a different path."
        )
        return 1

    summary = run(args.video, args.output_dir)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
