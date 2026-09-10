#!/usr/bin/env python3
"""
scripts/side_by_side_compare.py
────────────────────────────────
Generate a side-by-side comparison video: original vs. final lipsync result.

Usage:
    python scripts/side_by_side_compare.py \
        --original data/inputs/sample.mp4 \
        --result data/outputs/final_lipsync.mp4 \
        --output data/outputs/comparison.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("side_by_side")

# Label styling
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 1.2
_FONT_THICKNESS = 3
_LABEL_COLOR = (255, 255, 255)
_LABEL_BG_COLOR = (0, 0, 0)
_LABEL_MARGIN = 15
_LABEL_PAD = 10


def _draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a labelled banner at the top of a frame."""
    out = frame.copy()
    (text_w, text_h), baseline = cv2.getTextSize(
        text, _FONT, _FONT_SCALE, _FONT_THICKNESS
    )
    # Semi-transparent background bar
    bar_h = text_h + baseline + 2 * _LABEL_PAD
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], bar_h), _LABEL_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    # Text
    x = (out.shape[1] - text_w) // 2
    y = _LABEL_PAD + text_h
    cv2.putText(out, text, (x, y), _FONT, _FONT_SCALE, _LABEL_COLOR, _FONT_THICKNESS, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate side-by-side comparison video"
    )
    parser.add_argument(
        "--original", type=Path, required=True, help="Original source video"
    )
    parser.add_argument(
        "--result", type=Path, required=True, help="Lipsync result video"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/comparison.mp4"),
        help="Output comparison video",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Max height per panel (for reasonable file size)",
    )
    args = parser.parse_args()

    for p in (args.original, args.result):
        if not p.exists():
            parser.error(f"File not found: {p}")

    cap_orig = cv2.VideoCapture(str(args.original))
    cap_result = cv2.VideoCapture(str(args.result))

    if not cap_orig.isOpened():
        raise IOError(f"Cannot open: {args.original}")
    if not cap_result.isOpened():
        raise IOError(f"Cannot open: {args.result}")

    fps_orig = cap_orig.get(cv2.CAP_PROP_FPS) or 25.0
    fps_result = cap_result.get(cv2.CAP_PROP_FPS) or fps_orig
    fps = min(fps_orig, fps_result)

    n_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    n_result = int(cap_result.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_orig, n_result)

    logger.info(
        "Original: %d frames @ %.1f fps | Result: %d frames @ %.1f fps",
        n_orig, fps_orig, n_result, fps_result,
    )
    logger.info("Will compare %d frames at %.1f fps", n_frames, fps)

    # Determine output dimensions: both panels resized to same height
    h_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_result = int(cap_result.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_result = int(cap_result.get(cv2.CAP_PROP_FRAME_WIDTH))

    target_h = min(args.max_height, h_orig, h_result)
    scale_orig = target_h / h_orig
    scale_result = target_h / h_result
    panel_w_orig = int(w_orig * scale_orig)
    panel_w_result = int(w_result * scale_result)
    total_w = panel_w_orig + panel_w_result + 4  # 4px divider

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (total_w, target_h),
    )
    if not writer.isOpened():
        raise IOError(f"Cannot open writer: {args.output}")

    for i in range(n_frames):
        ret1, frame_orig = cap_orig.read()
        ret2, frame_result = cap_result.read()
        if not ret1 or not ret2:
            break

        # Resize both panels to target height
        panel_orig = cv2.resize(
            frame_orig,
            (panel_w_orig, target_h),
            interpolation=cv2.INTER_AREA,
        )
        panel_result = cv2.resize(
            frame_result,
            (panel_w_result, target_h),
            interpolation=cv2.INTER_AREA,
        )

        # Add labels
        panel_orig = _draw_label(panel_orig, "Original")
        panel_result = _draw_label(panel_result, "SmartDub")

        # Create divider (white vertical line)
        divider = np.full((target_h, 4, 3), 200, dtype=np.uint8)

        # Concatenate horizontally
        combined = np.hstack([panel_orig, divider, panel_result])
        writer.write(combined)

        if (i + 1) % 100 == 0:
            logger.info("  Processed %d/%d frames", i + 1, n_frames)

    writer.release()
    cap_orig.release()
    cap_result.release()

    logger.info("Side-by-side comparison written to: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
