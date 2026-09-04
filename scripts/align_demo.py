#!/usr/bin/env python3
"""Align dubbed audio and write a frame-indexed viseme timeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.forced_alignment import align_audio
from src.preprocessing.viseme_mapping import build_viseme_timeline, phonemes_to_visemes

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a WhisperX viseme timeline")
    parser.add_argument("audio", type=Path, help="Path to dubbed WAV audio")
    parser.add_argument("--transcript", type=Path, help="Optional transcript text file")
    parser.add_argument("--fps", type=float, required=True, help="Target video frame rate")
    parser.add_argument("--total-frames", type=int, required=True, help="Target video frame count")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/viseme_timeline.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    transcript = args.transcript.read_text(encoding="utf-8") if args.transcript else ""
    aligned = align_audio(str(args.audio), transcript, args.fps)
    visemes = phonemes_to_visemes(aligned)
    timeline = build_viseme_timeline(visemes, args.total_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "phonemes": visemes,
        "viseme_timeline": timeline,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("First aligned phonemes:")
    for item in aligned[:20]:
        print(
            f"  {item['phoneme']:<8} {item['start_time_sec']:.3f}s–"
            f"{item['end_time_sec']:.3f}s "
            f"(frames {item['start_frame']}–{item['end_frame']})"
        )
    print(f"Wrote {len(visemes)} phonemes and {len(timeline)} frame labels to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
