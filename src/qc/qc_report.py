"""
src/qc/qc_report.py
─────────────────────
Unified QC report generator.

Runs all four scoring modules (lip-sync, flicker, sharpness, boundary) and
generates a self-contained HTML report with:
  - Four numeric scores
  - Colour-coded pass/warn/fail indicators
  - Embedded video player
  - Explanatory notes

Threshold calibration note
──────────────────────────
The pass/warn/fail thresholds below are initial estimates based on:
  - Published literature values for LSE-D/LSE-C (Wav2Lip, VideoReTalking papers)
  - Reasonable engineering judgment for the other three metrics

⚠️  These thresholds NEED CALIBRATION against more real clips.  We currently
have only one real test clip, so the boundaries between warn/fail may shift
significantly once we process more diverse content (different faces, lighting,
speech rates, resolutions).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Threshold Constants ──────────────────────────────────────────────────────
# ⚠️  NEEDS CALIBRATION against more real clips.  These are starting points.

# LSE-D (lip-sync distance): lower = better sync
# Published baselines: Wav2Lip ≈ 6.5, ground truth ≈ 7.0-8.0, bad sync > 10
LSED_THRESHOLDS = {"pass": 8.0, "warn": 10.0}  # < pass = PASS, < warn = WARN, else FAIL

# LSE-C (lip-sync confidence): higher = better sync
# Published baselines: Wav2Lip ≈ 7.5, ground truth ≈ 6.0-7.0, bad sync < 3
LSEC_THRESHOLDS = {"pass": 5.0, "warn": 3.0}  # > pass = PASS, > warn = WARN, else FAIL

# Flicker (MSE between consecutive mouth ROIs): lower = smoother
# These are empirical — no published standard exists
# CALIBRATION STATUS: placeholder — based on visual inspection of one clip
FLICKER_THRESHOLDS = {"pass": 50.0, "warn": 150.0}  # < pass = PASS, < warn = WARN, else FAIL

# Sharpness ratio (mouth / reference): closer to 1.0 = better
# A ratio of 0.5 means the mouth is half as sharp as the rest of the face
# CALIBRATION STATUS: well-reasoned — ratio is inherently interpretable
SHARPNESS_THRESHOLDS = {"pass": 0.7, "warn": 0.4}  # > pass = PASS, > warn = WARN, else FAIL

# Boundary score (edge ratio at seam vs interior): closer to 1.0 = better
# CALIBRATION STATUS: placeholder — depends heavily on blur kernel, resolution
BOUNDARY_THRESHOLDS = {"pass": 1.3, "warn": 1.8}  # < pass = PASS, < warn = WARN, else FAIL


def _status(value: float, thresholds: dict, higher_is_better: bool = False) -> str:
    """Compute pass/warn/fail status from a value and threshold dict."""
    if value < 0:
        return "skip"  # GPU-dependent metric that couldn't be computed

    if higher_is_better:
        if value >= thresholds["pass"]:
            return "pass"
        elif value >= thresholds["warn"]:
            return "warn"
        else:
            return "fail"
    else:
        if value <= thresholds["pass"]:
            return "pass"
        elif value <= thresholds["warn"]:
            return "warn"
        else:
            return "fail"


_STATUS_COLORS = {
    "pass": "#22c55e",  # green
    "warn": "#f59e0b",  # amber
    "fail": "#ef4444",  # red
    "skip": "#6b7280",  # gray
}

_STATUS_EMOJI = {
    "pass": "✅",
    "warn": "⚠️",
    "fail": "❌",
    "skip": "⏭️",
}


def generate_qc_report(
    video_path: str,
    audio_path: str,
    roi_bboxes: list[tuple[int, int, int, int]],
    output_dir: Optional[str] = None,
) -> dict:
    """Run all QC scores and generate an HTML report.

    Args:
        video_path:  Path to the final composited video.
        audio_path:  Path to the dubbed audio file.
        roi_bboxes:  Per-frame ``(x, y, w, h)`` face bounding boxes.
        output_dir:  Directory for the HTML report.  Defaults to same dir
                     as *video_path*.

    Returns:
        dict with all scores and the path to the HTML report.
    """
    from src.qc.lip_sync_score import compute_lse_score
    from src.qc.flicker_score import compute_flicker_score
    from src.qc.sharpness_score import compute_sharpness_score
    from src.qc.boundary_score import compute_boundary_score

    video_p = Path(video_path)
    if output_dir is None:
        output_dir = str(video_p.parent)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Run all scorers ──────────────────────────────────────────────────────
    logger.info("Running QC scoring on %s ...", video_path)

    lse = compute_lse_score(video_path, audio_path, roi_bboxes)
    flicker = compute_flicker_score(video_path, roi_bboxes)
    sharpness = compute_sharpness_score(video_path, roi_bboxes)
    boundary = compute_boundary_score(video_path, roi_bboxes)

    # ── Compute statuses ─────────────────────────────────────────────────────
    lsed_status = _status(lse["lse_distance"], LSED_THRESHOLDS, higher_is_better=False)
    lsec_status = _status(lse["lse_confidence"], LSEC_THRESHOLDS, higher_is_better=True)
    flicker_status = _status(flicker, FLICKER_THRESHOLDS, higher_is_better=False)
    sharpness_status = _status(
        sharpness["sharpness_ratio"], SHARPNESS_THRESHOLDS, higher_is_better=True
    )
    boundary_status = _status(boundary, BOUNDARY_THRESHOLDS, higher_is_better=False)

    # ── Build results dict ───────────────────────────────────────────────────
    results = {
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "timestamp": datetime.now().isoformat(),
        "lip_sync": {
            "lse_distance": lse["lse_distance"],
            "lse_confidence": lse["lse_confidence"],
            "lse_distance_status": lsed_status,
            "lse_confidence_status": lsec_status,
            "gpu_available": lse["gpu_available"],
            "n_windows": lse["n_windows"],
        },
        "flicker": {
            "score": flicker,
            "status": flicker_status,
        },
        "sharpness": {
            "mouth_sharpness": sharpness["mouth_sharpness"],
            "reference_sharpness": sharpness["reference_sharpness"],
            "sharpness_ratio": sharpness["sharpness_ratio"],
            "status": sharpness_status,
            "n_frames_scored": sharpness["n_frames_scored"],
        },
        "boundary": {
            "score": boundary,
            "status": boundary_status,
        },
    }

    # ── Save JSON ────────────────────────────────────────────────────────────
    json_path = output_path / (video_p.stem + "_qc.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("QC JSON saved: %s", json_path)

    # ── Generate HTML ────────────────────────────────────────────────────────
    html_path = output_path / (video_p.stem + "_qc_report.html")
    html = _render_html(results, video_path)
    html_path.write_text(html, encoding="utf-8")
    logger.info("QC HTML report saved: %s", html_path)

    results["report_html_path"] = str(html_path)
    results["report_json_path"] = str(json_path)

    return results


def _render_html(results: dict, video_path: str) -> str:
    """Render a self-contained HTML report."""
    video_filename = Path(video_path).name

    def _score_row(label: str, value, status: str, description: str, fmt: str = ".4f") -> str:
        color = _STATUS_COLORS.get(status, "#6b7280")
        emoji = _STATUS_EMOJI.get(status, "")
        if isinstance(value, float) and value < 0:
            val_str = "N/A (no GPU)"
        elif isinstance(value, float):
            val_str = f"{value:{fmt}}"
        else:
            val_str = str(value)
        return f"""
        <tr>
          <td style="font-weight:600;">{label}</td>
          <td style="font-family:monospace; font-size:1.1em;">{val_str}</td>
          <td>
            <span style="display:inline-block; padding:2px 12px; border-radius:9999px;
                         background:{color}22; color:{color}; font-weight:700;
                         border:2px solid {color};">
              {emoji} {status.upper()}
            </span>
          </td>
          <td style="color:#6b7280; font-size:0.9em;">{description}</td>
        </tr>"""

    lip_sync = results["lip_sync"]
    flicker = results["flicker"]
    sharpness = results["sharpness"]
    boundary = results["boundary"]

    rows = "".join([
        _score_row(
            "LSE-D (Lip Sync Distance)",
            lip_sync["lse_distance"],
            lip_sync["lse_distance_status"],
            f"Lower = better sync. {lip_sync['n_windows']} windows scored.",
        ),
        _score_row(
            "LSE-C (Lip Sync Confidence)",
            lip_sync["lse_confidence"],
            lip_sync["lse_confidence_status"],
            "Higher = better sync (cosine similarity).",
        ),
        _score_row(
            "Flicker Score",
            flicker["score"],
            flicker["status"],
            "Mean squared diff between consecutive mouth ROIs. Lower = smoother.",
            ".2f",
        ),
        _score_row(
            "Sharpness Ratio",
            sharpness["sharpness_ratio"],
            sharpness["status"],
            f"Mouth / reference. mouth={sharpness['mouth_sharpness']:.1f}, "
            f"ref={sharpness['reference_sharpness']:.1f}. "
            f"Closer to 1.0 = better.",
        ),
        _score_row(
            "Boundary Score",
            boundary["score"],
            boundary["status"],
            "Edge ratio at seam vs. interior. ~1.0 = seamless.",
        ),
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QC Report — {video_filename}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --border: #334155;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 2rem;
      line-height: 1.6;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    h1 {{
      font-size: 1.8rem;
      margin-bottom: 0.25rem;
      background: linear-gradient(135deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; font-size: 0.9rem; }}
    .card {{
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .card h2 {{
      font-size: 1.1rem;
      margin-bottom: 1rem;
      color: #cbd5e1;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 0.75rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    th {{ color: var(--text-muted); font-weight: 500; font-size: 0.85rem; text-transform: uppercase; }}
    video {{
      width: 100%;
      border-radius: 8px;
      margin-top: 0.5rem;
    }}
    .caution {{
      background: #78350f22;
      border: 1px solid #f59e0b44;
      border-radius: 8px;
      padding: 1rem;
      margin-top: 1.5rem;
      color: #fbbf24;
      font-size: 0.85rem;
    }}
    .caution strong {{ color: #f59e0b; }}
    .timestamp {{ color: var(--text-muted); font-size: 0.8rem; margin-top: 1rem; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🎬 QC Report</h1>
    <p class="subtitle">{video_filename}</p>

    <div class="card">
      <h2>📊 Scores</h2>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Status</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>🎥 Output Video</h2>
      <video controls>
        <source src="{video_filename}" type="video/mp4">
        Your browser does not support video playback.
      </video>
    </div>

    <div class="caution">
      <strong>⚠️ Threshold Calibration Note:</strong>
      These pass/warn/fail thresholds are initial estimates and need calibration
      against more real clips.  We currently have only one real test clip, so
      boundary values may shift significantly with more diverse content.
    </div>

    <p class="timestamp">Generated: {results['timestamp']}</p>
  </div>
</body>
</html>"""

    return html
