# lipsync-pipeline

> **AI Lip-Sync Re-Alignment — Preserving the Original Human Vocal Performance**

An end-to-end research pipeline that synchronises lip movements in a source video to a
dubbed audio track (in a different language or voice) while keeping the actor's original
expression, timing feel, and vocal performance intact.

---

## Table of Contents

1. [Project Purpose](#project-purpose)
2. [Architecture Overview](#architecture-overview)
3. [Requirements](#requirements)
4. [Setup Instructions](#setup-instructions)
5. [Makefile Targets](#makefile-targets)
6. [Running the Pipeline](#running-the-pipeline)
7. [Pipeline Status](#pipeline-status)
8. [Project Structure](#project-structure)
9. [Contributing](#contributing)

---

## Project Purpose

Conventional dubbing replaces voice tracks but leaves the actor's lips out of sync with
the new language — a jarring visual mismatch for viewers.  This pipeline addresses that
by **re-animating only the lip region** of the original video to match the dubbed audio,
using a multi-stage approach:

| Goal | Approach |
|------|----------|
| Preserve actor expression / emotion | Operate strictly on the lip ROI; keep the rest of the face unchanged |
| Maintain temporal coherence | Optical-flow temporal smoothing + scene-aware processing |
| High visual fidelity | GAN-based refinement + super-resolution post-pass |
| Automated QA | SyncNet confidence + perceptual + audio quality metrics |

The pipeline is designed for **single-GPU inference** (RTX 4090 or A100, CUDA 12.1).

---

## Architecture Overview

```
Source Video ──┐
               ├─▶ [1] Preprocessing ──▶ [2] Generation ──▶ [3] Postprocessing ──▶ [4] QC ──▶ Output
Dubbed Audio ──┘
```

1. **Preprocessing** — Shot detection · face tracking · WhisperX word alignment · G2P phonemes
2. **Generation** — Coarse lip-sync model · high-detail refinement model
3. **Postprocessing** — Temporal smoothing · Poisson face blending · super-resolution
4. **QC** — SyncNet · FID · PESQ · SSIM · HTML report

---

## Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA driver | ≥ 525.85 (for CUDA 12.1) |
| GPU | NVIDIA RTX 4090 or A100 (≥ 16 GB VRAM) |
| PyTorch | 2.5.1 + cu121 |
| System packages | `ffmpeg`, `libsndfile1` |

Install system packages first (Ubuntu):

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 python3.10 python3.10-venv python3-pip
```

---

## Setup Instructions

### Option A — Python venv (recommended)

This is the primary supported setup method.  All Makefile targets assume `.venv/` in the
repo root.

```bash
# 1. Clone the repository
git clone <repo-url> lipsync-pipeline
cd lipsync-pipeline

# 2. Create the virtual environment, install deps, and verify CUDA
make setup

# 3. (Optional) Activate for interactive use
source .venv/bin/activate
```

`make setup` will:
- Create `.venv/` using `python3.10 -m venv`
- Upgrade pip / setuptools / wheel
- Install PyTorch 2.5.1 with CUDA 12.1 from the official PyTorch wheel index
- Install all other pinned dependencies from `requirements.txt`
- Run `scripts/check_cuda.py` — **the setup will FAIL loudly if CUDA is not accessible**

> **Why venv over conda?**  `pip` gives access to the official CUDA-variant torch wheels
> (`+cu121` suffix) the moment they are released, while the conda `pytorch` channel
> typically lags by days to weeks.  For reproducibility, all deps are pinned in
> `requirements.txt`.

### Option B — Conda (alternative)

```bash
conda env create -f environment.yml
conda activate lipsync
# CUDA check (conda env uses the same check_cuda.py script)
python scripts/check_cuda.py
```

### Verifying the CUDA environment manually

```bash
python scripts/check_cuda.py
```

Expected output on a healthy machine:

```
  torch version : 2.5.1+cu121
  CUDA compiled : 12.1

✓ CUDA is available — 1 device(s) found:
  [0] NVIDIA GeForce RTX 4090
      Total VRAM : 24.00 GB
      SM count   : 128
      Capability : 8.9

✓ Smoke test passed (tensor allocation + arithmetic on GPU).
```

If CUDA is **not** available the script prints a detailed diagnostic and exits with
code `1` — it will **never** silently fall back to CPU.

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make setup` | Create `.venv`, install all deps, verify CUDA |
| `make setup-conda` | Create conda env `lipsync` from `environment.yml` |
| `make check-cuda` | Re-run the CUDA verification script |
| `make test` | Run pytest on `tests/` with coverage report |
| `make demo` | Placeholder: run `src/pipeline.py` on sample data |
| `make lint` | flake8 + isort check |
| `make format` | Auto-format with black + isort |
| `make clean` | Delete `.venv/` and all `__pycache__/` dirs |

---

## Running the Pipeline

```bash
# Full pipeline (placeholder — no model logic yet)
.venv/bin/python src/pipeline.py \
  --input  data/inputs/sample.mp4 \
  --audio  data/inputs/sample_dub.wav \
  --output data/outputs/result.mp4

# Or via Makefile (requires sample files in data/inputs/)
make demo
```

> **Note on Testing Data:** The file `data/inputs/sample.mp4` provided in the repository might be a synthetic fallback video (generated via ffmpeg zoompan from a static public domain image). It is intended ONLY for smoke-testing the pipeline (e.g., verifying face tracking executes without crashing). Face tracking metrics from this synthetic file should not be used to judge real-world model performance. See `data/inputs/SOURCES.md` for media provenance.

---

## Pipeline Status

> Last updated: 2026-09-03

### ✅ Implemented

| Component | File | Notes |
|-----------|------|-------|
| Project scaffold | — | Full directory structure, git repo |
| Dependency spec | `requirements.txt`, `environment.yml` | Pinned, CUDA 12.1 compatible |
| CUDA verification | `scripts/check_cuda.py` | Hard-fails on no CUDA |
| Pipeline orchestrator | `src/pipeline.py` | Stub — wires all 4 stages |
| Unit tests | `tests/test_pipeline.py`, `tests/test_cuda.py` | 11 tests pass |
| Makefile | `Makefile` | setup, test, demo, lint, format, clean |

### 🚧 Planned / WIP

| Stage | Module | Status |
|-------|--------|--------|
| **Preprocessing** | | |
| Scene / shot detection | `src/preprocessing/scene_detection.py` | ✅ Implemented |
| Face detection + tracking | `src/preprocessing/face_tracking.py` | ✅ Implemented |
| Forced alignment + viseme timeline | `src/preprocessing/forced_alignment.py`, `src/preprocessing/viseme_mapping.py` | ✅ Implemented and validated with real audio; WhisperX words → `g2p_en` ARPAbet phonemes → Preston Blair visemes |
| MuseTalk coarse lip-sync wrapper | `src/generation/coarse_lipsync.py` | ✅ Implemented; Colab/T4 inference pending; see [Colab run instructions](docs/colab_run_instructions.md) |
| **Generation** | | |
| Coarse lip-sync model | `src/generation/coarse_model.py` | Planned |
| Refinement / detail GAN | `src/generation/refinement_model.py` | Planned |
| **Postprocessing** | | |
| Temporal smoother | `src/postprocessing/temporal_smoother.py` | Planned |
| Face blender | `src/postprocessing/face_blender.py` | Planned |
| Super-resolution | `src/postprocessing/super_res.py` | Planned |
| **Quality Control** | | |
| SyncNet metric | `src/qc/syncnet.py` | Planned |
| Perceptual metrics (FID, LPIPS) | `src/qc/perceptual.py` | Planned |
| Audio quality (PESQ) | `src/qc/audio_quality.py` | Planned |
| Report generator | `src/qc/reporter.py` | Planned |

### Alignment limitations

WhisperX supplies word timestamps and character alignments, not phonological
phonemes. The alignment stage therefore uses `g2p_en` to convert each English
transcript word to ARPAbet phonemes, then divides that word's WhisperX interval
uniformly among its phonemes. This produces real phoneme labels suitable for the
Preston Blair mapping, but the per-phoneme durations are an approximation rather
than wav2vec2 measurements. The generated JSON stores both the timestamped
`phonemes` and the frame-indexed `viseme_timeline`.

---

## Project Structure

```
lipsync-pipeline/
├── src/
│   ├── __init__.py
│   ├── pipeline.py              ← end-to-end orchestrator
│   ├── preprocessing/           ← scene detection, face tracking, forced alignment
│   ├── generation/              ← coarse + refinement lip-sync models
│   ├── postprocessing/          ← temporal smoothing, blending, super-res
│   └── qc/                      ← automated quality scoring
├── models/                      ← model checkpoints (gitignored)
├── data/
│   ├── inputs/                  ← source video + dub audio (gitignored)
│   └── outputs/                 ← generated results (gitignored)
├── notebooks/                   ← exploratory Jupyter notebooks
├── scripts/
│   └── check_cuda.py            ← CUDA environment verification
├── tests/
│   ├── test_cuda.py
│   └── test_pipeline.py
├── requirements.txt
├── environment.yml              ← conda alternative
├── .gitignore
├── Makefile
└── README.md
```

---

## Contributing

1. Fork the repo and create a feature branch
2. Run `make lint` before committing
3. Add or update tests for any new behaviour
4. Open a PR with a clear description of the change

---

*This project is a research prototype.  No production SLA is implied.*
