# ─────────────────────────────────────────────────────────────────────────────
# lipsync-pipeline Makefile
# Target OS: Ubuntu 22.04 | Python 3.10 | CUDA 12.8
# ─────────────────────────────────────────────────────────────────────────────

PYTHON      := python3.10
VENV        := .venv
PIP         := $(VENV)/bin/pip
PYTEST      := $(VENV)/bin/pytest
PYTHON_VENV := $(VENV)/bin/python

TORCH_INDEX := https://download.pytorch.org/whl/cu128

.PHONY: help setup setup-conda test demo check-cuda lint format clean

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "lipsync-pipeline — available targets:"
	@echo "  make setup        Create .venv and install all dependencies (pip)"
	@echo "  make setup-conda  Create conda env 'lipsync' from environment.yml"
	@echo "  make check-cuda   Verify CUDA is accessible via PyTorch"
	@echo "  make test         Run pytest on tests/"
	@echo "  make demo         Run the placeholder end-to-end pipeline demo"
	@echo "  make lint         Run flake8 + isort --check"
	@echo "  make format       Auto-format with black + isort"
	@echo "  make clean        Remove .venv and __pycache__ dirs"
	@echo ""

# ── pip / venv setup ──────────────────────────────────────────────────────────
setup:
	@echo ">>> Creating Python 3.10 virtual environment in .venv ..."
	$(PYTHON) -m venv $(VENV)
	@echo ">>> Upgrading pip / setuptools / wheel ..."
	$(PIP) install --upgrade pip setuptools wheel
	@echo ">>> Installing PyTorch 2.8.0 with CUDA 12.8 ..."
	$(PIP) install \
		torch==2.8.0+cu128 \
		torchvision==0.23.0+cu128 \
		torchaudio==2.8.0+cu128 \
		--index-url $(TORCH_INDEX)
	@echo ">>> Installing remaining dependencies ..."
	$(PIP) install -r requirements.txt \
		--extra-index-url $(TORCH_INDEX)
	@echo ""
	@echo "✓ Setup complete. Activate with:  source .venv/bin/activate"
	@$(MAKE) check-cuda

# ── conda setup (alternative) ─────────────────────────────────────────────────
setup-conda:
	@echo ">>> Creating conda env 'lipsync' from environment.yml ..."
	conda env create -f environment.yml
	@echo ""
	@echo "✓ Conda env ready. Activate with:  conda activate lipsync"

# ── CUDA check ────────────────────────────────────────────────────────────────
check-cuda:
	@echo ">>> Checking CUDA availability ..."
	$(PYTHON_VENV) scripts/check_cuda.py

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	@echo ">>> Running test suite ..."
	$(PYTEST) tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Demo (placeholder) ────────────────────────────────────────────────────────
demo:
	@echo ">>> Running end-to-end pipeline demo (placeholder) ..."
	$(PYTHON_VENV) src/pipeline.py \
		--input  data/inputs/sample_video.mp4 \
		--audio  data/inputs/sample_dub.wav \
		--output data/outputs/result.mp4

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	$(VENV)/bin/flake8 src/ tests/ scripts/ --max-line-length=100
	$(VENV)/bin/isort --check-only src/ tests/ scripts/

format:
	$(VENV)/bin/black src/ tests/ scripts/ --line-length=100
	$(VENV)/bin/isort src/ tests/ scripts/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned."
