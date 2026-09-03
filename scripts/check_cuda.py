"""
scripts/check_cuda.py
─────────────────────
CUDA availability verification script.

Called by `make check-cuda` and `make setup`.
Exits with code 1 if CUDA is not available so the caller (Makefile) fails loudly.
"""

import sys


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        print(f"[FATAL] Cannot import torch: {exc}", file=sys.stderr)
        print("        Run `make setup` first.", file=sys.stderr)
        sys.exit(1)

    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA compiled : {torch.version.cuda}")

    if not torch.cuda.is_available():
        # Diagnose common failure modes before aborting.
        diag: list[str] = []

        if torch.version.cuda is None:
            diag.append(
                "PyTorch was installed WITHOUT CUDA support "
                "(CPU-only wheel). Reinstall with the cu121 index URL."
            )
        else:
            diag.append(
                "PyTorch has CUDA compiled in, but the runtime cannot find "
                "a usable GPU. Possible causes:"
            )
            diag.append("  • No NVIDIA GPU present / driver not loaded")
            diag.append(
                "  • Driver version too old for CUDA 12.1 "
                "(need driver ≥ 525.85.12)"
            )
            diag.append(
                "  • Running inside a container without --gpus flag "
                "or nvidia-container-toolkit not installed"
            )
            diag.append(
                "  • CUDA_VISIBLE_DEVICES='NoDevFiles' or ='' "
                "set in environment"
            )

        print("\n[FATAL] torch.cuda.is_available() == False", file=sys.stderr)
        for line in diag:
            print(f"        {line}", file=sys.stderr)
        print(
            "\n        Stopping — CPU fallback is NOT acceptable for this pipeline.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── CUDA is available ──────────────────────────────────────────────────────
    device_count = torch.cuda.device_count()
    print(f"\n✓ CUDA is available — {device_count} device(s) found:")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        print(f"  [{i}] {props.name}")
        print(f"      Total VRAM : {total_gb:.2f} GB")
        print(f"      SM count   : {props.multi_processor_count}")
        print(f"      Capability : {props.major}.{props.minor}")

    # Quick smoke test — allocate a tiny tensor on GPU
    try:
        t = torch.zeros(1, device="cuda")
        _ = t + 1
        print("\n✓ Smoke test passed (tensor allocation + arithmetic on GPU).")
    except Exception as exc:
        print(f"\n[FATAL] Smoke test FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
