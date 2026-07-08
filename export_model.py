"""
export_model.py
───────────────
Export the YOLO fish-detection model to ONNX (Windows / any platform)
or directly to a TensorRT engine (Jetson / Linux with CUDA).

Usage
─────
  # Step 1 — on Windows: export to ONNX (portable, cross-platform)
  python export_model.py --format onnx

  # Step 2 — on Jetson: convert .pt or .onnx → TensorRT engine (FP16)
  python export_model.py --format engine --half

  # Step 2 alt — on Jetson: INT8 (fastest, slight accuracy trade-off)
  python export_model.py --format engine --int8

  # Override model path
  python export_model.py --format engine --half --model /path/to/best.pt

Notes
─────
• TensorRT .engine files are DEVICE-SPECIFIC.
  You MUST build the engine on the Jetson that will run it.
• ONNX is cross-platform and portable — export on Windows, ship to Jetson.
• imgsz must match what your model was trained on (default 640).
"""

import argparse
import sys
from pathlib import Path
import os

# ── Resolve default model path (same logic as core/config.py) ────────────────
_script_dir = Path(__file__).resolve().parent
_default_model = _script_dir.parent / "export" / "best.pt"
_model_path = Path(os.environ.get("FISH_MODEL_PATH", str(_default_model)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Export YOLO fish model to ONNX or TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", default=str(_model_path),
        help=f"Path to .pt weights file (default: {_model_path})",
    )
    p.add_argument(
        "--format", choices=["onnx", "engine"], default="onnx",
        help="Export format: 'onnx' (portable) or 'engine' (TensorRT, Jetson only)",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (must match training, default: 640)",
    )
    p.add_argument(
        "--half", action="store_true",
        help="FP16 precision — ~2x faster on Jetson GPU, recommended",
    )
    p.add_argument(
        "--int8", action="store_true",
        help="INT8 precision — fastest, may have slight accuracy trade-off",
    )
    p.add_argument(
        "--device", default="0",
        help="Device: '0' for first GPU, 'cpu' for CPU-only (default: 0)",
    )
    p.add_argument(
        "--batch", type=int, default=1,
        help="Batch size for engine (default: 1 for real-time inference)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌  Model not found: {model_path}")
        print(f"    Set FISH_MODEL_PATH env var or pass --model /path/to/best.pt")
        sys.exit(1)

    # ── Precision logic ───────────────────────────────────────────────────────
    use_half = args.half and not args.int8
    use_int8 = args.int8

    if use_int8 and args.format == "onnx":
        print("⚠️  INT8 is TensorRT-only. Exporting ONNX with FP32 instead.")
        use_int8 = False

    precision_label = "INT8" if use_int8 else ("FP16" if use_half else "FP32")

    print()
    print("=" * 60)
    print("  🐟  Fish Model Export")
    print("=" * 60)
    print(f"  Source    : {model_path}")
    print(f"  Format    : {args.format.upper()}")
    print(f"  Precision : {precision_label}")
    print(f"  Image size: {args.imgsz}x{args.imgsz}")
    print(f"  Device    : {args.device}")
    print()

    # ── Pre-flight checks for TensorRT ───────────────────────────────────────
    if args.format == "engine":
        try:
            import torch
            if not torch.cuda.is_available():
                print("❌  CUDA not available — TensorRT export requires a CUDA GPU.")
                print("    Run this script on your Jetson Orin Nano, not on Windows.")
                sys.exit(1)
            cuda_ver = torch.version.cuda
            print(f"  ✅ CUDA {cuda_ver} detected")
        except ImportError:
            print("❌  PyTorch not installed.")
            sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print("  Loading model…")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌  ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(model_path))
    print(f"  ✅ Loaded: {model_path.name}")
    print()

    if args.format == "engine":
        print("  ▶  Building TensorRT engine — this can take 5–15 minutes on first run.")
        print("     The engine is device-specific and cached for future runs.")
    else:
        print("  ▶  Exporting to ONNX…")
    print()

    # ── Export ────────────────────────────────────────────────────────────────
    export_kwargs = dict(
        format=args.format,
        imgsz=args.imgsz,
        half=use_half,
        int8=use_int8,
        device=args.device,
        batch=args.batch,
        simplify=True,
        verbose=True,
    )

    try:
        exported_path = model.export(**export_kwargs)
        print()
        print("=" * 60)
        print(f"  ✅ Export complete!")
        print(f"  Output: {exported_path}")
        print("=" * 60)

        if args.format == "engine":
            print()
            print("  To use the engine with the app, run:")
            print(f"    export FISH_MODEL_PATH={exported_path}")
            print("    python main.py")
            print()
            print("  For permanent effect, add to ~/.bashrc:")
            print(f"    echo 'export FISH_MODEL_PATH={exported_path}' >> ~/.bashrc")
            print("    source ~/.bashrc")

        else:  # onnx
            print()
            print("  ONNX is portable — copy it to your Jetson, then on the Jetson run:")
            print("    python export_model.py --format engine --half --model /path/to/best.onnx")
            print("  Or just copy best.pt to Jetson and run the engine export there directly.")

    except Exception as e:
        print(f"\n❌  Export failed: {e}")
        if "engine" in str(type(e).__name__).lower() or "tensorrt" in str(e).lower():
            print("\n  Common fixes:")
            print("  • Ensure TensorRT is installed: sudo apt install tensorrt python3-libnvinfer-dev")
            print("  • Check CUDA version matches TensorRT version")
            print("  • Try without --half flag first (FP32 is more compatible)")
        sys.exit(1)


if __name__ == "__main__":
    main()
