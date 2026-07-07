"""
fish_detection_app/main.py
───────────────────────────
Unified entry-point for the Fish Detection & Counting application.

Usage
─────
    # From the fish_detection_app/ directory:
    python main.py

    # From the parent (fish_dataset/) directory:
    python -m fish_detection_app
    python fish_detection_app/main.py
"""

import sys
import logging
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path bootstrap — must be first.
#
# Inserting fish_detection_app/ into sys.path means:
#   from core.xxx import ...    works regardless of CWD or invocation method.
# ─────────────────────────────────────────────────────────────────────────────
_app_dir = Path(__file__).resolve().parent          # .../fish_detection_app/
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print()
    print("=" * 60)
    print("  🐟  Fish Detection & Counting Application")
    print("=" * 60)

    # ── 1. Load the YOLO model (happens once, on the main thread) ─────────────
    print("  Loading model…")
    from core.detector import FishDetector
    detector = FishDetector()

    if not detector.is_loaded:
        # Still open the window — the UI will show a warning in the status bar.
        print("  ⚠️  Model not loaded. Check MODEL_PATH in core/config.py")
    else:
        print("  ✅ Model ready")

    print()

    # ── 2. Create Qt application and open the main window ─────────────────────
    from PyQt5.QtWidgets import QApplication
    from ui.app import FishDetectionApp

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("Fish Detection & Counting")

    window = FishDetectionApp(detector=detector)
    window.show()

    # ── 3. Run the Qt event loop (blocks until window closes) ─────────────────
    exit_code = qt_app.exec_()
    print("\n⏹️  Window closed — exiting.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
