"""
fish_detection_app/__main__.py
──────────────────────────────
Enables running as a module from the PARENT directory:
    python -m fish_detection_app
"""

import sys
from pathlib import Path

# Bootstrap: ensure fish_detection_app/ is in sys.path
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from main import main  # noqa: E402

sys.exit(main())
