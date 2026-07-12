"""
core/config.py
──────────────
Shared constants for the Fish Detection & Counting application.
Single source of truth — import from here, never define elsewhere.
"""

from pathlib import Path
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
# Model path — controlled by the FISH_MODEL_PATH environment variable.
# If the variable is not set, falls back to fish_dataset/export/best.pt
#
# To override on Jetson (add to ~/.bashrc or run before launching):
#   export FISH_MODEL_PATH=/path/to/your/best.pt
import os as _os
_default_model = Path(__file__).resolve().parent.parent.parent / "export" / "best.pt"
MODEL_PATH = Path(_os.environ.get("FISH_MODEL_PATH", str(_default_model)))

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Video ingestion
# ─────────────────────────────────────────────────────────────────────────────
TARGET_FPS = None   # None = keep original video fps
MAX_DIM    = 800    # downscale frames to this maximum dimension on ingest

# ─────────────────────────────────────────────────────────────────────────────
# Detection / tracking
# ─────────────────────────────────────────────────────────────────────────────
TRACKER_RESET_INTERVAL = 150  # reset ByteTrack state every N frames
PREVIEW_INTERVAL       = 4    # generate a live-preview frame every N frames
PREVIEW_MAX_DIM        = 480  # max dimension of the live-preview (pixels)

# ─────────────────────────────────────────────────────────────────────────────
# Live camera
# ─────────────────────────────────────────────────────────────────────────────
# USB camera device index — overridable via FISH_CAMERA_INDEX env variable.
# On Jetson: /dev/video0 → index 0 (most USB cameras), or 1+ if CSI cam is present.
CAMERA_INDEX   = int(_os.environ.get("FISH_CAMERA_INDEX", "0"))
CAMERA_MAX_DIM = 640   # resize camera frames to this max-dim before inference
