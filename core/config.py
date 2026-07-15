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

# ─────────────────────────────────────────────────────────────────────────────
# GPIO relay pins  (Jetson Orin Nano — BOARD pin numbering)
# Relay board: SRD-05VDC-SL-C (active-LOW: relay ON when IN pin is driven LOW)
# Indicator:   AD22-22DS
# ─────────────────────────────────────────────────────────────────────────────
# IN1 → Pin 11  — Application ready / model loaded
# IN2 → Pin 13  — Detection or camera stream running
# IN3 → Pin 15  — Detection or camera stream running (paired with IN2)
# IN4 → Pin 16  — Counting complete
GPIO_PIN_READY    = 11
GPIO_PIN_DETECT_A = 13
GPIO_PIN_DETECT_B = 15
GPIO_PIN_COMPLETE = 16
