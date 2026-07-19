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
# NOTE: No TRACKER_RESET_INTERVAL — ByteTrack is never reset mid-run.
# Resetting mid-run re-assigns track IDs and causes double-counting.

PREVIEW_INTERVAL = 4    # generate a live-preview frame every N frames
PREVIEW_MAX_DIM  = 480  # max dimension of the live-preview (pixels)

# ── FishTracker constants ─────────────────────────────────────────────────────
# A new track must be detected for this many consecutive frames before being
# eligible to be counted.  Prevents single-frame ghost detections.
TRACKER_MIN_HIT_STREAK  = 3

# Number of frames to keep a lost track in the re-ID registry.  During this
# window a new track ID that overlaps the lost track's last bounding box is
# treated as a re-ID (same fish), not a new fish.
TRACKER_LOST_TTL        = 60

# Minimum IoU between a new track's box and a lost track's last box for the
# re-ID match to fire.
TRACKER_REID_IOU_THRESH = 0.35

# ── Line-crossing counting ────────────────────────────────────────────────────
# Fish swim top → bottom through the frame.  A horizontal counting line is
# placed at COUNTING_LINE_POSITION (0.5 = 50% of frame height).  A fish is
# counted only when its centroid crosses the line from above to below.
COUNTING_LINE_ENABLED  = True    # set False to disable line-crossing gate
COUNTING_LINE_AXIS     = "y"     # 'y' = horizontal line (fixed Y coordinate)
COUNTING_LINE_POSITION = 0.5     # fraction of frame height (0.0 – 1.0)

# ── Display rotation ──────────────────────────────────────────────────────────
# Rotate the preview/annotated frame 90° counter-clockwise before displaying.
# This makes fish that physically travel top→bottom appear left→right in the UI.
DISPLAY_ROTATE_90_CCW  = True

# ─────────────────────────────────────────────────────────────────────────────
# Live camera
# ─────────────────────────────────────────────────────────────────────────────
# CSI camera (Jetson camera connector, e.g. IMX219 / IMX477)
# Set FISH_CAMERA_USE_CSI=0 to fall back to USB.
CAMERA_USE_CSI = bool(int(_os.environ.get("FISH_CAMERA_USE_CSI", "1")))

# GStreamer pipeline for nvarguscamerasrc (CSI).
# sensor-id=0 targets the first CSI port on the Jetson Orin Nano.
# Adjust width/height/framerate to match your sensor (e.g. IMX219: 1920x1080@30).
CAMERA_CSI_WIDTH     = int(_os.environ.get("FISH_CAMERA_CSI_WIDTH",  "1920"))
CAMERA_CSI_HEIGHT    = int(_os.environ.get("FISH_CAMERA_CSI_HEIGHT", "1080"))
CAMERA_CSI_FPS       = int(_os.environ.get("FISH_CAMERA_CSI_FPS",    "60"))   # IMX477 confirmed @ 60fps
CAMERA_CSI_SENSOR_ID = int(_os.environ.get("FISH_CAMERA_CSI_SENSOR", "0"))

def _build_csi_pipeline(width: int, height: int, fps: int, sensor_id: int) -> str:
    """Return the GStreamer pipeline string for the Jetson CSI camera."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )

CAMERA_CSI_PIPELINE = _build_csi_pipeline(
    CAMERA_CSI_WIDTH, CAMERA_CSI_HEIGHT, CAMERA_CSI_FPS, CAMERA_CSI_SENSOR_ID
)

# USB camera device index — used only when CAMERA_USE_CSI is False.
# Overridable via FISH_CAMERA_INDEX env variable.
CAMERA_INDEX   = int(_os.environ.get("FISH_CAMERA_INDEX", "0"))
CAMERA_MAX_DIM = 640   # resize camera frames to this max-dim before inference

# ─────────────────────────────────────────────────────────────────────────────
# GPIO relay pins  (Jetson Orin Nano — BOARD pin numbering)
# Relay board: SRD-05VDC-SL-C (active-LOW: relay ON when IN pin is driven LOW)
# Indicator:   AD22-22DS
# ─────────────────────────────────────────────────────────────────────────────
# IN1 → Pin 16  — Application ready / model loaded
# IN2 → Pin 15  — Detection or camera stream running
# IN3 → Pin 13  — Detection or camera stream running (paired with IN2)
# IN4 → Pin 11  — Counting complete
GPIO_PIN_READY    = 16
GPIO_PIN_DETECT_A = 15
GPIO_PIN_DETECT_B = 13
GPIO_PIN_COMPLETE = 11
