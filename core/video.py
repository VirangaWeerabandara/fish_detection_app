"""
core/video.py
─────────────
Video loading and frame extraction library.

Pure functions — no side effects, no PyQt5, no HTTP.
Import and call directly from any thread.
"""

import cv2
import numpy as np
import logging
from typing import Optional

from core.config import MAX_DIM, TARGET_FPS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional imageio
# ─────────────────────────────────────────────────────────────────────────────
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    logger.warning("imageio not installed — MP4 support may be limited")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resize_frame(frame: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Downscale frame so its longest side ≤ max_dim. Returns original if already small."""
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def _resample(frames_in: list, src_fps: float, tgt_fps: float) -> list:
    """Uniformly subsample frames_in from src_fps down to tgt_fps."""
    n = len(frames_in)
    if n == 0:
        return []
    duration = n / (src_fps if src_fps > 0 else tgt_fps)
    n_out = min(int(duration * tgt_fps), n)
    n_out = max(n_out, 1)
    if n_out == n:
        return frames_in
    idxs = np.linspace(0, n - 1, n_out, dtype=int)
    return [frames_in[i] for i in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(
    video_path: str,
    target_fps: Optional[float] = TARGET_FPS,
    max_dim: int = MAX_DIM,
) -> list:
    """
    Extract and return all frames from *video_path* as a list of BGR numpy arrays.

    Strategy
    ────────
    1. OpenCV (primary) — fast, works for most formats.
    2. imageio/ffmpeg (fallback) — handles edge-case codecs.

    Parameters
    ──────────
    video_path : path to the video file
    target_fps : resample to this fps; None = keep original fps
    max_dim    : downscale so the longest side ≤ max_dim

    Returns
    ───────
    List of BGR numpy arrays, one per frame.

    Raises
    ──────
    RuntimeError if both backends fail.
    """
    # ── OpenCV ────────────────────────────────────────────────────────────────
    try:
        logger.info("Extracting frames (OpenCV)…")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tgt     = target_fps if target_fps else src_fps
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"~{total} frames @ {src_fps:.1f} fps")

        all_frames, bad = [], 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.size == 0:
                bad += 1
                if bad > 10:
                    break
                continue
            all_frames.append(_resize_frame(frame, max_dim))
            bad = 0
        cap.release()

        if all_frames:
            result = _resample(all_frames, src_fps, tgt)
            logger.info(f"OpenCV: {len(all_frames)} raw → {len(result)} frames")
            return result
        logger.warning("OpenCV: 0 valid frames — trying imageio…")

    except Exception as e:
        logger.warning(f"OpenCV failed: {e}")

    # ── imageio v3 fallback ───────────────────────────────────────────────────
    if HAS_IMAGEIO:
        try:
            import imageio.v3 as iio
            meta    = iio.immeta(video_path, plugin="ffmpeg")
            src_fps = float(meta.get("fps", 30))
            tgt     = target_fps if target_fps else src_fps
            frames  = []
            for f in iio.imiter(video_path, plugin="ffmpeg"):
                frm = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.ndim == 3 else f
                frames.append(_resize_frame(frm, max_dim))
            if frames:
                result = _resample(frames, src_fps, tgt)
                logger.info(f"imageio: {len(result)} frames")
                return result
        except Exception as e:
            logger.warning(f"imageio failed: {e}")

    raise RuntimeError(f"Could not extract frames from {video_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Live camera
# ─────────────────────────────────────────────────────────────────────────────

def _gstreamer_available() -> bool:
    """Return True if this OpenCV build was compiled with GStreamer support."""
    build_info = cv2.getBuildInformation()
    for line in build_info.splitlines():
        if "GStreamer" in line:
            # Line looks like:  "    GStreamer:                    YES (1.x.y)"
            return "YES" in line
    return False


def _try_open(source, backend=None, label: str = "") -> "cv2.VideoCapture | None":
    """
    Attempt to open a VideoCapture.  Returns the cap on success, None on failure.
    Never raises.
    """
    try:
        cap = cv2.VideoCapture(source, backend) if backend is not None \
              else cv2.VideoCapture(source)
        if cap.isOpened():
            logger.info(f"Camera opened via {label or source}")
            return cap
        cap.release()
    except Exception as exc:
        logger.debug(f"_try_open({label}) raised: {exc}")
    return None


def open_camera(source: "int | str" = 0) -> cv2.VideoCapture:
    """
    Open a camera and return a ready-to-read VideoCapture.

    Parameters
    ──────────
    source : int  → USB/V4L2 device index (e.g. 0 = /dev/video0)
             str  → Primary GStreamer pipeline string (e.g. CAMERA_CSI_PIPELINE)
                    If it fails, fallback pipelines are tried automatically.

    Fallback order for CSI (str source)
    ────────────────────────────────────
      1. Primary pipeline as-is          (nvarguscamerasrc)
      2. v4l2src GStreamer pipeline       (/dev/video0 via GStreamer)
      3. Plain device index 0             (direct V4L2, no GStreamer)

    Raises
    ──────
    RuntimeError with a diagnostic message if every attempt fails.
    """
    gst_ok = _gstreamer_available()

    # ── CSI / GStreamer path ───────────────────────────────────────────────────
    if isinstance(source, str):
        logger.info("Opening CSI camera…")

        if not gst_ok:
            logger.error(
                "OpenCV was NOT built with GStreamer support. "
                "The pip-installed opencv-python does not include GStreamer. "
                "On Jetson, use the system OpenCV that ships with JetPack:\n"
                "  pip uninstall opencv-python opencv-python-headless -y\n"
                "  # The system cv2 at /usr/lib/python3/dist-packages is already correct."
            )
            # Still attempt the fallback below; on some setups the device index works.

        else:
            # Attempt 1 — primary pipeline (nvarguscamerasrc)
            logger.info(f"Attempt 1 — nvarguscamerasrc pipeline")
            cap = _try_open(source, cv2.CAP_GSTREAMER, "nvarguscamerasrc pipeline")
            if cap:
                _configure_cap(cap, "CSI/nvarguscamerasrc")
                return cap

            logger.warning(
                "nvarguscamerasrc pipeline failed. Possible causes:\n"
                "  • nvargus-daemon is not running  →  sudo systemctl start nvargus-daemon\n"
                "  • Another process holds the camera sensor (kill it and retry)\n"
                "  • Wrong sensor-id or resolution in config.py\n"
                "  • Camera ribbon cable not seated properly\n"
                "Trying v4l2src fallback…"
            )

            # Attempt 2 — v4l2src GStreamer pipeline (CSI exposed as /dev/video0)
            v4l2_pipeline = (
                "v4l2src device=/dev/video0 ! "
                "video/x-raw, format=UYVY ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink"
            )
            logger.info(f"Attempt 2 — v4l2src pipeline: {v4l2_pipeline}")
            cap = _try_open(v4l2_pipeline, cv2.CAP_GSTREAMER, "v4l2src pipeline")
            if cap:
                _configure_cap(cap, "CSI/v4l2src")
                return cap
            logger.warning("v4l2src GStreamer pipeline also failed. Trying direct index 0…")

        # Attempt 3 — plain device index (last resort)
        logger.info("Attempt 3 — plain cv2.VideoCapture(0)")
        cap = _try_open(0, label="device index 0")
        if cap:
            _configure_cap(cap, "device index 0 (fallback)")
            return cap

        raise RuntimeError(
            "Cannot open the CSI camera. All attempts failed.\n\n"
            "Checklist:\n"
            "  1. Is nvargus-daemon running?\n"
            "     → sudo systemctl status nvargus-daemon\n"
            "     → sudo systemctl start  nvargus-daemon\n"
            "  2. Is the camera detected?\n"
            "     → ls /dev/video*\n"
            "     → gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! fakesink\n"
            "  3. Is OpenCV GStreamer-enabled?\n"
            "     → python3 -c \"import cv2; print(cv2.getBuildInformation())\" | grep GStreamer\n"
            "     → If NO, uninstall pip opencv and use the JetPack system cv2.\n"
            "  4. Is the ribbon cable properly seated on both ends?\n"
            f"  GStreamer available in this OpenCV build: {gst_ok}"
        )

    # ── USB / device-index path ────────────────────────────────────────────────
    logger.info(f"Opening USB camera index {source}…")
    cap = _try_open(source, label=f"USB index {source}")
    if cap:
        _configure_cap(cap, f"USB index {source}")
        return cap

    raise RuntimeError(
        f"Cannot open USB camera at index {source}. "
        "Check it is plugged in and not in use by another process."
    )


def _configure_cap(cap: cv2.VideoCapture, label: str):
    """Apply common post-open settings and log the resolved resolution."""
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info(
        f"Camera ready ({label}) — "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.1f} fps"
    )

