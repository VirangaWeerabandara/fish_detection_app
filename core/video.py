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
