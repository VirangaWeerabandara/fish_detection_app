"""
core/detector.py
────────────────
YOLO model wrapper and drawing utilities.

No PyQt5, no HTTP, no asyncio — pure Python library.
Thread-safe for read-only inference (YOLO/PyTorch handles its own GIL).
"""

import cv2
import numpy as np
import logging
from typing import Optional

from core.config import MODEL_PATH, DEVICE, FAST_TRACKER_LINE_FRAC
from core.fast_tracker import FastFishTracker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FishDetector — wraps the YOLO model
# ─────────────────────────────────────────────────────────────────────────────
class FishDetector:
    """
    Loads the YOLO model once and exposes simple inference methods.

    Usage
    ─────
    detector = FishDetector()          # loads model at construction time
    results  = detector.track(frame, confidence=0.85)
    results  = detector.predict(frame, confidence=0.85)
    detector.reset_tracker()           # wipe ByteTrack state between runs
    """

    def __init__(self):
        from ultralytics import YOLO
        self.device = DEVICE
        self.model  = None
        try:
            self.model = YOLO(str(MODEL_PATH))
            logger.info(f"Model loaded: {MODEL_PATH}")
            print(f"✅ Model loaded: {MODEL_PATH}")
            self.tracker = FastFishTracker()
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            print(f"❌ Model load failed: {e}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def reset_tracker(self, full=False):
        """Wipe tracker internal state so track IDs start fresh."""
        try:
            if hasattr(self, 'tracker'):
                if full:
                    self.tracker.full_reset()
                else:
                    self.tracker.reset()
        except Exception as e:
            logger.warning(f"reset_tracker: {e}")

    def track(self, frame: np.ndarray, confidence: float) -> dict:
        """
        Run inference and custom fast tracking on a single frame.
        Returns a dict: {track_ids: list, boxes: list, total_counted: int}
        """
        results = self.model.predict(
            source=frame, conf=confidence,
            iou=0.45, device=self.device, verbose=False,
        )
        boxes_obj = results[0].boxes
        
        boxes_list = boxes_obj.xyxy.cpu().numpy().tolist() if boxes_obj else []
        confs_list = boxes_obj.conf.cpu().numpy().tolist() if boxes_obj else []
        
        out_ids, out_boxes, total = self.tracker.update(boxes_list, frame.shape[0])
        
        return {
            "track_ids": out_ids,
            "boxes": out_boxes,
            "confidences": [1.0] * len(out_boxes), # Mock confidences for tracked boxes
            "total_counted": total
        }

    def predict(self, frame: np.ndarray, confidence: float) -> list:
        """
        Run plain detection (no tracking) on a single frame.
        Used as fallback when track() fails.
        """
        return self.model.predict(
            source=frame, conf=confidence,
            iou=0.45, device=self.device, verbose=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Drawing utilities
# ─────────────────────────────────────────────────────────────────────────────
def draw_boxes_on_frame(
    frame: np.ndarray,
    boxes: list,
    confidences: list,
    track_ids: list,
) -> np.ndarray:
    """
    Draw detection bounding boxes on a *copy* of frame.

    Parameters
    ──────────
    frame       : BGR numpy array
    boxes       : list of [x1, y1, x2, y2] (pixel coords)
    confidences : list of float confidence scores
    track_ids   : list of integer track IDs (may be shorter than boxes)

    Returns
    ───────
    Annotated BGR numpy array (new copy, original unchanged).
    """
    out = frame.copy()
    
    # Draw Counting Line
    line_y = int(frame.shape[0] * FAST_TRACKER_LINE_FRAC)
    cv2.line(out, (0, line_y), (frame.shape[1], line_y), (255, 255, 0), 2)
    cv2.putText(out, "Counting Line", (10, line_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        tid   = track_ids[i] if i < len(track_ids) else None
        label = f"#{tid} {conf:.2f}" if tid is not None else f"Fish {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 80), 2)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 220, 80), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return out


def make_preview_frame(
    frame: np.ndarray,
    boxes: list,
    confidences: list,
    track_ids: list,
    max_dim: int = 480,
) -> np.ndarray:
    """
    Return a downscaled, annotated BGR frame suitable for live preview.
    Returns a numpy array (no encoding — the UI converts to QPixmap directly).
    """
    out = draw_boxes_on_frame(frame, boxes, confidences, track_ids)
    h, w = out.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        out = cv2.resize(out, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    return out
