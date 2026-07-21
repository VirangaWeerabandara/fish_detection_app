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

from core.config import MODEL_PATH, DEVICE

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
    results  = detector.predict(frame, confidence=0.85)
    """

    def __init__(self):
        from ultralytics import YOLO
        self.device = DEVICE
        self.model  = None
        try:
            self.model = YOLO(str(MODEL_PATH))
            logger.info(f"Model loaded: {MODEL_PATH}")
            print(f"✅ Model loaded: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            print(f"❌ Model load failed: {e}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, frame: np.ndarray, confidence: float) -> list:
        """
        Run plain detection on a single frame.
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
) -> np.ndarray:
    """
    Draw detection bounding boxes on a *copy* of frame.

    Parameters
    ──────────
    frame       : BGR numpy array
    boxes       : list of [x1, y1, x2, y2] (pixel coords)
    confidences : list of float confidence scores

    Returns
    ───────
    Annotated BGR numpy array (new copy, original unchanged).
    """
    out = frame.copy()
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        label = f"Fish {conf:.2f}"

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
    max_dim: int = 480,
) -> np.ndarray:
    """
    Return a downscaled, annotated BGR frame suitable for live preview.
    Returns a numpy array (no encoding — the UI converts to QPixmap directly).
    """
    out = draw_boxes_on_frame(frame, boxes, confidences)
    h, w = out.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        out = cv2.resize(out, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    return out
