import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DetectionLineCounter:
    """
    Counts fish crossing a line using raw YOLO detections (predict, not track).
    No persistent track IDs required -- dedup is done via short-window
    space-time proximity to already-counted crossings near the line.

    Handles multiple simultaneous fish in one frame correctly:
      - New detections are only matched against fish seen in PREVIOUS frames,
        never against other detections from the same current frame. This
        stops two distinct fish appearing together from merging into one
        count just because they happen to be within tolerance of each other.
      - Matching uses closest-pair greedy assignment (one-to-one), not
        first-found, so with several fish and several history entries active
        at once, each detection pairs with its nearest plausible match
        instead of an arbitrary earlier one.
    """
    def __init__(self, line_y=540, band_px=90, x_tolerance=150, y_tolerance=230,
                 max_frame_gap=5, conf_thresh=0.15):
        self.line_y = line_y
        self.band_px = band_px
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
        self.max_frame_gap = max_frame_gap
        self.conf_thresh = conf_thresh
        self.recent_counts = []  # list of dict: {frame_idx, x, y}
        self.count = 0

    def _in_tolerance(self, cx, cy, entry):
        return (abs(cx - entry["x"]) <= self.x_tolerance and
                abs(cy - entry["y"]) <= self.y_tolerance)

    def process_frame(self, frame_idx, detections):
        """detections: list of (x, y, w, h, conf) from YOLO predict(),
        already filtered to the fish class. x,y are top-left box coords."""

        self.recent_counts = [
            e for e in self.recent_counts
            if frame_idx - e["frame_idx"] <= self.max_frame_gap
        ]

        candidates = []
        for (x, y, w, h, conf) in detections:
            if conf < self.conf_thresh:
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            if abs(cy - self.line_y) > self.band_px:
                continue
            candidates.append((cx, cy))

        pairs = []
        for ci, (cx, cy) in enumerate(candidates):
            for hi, entry in enumerate(self.recent_counts):
                if self._in_tolerance(cx, cy, entry):
                    dist = ((cx - entry["x"]) ** 2 + (cy - entry["y"]) ** 2) ** 0.5
                    pairs.append((dist, ci, hi))
        pairs.sort(key=lambda p: p[0])

        matched_candidates = set()
        matched_history = set()
        for dist, ci, hi in pairs:
            if ci in matched_candidates or hi in matched_history:
                continue
            matched_candidates.add(ci)
            matched_history.add(hi)
            cx, cy = candidates[ci]
            
            # Check for line crossing (top to bottom)
            if not self.recent_counts[hi].get("counted", False) and cy >= self.line_y:
                self.count += 1
                self.recent_counts[hi]["counted"] = True
                logger.info(f"[Frame {frame_idx}] Fish crossed line at x={cx:.1f}, y={cy:.1f}")

            self.recent_counts[hi]["frame_idx"] = frame_idx
            self.recent_counts[hi]["x"] = cx
            self.recent_counts[hi]["y"] = cy

        for ci, (cx, cy) in enumerate(candidates):
            if ci not in matched_candidates:
                # If a new fish appears already below the line, it must have jumped the top half.
                # Since fish only go downward, we count it immediately to avoid missing it.
                is_counted = False
                if cy >= self.line_y:
                    self.count += 1
                    is_counted = True
                    logger.info(f"[Frame {frame_idx}] Fast fish counted below line at x={cx:.1f}, y={cy:.1f}")
                
                self.recent_counts.append({
                    "frame_idx": frame_idx, 
                    "x": cx, 
                    "y": cy, 
                    "counted": is_counted
                })

    def get_count(self):
        return self.count
        
    def flush(self, last_frame_idx: int):
        """
        No-op for this method, included for compatibility.
        """
        pass
        
    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the counting line, band region, and running count on the frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        
        # Draw band region
        band_top = max(0, self.line_y - self.band_px)
        band_bottom = min(h, self.line_y + self.band_px)
        
        overlay = out.copy()
        cv2.rectangle(overlay, (0, int(band_top)), (w, int(band_bottom)), (0, 100, 100), -1)
        cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)
        
        # Draw line
        cv2.line(out, (0, int(self.line_y)), (w, int(self.line_y)), (0, 255, 0), 2)
        
        # Draw text
        text = f"Count: {self.count}"
        (lw, lh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(out, (10, 10), (10 + lw + 20, 10 + lh + 20), (0, 0, 0), -1)
        cv2.putText(out, text, (20, 10 + lh + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        return out
