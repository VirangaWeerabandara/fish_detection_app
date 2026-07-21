"""
core/fast_tracker.py
────────────────────
Lightweight tracking algorithm utilizing Centroid Matching via LAP or SciPy,
and a Virtual Counting Line for robust, ultra-fast fish counting.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import lap (C++ optimized, required by ultralytics ByteTrack)
try:
    import lap
    USE_LAP = True
except ImportError:
    USE_LAP = False
    try:
        from scipy.optimize import linear_sum_assignment
        USE_SCIPY = True
    except ImportError:
        USE_SCIPY = False
        logger.error("Missing lap and scipy! FastFishTracker requires one of them.")

from core.config import (
    FAST_TRACKER_MAX_MISSING,
    FAST_TRACKER_LINE_FRAC,
    FAST_TRACKER_MAX_Y_DIST,
    FAST_TRACKER_MAX_X_DIST
)

class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.history = [self.center]
        self.missed_frames = 0
        self.state = "TRACKING"  # TRACKING -> COUNTED
        
    @property
    def center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

class FastFishTracker:
    def __init__(self):
        self.tracks = {}  # track_id -> Track
        self.next_id = 1
        self.total_counted = 0
        
    def reset(self):
        self.tracks.clear()
        
    def full_reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.total_counted = 0

    def update(self, boxes_xyxy, frame_height):
        """
        Takes raw boxes [N, 4] for the current frame.
        Returns active track IDs, corresponding active boxes, and total_counted.
        """
        line_y = frame_height * FAST_TRACKER_LINE_FRAC
        
        # 1. Increment missed frames for all tracks initially
        for t in self.tracks.values():
            t.missed_frames += 1
            
        if len(boxes_xyxy) == 0:
            self._prune_tracks()
            return [], [], self.total_counted
            
        active_track_ids = list(self.tracks.keys())
        num_tracks = len(active_track_ids)
        num_boxes = len(boxes_xyxy)
        
        matched_tracks = set()
        matched_boxes = set()

        # 2. Match if we have both tracks and boxes
        if num_tracks > 0:
            # Build cost matrix
            cost_matrix = np.full((num_tracks, num_boxes), 1e6, dtype=np.float32)
            
            for i, tid in enumerate(active_track_ids):
                t = self.tracks[tid]
                cx, cy = t.center
                
                for j, bbox in enumerate(boxes_xyxy):
                    bx = (bbox[0] + bbox[2]) / 2.0
                    by = (bbox[1] + bbox[3]) / 2.0
                    
                    dx = bx - cx
                    dy = by - cy
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    # Directional & Distance Constraints
                    # Fish strictly move down (dy > 0), allow slight negative margin (-30) for bbox jitter
                    if dy < -30:
                        continue
                    if dy > FAST_TRACKER_MAX_Y_DIST:
                        continue
                    if abs(dx) > FAST_TRACKER_MAX_X_DIST:
                        continue
                        
                    cost_matrix[i, j] = dist
                    
            # Solve Assignment
            if USE_LAP:
                cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=1e5)
                for i, j in enumerate(x):
                    if j >= 0 and cost_matrix[i, j] < 1e5:
                        matched_tracks.add(i)
                        matched_boxes.add(j)
            elif USE_SCIPY:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 1e5:
                        matched_tracks.add(r)
                        matched_boxes.add(c)
            else:
                # Greedy fallback if neither is available
                logger.warning("No LAP/SciPy found, using greedy matching (suboptimal).")
                for i in range(num_tracks):
                    best_j = -1
                    best_cost = 1e5
                    for j in range(num_boxes):
                        if j not in matched_boxes and cost_matrix[i, j] < best_cost:
                            best_cost = cost_matrix[i, j]
                            best_j = j
                    if best_j >= 0:
                        matched_tracks.add(i)
                        matched_boxes.add(best_j)
                        
            # 3. Update matched tracks
            for i in matched_tracks:
                tid = active_track_ids[i]
                t = self.tracks[tid]
                j = list(matched_boxes)[list(matched_tracks).index(i)] if not USE_LAP and not USE_SCIPY else (x[i] if USE_LAP else dict(zip(row_ind, col_ind))[i])
                
                t.bbox = boxes_xyxy[j]
                t.missed_frames = 0
                
                prev_y = t.history[-1][1]
                curr_y = t.center[1]
                t.history.append(t.center)
                
                # Check for line crossing (Option 1: Bounding Box Intersection or Center Crossing)
                if t.state == "TRACKING":
                    y1 = t.bbox[1]
                    y2 = t.bbox[3]
                    # Count if the bounding box straddles the line, or if the center completely jumped over it
                    if (y1 <= line_y and y2 >= line_y) or (prev_y < line_y and curr_y >= line_y):
                        t.state = "COUNTED"
                        self.total_counted += 1

        # 4. Create new tracks for unmatched boxes
        for j, bbox in enumerate(boxes_xyxy):
            if j not in matched_boxes:
                t = Track(self.next_id, bbox)
                
                # If a fish appears for the very first time exactly on the line, count it immediately
                y1 = bbox[1]
                y2 = bbox[3]
                if y1 <= line_y and y2 >= line_y:
                    t.state = "COUNTED"
                    self.total_counted += 1
                    
                self.tracks[self.next_id] = t
                self.next_id += 1
                
        # 5. Prune lost tracks
        self._prune_tracks()
        
        # 6. Prepare output (active valid boxes and IDs)
        out_ids = []
        out_boxes = []
        for tid, t in self.tracks.items():
            if t.missed_frames == 0:  # Only output if seen in current frame
                out_ids.append(tid)
                out_boxes.append(t.bbox)
                
        return out_ids, out_boxes, self.total_counted

    def _prune_tracks(self):
        # Remove tracks that have been missing for too long or successfully counted
        tids = list(self.tracks.keys())
        for tid in tids:
            t = self.tracks[tid]
            if t.missed_frames > FAST_TRACKER_MAX_MISSING:
                del self.tracks[tid]
