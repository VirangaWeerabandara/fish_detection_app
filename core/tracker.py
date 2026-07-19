"""
core/tracker.py
───────────────
Industry-standard fish counting tracker with defense-in-depth logic.

Counting strategy:  Line-crossing (fish pass top → bottom through the frame).
Display strategy:   Frame is rotated 90° CCW so fish appear left → right in UI.

The virtual counting line is a horizontal rule across the original frame at
COUNTING_LINE_POSITION (default 0.5 = 50% of frame height).  After the 90° CCW
rotation applied for display, this becomes a vertical line — fish appear to
cross it from left to right on screen.

Three protective layers
───────────────────────
1. Never reset ByteTrack mid-run → track IDs stay stable for the whole job.
2. Lost-track re-ID registry     → IoU match prevents re-IDed fish being
                                   counted again after a brief tracking gap.
3. Hit-streak gate               → a track must be detected for
                                   TRACKER_MIN_HIT_STREAK consecutive frames
                                   before it is eligible to be counted (kills
                                   single-frame ghost detections).
4. Line-crossing gate            → the centroid must actually cross the counting
                                   line (top→bottom transition) to be counted.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import (
    TRACKER_MIN_HIT_STREAK,
    TRACKER_LOST_TTL,
    TRACKER_REID_IOU_THRESH,
    COUNTING_LINE_ENABLED,
    COUNTING_LINE_AXIS,
    COUNTING_LINE_POSITION,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Box      = List[float]          # [x1, y1, x2, y2]
Centroid = Tuple[float, float]  # (cx, cy)


# ---------------------------------------------------------------------------
# Pure helpers (module-level so they can be imported independently)
# ---------------------------------------------------------------------------
def _centroid(box: Box) -> Centroid:
    """Return the centre point of a bounding box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _iou(a: Box, b: Box) -> float:
    """Axis-aligned Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _line_coord(centroid: Centroid, axis: str) -> float:
    """Return the coordinate component that is compared against the counting line."""
    return centroid[1] if axis == "y" else centroid[0]


def draw_counting_line(frame: np.ndarray) -> np.ndarray:
    """
    Draw the virtual counting line on a *copy* of frame (BGR).

    The line is drawn in the original (unrotated) frame coordinate space.
    When the frame is later rotated 90° CCW for display, a horizontal line
    (axis='y') becomes a vertical line — fish appear to cross left → right.

    No-op if COUNTING_LINE_ENABLED is False.
    """
    import cv2  # deferred import so the module stays importable without cv2
    if not COUNTING_LINE_ENABLED:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    color      = (0, 255, 255)   # cyan-yellow
    thickness  = 2
    if COUNTING_LINE_AXIS == "y":
        y = int(h * COUNTING_LINE_POSITION)
        cv2.line(out, (0, y), (w, y), color, thickness)
        cv2.putText(out, "COUNT LINE", (10, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    else:
        x = int(w * COUNTING_LINE_POSITION)
        cv2.line(out, (x, 0), (x, h), color, thickness)
        cv2.putText(out, "COUNT LINE", (x + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# FishTracker
# ---------------------------------------------------------------------------
class FishTracker:
    """
    Stateful fish counter for one processing job (video file or camera session).

    Create one instance per job.  Call ``update()`` on every frame in order.

    Parameters
    ──────────
    frame_w, frame_h : pixel dimensions of the frames that will be processed.
                       If 0/0, dimensions are inferred from the first ``update``
                       call via the ``frame_shape`` argument.

    Public API
    ──────────
    tracker.update(frame_idx, track_ids, boxes, frame_shape=None)
        → (new_fish_this_frame: int, total_unique: int)

    tracker.total                    → current unique fish count (int)
    tracker.draw_counting_line(frame) → annotated BGR frame copy
    """

    def __init__(self, frame_w: int = 0, frame_h: int = 0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._line_px = self._compute_line_px()

        # ── Global count ──────────────────────────────────────────────────
        # All track IDs that have been definitively counted for this job.
        self.seen_ids: set[int] = set()

        # ── Hit-streak gate ───────────────────────────────────────────────
        # {tid: consecutive_hit_count}
        # Counts how many frames in a row a track has been observed.
        self._hit_streak: Dict[int, int] = {}

        # ── Active-track state ────────────────────────────────────────────
        # {tid: (last_box, last_centroid, last_frame_idx)}
        # Updated every frame a track is present.
        self._active: Dict[int, Tuple[Box, Centroid, int]] = {}

        # ── Lost-track re-ID registry ─────────────────────────────────────
        # {tid: (last_box, last_centroid, expiry_frame_idx)}
        # Populated when a track disappears; retained for TRACKER_LOST_TTL frames.
        self._lost: Dict[int, Tuple[Box, Centroid, int]] = {}

        # ── Line-crossing state ───────────────────────────────────────────
        # {tid: previous_line_coordinate}
        # Stores the relevant coordinate (x or y) from the previous frame
        # so we can detect when the centroid crosses the counting line.
        self._line_side: Dict[int, float] = {}

        logger.info(
            "FishTracker initialised — "
            f"line_enabled={COUNTING_LINE_ENABLED}, "
            f"axis={COUNTING_LINE_AXIS}, "
            f"line_px={self._line_px:.0f}, "
            f"min_hit_streak={TRACKER_MIN_HIT_STREAK}, "
            f"lost_ttl={TRACKER_LOST_TTL}, "
            f"reid_iou_thresh={TRACKER_REID_IOU_THRESH}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────
    def update(
        self,
        frame_idx: int,
        track_ids: List[int],
        boxes:     List[Box],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Process one frame's ByteTrack output through all counting layers.

        Parameters
        ──────────
        frame_idx   : monotonically increasing frame index for this job
        track_ids   : list of integer track IDs returned by ByteTrack
        boxes       : parallel list of [x1,y1,x2,y2] bounding boxes (pixels)
        frame_shape : (height, width) — if provided, recalculates line_px
                      when the frame dimensions change (e.g. first frame of stream)

        Returns
        ───────
        (new_fish_this_frame, total_unique_count)
        """
        # Lazily update frame dimensions (e.g. first camera frame)
        if frame_shape is not None:
            h, w = frame_shape
            if h != self.frame_h or w != self.frame_w:
                self.frame_h, self.frame_w = h, w
                self._line_px = self._compute_line_px()
                logger.debug(f"FishTracker: frame_shape updated to {w}×{h}, "
                             f"line_px={self._line_px:.0f}")

        # ── 1. Expire stale lost-track entries ────────────────────────────
        self._expire_lost(frame_idx)

        # ── 2. Move tracks not seen this frame into the lost registry ─────
        seen_this_frame = set(track_ids)
        for tid in list(self._active.keys()):
            if tid not in seen_this_frame:
                box, centroid, _ = self._active.pop(tid)
                expiry = frame_idx + TRACKER_LOST_TTL
                self._lost[tid] = (box, centroid, expiry)
                self._hit_streak.pop(tid, None)
                # Keep _line_side so re-IDed fish can resume crossing tracking

        # ── 3. Process each detected track ────────────────────────────────
        new_fish_this_frame = 0

        for tid, box in zip(track_ids, boxes):
            centroid = _centroid(box)

            # Update active state
            self._active[tid] = (box, centroid, frame_idx)

            # Accumulate hit streak
            self._hit_streak[tid] = self._hit_streak.get(tid, 0) + 1

            # ── Already counted — just update line-side coord ──────────────
            if tid in self.seen_ids:
                if COUNTING_LINE_ENABLED:
                    self._line_side[tid] = _line_coord(centroid, COUNTING_LINE_AXIS)
                continue

            # ── Re-ID check ───────────────────────────────────────────────
            # If this brand-new track ID has high IoU with a known lost track,
            # it's the same physical fish re-IDed by ByteTrack — don't count.
            if self._is_reid(tid, box, centroid):
                # Register as already counted so future frames skip it
                self.seen_ids.add(tid)
                logger.debug(
                    f"Frame {frame_idx}: TID {tid} → re-ID of known fish (skipped)"
                )
                continue

            # ── Hit-streak gate ────────────────────────────────────────────
            # Reject single/double-frame ghost detections
            if self._hit_streak.get(tid, 0) < TRACKER_MIN_HIT_STREAK:
                continue

            # ── Line-crossing gate ─────────────────────────────────────────
            if COUNTING_LINE_ENABLED:
                coord      = _line_coord(centroid, COUNTING_LINE_AXIS)
                prev_coord = self._line_side.get(tid)
                self._line_side[tid] = coord

                if prev_coord is None:
                    # First observation of this track — record side, don't count yet
                    continue

                # Count only on the downward transition (top → bottom for axis='y')
                # Allow both directions if you want bidirectional counting.
                crossed = (prev_coord < self._line_px <= coord)
                if not crossed:
                    continue

            # ── Count this fish ────────────────────────────────────────────
            self.seen_ids.add(tid)
            new_fish_this_frame += 1
            logger.debug(
                f"Frame {frame_idx}: TID {tid} COUNTED "
                f"(streak={self._hit_streak.get(tid)}, "
                f"line={'crossing' if COUNTING_LINE_ENABLED else 'off'})"
            )

        return new_fish_this_frame, len(self.seen_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def total(self) -> int:
        """Current unique fish count for this job."""
        return len(self.seen_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # Drawing helper (delegates to module-level function)
    # ─────────────────────────────────────────────────────────────────────────
    def draw_counting_line(self, frame: np.ndarray) -> np.ndarray:
        """Draw the counting line on a copy of frame. No-op if line is disabled."""
        return draw_counting_line(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_line_px(self) -> float:
        """Convert the fractional line position to absolute pixels."""
        if COUNTING_LINE_AXIS == "y":
            return self.frame_h * COUNTING_LINE_POSITION
        return self.frame_w * COUNTING_LINE_POSITION

    def _is_reid(self, tid: int, box: Box, centroid: Centroid) -> bool:
        """
        Return True if a new track ID spatially matches a recently-lost counted fish.

        Strategy: find the lost track with the highest IoU against *box*.
        If that IoU ≥ TRACKER_REID_IOU_THRESH, the new track is considered a
        re-ID of that lost fish and should NOT be counted again.
        """
        best_iou  = 0.0
        best_ltid: Optional[int] = None

        for lost_tid, (lost_box, _lost_centroid, _expiry) in self._lost.items():
            # Only match against tracks that were actually counted
            if lost_tid not in self.seen_ids:
                continue
            iou = _iou(box, lost_box)
            if iou > best_iou:
                best_iou  = iou
                best_ltid = lost_tid

        if best_iou >= TRACKER_REID_IOU_THRESH and best_ltid is not None:
            # Transfer line-side state so crossing detection is seamless
            if best_ltid in self._line_side:
                self._line_side[tid] = self._line_side[best_ltid]
            # Consume the lost entry so it can't match another track
            self._lost.pop(best_ltid, None)
            logger.debug(
                f"Re-ID match: new TID {tid} ← lost TID {best_ltid} "
                f"(IoU={best_iou:.3f})"
            )
            return True
        return False

    def _expire_lost(self, frame_idx: int):
        """Remove lost-track entries that have exceeded their time-to-live."""
        expired = [
            tid for tid, (_box, _centroid, expiry) in self._lost.items()
            if frame_idx >= expiry
        ]
        for tid in expired:
            self._lost.pop(tid, None)
            self._line_side.pop(tid, None)
