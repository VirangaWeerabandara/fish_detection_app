"""
core/tracker.py
───────────────
Reliable fish counter for a pass-through (top → bottom) scenario.

Design principles
─────────────────
1. ByteTrack is NEVER reset mid-run — track IDs stay stable.
2. Line-crossing gate — a fish is counted the first time its centroid
   crosses the counting line (top → bottom).  Fish staying in frame or
   swimming back up are NOT double-counted.
3. Re-ID protection uses a HIGH IoU threshold (0.7) and a longer TTL
   (60 frames) so it fires for the same fish briefly lost but not for
   genuinely new fish that happen to be in the same channel area.
   When a re-ID match is found, the old TID is *removed* from seen_ids
   and the new TID is *added* — keeping the set size correct (no double
   count from a single fish occupying two slots).
4. Uncounted-lost registry: if an UNCOUNTED track is lost before reaching
   the hit-streak threshold, its partial streak is stored.  If ByteTrack
   reassigns a new ID to the same fish (IoU ≥ thresh), the new ID
   *inherits* the streak so it is not forced to start from 0 — preventing
   the same fish-pass from triggering a second count attempt.
5. Conservative fallback: if ByteTrack does not return track IDs for a
   detection (boxes_obj.id is None), the tracker receives an empty
   track_ids list and simply waits — it does not crash or mis-count.

Display rotation (90° CCW) is applied in app.py after this module runs.
The counting line and all coordinates are in the ORIGINAL frame space.
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
# Pure helpers
# ---------------------------------------------------------------------------
def _centroid(box: Box) -> Centroid:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _iou(a: Box, b: Box) -> float:
    """Axis-aligned IoU for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _line_coord(centroid: Centroid, axis: str) -> float:
    return centroid[1] if axis == "y" else centroid[0]


def draw_counting_line(frame: np.ndarray) -> np.ndarray:
    """
    Draw the virtual counting line on a *copy* of the frame (BGR).
    No-op if COUNTING_LINE_ENABLED is False.
    The line is drawn in the original (unrotated) frame coordinate space.
    """
    import cv2
    if not COUNTING_LINE_ENABLED:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    color     = (0, 255, 255)  # cyan
    thickness = 2
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
    Stateful per-job fish counter.

    One instance per video or camera session.  Call update() on every frame.

    Parameters
    ──────────
    frame_w, frame_h : pixel size of the frames that will be processed.
                       May be 0/0 — dimensions are auto-detected from the
                       first update() call via the frame_shape argument.
    """

    def __init__(self, frame_w: int = 0, frame_h: int = 0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._line_px = self._compute_line_px()

        # ── Counting state ────────────────────────────────────────────────
        # seen_ids: track IDs that have been definitively counted this job.
        self.seen_ids: set[int] = set()

        # ── Hit-streak gate ───────────────────────────────────────────────
        # How many consecutive frames a track has been observed.
        # {tid: int}
        self._hit_streak: Dict[int, int] = {}

        # ── Active tracks ─────────────────────────────────────────────────
        # {tid: (last_box, last_centroid, last_frame_idx)}
        self._active: Dict[int, Tuple[Box, Centroid, int]] = {}

        # ── Lost-track re-ID registry (COUNTED fish) ──────────────────────────
        # Stores the last known box of recently-lost COUNTED fish for a
        # short window.  Used to detect when ByteTrack re-assigns a new ID
        # to the same fish after a brief occlusion.
        # When a match is found the old TID slot is *transferred* to the
        # new TID so seen_ids stays at the correct size (no double-counting).
        # {tid: (last_box, expiry_frame_idx)}
        self._lost: Dict[int, Tuple[Box, int]] = {}

        # ── Lost-track re-ID registry (UNCOUNTED fish) ──────────────────────
        # Stores the last known box AND accumulated hit-streak of recently-lost
        # fish that were NOT yet counted.  If ByteTrack re-assigns a new ID to
        # the same fish the new track inherits the streak, preventing the fish
        # from being forced to restart from 0 and potentially crossing the
        # threshold a second time under a different ID.
        # {tid: (last_box, inherited_streak, expiry_frame_idx)}
        self._uncounted_lost: Dict[int, Tuple[Box, int, int]] = {}

        # ── Line-crossing state ───────────────────────────────────────────
        # Tracks which side of the counting line each track was on in the
        # previous frame.
        # {tid: prev_coord}
        self._prev_coord: Dict[int, Optional[float]] = {}

        logger.info(
            "FishTracker init — "
            f"line={COUNTING_LINE_ENABLED}, axis={COUNTING_LINE_AXIS}, "
            f"line_px={self._line_px:.0f}, "
            f"hit_streak={TRACKER_MIN_HIT_STREAK}, "
            f"lost_ttl={TRACKER_LOST_TTL}, "
            f"reid_thresh={TRACKER_REID_IOU_THRESH}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def update(
        self,
        frame_idx: int,
        track_ids: List[int],
        boxes:     List[Box],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Process one frame.

        Parameters
        ──────────
        frame_idx   : monotonically increasing frame index
        track_ids   : ByteTrack track IDs (may be empty if ByteTrack returned
                      no IDs — the tracker simply skips that frame safely)
        boxes       : parallel list of [x1,y1,x2,y2] boxes in original frame
                      pixel coordinates
        frame_shape : (height, width) of the frame; used to set the counting
                      line pixel position on the first call

        Returns
        ───────
        (new_fish_this_frame, total_unique_count)
        """
        # ── Resolve frame dimensions ──────────────────────────────────────
        if frame_shape is not None:
            h, w = frame_shape
            if h != self.frame_h or w != self.frame_w:
                self.frame_h, self.frame_w = h, w
                self._line_px = self._compute_line_px()
                logger.info(
                    f"FishTracker: frame size → {w}×{h}, "
                    f"line_px={self._line_px:.0f}"
                )

        # ── Expire stale lost-track entries ─────────────────────────────────
        expired = [t for t, (_, exp) in self._lost.items() if frame_idx >= exp]
        for t in expired:
            self._lost.pop(t, None)

        expired_u = [t for t, (_, _s, exp) in self._uncounted_lost.items()
                     if frame_idx >= exp]
        for t in expired_u:
            self._uncounted_lost.pop(t, None)

        # ── Move gone tracks to lost registry ────────────────────────────────
        seen_this_frame = set(track_ids)
        for tid in list(self._active.keys()):
            if tid not in seen_this_frame:
                last_box, _c, _ = self._active.pop(tid)
                streak = self._hit_streak.pop(tid, 0)
                self._prev_coord.pop(tid, None)  # prevent memory leak in 24/7 loops
                if tid in self.seen_ids:
                    # Counted fish → remember position for re-ID
                    self._lost[tid] = (last_box, frame_idx + TRACKER_LOST_TTL)
                else:
                    # Not yet counted → remember position + streak for re-ID
                    self._uncounted_lost[tid] = (
                        last_box, streak, frame_idx + TRACKER_LOST_TTL
                    )

        # ── Process each detected track ───────────────────────────────────
        new_fish_this_frame = 0

        for tid, box in zip(track_ids, boxes):
            centroid = _centroid(box)
            self._active[tid] = (box, centroid, frame_idx)

            # ── Inherit streak from uncounted-lost re-ID match ──────────────
            if tid not in self._hit_streak:
                inherited = self._check_uncounted_reid(tid, box)
                if inherited > 0:
                    self._hit_streak[tid] = inherited
                else:
                    self._hit_streak[tid] = 1
            else:
                self._hit_streak[tid] += 1

            # ── Already counted: keep prev_coord current and move on ───────
            if tid in self.seen_ids:
                if COUNTING_LINE_ENABLED:
                    self._prev_coord[tid] = _line_coord(centroid, COUNTING_LINE_AXIS)
                continue

            # ── Re-ID check (counted fish) ──────────────────────────────────
            # A new track ID can be a ByteTrack re-assignment of a fish we
            # already counted (brief occlusion).  Only block if IoU is VERY
            # high (≥ TRACKER_REID_IOU_THRESH, default 0.80) AND the lost
            # entry is very recent (TRACKER_LOST_TTL, default 60 frames).
            # This prevents the common failure mode where many fish from the
            # same channel area all match against a single past fish at 0.35.
            #
            # IMPORTANT: transfer the old TID slot in seen_ids to the new TID.
            # Previously the old TID was left in seen_ids AND the new TID was
            # added, so the same fish occupied two set slots (double-count).
            old_tid = self._check_reid(tid, box)
            if old_tid is not None:
                # Transfer: remove old entry, register new TID as counted
                self.seen_ids.discard(old_tid)
                self.seen_ids.add(tid)
                if COUNTING_LINE_ENABLED:
                    self._prev_coord[tid] = _line_coord(centroid, COUNTING_LINE_AXIS)
                logger.info(
                    f"  [tracker] Frame {frame_idx}: TID {tid} ← re-ID of "
                    f"counted TID {old_tid} (transferred, total={len(self.seen_ids)})"
                )
                continue

            # ── Hit-streak gate ───────────────────────────────────────────
            if self._hit_streak[tid] < TRACKER_MIN_HIT_STREAK:
                continue

            # ── Line-crossing gate ────────────────────────────────────────
            if COUNTING_LINE_ENABLED:
                coord      = _line_coord(centroid, COUNTING_LINE_AXIS)
                prev_coord = self._prev_coord.get(tid)
                self._prev_coord[tid] = coord   # always update

                if prev_coord is None:
                    # ── First observation of this track ───────────────────
                    # Fast fish may be first detected BELOW the line — they
                    # crossed it between frames.  Count them immediately.
                    # Fish above the line wait for an explicit crossing.
                    if coord < self._line_px:
                        # Above line — record side, wait for crossing
                        continue
                    # Below line on first sighting → must have entered from
                    # above; count it (fall through to counting block).
                else:
                    # ── Subsequent frame: detect top→bottom transition ────
                    crossed = (prev_coord < self._line_px <= coord)
                    if not crossed:
                        continue

            # ── Count ─────────────────────────────────────────────────────
            self.seen_ids.add(tid)
            new_fish_this_frame += 1
            logger.info(
                f"  [tracker] Frame {frame_idx}: TID {tid} COUNTED "
                f"(streak={self._hit_streak[tid]}, total={len(self.seen_ids)})"
            )

        return new_fish_this_frame, len(self.seen_ids)

    @property
    def total(self) -> int:
        return len(self.seen_ids)

    def draw_counting_line(self, frame: np.ndarray) -> np.ndarray:
        return draw_counting_line(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_line_px(self) -> float:
        if COUNTING_LINE_AXIS == "y":
            return self.frame_h * COUNTING_LINE_POSITION
        return self.frame_w * COUNTING_LINE_POSITION

    def _check_reid(self, tid: int, box: Box) -> Optional[int]:
        """
        Check whether this new track ID is almost certainly a ByteTrack
        re-assignment of a *counted* fish (brief occlusion).

        Returns the matched lost TID (int) on a hit so the caller can
        transfer the seen_ids entry, or None on a miss.

        Uses a HIGH IoU threshold (TRACKER_REID_IOU_THRESH, default 0.80) so
        that genuine new fish passing through the same channel area are NOT
        blocked.
        """
        best_iou  = 0.0
        best_ltid: Optional[int] = None

        for lost_tid, (lost_box, _expiry) in self._lost.items():
            iou = _iou(box, lost_box)
            if iou > best_iou:
                best_iou  = iou
                best_ltid = lost_tid

        if best_iou >= TRACKER_REID_IOU_THRESH and best_ltid is not None:
            self._lost.pop(best_ltid, None)
            logger.debug(
                f"Re-ID (counted): TID {tid} ← lost TID {best_ltid} "
                f"(IoU={best_iou:.2f})"
            )
            return best_ltid
        return None

    def _check_uncounted_reid(self, tid: int, box: Box) -> int:
        """
        Check whether this new track ID maps (by IoU) to a recently-lost
        UNCOUNTED fish.

        Returns the inherited hit-streak (> 0) on a match so the caller can
        prime self._hit_streak[tid] with it, or 0 on no match.

        Uses the same IoU threshold as _check_reid so the bar is equally high.
        """
        best_iou  = 0.0
        best_ltid: Optional[int] = None

        for lost_tid, (lost_box, _streak, _expiry) in self._uncounted_lost.items():
            iou = _iou(box, lost_box)
            if iou > best_iou:
                best_iou  = iou
                best_ltid = lost_tid

        if best_iou >= TRACKER_REID_IOU_THRESH and best_ltid is not None:
            _lb, inherited_streak, _exp = self._uncounted_lost.pop(best_ltid)
            logger.debug(
                f"Re-ID (uncounted): TID {tid} ← lost TID {best_ltid} "
                f"(IoU={best_iou:.2f}, inherited streak={inherited_streak})"
            )
            return inherited_streak
        return 0
