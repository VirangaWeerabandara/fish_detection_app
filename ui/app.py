"""
ui/app.py
─────────
PyQt5 application — Fish Detection & Counting.

The UI imports core/ directly as a Python library.
No HTTP, no FastAPI, no requests — everything runs in-process.

Communication pattern
─────────────────────
• UploadWorker  (QThread) → calls core.video.extract_frames()
                           → emits upload_finished(frame_count: int)
• DetectionWorker (QThread) → calls core.detector directly
                            → emits frame_ready(data: dict)  per frame
                            → emits processing_done(result: dict) when done
• show_frame()  → renders self.frames[idx] + self.detections[idx] in-memory
                  with cv2 drawing → QPixmap (zero HTTP, zero encoding)
"""

import sys
import threading
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QProgressBar, QFileDialog,
    QMessageBox, QTextEdit, QSizePolicy,
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from core.config import (
    TARGET_FPS, MAX_DIM,
    TRACKER_RESET_INTERVAL, PREVIEW_INTERVAL, PREVIEW_MAX_DIM,
    DEVICE,
)
from core.video import extract_frames
from core.detector import FishDetector, draw_boxes_on_frame, make_preview_frame

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bgr_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR numpy array to a QPixmap (no file I/O, no encoding)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    # tobytes() makes a copy so QImage owns the data safely
    qi = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qi)


# ─────────────────────────────────────────────────────────────────────────────
# Upload Worker  — extracts frames in a background thread
# ─────────────────────────────────────────────────────────────────────────────
class UploadWorker(QThread):
    """
    Runs frame extraction (CPU-bound) in a dedicated thread so the UI
    stays responsive during large video loads.

    After the thread finishes, retrieve the frames via worker.frames.
    """
    upload_finished = pyqtSignal(int)   # emits frame_count
    upload_error    = pyqtSignal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self.frames: list = []           # populated by run(); read by main thread

    def run(self):
        try:
            self.frames = extract_frames(self.video_path, TARGET_FPS, MAX_DIM)
            self.upload_finished.emit(len(self.frames))
        except Exception as e:
            self.upload_error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Detection Worker  — runs YOLO inference in a background thread
# ─────────────────────────────────────────────────────────────────────────────
class DetectionWorker(QThread):
    """
    Runs YOLO + ByteTrack inference directly (no HTTP, no asyncio).

    Signals
    ───────
    frame_ready(dict)     — emitted per frame with live stats + preview image
    processing_done(dict) — emitted once when all frames are processed
    error_occurred(str)   — emitted on unrecoverable error

    The dict passed to frame_ready contains:
        frame_idx, total_frames, fish_count, unique_fish_count,
        image (numpy BGR array or None), is_processing (True)

    After processing_done, the caller can read worker.detections for the
    full per-frame detection list.
    """
    frame_ready     = pyqtSignal(dict)
    processing_done = pyqtSignal(dict)
    error_occurred  = pyqtSignal(str)

    def __init__(self,
                 frames: list,
                 detector: FishDetector,
                 confidence: float,
                 threshold: Optional[int] = None):
        super().__init__()
        self.frames     = frames
        self.detector   = detector
        self.confidence = confidence
        self.threshold  = threshold
        self._stop      = threading.Event()
        self.detections: list = []        # available after processing_done

    def stop(self):
        """Request a graceful stop after the current frame."""
        self._stop.set()

    def run(self):
        if not self.detector.is_loaded:
            self.error_occurred.emit("Model not loaded — check MODEL_PATH in core/config.py")
            return

        seen_ids:      set           = set()
        detections:    list          = []
        stopped_at:    int           = -1
        total:         int           = len(self.frames)
        last_preview:  Optional[np.ndarray] = None
        tracker_epoch: int           = 0

        self.detector.reset_tracker()
        logger.info(f"DetectionWorker started — {total} frames, "
                    f"conf={self.confidence}, threshold={self.threshold}, "
                    f"device={DEVICE}")

        for idx, frame in enumerate(self.frames):

            # ── Stop-flag check ───────────────────────────────────────────────
            if self._stop.is_set():
                logger.info(f"Worker stopped at frame {idx}")
                break

            # ── Periodic ByteTrack reset to cap state growth ──────────────────
            if idx > 0 and idx % TRACKER_RESET_INTERVAL == 0:
                self.detector.reset_tracker()
                tracker_epoch += 1
                logger.info(f"Tracker reset #{tracker_epoch} at frame {idx}")

            # ── YOLO inference ────────────────────────────────────────────────
            try:
                results = self.detector.track(frame, self.confidence)
            except Exception as e:
                logger.warning(f"track() failed frame {idx}: {e} — falling back")
                try:
                    results = self.detector.predict(frame, self.confidence)
                except Exception as e2:
                    logger.error(f"predict() also failed frame {idx}: {e2} — skip")
                    continue

            # ── Parse results ─────────────────────────────────────────────────
            boxes_obj     = results[0].boxes
            fish_in_frame = len(boxes_obj) if boxes_obj else 0
            track_ids: list = []
            new_fish = 0

            if boxes_obj is not None and boxes_obj.id is not None:
                for tid in boxes_obj.id.cpu().numpy().tolist():
                    tid_int = int(tid)
                    track_ids.append(tid_int)
                    if tid_int not in seen_ids:
                        seen_ids.add(tid_int)
                        new_fish += 1

            unique_so_far = len(seen_ids)
            boxes_list = boxes_obj.xyxy.cpu().numpy().tolist() if fish_in_frame > 0 else []
            confs_list  = boxes_obj.conf.cpu().numpy().tolist() if fish_in_frame > 0 else []

            detections.append({
                "frame_idx":           idx,
                "fish_count":          fish_in_frame,
                "new_fish_this_frame": new_fish,
                "unique_fish_count":   unique_so_far,
                "track_ids":           track_ids,
                "boxes":               boxes_list,
                "confidences":         confs_list,
            })

            # ── Live preview (downscaled numpy array, no encoding) ────────────
            encode_now = (idx % PREVIEW_INTERVAL == 0)
            if encode_now:
                try:
                    last_preview = make_preview_frame(
                        frame, boxes_list, confs_list, track_ids, PREVIEW_MAX_DIM)
                except Exception as e:
                    logger.error(f"Preview failed frame {idx}: {e}")

            # ── Emit live update to UI ────────────────────────────────────────
            self.frame_ready.emit({
                "frame_idx":         idx,
                "total_frames":      total,
                "fish_count":        fish_in_frame,
                "unique_fish_count": unique_so_far,
                "image":             last_preview if encode_now else None,
                "is_processing":     True,
            })

            # ── Periodic GPU cache flush ──────────────────────────────────────
            if idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── Threshold check ───────────────────────────────────────────────
            if self.threshold is not None and unique_so_far >= self.threshold:
                stopped_at = idx
                logger.info(f"Threshold {self.threshold} reached at frame {idx}")
                break

        # ── Finalize ──────────────────────────────────────────────────────────
        self.detections = detections
        unique_fish = len(seen_ids)
        n_proc      = len(detections)
        avg         = round(sum(d["fish_count"] for d in detections) / n_proc, 2) \
                      if n_proc else 0

        self.processing_done.emit({
            "status":                 "stopped" if stopped_at >= 0 else "completed",
            "total_frames_processed": n_proc,
            "total_fish_detected":    unique_fish,
            "total_fish_counted":     unique_fish,
            "average_fish_per_frame": avg,
            "stopped_at_frame":       stopped_at,
            "threshold":              self.threshold,
            "detections":             detections,
        })
        logger.info(f"DetectionWorker done — unique fish: {unique_fish}, frames: {n_proc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Window
# ─────────────────────────────────────────────────────────────────────────────
class FishDetectionApp(QMainWindow):
    """
    Main PyQt5 window.

    Owns:
      self.detector    — FishDetector instance (shared with DetectionWorker)
      self.frames      — list of BGR numpy arrays (set after upload)
      self.detections  — list of per-frame detection dicts (set after processing)
    """

    def __init__(self, detector: FishDetector):
        super().__init__()
        self.detector = detector

        self.setWindowTitle("🐟 Fish Detection & Counting System")
        self.setGeometry(100, 100, 1920, 1080)
        self.showMaximized()

        # App state
        self.video_path:        Optional[str]  = None
        self.frames:            list           = []
        self.detections:        list           = []
        self.current_frame_idx: int            = 0
        self.upload_worker:     Optional[UploadWorker]    = None
        self.det_worker:        Optional[DetectionWorker] = None

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        section_font = QFont()
        section_font.setPointSize(11)
        section_font.setBold(True)

        # ── Left control panel ────────────────────────────────────────────────
        left = QVBoxLayout()

        title_lbl = QLabel("Fish Detection Control Panel")
        title_font = QFont(); title_font.setPointSize(14); title_font.setBold(True)
        title_lbl.setFont(title_font)
        left.addWidget(title_lbl)

        # 1. Upload
        left.addWidget(self._section_lbl("1. Upload Video", section_font))
        self.file_label = QLabel("No video selected")
        self.file_label.setStyleSheet("color: gray; padding: 5px;")
        left.addWidget(self.file_label)

        self.upload_btn = QPushButton("📁 Select & Upload Video")
        self.upload_btn.clicked.connect(self.select_and_upload_video)
        self.upload_btn.setStyleSheet(
            "QPushButton { background-color:#4CAF50; color:white; padding:10px;"
            " border-radius:5px; font-weight:bold; }")
        left.addWidget(self.upload_btn)

        self.upload_progress = QProgressBar()
        self.upload_progress.setRange(0, 0)   # indeterminate while extracting
        self.upload_progress.setVisible(False)
        left.addWidget(self.upload_progress)

        # 2. Configure
        left.addWidget(self._section_lbl("2. Configure Detection", section_font))

        row_thr = QHBoxLayout()
        row_thr.addWidget(QLabel("Fish Count Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 10000)
        self.threshold_spin.setValue(10)
        row_thr.addWidget(self.threshold_spin)
        left.addLayout(row_thr)

        row_conf = QHBoxLayout()
        row_conf.addWidget(QLabel("Detection Confidence (0–100):"))
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(85)
        self.confidence_spin.setSuffix("%")
        row_conf.addWidget(self.confidence_spin)
        left.addLayout(row_conf)

        # 3. Process
        left.addWidget(self._section_lbl("3. Process Video", section_font))

        self.process_threshold_btn = QPushButton("▶️ Start Detection (Stop at Threshold)")
        self.process_threshold_btn.clicked.connect(self.process_with_threshold)
        self.process_threshold_btn.setEnabled(False)
        self.process_threshold_btn.setStyleSheet(
            "QPushButton { background-color:#2196F3; color:white; padding:10px;"
            " border-radius:5px; font-weight:bold; }"
            "QPushButton:disabled { background-color:#CCCCCC; }")
        left.addWidget(self.process_threshold_btn)

        self.process_all_btn = QPushButton("▶️ Analyze All Frames")
        self.process_all_btn.clicked.connect(self.process_all_frames)
        self.process_all_btn.setEnabled(False)
        self.process_all_btn.setStyleSheet(
            "QPushButton { background-color:#FF9800; color:white; padding:10px;"
            " border-radius:5px; font-weight:bold; }"
            "QPushButton:disabled { background-color:#CCCCCC; }")
        left.addWidget(self.process_all_btn)

        self.stop_btn = QPushButton("⏹ Stop Processing")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color:#e53935; color:white; padding:10px;"
            " border-radius:5px; font-weight:bold; }"
            "QPushButton:disabled { background-color:#CCCCCC; }")
        left.addWidget(self.stop_btn)

        # 4. Results
        left.addWidget(self._section_lbl("4. Results", section_font))
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        left.addWidget(self.status_text)

        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        self.clear_btn.setStyleSheet(
            "QPushButton { background-color:#f44336; color:white;"
            " padding:8px; border-radius:5px; }")
        left.addWidget(self.clear_btn)
        left.addStretch()

        # ── Right display panel ───────────────────────────────────────────────
        right = QVBoxLayout()

        right.addWidget(self._section_lbl("Frame Visualization", section_font))

        # LIVE badge
        self.live_label = QLabel("🔴 LIVE")
        self.live_label.setAlignment(Qt.AlignCenter)
        self.live_label.setStyleSheet(
            "QLabel { background:#1a1a2e; color:#e94560; font-weight:bold;"
            " font-size:13px; padding:4px 10px; border-radius:4px;"
            " letter-spacing:2px; }")
        self.live_label.setVisible(False)
        right.addWidget(self.live_label)

        # Detection progress bar
        self.det_progress = QProgressBar()
        self.det_progress.setRange(0, 100)
        self.det_progress.setValue(0)
        self.det_progress.setVisible(False)
        self.det_progress.setStyleSheet(
            "QProgressBar { border:1px solid #ccc; border-radius:4px;"
            " text-align:center; height:18px; }"
            "QProgressBar::chunk { background:qlineargradient("
            "x1:0,y1:0,x2:1,y2:0,stop:0 #00c6ff,stop:1 #0072ff); }")
        right.addWidget(self.det_progress)

        # Frame display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border:2px solid #ddd; background:#f5f5f5;")
        right.addWidget(self.image_label, stretch=1)

        # Navigation row
        nav = QHBoxLayout()

        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.clicked.connect(self.show_previous_frame)
        self.prev_btn.setEnabled(False)
        nav.addWidget(self.prev_btn)

        self.frame_info_lbl = QLabel("No frame loaded")
        self.frame_info_lbl.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.frame_info_lbl, stretch=1)

        jump = QHBoxLayout(); jump.setSpacing(4)
        jump.addWidget(QLabel("Go to frame:"))
        self.jump_spin = QSpinBox()
        self.jump_spin.setRange(1, 1)
        self.jump_spin.setEnabled(False)
        self.jump_spin.setFixedWidth(80)
        self.jump_spin.editingFinished.connect(self.go_to_frame)
        jump.addWidget(self.jump_spin)
        self.jump_btn = QPushButton("🔍 Go")
        self.jump_btn.clicked.connect(self.go_to_frame)
        self.jump_btn.setEnabled(False)
        self.jump_btn.setFixedWidth(60)
        self.jump_btn.setStyleSheet(
            "QPushButton { background-color:#607D8B; color:white;"
            " border-radius:4px; padding:4px 8px; font-weight:bold; }"
            "QPushButton:disabled { background-color:#CCCCCC; }")
        jump.addWidget(self.jump_btn)
        nav.addLayout(jump)

        self.next_btn = QPushButton("Next ➡️")
        self.next_btn.clicked.connect(self.show_next_frame)
        self.next_btn.setEnabled(False)
        nav.addWidget(self.next_btn)

        right.addLayout(nav)

        root.addLayout(left, 1)
        root.addLayout(right, 2)
        self.statusBar().showMessage(
            "Ready" if self.detector.is_loaded else
            "⚠️  Model not loaded — check MODEL_PATH in core/config.py")

    # ─────────────────────────────────────────────────────────────────────────
    # Small helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _section_lbl(text: str, font: QFont) -> QLabel:
        lbl = QLabel(text); lbl.setFont(font); return lbl

    def _set_process_btns(self, enabled: bool):
        self.process_threshold_btn.setEnabled(enabled)
        self.process_all_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)

    def _show_live_ui(self):
        self.live_label.setVisible(True)
        self.det_progress.setVisible(True)
        self.det_progress.setValue(0)
        self.image_label.setPixmap(QPixmap())
        self.frame_info_lbl.setText("⏳ Processing…")

    def _display_pixmap(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def log(self, msg: str):
        self.status_text.append(msg)
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum())

    # ─────────────────────────────────────────────────────────────────────────
    # Upload flow
    # ─────────────────────────────────────────────────────────────────────────
    def select_and_upload_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if not path:
            return

        self.video_path = path
        self.file_label.setText(f"Selected: {Path(path).name}")
        self.file_label.setStyleSheet("color:green; padding:5px;")
        self.upload_progress.setVisible(True)
        self.upload_btn.setEnabled(False)
        self.statusBar().showMessage("Extracting frames…")

        self.upload_worker = UploadWorker(path)
        self.upload_worker.upload_finished.connect(self.on_upload_done)
        self.upload_worker.upload_error.connect(self.on_upload_error)
        self.upload_worker.start()

    def on_upload_done(self, frame_count: int):
        # Retrieve the extracted frames from the finished worker
        self.frames = self.upload_worker.frames
        self.upload_progress.setVisible(False)
        self.upload_btn.setEnabled(True)
        self.statusBar().showMessage(f"✅ Ready: {frame_count} frames extracted")
        self._set_process_btns(True)
        self.log(
            f"✅ Video Ready\n"
            f"   File  : {Path(self.video_path).name}\n"
            f"   Frames: {frame_count}")

    def on_upload_error(self, err: str):
        self.upload_progress.setVisible(False)
        self.upload_btn.setEnabled(True)
        QMessageBox.critical(self, "Load Error", err)
        self.statusBar().showMessage("❌ Frame extraction failed")

    # ─────────────────────────────────────────────────────────────────────────
    # Detection flow — start / stop
    # ─────────────────────────────────────────────────────────────────────────
    def _start_detection(self, confidence: float, threshold: Optional[int] = None):
        self._set_process_btns(False)
        self._show_live_ui()

        self.det_worker = DetectionWorker(
            self.frames, self.detector, confidence, threshold)
        self.det_worker.frame_ready.connect(self.on_frame_update)
        self.det_worker.processing_done.connect(self.on_processing_done)
        self.det_worker.error_occurred.connect(self.on_processing_error)
        self.det_worker.start()

    def process_with_threshold(self):
        if not self.frames:
            QMessageBox.warning(self, "No Video", "Load a video first"); return
        thr  = self.threshold_spin.value()
        conf = self.confidence_spin.value() / 100.0
        self.statusBar().showMessage(f"Processing… (stop at {thr} unique fish)")
        self.log(f"🔍 Detection — threshold: {thr}, conf: {conf:.2f}")
        self._start_detection(conf, thr)

    def process_all_frames(self):
        if not self.frames:
            QMessageBox.warning(self, "No Video", "Load a video first"); return
        conf = self.confidence_spin.value() / 100.0
        self.statusBar().showMessage("Analyzing all frames…")
        self.log(f"🔍 Analyze all — conf: {conf:.2f}")
        self._start_detection(conf, None)

    def stop_processing(self):
        if self.det_worker:
            self.det_worker.stop()
        self._set_process_btns(True)
        self.live_label.setVisible(False)
        self.det_progress.setVisible(False)
        self.statusBar().showMessage("Processing stopped")

    # ─────────────────────────────────────────────────────────────────────────
    # Detection callbacks  (delivered on the main/GUI thread via Qt signals)
    # ─────────────────────────────────────────────────────────────────────────
    def on_frame_update(self, data: dict):
        """Called for every frame — updates progress bar and live preview."""
        try:
            idx     = data.get("frame_idx", 0)
            total   = data.get("total_frames", 1)
            unique  = data.get("unique_fish_count", 0)
            visible = data.get("fish_count", 0)
            pct     = int((idx + 1) / max(total, 1) * 100)

            self.det_progress.setValue(pct)
            self.live_label.setText(f"🔴 LIVE  {pct}%")

            # image is a numpy BGR array (or None if not a preview frame)
            preview: Optional[np.ndarray] = data.get("image")
            if preview is not None:
                self._display_pixmap(_bgr_to_pixmap(preview))

            self.frame_info_lbl.setText(
                f"⚡ Frame {idx + 1}/{total}  │  "
                f"Visible: {visible}  │  🐟 Unique: {unique}")
        except Exception:
            pass

    def on_processing_done(self, result: dict):
        """Called once when the DetectionWorker finishes."""
        self.live_label.setVisible(False)
        self.det_progress.setVisible(False)
        self._set_process_btns(True)

        # Grab detections from the finished worker
        self.detections        = self.det_worker.detections
        self.current_frame_idx = 0

        if self.detections:
            n = len(self.detections)
            self.show_frame(0)
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.jump_spin.setMaximum(n)
            self.jump_spin.setValue(1)
            self.jump_spin.setEnabled(True)
            self.jump_btn.setEnabled(True)

        status      = result.get("status", "?").upper()
        frames_done = result.get("total_frames_processed", len(self.detections))
        unique      = (result.get("total_fish_detected")
                       or result.get("total_fish_counted", 0))
        avg         = result.get("average_fish_per_frame", "N/A")
        sf          = result.get("stopped_at_frame", -1)

        msg  = f"✅ Done!\n\nStatus : {status}\nFrames : {frames_done}\n"
        msg += f"🐟 Unique fish: {unique}\n"
        if avg != "N/A":
            msg += f"   avg {avg}/frame\n"
        if sf >= 0:
            msg += f"   Threshold hit at frame {sf}\n"
        msg += "\nℹ️  ByteTrack — no double-counting."
        self.log(msg)
        self.statusBar().showMessage("✅ Detection complete — browse frames with arrows")

    def on_processing_error(self, err: str):
        self._set_process_btns(True)
        self.live_label.setVisible(False)
        self.det_progress.setVisible(False)
        QMessageBox.critical(self, "Detection Error", err)
        self.statusBar().showMessage("❌ Error during detection")

    # ─────────────────────────────────────────────────────────────────────────
    # Frame browsing  — renders directly from in-memory numpy arrays
    # ─────────────────────────────────────────────────────────────────────────
    def show_frame(self, idx: int):
        """
        Render frame idx from in-memory data.
        No HTTP call — draws boxes with OpenCV and converts to QPixmap directly.
        """
        if not self.detections or not self.frames or idx >= len(self.detections):
            return
        self.current_frame_idx = idx

        try:
            det       = self.detections[idx]
            frame_raw = self.frames[idx]
            annotated = draw_boxes_on_frame(
                frame_raw,
                det.get("boxes", []),
                det.get("confidences", []),
                det.get("track_ids", []),
            )
            self._display_pixmap(_bgr_to_pixmap(annotated))

            self.frame_info_lbl.setText(
                f"Frame {idx + 1}/{len(self.detections)}  │  "
                f"Visible: {det['fish_count']}  │  "
                f"Unique total: {det.get('unique_fish_count', '?')}")
            self.jump_spin.blockSignals(True)
            self.jump_spin.setValue(idx + 1)
            self.jump_spin.blockSignals(False)
        except Exception as e:
            self.log(f"⚠️ Frame render error: {e}")

    def show_previous_frame(self):
        if self.current_frame_idx > 0:
            self.show_frame(self.current_frame_idx - 1)

    def show_next_frame(self):
        if self.detections and self.current_frame_idx < len(self.detections) - 1:
            self.show_frame(self.current_frame_idx + 1)

    def go_to_frame(self):
        if not self.detections:
            return
        idx = max(0, min(self.jump_spin.value() - 1, len(self.detections) - 1))
        self.show_frame(idx)

    # ─────────────────────────────────────────────────────────────────────────
    # Clear
    # ─────────────────────────────────────────────────────────────────────────
    def clear_all(self):
        if QMessageBox.question(self, "Clear All", "Clear all data?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        if self.det_worker and self.det_worker.isRunning():
            self.det_worker.stop()
        self.video_path        = None
        self.frames            = []
        self.detections        = []
        self.current_frame_idx = 0
        self.file_label.setText("No video selected")
        self.file_label.setStyleSheet("color:gray; padding:5px;")
        self.image_label.setPixmap(QPixmap())
        self.frame_info_lbl.setText("No frame loaded")
        self.status_text.clear()
        self._set_process_btns(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.jump_spin.setValue(1)
        self.jump_spin.setMaximum(1)
        self.jump_spin.setEnabled(False)
        self.jump_btn.setEnabled(False)
        self.statusBar().showMessage("Cleared")
