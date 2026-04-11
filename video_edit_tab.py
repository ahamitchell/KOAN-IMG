"""video_edit_tab.py — VIDEO EDIT (experiment) tab for KOAN.img

Library of finished videos + basic beat-sync speed-ramp editing.
Phase 1: video library with thumbnails, add/remove, preview.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

# ── constants ────────────────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
THUMB_W, THUMB_H = 240, 135
LIBRARY_FILE = Path(__file__).parent / "video_edit_library.json"

# ── shared style ─────────────────────────────────────────────────────────────
_STYLE = """
QWidget          { background: #050607; color: #40ff6b; font-family: 'Consolas','Courier New',monospace; }
QGroupBox        { border: 1px solid rgba(64,255,107,0.25); border-radius: 6px;
                   margin-top: 10px; padding-top: 14px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QPushButton      { background: rgba(64,255,107,0.12); border: 1px solid rgba(64,255,107,0.3);
                   border-radius: 4px; padding: 6px 14px; font-weight: 600; }
QPushButton:hover { background: rgba(64,255,107,0.22); }
QLabel#empty     { color: rgba(64,255,107,0.4); font-size: 14px; }
QFrame#card      { background: rgba(64,255,107,0.06); border: 1px solid rgba(64,255,107,0.15);
                   border-radius: 6px; }
QFrame#card:hover { border-color: rgba(64,255,107,0.4); }
QFrame#card[sel="1"] { border: 2px solid #40ff6b; background: rgba(64,255,107,0.12); }
"""


def _extract_thumbnail(video_path: str, w: int = THUMB_W, h: int = THUMB_H) -> Optional[QPixmap]:
    """Extract first frame from video using ffmpeg, return as QPixmap."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ih, iw = frame.shape[:2]
        # scale to fit
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        frame = cv2.resize(frame, (nw, nh))
        qimg = QImage(frame.data, nw, nh, nw * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())
    except Exception:
        return None


# ── VideoCard ────────────────────────────────────────────────────────────────
class VideoCard(QFrame):
    """Single video tile in the library grid."""
    clicked   = pyqtSignal(str)          # path
    remove_me = pyqtSignal(str)          # path

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setObjectName("card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(THUMB_W + 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(4)

        # thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(THUMB_W, THUMB_H)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet("background:#070a08; border-radius:4px;")
        pix = _extract_thumbnail(path)
        if pix:
            self._thumb.setPixmap(pix)
        else:
            self._thumb.setText("(no preview)")
        vl.addWidget(self._thumb)

        # filename
        name = Path(path).name
        if len(name) > 35:
            name = name[:32] + "…"
        lbl = QLabel(name)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size:10px; color:rgba(64,255,107,0.7);")
        vl.addWidget(lbl)

        # remove button
        btn_rm = QPushButton("✕ Remove")
        btn_rm.setFixedHeight(24)
        btn_rm.clicked.connect(lambda: self.remove_me.emit(self.path))
        vl.addWidget(btn_rm)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.path)
        super().mousePressEvent(ev)


# ── VideoEditTab ─────────────────────────────────────────────────────────────
class VideoEditTab(QWidget):
    """VIDEO EDIT tab — library of finished videos + future beat-sync editing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_STYLE)
        self._videos: List[str] = []   # list of video file paths
        self._cards: List[VideoCard] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── left: library ────────────────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("VIDEO LIBRARY")
        grp_lay = QVBoxLayout(grp)

        # buttons row
        btn_row = QHBoxLayout()
        btn_add = QPushButton("＋  Add Video")
        btn_add.clicked.connect(self._add_video_dialog)
        btn_row.addWidget(btn_add)

        btn_scan = QPushButton("📁  Scan Folder")
        btn_scan.clicked.connect(self._scan_folder)
        btn_row.addWidget(btn_scan)

        btn_row.addStretch()
        grp_lay.addLayout(btn_row)

        # grid scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._grid_inner = QWidget()
        self._grid = QGridLayout(self._grid_inner)
        self._grid.setSpacing(10)
        self._grid.setContentsMargins(10, 10, 10, 10)
        self._scroll.setWidget(self._grid_inner)

        grp_lay.addWidget(self._scroll, 1)

        # empty label
        self._empty_lbl = QLabel("No videos yet — add videos or scan a folder.", objectName="empty")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grp_lay.addWidget(self._empty_lbl)

        left_lay.addWidget(grp)
        splitter.addWidget(left)

        # ── right: preview / future editing area ─────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        grp_edit = QGroupBox("PREVIEW")
        edit_lay = QVBoxLayout(grp_edit)

        self._preview_lbl = QLabel("Select a video from the library to preview.")
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setStyleSheet("color: rgba(64,255,107,0.4); font-size: 13px; padding: 40px;")
        self._preview_lbl.setWordWrap(True)
        edit_lay.addWidget(self._preview_lbl)

        self._preview_thumb = QLabel()
        self._preview_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_thumb.setFixedHeight(300)
        self._preview_thumb.hide()
        edit_lay.addWidget(self._preview_thumb)

        self._preview_path = QLabel()
        self._preview_path.setWordWrap(True)
        self._preview_path.setStyleSheet("font-size:11px; color:rgba(64,255,107,0.6); padding: 8px;")
        self._preview_path.hide()
        edit_lay.addWidget(self._preview_path)

        btn_open = QPushButton("▶  Open in Player")
        btn_open.clicked.connect(self._open_selected)
        edit_lay.addWidget(btn_open)

        edit_lay.addStretch()

        # placeholder for future editing tools
        future_lbl = QLabel("Beat-sync editing tools coming soon.")
        future_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        future_lbl.setStyleSheet("color: rgba(64,255,107,0.2); font-size: 11px; padding: 20px;")
        edit_lay.addWidget(future_lbl)

        right_lay.addWidget(grp_edit)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        # load saved library
        self._selected_path: Optional[str] = None
        self._load_library()
        self._rebuild_grid()

    # ── library persistence ──────────────────────────────────────────────
    def _load_library(self) -> None:
        if LIBRARY_FILE.exists():
            try:
                data = json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
                self._videos = [p for p in data.get("videos", []) if Path(p).exists()]
            except Exception:
                self._videos = []

    def save_state(self) -> None:
        try:
            LIBRARY_FILE.write_text(
                json.dumps({"videos": self._videos}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── add / scan ───────────────────────────────────────────────────────
    def add_video(self, path: str) -> None:
        """Add a video to the library (called externally or internally)."""
        if path not in self._videos and Path(path).exists():
            self._videos.append(path)
            self._rebuild_grid()
            self.save_state()

    def _add_video_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Videos",
            "",
            "Video files (*.mp4 *.mov *.m4v *.avi *.mkv *.webm);;All files (*)",
        )
        for p in paths:
            if p not in self._videos:
                self._videos.append(p)
        self._rebuild_grid()
        self.save_state()

    def _scan_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Scan Folder for Videos")
        if not folder:
            return
        count = 0
        for f in Path(folder).rglob("*"):
            if f.suffix.lower() in VIDEO_EXTS and str(f) not in self._videos:
                self._videos.append(str(f))
                count += 1
        self._rebuild_grid()
        self.save_state()

    def _remove_video(self, path: str) -> None:
        self._videos = [v for v in self._videos if v != path]
        if self._selected_path == path:
            self._selected_path = None
            self._preview_thumb.hide()
            self._preview_path.hide()
            self._preview_lbl.show()
        self._rebuild_grid()
        self.save_state()

    # ── grid ─────────────────────────────────────────────────────────────
    def _rebuild_grid(self) -> None:
        # clear
        for card in self._cards:
            self._grid.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        self._empty_lbl.setVisible(len(self._videos) == 0)

        cols = max(1, (self._scroll.viewport().width()) // (THUMB_W + 28))

        for i, vpath in enumerate(self._videos):
            card = VideoCard(vpath)
            card.clicked.connect(self._on_card_clicked)
            card.remove_me.connect(self._remove_video)
            self._grid.addWidget(card, i // cols, i % cols)
            self._cards.append(card)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._videos:
            self._rebuild_grid()

    # ── preview ──────────────────────────────────────────────────────────
    def _on_card_clicked(self, path: str) -> None:
        self._selected_path = path
        self._preview_lbl.hide()

        pix = _extract_thumbnail(path, 480, 270)
        if pix:
            self._preview_thumb.setPixmap(pix)
            self._preview_thumb.show()
        else:
            self._preview_thumb.hide()

        self._preview_path.setText(path)
        self._preview_path.show()

    def _open_selected(self) -> None:
        if not self._selected_path:
            return
        import os, platform
        p = self._selected_path
        if platform.system() == "Windows":
            os.startfile(p)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
