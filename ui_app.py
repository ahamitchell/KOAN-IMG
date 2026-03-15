"""KOAN.img — PyQt6 desktop UI  (PICK + INDEX tabs)

Launch:  python ui_app.py
Prereq:  pip install PyQt6
"""
from __future__ import annotations

import json
import random
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from PyQt6.QtCore import (
    QObject, QSize, QThread, QTimer, Qt,
    pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import (
    QFont, QImage, QKeySequence, QMovie, QPixmap, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

from embedder import Embedder
from query import load_artifacts_cacheable, pick_similar_cached
from video_tab import VideoTab

# ── lazy embedder singleton ────────────────────────────────────────────────────
_embedder: Optional["Embedder"] = None

def _get_embedder() -> "Embedder":
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

# ── paths ─────────────────────────────────────────────────────────────────────
APP_ROOT     = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
STATE_FILE   = APP_ROOT / ".koan_ui_state.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "ai_index"
DEFAULT_SRC_DIR   = PROJECT_ROOT / "images"
DEFAULT_SETS_DIR  = PROJECT_ROOT / "sets"
PY           = sys.executable
CHUNK_SCRIPT = str((APP_ROOT / "index_images_chunked.py").resolve())
VIDEO_EXTS   = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
THUMB_W, THUMB_H = 280, 220
GRID_COLS = 4


# ── QSS green theme ───────────────────────────────────────────────────────────
QSS = """
QMainWindow, QWidget, QDialog { background: #050607; color: #40ff6b; }
QTabWidget::pane { border: 1px solid rgba(64,255,107,0.3); background: #050607; }
QTabBar::tab {
    background: #0a0f0b; color: #40ff6b;
    border: 1px solid rgba(64,255,107,0.3);
    padding: 6px 20px; margin-right: 2px;
}
QTabBar::tab:selected { background: #0d1a0f; border-bottom: 2px solid #40ff6b; }
QPushButton {
    background: #0a0f0b; border: 1px solid rgba(64,255,107,0.45);
    color: #40ff6b; padding: 6px 14px; border-radius: 4px; font-weight: bold;
}
QPushButton:hover  { background: #0d1a0f; border-color: #40ff6b; }
QPushButton:pressed { background: #0a2010; }
QPushButton:disabled { color: rgba(64,255,107,0.28); border-color: rgba(64,255,107,0.15); }
QPushButton#runBtn { font-size: 14px; padding: 10px; letter-spacing: 2px; }
QLineEdit, QTextEdit, QSpinBox {
    background: #070a08; border: 1px solid rgba(64,255,107,0.25);
    color: #40ff6b; padding: 4px 6px; border-radius: 3px;
    selection-background-color: #1a4a20;
}
QLineEdit:focus, QSpinBox:focus { border-color: rgba(64,255,107,0.7); }
QSlider::groove:horizontal {
    background: rgba(64,255,107,0.15); height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #40ff6b; width: 14px; height: 14px;
    border-radius: 7px; margin: -5px 0;
}
QSlider::sub-page:horizontal { background: rgba(64,255,107,0.5); border-radius: 2px; }
QCheckBox { color: #40ff6b; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid rgba(64,255,107,0.5); border-radius: 2px; background: #070a08;
}
QCheckBox::indicator:checked { background: #40ff6b; border-color: #40ff6b; }
QScrollArea { background: #050607; border: none; }
QScrollBar:vertical   { width: 7px;  background: #050607; }
QScrollBar:horizontal { height: 7px; background: #050607; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: rgba(64,255,107,0.28); border-radius: 3px; min-height: 24px; min-width: 24px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: rgba(64,255,107,0.55);
}
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
QGroupBox {
    border: 1px solid rgba(64,255,107,0.3); border-radius: 6px;
    margin-top: 12px; padding-top: 6px; color: #40ff6b; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
QFrame#card {
    border: 1px solid rgba(64,255,107,0.35); border-radius: 6px;
    background: rgba(5,9,7,0.55);
}
QFrame#card[sel="1"] {
    border-color: rgba(64,255,107,0.95); background: rgba(10,22,12,0.82);
}
QLabel#rank    { color: rgba(64,255,107,0.55); font-size: 11px; }
QLabel#caption { color: rgba(64,255,107,0.65); font-size: 10px; }
QLabel#hdr     { font-size: 34px; font-weight: 900; }
QLabel#cnt     { font-size: 34px; font-weight: 900; }
QLabel#selcnt  { font-size: 26px; font-weight: 900; color: rgba(64,255,107,0.85); }
QLabel#small   { color: rgba(64,255,107,0.6); font-size: 11px; }
"""


# ── state persistence ─────────────────────────────────────────────────────────
def _load_state() -> Dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_state(d: Dict[str, Any]) -> None:
    try:
        STATE_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

# ── thumbnail cache (synchronous PIL, in-process) ─────────────────────────────
_THUMB_CACHE: Dict[tuple, QPixmap] = {}

def _load_thumb(path: str, w: int = THUMB_W, h: int = THUMB_H) -> QPixmap:
    key = (path, w, h)
    if key in _THUMB_CACHE:
        return _THUMB_CACHE[key]
    try:
        ext = Path(path).suffix.lower()
        if ext in VIDEO_EXTS:
            import cv2
            cap = cv2.VideoCapture(path)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return QPixmap()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
        else:
            img = Image.open(path)
            try:
                img.seek(0)
            except Exception:
                pass
            img = img.convert("RGB")
        img.thumbnail((w, h), Image.LANCZOS)
        data = img.tobytes("raw", "RGB")
        qi = QImage(data, img.width, img.height,
                    img.width * 3, QImage.Format.Format_RGB888)
        px = QPixmap.fromImage(qi)
        _THUMB_CACHE[key] = px
        return px
    except Exception:
        return QPixmap()

class _VideoFrameCycler(QObject):
    """Cycles cv2 video frames on a QLabel using a QTimer. Parent it to the label."""
    def __init__(self, label: QLabel, frames: list, fps: int = 12):
        super().__init__(label)
        self._label  = label
        self._frames = frames
        self._idx    = 0
        if frames:
            label.setPixmap(frames[0])
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next)
        self._timer.start(max(1, 1000 // fps))

    def _next(self) -> None:
        if not self._frames:
            return
        self._idx = (self._idx + 1) % len(self._frames)
        self._label.setPixmap(self._frames[self._idx])


def _extract_video_frames(path: str, w: int, h: int, n: int = 24) -> list:
    """Return up to n evenly-spaced frames as QPixmaps."""
    try:
        import cv2
        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs  = [int(total * i / n) for i in range(n)] if total >= n else list(range(max(total, 1)))
        frames = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(frame)
            img.thumbnail((w, h), Image.LANCZOS)
            data  = img.tobytes("raw", "RGB")
            qi    = QImage(data, img.width, img.height,
                           img.width * 3, QImage.Format.Format_RGB888)
            frames.append(QPixmap.fromImage(qi))
        cap.release()
        return frames
    except Exception:
        return []


def _attach_thumb(label: QLabel, path: str, w: int, h: int) -> Optional[object]:
    """Attach thumbnail/animation to label. Returns object caller must keep alive."""
    ext = Path(path).suffix.lower()
    if ext == ".gif":
        movie = QMovie(path)
        movie.jumpToFrame(0)
        orig = movie.currentPixmap().size()
        if orig.width() > 0 and orig.height() > 0:
            ratio = min(w / orig.width(), h / orig.height())
            movie.setScaledSize(QSize(int(orig.width() * ratio), int(orig.height() * ratio)))
        else:
            movie.setScaledSize(QSize(w, h))
        label.setMovie(movie)
        movie.start()
        return movie
    elif ext in VIDEO_EXTS:
        frames = _extract_video_frames(path, w, h)
        if frames:
            return _VideoFrameCycler(label, frames)
        label.setText("▶")
        return None
    else:
        px = _load_thumb(path, w=w, h=h)
        if not px.isNull():
            label.setPixmap(px.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        else:
            label.setText("⚠")
        return None


# ── helpers ───────────────────────────────────────────────────────────────────
def _count_indexed(index_dir: Path) -> int:
    db = index_dir / "catalog.sqlite"
    if not db.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute("SELECT COUNT(*) FROM images;").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception:
        return 0

def _slugify(text: str) -> str:
    t = re.sub(r"[^a-z0-9\s]", " ", (text or "").strip().lower())
    return "-".join(t.split()[:5])

def _sanitize(name: str) -> str:
    n = re.sub(r"[^a-z0-9\-]", "-", (name or "").strip().lower())
    return re.sub(r"-{2,}", "-", n).strip("-") or "set"

def _unique_dir(root: Path, base: str) -> Path:
    cand = root / base
    if not cand.exists():
        return cand
    i = 2
    while (root / f"{base}-{i}").exists():
        i += 1
    return root / f"{base}-{i}"

def _browse_dir(parent: QWidget, edit: QLineEdit) -> None:
    d = QFileDialog.getExistingDirectory(parent, "Select folder",
                                          edit.text() or str(Path.home()))
    if d:
        edit.setText(d)

def _add_browse_row(layout, label: str, edit: QLineEdit, parent: QWidget) -> None:
    layout.addWidget(QLabel(label, objectName="small"))
    row = QHBoxLayout()
    row.addWidget(edit, 1)
    btn = QPushButton("…")
    btn.setMaximumWidth(32)
    btn.clicked.connect(lambda: _browse_dir(parent, edit))
    row.addWidget(btn)
    layout.addLayout(row)


# ── ImageCard ─────────────────────────────────────────────────────────────────
class ImageCard(QFrame):
    """Single result tile: thumbnail + checkbox + score label + MAKE SEED button."""
    selection_toggled = pyqtSignal(str, bool)
    seed_requested    = pyqtSignal(str)

    def __init__(self, path: str, top_label: str, caption: str,
                 selected: bool, already_seed: bool,
                 thumb_w: int = THUMB_W, thumb_h: int = THUMB_H,
                 parent=None) -> None:
        super().__init__(parent)
        self.path = path
        self.setObjectName("card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(thumb_w + 18)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(4)

        # top row: label + checkbox
        top = QHBoxLayout()
        lbl = QLabel(top_label, objectName="rank")
        lbl.setWordWrap(True)
        top.addWidget(lbl, 1)
        self._cb = QCheckBox()
        self._cb.setChecked(selected)
        self._cb.toggled.connect(self._on_check)
        top.addWidget(self._cb)
        vl.addLayout(top)

        # thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(thumb_w, thumb_h)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet("background:#070a08; border-radius:4px;")
        self._movie = _attach_thumb(self._thumb, path, thumb_w, thumb_h)
        vl.addWidget(self._thumb)

        # caption
        if caption:
            cap = QLabel(caption[:90] + ("…" if len(caption) > 90 else ""),
                         objectName="caption")
            cap.setWordWrap(True)
            vl.addWidget(cap)

        # MAKE SEED
        seed_btn = QPushButton("✓ IN SEEDS" if already_seed else "MAKE SEED")
        seed_btn.setEnabled(not already_seed)
        seed_btn.clicked.connect(lambda: self.seed_requested.emit(path))
        vl.addWidget(seed_btn)
        vl.addStretch()

        self._refresh_style(selected)

    def _on_check(self, v: bool) -> None:
        self._refresh_style(v)
        self.selection_toggled.emit(self.path, v)

    def _refresh_style(self, sel: bool) -> None:
        self.setProperty("sel", "1" if sel else "0")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_checked(self, v: bool) -> None:
        self._cb.blockSignals(True)
        self._cb.setChecked(v)
        self._cb.blockSignals(False)
        self._refresh_style(v)


# ── ImageGrid ─────────────────────────────────────────────────────────────────
class ImageGrid(QScrollArea):
    """Scrollable grid of ImageCards. Owns selected_paths across searches."""
    selection_changed = pyqtSignal(int)   # total selected count
    seed_requested    = pyqtSignal(str)   # path of image to make seed

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._inner = QWidget()
        self._grid  = QGridLayout(self._inner)
        self._grid.setSpacing(10)
        self._grid.setContentsMargins(10, 10, 10, 10)
        self.setWidget(self._inner)
        self._cards: List[ImageCard] = []
        self.selected_paths: Dict[str, bool] = {}
        self._current_items: List[Dict] = []
        self._current_seeds: set = set()
        self._last_cols: int = 0
        self._thumb_w: int = THUMB_W
        self._thumb_h: int = THUMB_H

    def _compute_cols(self) -> int:
        vw = max(1, self.viewport().width())
        card_w = self._thumb_w + 18 + self._grid.spacing()
        return max(1, vw // card_w)

    def set_thumb_size(self, w: int, h: int) -> None:
        self._thumb_w = w
        self._thumb_h = h
        self._last_cols = 0   # force full rebuild
        if self._current_items:
            cols = self._compute_cols()
            self._last_cols = cols
            self._rebuild(self._current_items, self._current_seeds, cols)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cols = self._compute_cols()
        if cols != self._last_cols and self._current_items:
            self._last_cols = cols
            self._rebuild(self._current_items, self._current_seeds, cols)

    def populate(self, items: List[Dict], seed_paths: set) -> None:
        """Rebuild the grid from a list of result dicts."""
        self._current_items = items
        self._current_seeds = seed_paths
        cols = self._compute_cols()
        self._last_cols = cols
        self._rebuild(items, seed_paths, cols)

    def _rebuild(self, items: List[Dict], seed_paths: set, cols: int) -> None:
        # clear existing cards
        for card in self._cards:
            self._grid.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, item in enumerate(items):
            p = item.get("path", "")
            is_seed   = item.get("is_seed", False)
            is_pinned = item.get("is_pinned", False)
            rank  = item.get("rank", 0)
            score = item.get("score", 0.0)

            if is_seed:
                label = f"🌱 {item.get('label', Path(p).name)}"
            elif is_pinned:
                label = "★ SELECTED"
            else:
                label = f"#{rank:02d}  {score:.4f}"

            self.selected_paths.setdefault(p, False)
            card = ImageCard(
                path        = p,
                top_label   = label,
                caption     = item.get("caption", ""),
                selected    = bool(self.selected_paths.get(p, False)),
                already_seed= (p in seed_paths),
                thumb_w     = self._thumb_w,
                thumb_h     = self._thumb_h,
            )
            card.selection_toggled.connect(self._on_toggle)
            card.seed_requested.connect(self.seed_requested)
            row, col = divmod(i, cols)
            self._grid.addWidget(card, row, col)
            self._cards.append(card)

    def _on_toggle(self, path: str, checked: bool) -> None:
        self.selected_paths[path] = checked
        self.selection_changed.emit(sum(v for v in self.selected_paths.values()))

    def select_all(self) -> None:
        for card in self._cards:
            self.selected_paths[card.path] = True
            card.set_checked(True)
        self.selection_changed.emit(len(self._cards))

    def clear_selection(self) -> None:
        for card in self._cards:
            self.selected_paths[card.path] = False
            card.set_checked(False)
        self.selection_changed.emit(0)

    def get_selected(self) -> List[str]:
        return [p for p, v in self.selected_paths.items() if v]



# ── SeedTile ──────────────────────────────────────────────────────────────────

class SeedTile(QFrame):
    remove_clicked    = pyqtSignal(int)          # index in seed list
    weight_changed    = pyqtSignal(int, float)   # index, new_weight
    selection_toggled = pyqtSignal(str, bool)    # path, checked

    def __init__(self, index: int, path: str, weight: float = 1.0, parent=None):
        super().__init__(parent)
        self.index = index
        self.path  = path
        self.setObjectName("card")
        self.setFixedWidth(330)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # checkbox + thumbnail row
        thumb_row = QHBoxLayout()
        thumb_row.setContentsMargins(0, 0, 0, 0)
        self._cb = QCheckBox()
        self._cb.setToolTip("Include in selection")
        self._cb.toggled.connect(lambda v: self.selection_toggled.emit(self.path, v))
        thumb_row.addWidget(self._cb, 0, Qt.AlignmentFlag.AlignTop)

        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(296, 160)
        thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_lbl.setStyleSheet("background:#070a08; border-radius:4px;")
        self._movie = _attach_thumb(thumb_lbl, path, 296, 160)
        thumb_row.addWidget(thumb_lbl)
        lay.addLayout(thumb_row)

        # filename label
        name_lbl = QLabel(Path(path).name)
        name_lbl.setWordWrap(True)
        name_lbl.setStyleSheet("font-size:10px; color:rgba(64,255,107,0.6);")
        lay.addWidget(name_lbl)

        # weight row
        wrow = QHBoxLayout()
        wlbl = QLabel("Weight:")
        wlbl.setStyleSheet("font-size:10px;")
        wrow.addWidget(wlbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(1, 30)
        self._slider.setValue(max(1, min(30, int(round(weight * 10)))))
        self._slider.setFixedWidth(120)
        wrow.addWidget(self._slider)

        self._wval = QLabel(f"{weight:.1f}")
        self._wval.setStyleSheet("font-size:10px; min-width:28px;")
        wrow.addWidget(self._wval)
        wrow.addStretch()
        lay.addLayout(wrow)

        self._slider.valueChanged.connect(self._on_slider)

        # remove button
        btn_rm = QPushButton("✕ Remove")
        btn_rm.setFixedHeight(24)
        btn_rm.clicked.connect(lambda: self.remove_clicked.emit(self.index))
        lay.addWidget(btn_rm)

    def _on_slider(self, v: int) -> None:
        w = v / 10.0
        self._wval.setText(f"{w:.1f}")
        self.weight_changed.emit(self.index, w)

    def get_weight(self) -> float:
        return self._slider.value() / 10.0


# ── SearchWorker ──────────────────────────────────────────────────────────────

class SearchWorker(QThread):
    finished = pyqtSignal(object)   # List[Dict]
    error    = pyqtSignal(str)

    def __init__(
        self,
        index_dir:   str,
        seeds:       List[Dict],   # [{"path": str, "weight": float}, ...]
        text_prompt: str,
        neg_prompt:  str,
        w_text:      float,
        n_results:   int,
        top_k:       int,
        w_clip:      float,
        dedupe:      bool,
        dedupe_thr:  float,
        parent=None,
    ):
        super().__init__(parent)
        self.index_dir   = index_dir
        self.seeds       = seeds
        self.text_prompt = text_prompt
        self.neg_prompt  = neg_prompt
        self.w_text      = w_text
        self.n_results   = n_results
        self.top_k       = top_k
        self.w_clip      = w_clip
        self.dedupe      = dedupe
        self.dedupe_thr  = dedupe_thr

    def run(self) -> None:
        try:
            arts = load_artifacts_cacheable(Path(self.index_dir))
            emb  = _get_embedder()

            # Build seeds — extract a temp frame PNG for video files so the
            # embedder (which only accepts images) can process them.
            import tempfile, os, cv2 as _cv2
            _tmp_frames: list[str] = []
            api_seeds = []
            for s in self.seeds:
                p = s.get("path", "")
                if not p:
                    continue
                if Path(p).suffix.lower() in VIDEO_EXTS:
                    cap = _cv2.VideoCapture(p)
                    ok, frame = cap.read()
                    cap.release()
                    if ok:
                        tmp = tempfile.NamedTemporaryFile(
                            suffix=".png", delete=False)
                        tmp.close()
                        _cv2.imwrite(tmp.name, frame)
                        _tmp_frames.append(tmp.name)
                        p = tmp.name
                api_seeds.append({"path": p, "w_concept": s.get("weight", 1.0)})

            try:
                report = pick_similar_cached(
                    artifacts        = arts,
                    emb              = emb,
                    seeds            = api_seeds or None,
                    text_prompt      = self.text_prompt,
                    neg_prompt       = self.neg_prompt,
                    w_text           = self.w_text,
                    n_results        = self.n_results,
                    top_k            = self.top_k,
                    w_clip           = self.w_clip,
                    dedupe           = self.dedupe,
                    dedupe_threshold = self.dedupe_thr,
                )
            finally:
                for tf in _tmp_frames:
                    try:
                        os.unlink(tf)
                    except OSError:
                        pass

            from dataclasses import asdict
            items = [asdict(r) for r in report.results]
            self.finished.emit(items)
        except Exception as exc:
            self.error.emit(str(exc))


# ── IndexWorker ───────────────────────────────────────────────────────────────

class IndexWorker(QThread):
    log_line  = pyqtSignal(str)
    finished  = pyqtSignal(str)   # summary line

    def __init__(
        self,
        src_folder:    str,
        out_folder:    str,
        recursive:     bool,
        reset_progress: bool,
        chunk_size:    int,
        batch_commit:  int,
        parent=None,
    ):
        super().__init__(parent)
        self.src_folder     = src_folder
        self.out_folder     = out_folder
        self.recursive      = recursive
        self.reset_progress = reset_progress
        self.chunk_size     = chunk_size
        self.batch_commit   = batch_commit
        self._stop          = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        import subprocess, sys

        script = str(APP_ROOT / "index_images_chunked.py")
        cmd = [
            sys.executable, script,
            self.src_folder,
            self.out_folder,
            "--chunk_size",   str(self.chunk_size),
            "--batch_commit", str(self.batch_commit),
        ]
        if self.recursive:
            cmd.append("--recursive")
        if self.reset_progress:
            cmd.append("--reset_progress")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            summary = ""
            for line in proc.stdout:
                if self._stop:
                    proc.terminate()
                    self.finished.emit("Cancelled.")
                    return
                line = line.rstrip()
                self.log_line.emit(line)
                if line.startswith("KOAN_SUMMARY"):
                    summary = line
            proc.wait()
            self.finished.emit(summary or "Done.")
        except Exception as exc:
            self.finished.emit(f"Error: {exc}")


# ── IndexTab ──────────────────────────────────────────────────────────────────

class IndexTab(QWidget):
    def __init__(self, state: Dict, parent=None):
        super().__init__(parent)
        self._state  = state
        self._worker: Optional[IndexWorker] = None

        root_lay = QVBoxLayout(self)
        root_lay.setContentsMargins(16, 16, 16, 16)
        root_lay.setSpacing(12)

        grp = QGroupBox("INDEX BUILD")
        grp_lay = QVBoxLayout(grp)
        grp_lay.setSpacing(8)

        # source folder
        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Source folder:"))
        self._src = QLineEdit(state.get("idx_src_folder", ""))
        self._src.setPlaceholderText("Folder containing images…")
        row_src.addWidget(self._src, 1)
        btn_src = QPushButton("…")
        btn_src.setFixedWidth(32)
        btn_src.clicked.connect(self._browse_src)
        row_src.addWidget(btn_src)
        grp_lay.addLayout(row_src)

        # output folder
        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Index output:"))
        self._out = QLineEdit(state.get("idx_out_index_folder", ""))
        self._out.setPlaceholderText("Where to write catalog…")
        row_out.addWidget(self._out, 1)
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(32)
        btn_out.clicked.connect(self._browse_out)
        row_out.addWidget(btn_out)
        grp_lay.addLayout(row_out)

        # options row
        opt_row = QHBoxLayout()
        self._chk_recursive = QCheckBox("Index subfolders")
        self._chk_recursive.setChecked(bool(state.get("idx_recursive", True)))
        opt_row.addWidget(self._chk_recursive)

        self._chk_reset = QCheckBox("Reset progress")
        self._chk_reset.setChecked(bool(state.get("idx_reset_progress", False)))
        opt_row.addWidget(self._chk_reset)
        opt_row.addStretch()
        grp_lay.addLayout(opt_row)

        # numbers row
        num_row = QHBoxLayout()
        num_row.addWidget(QLabel("Chunk:"))
        self._chunk_spin = QSpinBox()
        self._chunk_spin.setRange(100, 50000)
        self._chunk_spin.setValue(int(state.get("idx_chunk_size", 2000)))
        self._chunk_spin.setSingleStep(500)
        num_row.addWidget(self._chunk_spin)
        num_row.addSpacing(16)
        num_row.addWidget(QLabel("DB batch:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(10, 5000)
        self._batch_spin.setValue(int(state.get("idx_batch_commit", 200)))
        num_row.addWidget(self._batch_spin)
        num_row.addStretch()
        grp_lay.addLayout(num_row)

        root_lay.addWidget(grp)

        # log output (fills space between form and buttons)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        self._log.setStyleSheet(
            "background:#020403; color:rgba(64,255,107,0.8);"
            "border:1px solid rgba(64,255,107,0.2);"
        )
        root_lay.addWidget(self._log, 1)

        # buttons row
        btn_row = QHBoxLayout()
        self._btn_index = QPushButton("⬡  INDEX")
        self._btn_index.setFixedHeight(40)
        self._btn_index.setStyleSheet(
            "font-size:15px; font-weight:700; background:#0a1f0d;"
            "border:1px solid rgba(64,255,107,0.7);"
        )
        self._btn_index.clicked.connect(self._start_index)
        btn_row.addWidget(self._btn_index, 1)

        self._btn_cancel = QPushButton("✕  CANCEL")
        self._btn_cancel.setFixedHeight(40)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_index)
        btn_row.addWidget(self._btn_cancel)
        root_lay.addLayout(btn_row)

    # ── helpers ──────────────────────────────────────────────────────────
    def _browse_src(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Source Folder", self._src.text())
        if d:
            self._src.setText(d)

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Index Output Folder", self._out.text())
        if d:
            self._out.setText(d)


    def _start_index(self) -> None:
        src = self._src.text().strip()
        out = self._out.text().strip()
        if not src or not Path(src).is_dir():
            QMessageBox.warning(self, "KOAN.img", "Source folder not found.")
            return
        if not out:
            QMessageBox.warning(self, "KOAN.img", "Please set an index output folder.")
            return
        Path(out).mkdir(parents=True, exist_ok=True)

        self._log.clear()
        self._btn_index.setEnabled(False)
        self._btn_cancel.setEnabled(True)

        self._worker = IndexWorker(
            src_folder     = src,
            out_folder     = out,
            recursive      = self._chk_recursive.isChecked(),
            reset_progress = self._chk_reset.isChecked(),
            chunk_size     = self._chunk_spin.value(),
            batch_commit   = self._batch_spin.value(),
        )
        self._worker.log_line.connect(self._append_log)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _cancel_index(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._btn_cancel.setEnabled(False)

    @pyqtSlot(str)
    def _append_log(self, line: str) -> None:
        self._log.append(line)

    @pyqtSlot(str)
    def _on_done(self, summary: str) -> None:
        self._btn_index.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._log.append(f"\n── {summary} ──")

    def snapshot(self) -> Dict:
        return {
            "idx_src_folder":       self._src.text(),
            "idx_out_index_folder": self._out.text(),
            "idx_recursive":        self._chk_recursive.isChecked(),
            "idx_reset_progress":   self._chk_reset.isChecked(),
            "idx_chunk_size":       self._chunk_spin.value(),
            "idx_batch_commit":     self._batch_spin.value(),
        }


# ── PickTab ───────────────────────────────────────────────────────────────────

class PickTab(QWidget):
    push_to_video_signal = pyqtSignal(list)   # list[str] of selected image paths

    def __init__(self, state: Dict, parent=None):
        super().__init__(parent)
        self._state   = state
        self._seeds:  List[Dict] = []   # {"path": str, "weight": float}
        self._worker: Optional[SearchWorker] = None
        self._item_cache: Dict[str, Dict] = {}   # path → last known result dict

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        # ── LEFT PANEL ────────────────────────────────────────────────────
        left_outer = QScrollArea()
        left_outer.setWidgetResizable(True)
        left_outer.setFixedWidth(360)
        left_outer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_w = QWidget()
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(10)

        # ── SETTINGS group
        grp_set = QGroupBox("SETTINGS")
        gs_lay  = QVBoxLayout(grp_set)
        gs_lay.setSpacing(6)

        # index folder
        r_idx = QHBoxLayout()
        r_idx.addWidget(QLabel("Index folder:"))
        self._idx_dir = QLineEdit(state.get("pick_index_dir", ""))
        self._idx_dir.setPlaceholderText("catalog.sqlite folder…")
        r_idx.addWidget(self._idx_dir, 1)
        btn_idx = QPushButton("…"); btn_idx.setFixedWidth(28)
        btn_idx.clicked.connect(self._browse_index)
        r_idx.addWidget(btn_idx)
        gs_lay.addLayout(r_idx)

        # export folder
        r_exp = QHBoxLayout()
        r_exp.addWidget(QLabel("Export folder:"))
        self._exp_dir = QLineEdit(state.get("pick_export_root", ""))
        self._exp_dir.setPlaceholderText("Where to copy picks…")
        r_exp.addWidget(self._exp_dir, 1)
        btn_exp = QPushButton("…"); btn_exp.setFixedWidth(28)
        btn_exp.clicked.connect(self._browse_export)
        r_exp.addWidget(btn_exp)
        gs_lay.addLayout(r_exp)

        # results + pool row
        r_np = QHBoxLayout()
        r_np.addWidget(QLabel("Results:"))
        self._n_results = QSpinBox()
        self._n_results.setRange(1, 500)
        self._n_results.setValue(int(state.get("pick_n_results", 20)))
        r_np.addWidget(self._n_results)
        r_np.addSpacing(12)
        r_np.addWidget(QLabel("Pool:"))
        self._top_k = QSpinBox()
        self._top_k.setRange(1, 999999)
        self._top_k.setValue(int(state.get("pick_top_k", 200)))
        r_np.addWidget(self._top_k)
        r_np.addStretch()
        gs_lay.addLayout(r_np)

        # concept weight
        r_cw = QHBoxLayout()
        r_cw.addWidget(QLabel("Concept:"))
        self._w_clip = QSlider(Qt.Orientation.Horizontal)
        self._w_clip.setRange(0, 100)
        self._w_clip.setValue(int(state.get("pick_w_clip", 100)))
        r_cw.addWidget(self._w_clip, 1)
        self._w_clip_lbl = QLabel(f"{self._w_clip.value()/100:.2f}")
        self._w_clip_lbl.setFixedWidth(36)
        self._w_clip.valueChanged.connect(
            lambda v: self._w_clip_lbl.setText(f"{v/100:.2f}")
        )
        r_cw.addWidget(self._w_clip_lbl)
        gs_lay.addLayout(r_cw)

        # dedupe row
        r_dd = QHBoxLayout()
        self._chk_dedupe = QCheckBox("Dedupe")
        self._chk_dedupe.setChecked(bool(state.get("pick_dedupe", True)))
        r_dd.addWidget(self._chk_dedupe)
        self._dd_slider = QSlider(Qt.Orientation.Horizontal)
        self._dd_slider.setRange(50, 99)
        self._dd_slider.setValue(int(float(state.get("pick_dedupe_threshold", 0.95)) * 100))
        r_dd.addWidget(self._dd_slider, 1)
        self._dd_lbl = QLabel(f"{self._dd_slider.value()/100:.2f}")
        self._dd_lbl.setFixedWidth(36)
        self._dd_slider.valueChanged.connect(
            lambda v: self._dd_lbl.setText(f"{v/100:.2f}")
        )
        r_dd.addWidget(self._dd_lbl)
        gs_lay.addLayout(r_dd)

        left_lay.addWidget(grp_set)

        # ── SEEDS group
        grp_seeds = QGroupBox("SEEDS")
        gseed_lay = QVBoxLayout(grp_seeds)
        gseed_lay.setSpacing(6)

        # positive / negative prompts
        gseed_lay.addWidget(QLabel("POSITIVE PROMPT"))
        self._pos_prompt = QLineEdit(state.get("pick_text_prompt", ""))
        self._pos_prompt.setPlaceholderText("e.g. golden hour portrait")
        gseed_lay.addWidget(self._pos_prompt)

        gseed_lay.addWidget(QLabel("NEGATIVE PROMPT"))
        self._neg_prompt = QLineEdit(state.get("pick_neg_prompt", ""))
        self._neg_prompt.setPlaceholderText("e.g. blurry, dark")
        gseed_lay.addWidget(self._neg_prompt)

        # text influence slider (always visible; only used when text set)
        r_tw = QHBoxLayout()
        r_tw.addWidget(QLabel("Text influence:"))
        self._w_text = QSlider(Qt.Orientation.Horizontal)
        self._w_text.setRange(0, 100)
        self._w_text.setValue(int(float(state.get("pick_w_text", 0.5)) * 100))
        r_tw.addWidget(self._w_text, 1)
        self._w_text_lbl = QLabel(f"{self._w_text.value()/100:.2f}")
        self._w_text_lbl.setFixedWidth(36)
        self._w_text.valueChanged.connect(
            lambda v: self._w_text_lbl.setText(f"{v/100:.2f}")
        )
        r_tw.addWidget(self._w_text_lbl)
        gseed_lay.addLayout(r_tw)

        # seed tiles container
        self._seeds_container = QWidget()
        self._seeds_lay = QVBoxLayout(self._seeds_container)
        self._seeds_lay.setContentsMargins(0, 0, 0, 0)
        self._seeds_lay.setSpacing(6)
        gseed_lay.addWidget(self._seeds_container)

        # add seed button
        btn_add_seed = QPushButton("＋  Add Seed Image")
        btn_add_seed.clicked.connect(self._add_seed_dialog)
        gseed_lay.addWidget(btn_add_seed)

        left_lay.addWidget(grp_seeds)
        left_lay.addStretch()
        left_outer.setWidget(left_w)

        # ── RUN button (below left scroll, above splitter boundary)
        left_wrap = QWidget()
        left_wrap_lay = QVBoxLayout(left_wrap)
        left_wrap_lay.setContentsMargins(0, 0, 0, 0)
        left_wrap_lay.addWidget(left_outer, 1)

        self._btn_run = QPushButton("▶  RUN")
        self._btn_run.setFixedHeight(44)
        self._btn_run.setStyleSheet(
            "font-size:16px; font-weight:800; background:#061408;"
            "border:2px solid rgba(64,255,107,0.8);"
        )
        self._btn_run.clicked.connect(self._run_search)
        left_wrap_lay.addWidget(self._btn_run)

        splitter.addWidget(left_wrap)
        splitter.setStretchFactor(0, 0)

        # ── RIGHT PANEL ───────────────────────────────────────────────────
        right_w = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(4, 4, 4, 4)
        right_lay.setSpacing(6)

        # toolbar
        toolbar = QHBoxLayout()

        btn_all = QPushButton("SELECT ALL")
        btn_all.clicked.connect(self._select_all)
        toolbar.addWidget(btn_all)

        btn_clr = QPushButton("CLEAR")
        btn_clr.clicked.connect(self._clear_sel)
        toolbar.addWidget(btn_clr)

        self._sel_lbl = QLabel("0 selected")
        self._sel_lbl.setStyleSheet("font-size:11px; color:rgba(64,255,107,0.7);")
        toolbar.addWidget(self._sel_lbl)

        toolbar.addStretch()

        toolbar.addWidget(QLabel("Name:"))
        self._export_name = QLineEdit(state.get("pick_export_name", ""))
        self._export_name.setPlaceholderText("subfolder name (optional)")
        self._export_name.setFixedWidth(160)
        toolbar.addWidget(self._export_name)

        self._chk_clean = QCheckBox("Clean")
        self._chk_clean.setChecked(bool(state.get("pick_clean_export", False)))
        self._chk_clean.setToolTip("Use clean numeric filenames on export")
        toolbar.addWidget(self._chk_clean)

        self._btn_export = QPushButton("⬇  EXPORT SELECTED")
        self._btn_export.setStyleSheet("font-weight:700;")
        self._btn_export.clicked.connect(self._export_selected)
        toolbar.addWidget(self._btn_export)

        self._btn_push_video = QPushButton("→ VIDEO")
        self._btn_push_video.setToolTip("Send selected images to the VIDEO tab as key-frames")
        self._btn_push_video.setStyleSheet(
            "QPushButton { color: #050607; background: #40ff6b; font-weight: 700; "
            "border: none; border-radius: 4px; padding: 4px 12px; }"
            "QPushButton:hover { background: #60ff8b; }"
        )
        self._btn_push_video.clicked.connect(self._push_to_video)
        toolbar.addWidget(self._btn_push_video)

        right_lay.addLayout(toolbar)

        # status label
        self._status_lbl = QLabel("No results yet — add seeds and hit RUN.")
        self._status_lbl.setStyleSheet("font-size:11px; color:rgba(64,255,107,0.5);")
        right_lay.addWidget(self._status_lbl)

        # image grid
        self._grid = ImageGrid()
        self._grid.selection_changed.connect(self._on_sel_changed)
        self._grid.seed_requested.connect(self._add_seed_from_result)
        right_lay.addWidget(self._grid, 1)

        # ── thumb-size bar (bottom-right) ─────────────────────────────────
        self._thumb_steps = [140, 180, 220, 280, 340, 420]
        self._thumb_step_idx = 3   # default = 280px

        size_bar = QHBoxLayout()
        size_bar.addStretch()

        self._size_lbl = QLabel(f"{THUMB_W}px")
        self._size_lbl.setFixedWidth(46)
        self._size_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._size_lbl.setStyleSheet("font-size:11px; color:rgba(64,255,107,0.7);")
        size_bar.addWidget(self._size_lbl)

        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(0, len(self._thumb_steps) - 1)
        self._size_slider.setValue(self._thumb_step_idx)
        self._size_slider.setFixedWidth(120)
        self._size_slider.setToolTip("Thumbnail size")
        self._size_slider.valueChanged.connect(self._on_size_slider)
        size_bar.addWidget(self._size_slider)

        right_lay.addLayout(size_bar)

        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 1)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        # keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self._run_search)
        QShortcut(QKeySequence("Ctrl+A"),      self).activated.connect(self._select_all)
        QShortcut(QKeySequence("Ctrl+E"),      self).activated.connect(self._export_selected)

    # ── seed management ──────────────────────────────────────────────────
    def _rebuild_seed_tiles(self) -> None:
        """Repopulate the seeds container from self._seeds."""
        while self._seeds_lay.count():
            item = self._seeds_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, s in enumerate(self._seeds):
            tile = SeedTile(index=i, path=s["path"], weight=s["weight"])
            tile.remove_clicked.connect(self._remove_seed)
            tile.weight_changed.connect(self._update_weight)
            tile.selection_toggled.connect(self._on_seed_selected)
            self._seeds_lay.addWidget(tile)

    def _add_seed_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Seed Image(s)", "",
            "Images (*.jpg *.jpeg *.png *.gif *.webp *.bmp *.tif *.tiff)"
        )
        for p in paths:
            if not any(s["path"] == p for s in self._seeds):
                self._seeds.append({"path": p, "weight": 1.0})
        self._rebuild_seed_tiles()

    def _add_seed_from_result(self, path: str) -> None:
        if not any(s["path"] == path for s in self._seeds):
            self._seeds.append({"path": path, "weight": 1.0})
        self._rebuild_seed_tiles()

    @pyqtSlot(int)
    def _remove_seed(self, index: int) -> None:
        if 0 <= index < len(self._seeds):
            self._seeds.pop(index)
        self._rebuild_seed_tiles()

    @pyqtSlot(int, float)
    def _update_weight(self, index: int, weight: float) -> None:
        if 0 <= index < len(self._seeds):
            self._seeds[index]["weight"] = weight

    @pyqtSlot(str, bool)
    def _on_seed_selected(self, path: str, checked: bool) -> None:
        """Toggle a seed image in/out of the export selection."""
        self._grid.selected_paths[path] = checked
        self._grid.selection_changed.emit(
            sum(v for v in self._grid.selected_paths.values())
        )

    # ── browse ───────────────────────────────────────────────────────────
    def _browse_index(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Index Folder", self._idx_dir.text())
        if d:
            self._idx_dir.setText(d)

    def _browse_export(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Export Folder", self._exp_dir.text())
        if d:
            self._exp_dir.setText(d)

    # ── search ───────────────────────────────────────────────────────────
    def _run_search(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        idx_dir = self._idx_dir.text().strip()
        if not idx_dir or not Path(idx_dir).is_dir():
            QMessageBox.warning(self, "KOAN.img", "Index folder not found.")
            return

        self._btn_run.setText("⏳  Running…")
        self._btn_run.setEnabled(False)
        self._status_lbl.setText("Searching…")

        self._worker = SearchWorker(
            index_dir    = idx_dir,
            seeds        = list(self._seeds),
            text_prompt  = self._pos_prompt.text().strip(),
            neg_prompt   = self._neg_prompt.text().strip(),
            w_text       = self._w_text.value() / 100.0,
            n_results    = self._n_results.value(),
            top_k        = self._top_k.value(),
            w_clip       = self._w_clip.value() / 100.0,
            dedupe       = self._chk_dedupe.isChecked(),
            dedupe_thr   = self._dd_slider.value() / 100.0,
        )
        self._worker.finished.connect(self._on_search_done)
        self._worker.error.connect(self._on_search_error)
        self._worker.start()

    @pyqtSlot(object)
    def _on_search_done(self, items: List[Dict]) -> None:
        self._btn_run.setText("▶  RUN")
        self._btn_run.setEnabled(True)

        # update cache so pinned cards keep their captions
        for item in items:
            self._item_cache[item["path"]] = item

        # pin currently-selected images to the top of the new results
        selected = self._grid.get_selected()
        selected_set = set(selected)
        pinned = [
            {**self._item_cache.get(p, {"path": p, "caption": ""}), "is_pinned": True}
            for p in selected
        ]
        rest = [item for item in items if item["path"] not in selected_set]

        seed_set = {s["path"] for s in self._seeds}
        self._grid.populate(pinned + rest, seed_set)
        n_pinned = len(pinned)
        n_total  = len(pinned) + len(rest)
        status = f"{n_total} results"
        if n_pinned:
            status += f"  ·  {n_pinned} pinned"
        self._status_lbl.setText(status)

    @pyqtSlot(str)
    def _on_search_error(self, msg: str) -> None:
        self._btn_run.setText("▶  RUN")
        self._btn_run.setEnabled(True)
        self._status_lbl.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Search error", msg)

    # ── selection / export ───────────────────────────────────────────────
    @pyqtSlot(int)
    def _on_sel_changed(self, n: int) -> None:
        self._sel_lbl.setText(f"{n} selected")

    def _select_all(self) -> None:
        self._grid.select_all()

    def _clear_sel(self) -> None:
        self._grid.clear_selection()

    def _export_selected(self) -> None:
        import shutil
        selected = self._grid.get_selected()
        if not selected:
            QMessageBox.information(self, "KOAN.img", "No images selected.")
            return

        export_root = self._exp_dir.text().strip()
        if not export_root:
            QMessageBox.warning(self, "KOAN.img", "Export folder is not set.")
            return

        subfolder = self._export_name.text().strip()
        if not subfolder:
            # auto-name from search context
            pos = self._pos_prompt.text().strip()
            if pos:
                subfolder = _slugify(pos)
            elif self._seeds:
                subfolder = _slugify(Path(self._seeds[0]["path"]).stem)
            else:
                from datetime import datetime
                subfolder = datetime.now().strftime("export-%Y%m%d-%H%M%S")
        dest = Path(export_root) / subfolder
        dest.mkdir(parents=True, exist_ok=True)

        clean = self._chk_clean.isChecked()
        errors = []
        for i, src in enumerate(selected):
            try:
                src_p = Path(src)
                ext = src_p.suffix.lower()

                if ext == ".webp":
                    # Convert WebP: animated → .gif, static → .jpg
                    img = Image.open(str(src_p))
                    animated = getattr(img, "is_animated", False)
                    if animated:
                        out_ext = ".gif"
                        stem = f"{i+1:04d}" if clean else src_p.stem
                        dst = dest / f"{stem}{out_ext}"
                        frames = []
                        durations = []
                        for frame_idx in range(img.n_frames):
                            img.seek(frame_idx)
                            frames.append(img.convert("RGBA").copy())
                            durations.append(img.info.get("duration", 100))
                        frames[0].save(
                            str(dst), save_all=True,
                            append_images=frames[1:],
                            duration=durations, loop=0,
                        )
                    else:
                        out_ext = ".jpg"
                        stem = f"{i+1:04d}" if clean else src_p.stem
                        dst = dest / f"{stem}{out_ext}"
                        img.convert("RGB").save(str(dst), "JPEG", quality=95)
                else:
                    if clean:
                        dst_name = f"{i+1:04d}{ext}"
                    else:
                        dst_name = src_p.name
                    shutil.copy2(str(src_p), str(dest / dst_name))
            except Exception as exc:
                errors.append(f"{src}: {exc}")

        msg = f"Exported {len(selected) - len(errors)} file(s) to:\n{dest}"
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "Export complete", msg)

    # ── thumb size controls ──────────────────────────────────────────────
    def _on_size_slider(self, idx: int) -> None:
        w = self._thumb_steps[idx]
        h = int(round(w * THUMB_H / THUMB_W))
        self._size_lbl.setText(f"{w}px")
        self._grid.set_thumb_size(w, h)

    # ── state snapshot ───────────────────────────────────────────────────
    def snapshot(self) -> Dict:
        return {
            "pick_index_dir":         self._idx_dir.text(),
            "pick_export_root":       self._exp_dir.text(),
            "pick_export_name":       self._export_name.text(),
            "pick_n_results":         self._n_results.value(),
            "pick_top_k":             self._top_k.value(),
            "pick_w_clip":            self._w_clip.value(),
            "pick_text_prompt":       self._pos_prompt.text(),
            "pick_neg_prompt":        self._neg_prompt.text(),
            "pick_w_text":            self._w_text.value() / 100.0,
            "pick_dedupe":            self._chk_dedupe.isChecked(),
            "pick_dedupe_threshold":  self._dd_slider.value() / 100.0,
            "pick_clean_export":      self._chk_clean.isChecked(),
        }

    def _push_to_video(self) -> None:
        """Emit selected image paths to the VIDEO tab."""
        selected = self._grid.get_selected()
        if selected:
            self.push_to_video_signal.emit(selected)


# ── MainWindow ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KOAN.img")
        self.resize(1400, 860)

        state = _load_state()

        # ── central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(12, 10, 12, 10)
        main_lay.setSpacing(8)

        # ── header row
        hdr_row = QHBoxLayout()
        title = QLabel("KOAN.img")
        title.setObjectName("hdr")
        hdr_row.addWidget(title)
        hdr_row.addStretch()
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet("font-size:12px; color:rgba(64,255,107,0.55);")
        hdr_row.addWidget(self._count_lbl)
        main_lay.addLayout(hdr_row)

        # ── tabs
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabBar::tab { padding: 6px 20px; font-size:13px; font-weight:600; }"
            "QTabBar::tab:selected { border-bottom: 2px solid #40ff6b; }"
        )

        self._pick_tab  = PickTab(state)
        self._index_tab = IndexTab(state)
        self._video_tab = VideoTab()
        self._tabs.addTab(self._pick_tab,  "PICK")
        self._tabs.addTab(self._index_tab, "INDEX")
        self._tabs.addTab(self._video_tab, "VIDEO")
        self._pick_tab.push_to_video_signal.connect(self._on_push_to_video)
        main_lay.addWidget(self._tabs, 1)

        # ── auto-save timer
        self._timer = QTimer(self)
        self._timer.setInterval(30_000)
        self._timer.timeout.connect(self._save_state)
        self._timer.start()

        self._refresh_count(state)

    def _refresh_count(self, state: Optional[Dict] = None) -> None:
        if state is None:
            state = _load_state()
        idx_dir = state.get("pick_index_dir", "")
        if not idx_dir:
            idx_dir = state.get("idx_out_index_folder", "")
        db = Path(idx_dir) / "catalog.sqlite" if idx_dir else None
        if db and db.exists():
            try:
                import sqlite3
                c = sqlite3.connect(str(db))
                n = c.execute("SELECT COUNT(*) FROM images;").fetchone()[0]
                c.close()
                self._count_lbl.setText(f"{n:,} images indexed")
                return
            except Exception:
                pass
        self._count_lbl.setText("")

    def _on_push_to_video(self, paths: List[str]) -> None:
        """Receive paths from PICK tab, push to VIDEO tab, switch to it."""
        self._video_tab.push_frames(paths)
        self._tabs.setCurrentWidget(self._video_tab)

    def _save_state(self) -> None:
        combined = {}
        combined.update(self._pick_tab.snapshot())
        combined.update(self._index_tab.snapshot())
        _save_state(combined)
        self._video_tab.save_state()

    def closeEvent(self, event) -> None:
        self._save_state()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName("KOAN.img")
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
