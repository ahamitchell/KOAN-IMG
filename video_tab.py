"""video_tab.py — VIDEO tab for KOAN.img

Sequence strip of key-frames → per-transition settings (hint, style, model, duration…)
→ Higgsfield clip generation → ffmpeg stitch → inline preview → export.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from PIL import Image

from PyQt6.QtCore import (
    QMimeData, QObject, QPoint, QSize, Qt, QThread, QTimer, pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QCursor, QDrag, QFont, QImage, QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from video_api import MODELS, HiggsfieldClient, HiggsfieldError, FalClient, FalError, KlingClient, KlingError, load_client


def _find_ffmpeg() -> Optional[str]:
    """Return path to ffmpeg binary, searching PATH then common Windows locations."""
    import shutil, glob
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        str(Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    # WinGet installs under a version-stamped subfolder — glob for it
    winget_base = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    for hit in glob.glob(str(winget_base / "Gyan.FFmpeg_*" / "**" / "ffmpeg.exe"),
                         recursive=True):
        return hit
    return None

# ── palette (matches main app) ────────────────────────────────────────────────
GREEN       = "#40ff6b"
BG          = "#050607"
BG2         = "#070a08"
BG3         = "#0a0d0b"
BORDER      = "rgba(64,255,107,0.35)"
BORDER_SEL  = "rgba(64,255,107,0.95)"
DIM         = "#1a2e1e"

# ── thumbnail sizes ───────────────────────────────────────────────────────────
FRAME_W     = 150
FRAME_H     = 110
STRIP_H     = 168   # height of the scrollable strip widget
TRANS_W     = 76    # width of the between-frame connector

_THUMB_CACHE: Dict[tuple, QPixmap] = {}
_THUMB_ORDER: list = []
_THUMB_MAX = 200

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


# ── thumbnail loader ──────────────────────────────────────────────────────────
def _first_video_frame(path: str, w: int, h: int) -> QPixmap:
    """Extract first frame of a video as a scaled QPixmap."""
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return QPixmap()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.thumbnail((w, h), Image.LANCZOS)
    data = img.tobytes("raw", "RGB")
    qi = QImage(data, img.width, img.height,
                img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi.copy())


def _load_thumb(path: str, w: int = FRAME_W, h: int = FRAME_H) -> QPixmap:
    key = (path, w, h)
    if key in _THUMB_CACHE:
        return _THUMB_CACHE[key]
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        px = _first_video_frame(path, w, h)
    else:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((w, h), Image.LANCZOS)
                data = img.tobytes("raw", "RGB")
                qi = QImage(data, img.width, img.height,
                            img.width * 3, QImage.Format.Format_RGB888)
                px = QPixmap.fromImage(qi.copy())
        except Exception:
            px = QPixmap()
    _THUMB_CACHE[key] = px
    _THUMB_ORDER.append(key)
    while len(_THUMB_ORDER) > _THUMB_MAX:
        old = _THUMB_ORDER.pop(0)
        _THUMB_CACHE.pop(old, None)
    return px


def _slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:maxlen] or "video"


def _default_transition() -> Dict:
    return {
        "hint":         "",
        "style":        "literal",
        "enhance":      True,
        "prompt":       "",
        "model":        list(MODELS.keys())[0],
        "duration":     5,
        "audio":        True,
        "camera_fixed": False,
        "seed":         -1,
        "max_renders":  1,    # max versions to keep for this transition
        "status":       "",   # "" | "enhancing" | "generating" | "done" | "error"
        "clip_path":    "",   # currently previewed render
        "clip_paths":   [],   # all renders for this transition (history)
        "error_msg":    "",
    }


# ── workers ───────────────────────────────────────────────────────────────────
class EnhanceWorker(QThread):
    """Calls Claude Haiku for one transition. Emits (idx, prompt) or (idx, err)."""
    finished = pyqtSignal(int, str)
    error    = pyqtSignal(int, str)

    def __init__(self, idx: int, first_frame: str, last_frame: Optional[str],
                 hint: str, style: str, model_name: str,
                 llm_model: str = "claude-3-haiku-20240307",
                 global_style: str = ""):
        super().__init__()
        self.idx          = idx
        self.first_frame  = first_frame
        self.last_frame   = last_frame
        self.hint         = hint
        self.style        = style
        self.model_name   = model_name
        self.llm_model    = llm_model
        self.global_style = global_style

    def run(self):
        try:
            from video_llm import enhance_prompt
            result = enhance_prompt(
                first_frame  = self.first_frame,
                last_frame   = self.last_frame,
                hint         = self.hint,
                style        = self.style,
                model_name   = self.model_name,
                llm_model    = self.llm_model,
                global_style = self.global_style,
            )
            self.finished.emit(self.idx, result)
        except Exception as exc:
            self.error.emit(self.idx, str(exc))


class ClipWorker(QThread):
    """Generates one clip via Higgsfield or fal.ai. Emits progress, finished, or error."""
    progress = pyqtSignal(int, str)   # (idx, status_text)
    finished = pyqtSignal(int, str)   # (idx, dest_path)
    error    = pyqtSignal(int, str)   # (idx, error_msg)

    def __init__(self, idx: int, client,
                 first_frame: str, last_frame: Optional[str],
                 prompt: str, dest_path: str, **gen_kwargs):
        super().__init__()
        self._cancelled  = False
        self.idx         = idx
        self.client      = client
        self.first_frame = first_frame
        self.last_frame  = last_frame
        self.prompt      = prompt
        self.dest_path   = dest_path
        self.gen_kwargs  = gen_kwargs

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            result = self.client.generate(
                model_id     = self.gen_kwargs.pop("model_id"),
                first_frame  = self.first_frame,
                last_frame   = self.last_frame,
                prompt       = self.prompt,
                dest_path    = self.dest_path,
                progress_cb  = lambda s: self.progress.emit(self.idx, s),
                cancel_check = lambda: self._cancelled,
                **self.gen_kwargs,
            )
            self.finished.emit(self.idx, result)
        except Exception as exc:
            self.error.emit(self.idx, str(exc))


# ── sequence strip ────────────────────────────────────────────────────────────
class _TransitionHandle(QWidget):
    """Clickable panel connector between two frame cards — click to edit."""
    clicked_signal    = pyqtSignal(int)
    render_selected   = pyqtSignal(int, str)   # (trans_idx, path)
    compare_requested = pyqtSignal(int)         # trans_idx — show all renders

    _STATUS_COLOR = {
        "":           "#1e3824",
        "enhancing":  "#3a3000",
        "generating": "#3a2000",
        "done":       "#0d2e12",
        "error":      "#2e0d0d",
    }
    _STATUS_BORDER = {
        "":           "rgba(64,255,107,0.25)",
        "enhancing":  "rgba(220,180,0,0.7)",
        "generating": "rgba(230,120,0,0.7)",
        "done":       "rgba(64,255,107,0.8)",
        "error":      "rgba(220,40,40,0.8)",
    }
    _STATUS_ICON = {
        "":           "⟶",
        "enhancing":  "✦",
        "generating": "⏳",
        "done":       "✓",
        "error":      "✗",
    }
    _STATUS_LABEL = {
        "":           "edit",
        "enhancing":  "enhancing",
        "generating": "generating",
        "done":       "done",
        "error":      "error",
    }

    def __init__(self, idx: int, selected: bool = False,
                 status: str = "", parent=None):
        super().__init__(parent)
        self.idx = idx
        self._selected = selected
        self._status   = status
        self.setFixedSize(TRANS_W, FRAME_H + 28)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setToolTip(f"Clip {idx + 1}  (frame {idx+1} → {idx+2})  —  click to edit settings")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._num_lbl = QLabel(f"T{idx + 1}")
        self._num_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._num_lbl.setStyleSheet("font-size: 10px; font-weight: bold; border: none;")

        self._icon_lbl = QLabel("⟶")
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet("font-size: 20px; border: none;")

        self._sub_lbl = QLabel("▼ click\nto edit")
        self._sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub_lbl.setStyleSheet("font-size: 9px; border: none;")

        self._renders_combo = QComboBox()
        self._renders_combo.setFixedWidth(TRANS_W - 8)
        self._renders_combo.setMaximumHeight(20)
        self._renders_combo.setStyleSheet(
            f"QComboBox {{ background: #0d1a10; color: {GREEN}; "
            f"border: 1px solid rgba(64,255,107,0.55); "
            f"border-radius: 3px; font-size: 11px; font-weight: bold; padding: 1px 3px; }}"
            f"QComboBox::drop-down {{ border: none; width: 12px; }}"
            f"QComboBox QAbstractItemView {{ background: #0d1a10; color: {GREEN}; font-size: 11px; }}"
        )
        self._renders_combo.currentIndexChanged.connect(self._on_combo_changed)
        self._renders_combo.hide()

        layout.addWidget(self._num_lbl)
        layout.addWidget(self._icon_lbl)
        layout.addStretch()
        layout.addWidget(self._renders_combo)
        layout.addWidget(self._sub_lbl)

        self._refresh()

    def _refresh(self):
        bg     = "#2a5a38" if self._selected else self._STATUS_COLOR.get(self._status, "#1e3824")
        border = "rgba(64,255,107,0.95)" if self._selected else self._STATUS_BORDER.get(self._status, "rgba(64,255,107,0.25)")
        icon   = self._STATUS_ICON.get(self._status, "⟶")
        sub    = "▼ selected" if self._selected else ("▼ click\nto edit" if self._status == "" else self._STATUS_LABEL.get(self._status, ""))
        txt_c  = GREEN if self._selected else ("rgba(64,255,107,0.55)" if self._status == "" else GREEN)

        self.setStyleSheet(
            f"QWidget {{ background: {bg}; border: 1px solid {border}; border-radius: 5px; }}"
        )
        self._num_lbl.setStyleSheet(f"color: {txt_c}; font-size: 10px; font-weight: bold; border: none;")
        self._icon_lbl.setText(icon)
        self._icon_lbl.setStyleSheet(f"color: {GREEN if self._selected else txt_c}; font-size: 20px; border: none;")
        self._sub_lbl.setText(sub)
        self._sub_lbl.setStyleSheet(f"color: {txt_c}; font-size: 9px; border: none;")

    def set_selected(self, sel: bool):
        self._selected = sel
        self._refresh()

    def set_status(self, status: str):
        self._status = status
        self._refresh()

    def mousePressEvent(self, event):  # type: ignore[override]
        self.clicked_signal.emit(self.idx)

    def _on_combo_changed(self, combo_idx: int):
        path = self._renders_combo.itemData(combo_idx)
        if path == "__ALL__":
            self.compare_requested.emit(self.idx)
            return
        if path:
            self.render_selected.emit(self.idx, path)

    def set_renders(self, paths: List[str], current: str):
        """Show/update the renders dropdown. Hides sub_lbl when renders exist."""
        self._renders_combo.blockSignals(True)
        self._renders_combo.clear()
        for n, p in enumerate(paths):
            self._renders_combo.addItem(f"{n + 1}", p)
        if len(paths) >= 2:
            self._renders_combo.addItem("ALL", "__ALL__")
        for i in range(self._renders_combo.count()):
            if self._renders_combo.itemData(i) == current:
                self._renders_combo.setCurrentIndex(i)
                break
        self._renders_combo.blockSignals(False)
        has_renders = len(paths) > 0
        self._renders_combo.setVisible(has_renders)
        self._sub_lbl.setVisible(not has_renders)


# ── drag-reorderable frame card ───────────────────────────────────────────────
class _FrameCard(QFrame):
    """Frame card that initiates a QDrag on mouse-move for reordering."""
    def __init__(self, frame_idx: int, parent=None):
        super().__init__(parent)
        self._frame_idx     = frame_idx
        self._drag_start    = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._drag_start is not None
                and event.buttons() & Qt.MouseButton.LeftButton
                and (event.pos() - self._drag_start).manhattanLength() >= 8):
            drag = QDrag(self)
            md = QMimeData()
            md.setText(str(self._frame_idx))
            drag.setMimeData(md)
            px = self.grab()
            small = px.scaled(px.width() * 2 // 3, px.height() * 2 // 3,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            drag.setPixmap(small)
            drag.setHotSpot(QPoint(small.width() // 2, small.height() // 2))
            drag.exec(Qt.DropAction.MoveAction)
            self._drag_start = None
        super().mouseMoveEvent(event)


# ── drop-accepting container ──────────────────────────────────────────────────
class _StripContainer(QWidget):
    """HBox container that accepts drops from _FrameCard and emits reorder signal."""
    reorder_requested = pyqtSignal(int, int)   # (from_idx, to_idx)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._frame_cards: List["_FrameCard"] = []

    def dragEnterEvent(self, event):
        if event.mimeData().hasText() and event.mimeData().text().isdigit():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText() and event.mimeData().text().isdigit():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not (event.mimeData().hasText() and event.mimeData().text().isdigit()):
            return
        from_idx = int(event.mimeData().text())
        drop_x   = event.position().x()
        to_idx   = self._nearest_card_idx(drop_x)
        if to_idx != from_idx:
            self.reorder_requested.emit(from_idx, to_idx)
        event.acceptProposedAction()

    def _nearest_card_idx(self, x: float) -> int:
        best_idx, best_dist = 0, float("inf")
        for card in self._frame_cards:
            cx = card.geometry().center().x()
            d  = abs(cx - x)
            if d < best_dist:
                best_dist = d
                best_idx  = card._frame_idx
        return best_idx


class SequenceStrip(QScrollArea):
    """Horizontal strip: [Frame] ──[T0]── [Frame] ──[T1]── [Frame] … [+]"""
    frames_changed       = pyqtSignal()
    transition_selected  = pyqtSignal(int)
    render_selected      = pyqtSignal(int, str)   # (trans_idx, path)
    compare_requested    = pyqtSignal(int)         # trans_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_paths:   List[str]  = []
        self._selected_t:    int        = -1
        self._t_handles:     List[_TransitionHandle] = []

        self._container = _StripContainer()
        self._container.reorder_requested.connect(self._move_to)
        self._hbox = QHBoxLayout(self._container)
        self._hbox.setContentsMargins(6, 4, 6, 4)
        self._hbox.setSpacing(0)
        self._hbox.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.setWidget(self._container)
        self.setWidgetResizable(True)
        self.setFixedHeight(STRIP_H)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(f"QScrollArea {{ background: {BG2}; border: 1px solid {BORDER}; border-radius: 6px; }}")

        self._rebuild()

    # ── public ────────────────────────────────────────────────────────────────
    def add_frames(self, paths: List[str]):
        self._frame_paths.extend(paths)
        self._rebuild()
        self.frames_changed.emit()

    def get_frame_paths(self) -> List[str]:
        return list(self._frame_paths)

    def get_transitions(self) -> List[tuple]:
        """Return [(first_path, last_path), …] for each gap."""
        p = self._frame_paths
        return [(p[i], p[i + 1]) for i in range(len(p) - 1)]

    def get_selected_transition(self) -> int:
        return self._selected_t

    def set_transition_status(self, idx: int, status: str):
        if 0 <= idx < len(self._t_handles):
            self._t_handles[idx].set_status(status)

    def set_transition_renders(self, idx: int, paths: List[str], current: str):
        """Pass render history to the handle at idx."""
        if 0 <= idx < len(self._t_handles):
            self._t_handles[idx].set_renders(paths, current)

    def remove_frame(self, idx: int):
        self._remove(idx)

    # ── internal ──────────────────────────────────────────────────────────────
    def _rebuild(self):
        # Remove all widgets
        while self._hbox.count():
            item = self._hbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._t_handles.clear()
        self._container._frame_cards.clear()

        for i, path in enumerate(self._frame_paths):
            if i > 0:
                sel = (i - 1 == self._selected_t)
                h = _TransitionHandle(i - 1, selected=sel)
                h.clicked_signal.connect(self._on_transition_click)
                h.render_selected.connect(self.render_selected)
                h.compare_requested.connect(self.compare_requested)
                self._t_handles.append(h)
                self._hbox.addWidget(h)
            card = self._make_frame_card(i, path)
            self._container._frame_cards.append(card)
            self._hbox.addWidget(card)

        # "+" add button
        add_btn = QPushButton("＋\nAdd")
        add_btn.setFixedSize(54, FRAME_H + 28)
        add_btn.setStyleSheet(
            f"QPushButton {{ color: {GREEN}; background: {BG3}; border: 1px dashed {BORDER}; "
            f"border-radius: 5px; font-size: 13px; }}"
            f"QPushButton:hover {{ background: {DIM}; }}"
        )
        add_btn.clicked.connect(self._on_add)
        self._hbox.addWidget(add_btn)
        self._hbox.addStretch()

    def _make_frame_card(self, idx: int, path: str) -> "_FrameCard":
        card = _FrameCard(idx)
        card.setFixedSize(FRAME_W + 12, FRAME_H + 28)
        card.setStyleSheet(
            f"_FrameCard {{ background: {BG3}; border: 1px solid {BORDER}; border-radius: 5px; }}"
        )
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(3, 3, 3, 3)
        vbox.setSpacing(2)

        # ── top row: index label + X ──────────────────────────────────────────
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        idx_lbl = QLabel(f"#{idx + 1}")
        idx_lbl.setStyleSheet(f"color: {GREEN}; font-size: 10px; border: none;")
        x_btn = QPushButton("×")
        x_btn.setFixedSize(20, 20)
        x_btn.setStyleSheet(
            f"QPushButton {{ color: #ff4444; background: rgba(180,30,30,0.3); "
            f"border: 1px solid rgba(220,60,60,0.6); border-radius: 3px; "
            f"font-size: 15px; font-weight: bold; }}"
            f"QPushButton:hover {{ color: #ffffff; background: rgba(220,50,50,0.7); }}"
        )
        x_btn.clicked.connect(lambda checked=False, i=idx: self._remove(i))
        top_row.addWidget(idx_lbl)
        top_row.addStretch()
        top_row.addWidget(x_btn)
        vbox.addLayout(top_row)

        # ── thumbnail ─────────────────────────────────────────────────────────
        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(FRAME_W, FRAME_H)
        thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_lbl.setStyleSheet("border: none;")
        px = _load_thumb(path, FRAME_W, FRAME_H)
        if not px.isNull():
            thumb_lbl.setPixmap(px.scaled(FRAME_W, FRAME_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        else:
            thumb_lbl.setText("⚠")
        vbox.addWidget(thumb_lbl)

        # ── drag hint label ────────────────────────────────────────────────────
        drag_lbl = QLabel("⠿ drag to reorder")
        drag_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drag_lbl.setStyleSheet(f"color: rgba(64,255,107,0.35); font-size: 9px; border: none;")
        vbox.addWidget(drag_lbl)

        return card

    def _on_transition_click(self, idx: int):
        self._selected_t = idx
        for i, h in enumerate(self._t_handles):
            h.set_selected(i == idx)
        self.transition_selected.emit(idx)

    def _on_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Key Frames", "",
            "Images & Video (*.jpg *.jpeg *.png *.webp *.gif *.mp4 *.mov *.avi)"
        )
        if paths:
            self._frame_paths.extend(paths)
            self._rebuild()
            self.frames_changed.emit()

    def _remove(self, idx: int):
        if 0 <= idx < len(self._frame_paths):
            self._frame_paths.pop(idx)
            if self._selected_t >= len(self._frame_paths) - 1:
                self._selected_t = len(self._frame_paths) - 2
            self._rebuild()
            self.frames_changed.emit()

    def _move_to(self, from_idx: int, to_idx: int):
        """Drag-reorder: move frame at from_idx to to_idx position."""
        if from_idx == to_idx:
            return
        n = len(self._frame_paths)
        if not (0 <= from_idx < n and 0 <= to_idx < n):
            return
        item = self._frame_paths.pop(from_idx)
        self._frame_paths.insert(to_idx, item)
        self._selected_t = -1
        self._rebuild()
        self.frames_changed.emit()


# ── per-transition settings panel ────────────────────────────────────────────
class TransitionSettings(QWidget):
    """Settings for the currently selected transition slot."""

    prompt_changed     = pyqtSignal(int, str)   # (idx, new_prompt_text)
    generate_requested = pyqtSignal(int)        # (idx) — render just this clip
    enhance_requested  = pyqtSignal(int)        # (idx) — enhance just this transition

    _BTN_ON  = (f"QPushButton {{ color: {BG}; background: {GREEN}; border: none; "
                f"border-radius: 3px; padding: 3px 10px; font-weight: bold; }}")
    _BTN_OFF = (f"QPushButton {{ color: {GREEN}; background: {DIM}; border: 1px solid {BORDER}; "
                f"border-radius: 3px; padding: 3px 10px; }}"
                f"QPushButton:hover {{ background: #2a5a32; }}")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._idx         = -1
        self._transitions: List[Dict] = []
        self._loading     = False
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # placeholder shown when nothing selected
        self._placeholder = QLabel("← Select a transition above to configure it")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: #446648; font-size: 13px;")
        root.addWidget(self._placeholder)

        # settings widget (hidden until selection)
        self._sw = QWidget()
        sl = QVBoxLayout(self._sw)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # ── title + per-clip action buttons ──────────────────────────────────
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self._title = QLabel("Transition 1 → 2")
        self._title.setStyleSheet(f"color: {GREEN}; font-weight: bold; font-size: 13px;")
        title_row.addWidget(self._title, stretch=1)

        self._btn_enhance = QPushButton("✦ Enhance")
        self._btn_enhance.setFixedHeight(22)
        self._btn_enhance.setStyleSheet(
            f"QPushButton {{ color: {BG}; background: #c8a800; border: none; "
            f"border-radius: 3px; padding: 2px 8px; font-size: 11px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #f0c800; }}"
        )
        self._btn_enhance.setToolTip("Enhance this prompt with AI (Claude Haiku)")
        self._btn_enhance.clicked.connect(
            lambda: self.enhance_requested.emit(self._idx)
        )
        title_row.addWidget(self._btn_enhance)

        self._btn_render_one = QPushButton("▶ Generate")
        self._btn_render_one.setFixedHeight(22)
        self._btn_render_one.setStyleSheet(
            f"QPushButton {{ color: {BG}; background: {GREEN}; border: none; "
            f"border-radius: 3px; padding: 2px 8px; font-size: 11px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #60ff8b; }}"
        )
        self._btn_render_one.setToolTip("Generate clips for this transition (runs × count)")
        self._btn_render_one.clicked.connect(
            lambda: self.generate_requested.emit(self._idx)
        )
        title_row.addWidget(self._btn_render_one)

        _x_lbl = QLabel("×")
        _x_lbl.setStyleSheet(f"color: rgba(64,255,107,0.6); font-size: 11px; border: none;")
        _x_lbl.setFixedWidth(10)
        title_row.addWidget(_x_lbl)

        self._renders_spin = QSpinBox()
        self._renders_spin.setRange(1, 20)
        self._renders_spin.setValue(1)
        self._renders_spin.setFixedSize(38, 22)
        self._renders_spin.setToolTip("How many renders to generate per click")
        self._renders_spin.setStyleSheet(
            f"QSpinBox {{ background: {BG3}; color: {GREEN}; "
            f"border: 1px solid {BORDER}; border-radius: 3px; padding: 1px 2px; font-size: 11px; }}"
            f"QSpinBox::up-button, QSpinBox::down-button {{ width: 12px; }}"
        )
        self._renders_spin.valueChanged.connect(
            lambda v: self._save_current() if not self._loading else None
        )
        title_row.addWidget(self._renders_spin)
        sl.addLayout(title_row)

        # ── hint ──────────────────────────────────────────────────────────────
        sl.addWidget(self._lbl("Concept / hint:"))
        self._hint = QLineEdit()
        self._hint.setPlaceholderText("e.g. slow zoom into sunrise, mist rising…")
        sl.addWidget(self._hint)

        # ── style toggle ──────────────────────────────────────────────────────
        style_row = QHBoxLayout()
        style_row.setSpacing(4)
        self._btn_literal  = QPushButton("Literal")
        self._btn_abstract = QPushButton("Abstract")
        self._btn_literal.setCheckable(False)
        self._btn_abstract.setCheckable(False)
        self._btn_literal.clicked.connect(lambda: self._set_style("literal"))
        self._btn_abstract.clicked.connect(lambda: self._set_style("abstract"))
        style_row.addWidget(self._btn_literal)
        style_row.addWidget(self._btn_abstract)
        sl.addLayout(style_row)

        # ── prompt ────────────────────────────────────────────────────────────
        sl.addWidget(self._lbl("Prompt (editable):"))
        self._prompt = QTextEdit()
        self._prompt.setMaximumHeight(72)
        self._prompt.setPlaceholderText("Prompt will appear here after generation or enhancement…")
        sl.addWidget(self._prompt)

        # ── model ─────────────────────────────────────────────────────────────
        sl.addWidget(self._lbl("Model:"))
        self._model_combo = QComboBox()
        for m in MODELS:
            self._model_combo.addItem(m)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        sl.addWidget(self._model_combo)

        # ── duration ─────────────────────────────────────────────────────────
        dur_row = QHBoxLayout()
        dur_row.addWidget(self._lbl("Duration:"))
        self._dur_slider = QSlider(Qt.Orientation.Horizontal)
        self._dur_slider.setRange(4, 12)
        self._dur_slider.setSingleStep(1)
        self._dur_slider.setValue(5)
        self._dur_lbl = QLabel("5s")
        self._dur_lbl.setFixedWidth(28)
        self._dur_slider.valueChanged.connect(lambda v: self._dur_lbl.setText(f"{v}s"))
        dur_row.addWidget(self._dur_slider)
        dur_row.addWidget(self._dur_lbl)
        sl.addLayout(dur_row)

        # ── audio + camera fixed ───────────────────────────────────────────────
        misc_row = QHBoxLayout()
        self._audio_cb    = QCheckBox("Audio")
        self._cam_fixed_cb = QCheckBox("Camera Fixed")
        self._audio_cb.setChecked(True)
        misc_row.addWidget(self._audio_cb)
        misc_row.addWidget(self._cam_fixed_cb)
        misc_row.addStretch()
        sl.addLayout(misc_row)

        # ── seed ─────────────────────────────────────────────────────────────
        seed_row = QHBoxLayout()
        seed_row.addWidget(self._lbl("Seed:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(-1, 9_999_999)
        self._seed_spin.setValue(-1)
        self._seed_spin.setSpecialValueText("random")
        seed_row.addWidget(self._seed_spin)
        seed_row.addStretch()
        sl.addLayout(seed_row)

        sl.addStretch()
        root.addWidget(self._sw)
        self._sw.hide()
        self._set_style("literal")   # initialise button styles

    @staticmethod
    def _lbl(text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(f"color: {GREEN}; font-size: 11px;")
        return l

    # ── style toggle helper ───────────────────────────────────────────────────
    def _set_style(self, style: str):
        self._btn_literal.setStyleSheet (self._BTN_ON if style == "literal"  else self._BTN_OFF)
        self._btn_abstract.setStyleSheet(self._BTN_ON if style == "abstract" else self._BTN_OFF)
        self._current_style = style
        if not self._loading:
            self._save_current()

    # ── model changed: update AR / res / dur range ────────────────────────────
    def _on_model_changed(self, model_name: str):
        info = MODELS.get(model_name, {})
        # duration range
        dmin = info.get("duration_min", 4)
        dmax = info.get("duration_max", 12)
        self._dur_slider.setRange(dmin, dmax)
        if self._dur_slider.value() < dmin:
            self._dur_slider.setValue(dmin)
        if self._dur_slider.value() > dmax:
            self._dur_slider.setValue(dmax)
        # audio only for supported models
        self._audio_cb.setEnabled(info.get("has_audio", False))
        if not info.get("has_audio", False):
            self._audio_cb.setChecked(False)
        if not self._loading:
            self._save_current()

    # ── public interface ──────────────────────────────────────────────────────
    def bind(self, idx: int, transitions: List[Dict]):
        """Show settings for transition at idx."""
        self._save_current()   # persist previous
        self._idx         = idx
        self._transitions = transitions
        self._placeholder.hide()
        self._sw.show()
        self._load(idx)

    def unbind(self):
        self._save_current()
        self._idx = -1
        self._sw.hide()
        self._placeholder.show()

    def set_prompt(self, idx: int, text: str):
        """Called after AI enhancement to update the prompt field."""
        if 0 <= idx < len(self._transitions):
            self._transitions[idx]["prompt"] = text
        if idx == self._idx:
            self._prompt.blockSignals(True)
            self._prompt.setPlainText(text)
            self._prompt.blockSignals(False)

    def get_all_transitions(self) -> List[Dict]:
        self._save_current()
        return self._transitions

    # ── internal load / save ──────────────────────────────────────────────────
    def _load(self, idx: int):
        t = self._transitions[idx]
        self._loading = True
        self._title.setText(f"Transition {idx + 1}  →  {idx + 2}")
        self._hint.setText(t.get("hint", ""))
        style = t.get("style", "literal")
        self._current_style = style
        self._set_style(style)
        self._renders_spin.setValue(t.get("max_renders", 1))
        self._prompt.setPlainText(t.get("prompt", ""))

        model = t.get("model", list(MODELS.keys())[0])
        self._model_combo.setCurrentText(model)
        self._on_model_changed(model)   # repopulate dur / audio

        self._dur_slider.setValue(t.get("duration", 5))
        self._audio_cb.setChecked(t.get("audio", True))
        self._cam_fixed_cb.setChecked(t.get("camera_fixed", False))
        self._seed_spin.setValue(t.get("seed", -1))
        self._loading = False

    def _save_current(self):
        if self._idx < 0 or self._idx >= len(self._transitions):
            return
        t = self._transitions[self._idx]
        t["hint"]         = self._hint.text()
        t["style"]        = self._current_style
        t["max_renders"]  = self._renders_spin.value()
        t["prompt"]       = self._prompt.toPlainText()
        t["model"]        = self._model_combo.currentText()
        t["duration"]     = self._dur_slider.value()
        t["audio"]        = self._audio_cb.isChecked()
        t["camera_fixed"] = self._cam_fixed_cb.isChecked()
        t["seed"]         = self._seed_spin.value()


# ── simple cv2 video preview ──────────────────────────────────────────────────
class _VideoPreview(QLabel):
    """Plays a local MP4 by cycling frames with cv2 + QTimer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background: #000; border: 1px solid {BORDER}; border-radius: 4px;")
        self.setText("No clip loaded")
        self._frames: List[QPixmap] = []
        self._idx   = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

    def load(self, path: str):
        self._timer.stop()
        self._frames.clear()
        self._idx = 0
        cap = cv2.VideoCapture(path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 24
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total < 1:
            self.setText("⚠ Could not read video")
            return
        # Read ALL frames — thumbnails are small enough to fit in memory
        cap = cv2.VideoCapture(path)
        pw, ph = self.width() or 480, self.height() or 270
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((pw, ph), Image.LANCZOS)
            qi = QImage(img.tobytes("raw", "RGB"), img.width, img.height,
                        img.width * 3, QImage.Format.Format_RGB888)
            self._frames.append(QPixmap.fromImage(qi))
        cap.release()
        if self._frames:
            self.setPixmap(self._frames[0])
            interval = max(30, int(1000 / min(fps, 24)))
            self._timer.start(interval)
        else:
            self.setText("⚠ Could not decode frames")

    def stop(self):
        self._timer.stop()

    def _next_frame(self):
        if self._frames:
            self._idx = (self._idx + 1) % len(self._frames)
            self.setPixmap(self._frames[self._idx])


# ── compare dialog ─────────────────────────────────────────────────────────────
class _CompareDialog(QDialog):
    """Grid of all renders for a single transition, playing simultaneously.

    Clicking a render selects it and closes the dialog.
    """
    render_picked = pyqtSignal(int, str)   # (trans_idx, path)

    def __init__(self, trans_idx: int, paths: List[str], parent=None):
        super().__init__(parent)
        self._trans_idx = trans_idx
        self._paths     = paths
        self._previews: List[_VideoPreview] = []

        self.setWindowTitle(f"Compare renders — Transition {trans_idx + 1}")
        self.setStyleSheet(f"background: #050607; color: {GREEN};")
        self.setMinimumSize(800, 500)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        hint = QLabel("Click a render to select it")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(f"color: rgba(64,255,107,0.6); font-size: 12px;")
        outer.addWidget(hint)

        # Determine grid dimensions — aim for roughly square
        n = len(paths)
        cols = 2 if n <= 4 else 3
        grid = QGridLayout()
        grid.setSpacing(8)

        for i, p in enumerate(paths):
            cell = QFrame()
            cell.setStyleSheet(
                f"QFrame {{ background: #0a0c0a; border: 1px solid {BORDER}; border-radius: 5px; }}"
                f"QFrame:hover {{ border: 1px solid {GREEN}; }}"
            )
            cell.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(4, 4, 4, 4)
            cl.setSpacing(2)

            lbl = QLabel(f"Render {i + 1}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px; font-weight: bold; border: none;")
            cl.addWidget(lbl)

            pv = _VideoPreview()
            pv.setFixedSize(360, 220)
            pv.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            cl.addWidget(pv)

            # Make the whole cell clickable
            cell.mousePressEvent = lambda _ev, path=p: self._pick(path)
            pv.mousePressEvent   = lambda _ev, path=p: self._pick(path)

            self._previews.append(pv)
            grid.addWidget(cell, i // cols, i % cols)

        outer.addLayout(grid)

        # Load all videos (starts playback automatically)
        for pv, p in zip(self._previews, paths):
            pv.load(p)

    def _pick(self, path: str):
        self.render_picked.emit(self._trans_idx, path)
        self.accept()

    def closeEvent(self, event):
        for pv in self._previews:
            pv.stop()
        super().closeEvent(event)


# ── main VIDEO tab ────────────────────────────────────────────────────────────
class VideoTab(QWidget):
    """Full VIDEO tab widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._transitions: List[Dict] = []
        self._enhance_workers: Dict[int, EnhanceWorker] = {}
        self._clip_workers:    Dict[int, ClipWorker]    = {}
        self._pending_renders: Dict[int, int]           = {}   # idx → remaining queued renders
        self._output_dir: Optional[str] = None
        self._clients: dict = {}  # provider → client (HiggsfieldClient or FalClient)
        self._setup_ui()
        self._load_state()   # restore last session

    # ── UI construction ───────────────────────────────────────────────────────
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── toolbar ───────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._btn_add         = self._make_btn("＋ Add Frames", self._on_add_frames)
        self._btn_enhance_all = self._make_btn("✦ Enhance All", self._on_enhance_all, accent=False)
        self._btn_stitch      = self._make_btn("⛓ Stitch",      self._on_stitch,      accent=False)
        self._btn_export      = self._make_btn("⬆ Export",      self._on_export,      accent=False)
        self._btn_loop        = self._make_btn("⟲ Loop",        self._on_loop,        accent=False)
        self._btn_clear       = self._make_btn("✕ Clear All",   self._clear_all,      accent=False)
        self._btn_clear.setStyleSheet(
            "QPushButton { color: #cc3333; background: #1a0a0a; border: 1px solid rgba(204,51,51,0.4); "
            "border-radius: 4px; padding: 5px 14px; }"
            "QPushButton:hover { background: #2a1010; }"
        )

        for b in [self._btn_add, self._btn_enhance_all,
                  self._btn_stitch, self._btn_export, self._btn_loop, self._btn_clear]:
            toolbar.addWidget(b)

        toolbar.addStretch()

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px;")
        toolbar.addWidget(self._status_lbl)

        self._btn_api_keys = self._make_btn("⚙ API Keys", self._toggle_api_panel, accent=False)
        toolbar.addWidget(self._btn_api_keys)

        root.addLayout(toolbar)

        # ── collapsible API keys panel ────────────────────────────────────────
        self._api_panel = self._build_api_panel()
        self._api_panel.hide()
        root.addWidget(self._api_panel)

        # ── sequence strip ────────────────────────────────────────────────────
        self._strip = SequenceStrip()
        self._strip.frames_changed.connect(self._on_frames_changed)
        self._strip.transition_selected.connect(self._on_transition_selected)
        self._strip.render_selected.connect(self._on_render_selected)
        self._strip.compare_requested.connect(self._on_compare_renders)
        root.addWidget(self._strip)

        # ── splitter: settings | preview ─────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # Left: TransitionSettings (top) + Global Style (bottom)
        left_widget = QWidget()
        left_vbox   = QVBoxLayout(left_widget)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(4)
        left_widget.setMinimumWidth(260)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._settings_panel = TransitionSettings()
        self._settings_panel.generate_requested.connect(self._on_generate_single)
        self._settings_panel.enhance_requested.connect(self._on_enhance_single)
        left_scroll.setWidget(self._settings_panel)
        left_vbox.addWidget(left_scroll, stretch=1)

        # ── Generate All button ───────────────────────────────────────────────
        self._btn_generate = self._make_btn("▶ Generate All", self._on_generate, accent=True)
        self._btn_generate.setMinimumHeight(30)
        left_vbox.addWidget(self._btn_generate)

        # ── Global Style panel ────────────────────────────────────────────────
        gs_frame = QFrame()
        gs_frame.setStyleSheet(
            f"QFrame {{ background: {BG2}; border: 1px solid {BORDER}; border-radius: 5px; }}"
        )
        gs_layout = QVBoxLayout(gs_frame)
        gs_layout.setContentsMargins(8, 6, 8, 6)
        gs_layout.setSpacing(4)

        gs_header = QHBoxLayout()
        gs_lbl = QLabel("✦ Global Style")
        gs_lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px; font-weight: bold; border: none;")
        gs_tip = QLabel("Applied to every LLM enhance — set mood, palette, visual rules.")
        gs_tip.setStyleSheet(f"color: {DIM.replace('#','#')}; font-size: 10px; border: none; color: rgba(64,255,107,0.5);")
        gs_header.addWidget(gs_lbl)
        gs_header.addSpacing(8)
        gs_header.addWidget(gs_tip, stretch=1)
        gs_layout.addLayout(gs_header)

        self._global_style_edit = QTextEdit()
        self._global_style_edit.setPlaceholderText(
            "e.g.  black and white, shallow depth of field, slow motion, cinematic grain, "
            "no cartoons, no CGI, no colour grading…"
        )
        self._global_style_edit.setFixedHeight(68)
        self._global_style_edit.setStyleSheet(
            f"QTextEdit {{ background: {BG3}; color: {GREEN}; border: none; "
            f"border-radius: 3px; padding: 4px 6px; font-size: 11px; }}"
        )
        self._global_style_edit.textChanged.connect(self._save_state)
        gs_layout.addWidget(self._global_style_edit)

        # ── Global AR + Resolution row ────────────────────────────────────────
        # Collect union of all model options
        _all_ars  = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
        _all_res  = ["480p", "720p", "1080p"]

        ar_res_row = QHBoxLayout()
        ar_res_row.setSpacing(6)

        _ar_lbl = QLabel("Aspect:")
        _ar_lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px; border: none;")
        ar_res_row.addWidget(_ar_lbl)

        self._global_ar_combo = QComboBox()
        self._global_ar_combo.addItems(_all_ars)
        self._global_ar_combo.setCurrentText("16:9")
        self._global_ar_combo.setFixedWidth(70)
        self._global_ar_combo.currentTextChanged.connect(self._save_state)
        ar_res_row.addWidget(self._global_ar_combo)

        ar_res_row.addSpacing(12)

        _res_lbl = QLabel("Res:")
        _res_lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px; border: none;")
        ar_res_row.addWidget(_res_lbl)

        self._global_res_combo = QComboBox()
        self._global_res_combo.addItems(_all_res)
        self._global_res_combo.setCurrentText("720p")
        self._global_res_combo.setFixedWidth(68)
        self._global_res_combo.currentTextChanged.connect(self._save_state)
        ar_res_row.addWidget(self._global_res_combo)

        ar_res_row.addStretch()
        gs_layout.addLayout(ar_res_row)

        left_vbox.addWidget(gs_frame)
        splitter.addWidget(left_widget)

        # Right: preview + log
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        self._preview = _VideoPreview()
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rl.addWidget(self._preview, stretch=3)

        # log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(130)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet(
            f"QTextEdit {{ background: {BG2}; color: {GREEN}; border: 1px solid {BORDER}; "
            f"border-radius: 4px; }}"
        )
        rl.addWidget(self._log, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([290, 500])
        root.addWidget(splitter, stretch=1)

    # ── API settings panel ────────────────────────────────────────────────────
    def _build_api_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(
            f"QFrame {{ background: {BG3}; border: 1px solid {BORDER}; "
            f"border-radius: 6px; }}"
        )
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        title = QLabel("API Keys  —  saved to koan_config.json")
        title.setStyleSheet(f"color: {GREEN}; font-weight: bold; font-size: 12px; border: none;")
        outer.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setColumnStretch(1, 1)

        _FIELDS = [
            ("higgsfield_api_key",    "Higgsfield Key",    "Key ID from platform.higgsfield.ai → Settings"),
            ("higgsfield_api_secret", "Higgsfield Secret", "Secret from platform.higgsfield.ai → Settings"),
            ("fal_api_key",           "fal.ai Key",        "key_id:key_secret  from fal.ai dashboard"),
            ("kling_access_key",      "Kling Access Key",  "Access Key from app.klingai.com → API"),
            ("kling_secret_key",      "Kling Secret Key",  "Secret Key from app.klingai.com → API"),
            ("anthropic_api_key",     "Anthropic Key",     "sk-ant-…  (Claude Haiku for prompt enhancement)"),
        ]

        self._api_fields: Dict[str, QLineEdit] = {}
        for row, (cfg_key, label_text, tip) in enumerate(_FIELDS):
            lbl = QLabel(label_text + ":")
            lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px; border: none;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl.setFixedWidth(130)
            lbl.setToolTip(tip)

            field = QLineEdit()
            field.setEchoMode(QLineEdit.EchoMode.Password)
            field.setPlaceholderText(tip)
            field.setToolTip(tip)
            field.setStyleSheet(
                f"QLineEdit {{ background: {BG2}; color: {GREEN}; "
                f"border: 1px solid {BORDER}; border-radius: 3px; padding: 3px 6px; }}"
                f"QLineEdit:focus {{ border-color: {GREEN}; }}"
            )
            self._api_fields[cfg_key] = field

            eye_btn = QPushButton("👁")
            eye_btn.setFixedSize(26, 26)
            eye_btn.setCheckable(True)
            eye_btn.setStyleSheet(
                "QPushButton { background: transparent; border: none; font-size: 14px; }"
                "QPushButton:checked { opacity: 0.5; }"
            )
            eye_btn.toggled.connect(
                lambda checked, f=field: f.setEchoMode(
                    QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
                )
            )

            grid.addWidget(lbl,     row, 0)
            grid.addWidget(field,   row, 1)
            grid.addWidget(eye_btn, row, 2)

        outer.addLayout(grid)

        # ── Claude model selector ─────────────────────────────────────────────
        model_row = QHBoxLayout()
        model_lbl = QLabel("Claude model:")
        model_lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px; border: none;")
        model_lbl.setFixedWidth(130)
        model_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._llm_model_combo = QComboBox()
        self._llm_model_combo.setEditable(True)   # allow typing any model ID
        self._llm_model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        _LLM_MODELS = [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
        ]
        for model_id in _LLM_MODELS:
            self._llm_model_combo.addItem(model_id, model_id)
        self._llm_model_combo.setStyleSheet(
            f"QComboBox {{ background: {BG2}; color: {GREEN}; border: 1px solid {BORDER}; "
            f"border-radius: 3px; padding: 3px 6px; }}"
        )
        model_row.addWidget(model_lbl)
        model_row.addWidget(self._llm_model_combo)
        model_row.addStretch()
        outer.addLayout(model_row)

        # ── save + test row ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._api_save_lbl = QLabel("")
        self._api_save_lbl.setStyleSheet(f"color: {GREEN}; font-size: 11px; border: none;")
        btn_row.addWidget(self._api_save_lbl)
        btn_row.addStretch()

        test_btn = QPushButton("🔍  Test Claude key")
        test_btn.setStyleSheet(
            f"QPushButton {{ color: {GREEN}; background: {DIM}; border: 1px solid {BORDER}; "
            f"border-radius: 4px; padding: 5px 14px; }}"
            f"QPushButton:hover {{ background: #2a5a32; }}"
        )
        test_btn.clicked.connect(self._test_anthropic_key)
        btn_row.addWidget(test_btn)

        save_btn = QPushButton("💾  Save")
        save_btn.setStyleSheet(
            f"QPushButton {{ color: {BG}; background: {GREEN}; border: none; "
            f"border-radius: 4px; padding: 5px 18px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #60ff8b; }}"
        )
        save_btn.clicked.connect(self._save_api_keys)
        btn_row.addWidget(save_btn)

        outer.addLayout(btn_row)

        # Populate fields from disk
        self._load_api_keys()
        return panel

    def _toggle_api_panel(self):
        if self._api_panel.isVisible():
            self._api_panel.hide()
        else:
            self._load_api_keys()   # refresh from disk each time we open
            self._api_panel.show()

    def _load_api_keys(self):
        cfg_path = Path(__file__).parent / "koan_config.json"
        if not cfg_path.exists():
            return
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            for key, field in self._api_fields.items():
                field.setText(cfg.get(key, ""))
        except Exception:
            pass

    def _save_api_keys(self):
        cfg_path = Path(__file__).parent / "koan_config.json"
        # Load existing config so we don't wipe unknown keys
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        except Exception:
            cfg = {}
        for key, field in self._api_fields.items():
            cfg[key] = field.text().strip()
        try:
            cfg_path.write_text(
                json.dumps(cfg, indent=4, ensure_ascii=False),
                encoding="utf-8"
            )
            self._api_save_lbl.setText("✓ Saved")
            QTimer.singleShot(3000, lambda: self._api_save_lbl.setText(""))
        except Exception as exc:
            self._api_save_lbl.setText(f"✗ {exc}")

    def _test_anthropic_key(self):
        key = self._api_fields.get("anthropic_api_key", None)
        key_val = key.text().strip() if key else ""

        if not key_val:
            self._api_save_lbl.setText("✗ Anthropic key is empty")
            return
        if not key_val.startswith("sk-ant-"):
            self._api_save_lbl.setText("✗ Key should start with  sk-ant-")
            return

        self._api_save_lbl.setText("Testing…")
        QApplication.processEvents()

        try:
            import anthropic
            pkg_ver = getattr(anthropic, "__version__", "?")
            client  = anthropic.Anthropic(api_key=key_val)

            # First: list available models (works on any valid key)
            try:
                models_page = client.models.list()
                ids = [m.id for m in models_page.data]
                if ids:
                    # populate the combo with real model IDs
                    self._llm_model_combo.blockSignals(True)
                    self._llm_model_combo.clear()
                    for mid in ids:
                        self._llm_model_combo.addItem(mid, mid)
                    self._llm_model_combo.blockSignals(False)
                    self._api_save_lbl.setText(
                        f"✓ Key OK · SDK {pkg_ver} · {len(ids)} models loaded into dropdown"
                    )
                    return
            except AttributeError:
                pass   # older SDK has no models.list() — fall through to test call

            # Fallback: cheapest messages call
            resp = client.messages.create(
                model      = self._llm_model_combo.currentText().strip(),
                max_tokens = 5,
                messages   = [{"role": "user", "content": "Hi"}],
            )
            self._api_save_lbl.setText(f"✓ Key works · SDK {pkg_ver} · model {resp.model}")

        except Exception as exc:
            import anthropic as _a
            pkg_ver = getattr(_a, "__version__", "?")
            msg = str(exc)
            if "401" in msg or "invalid" in msg.lower() or "auth" in msg.lower():
                hint = f"✗ SDK {pkg_ver} · Invalid key — regenerate at console.anthropic.com"
            elif "credit" in msg.lower() or "billing" in msg.lower():
                hint = f"✗ SDK {pkg_ver} · No credits — add billing at console.anthropic.com"
            elif "404" in msg:
                hint = f"✗ SDK {pkg_ver} · 404 model not found — try upgrading: pip install -U anthropic"
            else:
                hint = f"✗ SDK {pkg_ver} · {msg[:100]}"
            self._api_save_lbl.setText(hint)

    @staticmethod
    def _make_btn(label: str, slot, accent: bool = False) -> QPushButton:
        b = QPushButton(label)
        if accent:
            b.setStyleSheet(
                f"QPushButton {{ color: {BG}; background: {GREEN}; border: none; "
                f"border-radius: 4px; padding: 5px 14px; font-weight: bold; }}"
                f"QPushButton:hover {{ background: #60ff8b; }}"
                f"QPushButton:disabled {{ background: #1a4a22; color: #446644; }}"
            )
        else:
            b.setStyleSheet(
                f"QPushButton {{ color: {GREEN}; background: {DIM}; border: 1px solid {BORDER}; "
                f"border-radius: 4px; padding: 5px 14px; }}"
                f"QPushButton:hover {{ background: #2a5a32; }}"
                f"QPushButton:disabled {{ color: #446644; }}"
            )
        b.clicked.connect(slot)
        return b

    # ── signal handlers ───────────────────────────────────────────────────────
    def _on_frames_changed(self):
        """Sync transition list when frames added/removed/reordered."""
        n_needed = max(0, len(self._strip.get_frame_paths()) - 1)
        while len(self._transitions) < n_needed:
            self._transitions.append(_default_transition())
        while len(self._transitions) > n_needed:
            self._transitions.pop()
        self._settings_panel.unbind()
        self._save_state()

    def _on_transition_selected(self, idx: int):
        self._settings_panel.bind(idx, self._transitions)
        if 0 <= idx < len(self._transitions):
            self._update_renders_selector(idx)
            clip = self._transitions[idx].get("clip_path", "")
            if clip and Path(clip).exists():
                self._preview.load(clip)
            else:
                self._preview.stop()
                self._preview.setText("No clip yet — click ▶ Generate")

    def _update_renders_selector(self, idx: int):
        """Pass render history to the handle widget for transition idx."""
        paths = [p for p in self._transitions[idx].get("clip_paths", [])
                 if Path(p).exists()]
        cur = self._transitions[idx].get("clip_path", "")
        self._strip.set_transition_renders(idx, paths, cur)

    def _on_render_selected(self, trans_idx: int, path: str):
        """Load the render chosen in a transition handle into the preview."""
        if path and Path(path).exists():
            if 0 <= trans_idx < len(self._transitions):
                self._transitions[trans_idx]["clip_path"] = path
            self._preview.load(path)

    def _on_compare_renders(self, trans_idx: int):
        """Open the comparison grid for all renders of a transition."""
        if trans_idx < 0 or trans_idx >= len(self._transitions):
            return
        paths = [p for p in self._transitions[trans_idx].get("clip_paths", [])
                 if Path(p).exists()]
        if len(paths) < 2:
            return
        dlg = _CompareDialog(trans_idx, paths, parent=self)
        dlg.render_picked.connect(self._on_render_selected)
        dlg.exec()
        # Reset the combo away from "ALL" back to the selected render
        self._update_renders_selector(trans_idx)

    def _on_add_frames(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Key Frames", "",
            "Images & Video (*.jpg *.jpeg *.png *.webp *.gif *.mp4 *.mov *.avi)"
        )
        if paths:
            self._strip.add_frames(paths)

    def _on_loop(self):
        """Append the first frame at the end so the sequence loops back to start."""
        paths = self._strip.get_frame_paths()
        if len(paths) < 2:
            self._set_status("Need at least 2 frames to loop.")
            return
        if paths[-1] == paths[0]:
            self._set_status("Already looped — last frame is already the first frame.")
            return
        self._strip.add_frames([paths[0]])
        self._log_line(f"⟲ Loop: appended first frame ({Path(paths[0]).name}) at end.")
        self._set_status("Loop frame added.")

    # ── push selected images from PICK tab ───────────────────────────────────
    def push_frames(self, paths: List[str]):
        """Called by PickTab 'Push to Video' button."""
        if paths:
            self._strip.add_frames(paths)
            self._log_line(f"Added {len(paths)} frame(s) from PICK tab.")

    # ── enhance ───────────────────────────────────────────────────────────────
    def _on_enhance_all(self):
        transitions = self._settings_panel.get_all_transitions()
        pairs = self._strip.get_transitions()
        if not pairs:
            self._set_status("No transitions to enhance.")
            return
        n_started = 0
        for i, (first, last) in enumerate(pairs):
            t = transitions[i]
            if i in self._enhance_workers and self._enhance_workers[i].isRunning():
                continue
            t["status"] = "enhancing"
            self._strip.set_transition_status(i, "enhancing")
            model        = t.get("model", list(MODELS.keys())[0])
            llm_model    = self._llm_model_combo.currentText().strip()
            global_style = self._global_style_edit.toPlainText().strip()
            w = EnhanceWorker(i, first, last, t.get("hint", ""),
                              t.get("style", "literal"), model, llm_model,
                              global_style=global_style)
            w.finished.connect(self._on_enhance_done)
            w.error.connect(self._on_enhance_error)
            self._enhance_workers[i] = w
            w.start()
            n_started += 1
        if n_started:
            self._set_status(f"Enhancing {n_started} transition(s)…")
        else:
            self._set_status("All transitions already enhancing — please wait.")

    def _on_enhance_done(self, idx: int, prompt: str):
        self._transitions[idx]["prompt"] = prompt
        self._transitions[idx]["status"] = ""
        self._settings_panel.set_prompt(idx, prompt)
        self._strip.set_transition_status(idx, "")
        self._log_line(f"[T{idx+1}] Enhanced: {prompt[:80]}…")
        self._set_status("Enhancement done.")
        self._save_state()

    def _on_enhance_error(self, idx: int, msg: str):
        self._transitions[idx]["status"] = "error"
        self._transitions[idx]["error_msg"] = msg
        self._strip.set_transition_status(idx, "error")
        self._log_line(f"[T{idx+1}] Enhance error: {msg}")

    def _on_enhance_single(self, idx: int):
        """Enhance the prompt for transition idx only (✦ Enhance button)."""
        transitions = self._settings_panel.get_all_transitions()
        pairs = self._strip.get_transitions()
        if idx < 0 or idx >= len(pairs):
            self._set_status(f"T{idx+1} doesn't exist.")
            return
        if idx in self._enhance_workers and self._enhance_workers[idx].isRunning():
            self._set_status(f"T{idx+1} is already enhancing.")
            return
        t = transitions[idx]
        first, last = pairs[idx]
        t["status"] = "enhancing"
        self._strip.set_transition_status(idx, "enhancing")
        model        = t.get("model", list(MODELS.keys())[0])
        llm_model    = self._llm_model_combo.currentText().strip()
        global_style = self._global_style_edit.toPlainText().strip()
        w = EnhanceWorker(idx, first, last, t.get("hint", ""),
                          t.get("style", "literal"), model, llm_model,
                          global_style=global_style)
        w.finished.connect(self._on_enhance_done)
        w.error.connect(self._on_enhance_error)
        self._enhance_workers[idx] = w
        w.start()
        self._set_status(f"Enhancing T{idx+1}…")

    # ── generate ──────────────────────────────────────────────────────────────
    def _on_generate(self):
        transitions = self._settings_panel.get_all_transitions()
        pairs = self._strip.get_transitions()
        if not pairs:
            self._set_status("Add at least 2 frames first.")
            return

        # Build output dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.home() / "koan_video" / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = str(out_dir)
        self._log_line(f"Output: {out_dir}")

        # Pre-load clients for all providers needed
        providers_needed = set()
        for i, _ in enumerate(pairs):
            model_name = transitions[i].get("model", list(MODELS.keys())[0])
            providers_needed.add(MODELS.get(model_name, {}).get("provider", "higgsfield"))

        self._clients = {}
        for prov in providers_needed:
            try:
                self._clients[prov] = load_client(prov)
            except Exception as exc:
                self._log_line(f"API error ({prov}): {exc}")
                self._set_status(f"API key error for {prov} — check ⚙ API Keys.")
                return

        for i, (first, last) in enumerate(pairs):
            count = transitions[i].get("max_renders", 1)
            self._queue_renders(i, transitions[i], first, last, out_dir, count)

        self._set_status(f"Generating {len(pairs)} transition(s)…")

    def _queue_renders(self, i: int, t: Dict, first: str, last: str,
                       out_dir: Path, count: int):
        """Queue `count` sequential renders for transition i. Fires the first one now."""
        if count < 1:
            return
        self._pending_renders[i] = count   # includes the one we're about to start
        self._launch_clip(i, t, first, last, out_dir)

    def _launch_clip(self, i: int, t: Dict, first: str, last: str, out_dir: Path):
        """Start a ClipWorker for transition i. No-op if already running."""
        if i in self._clip_workers and self._clip_workers[i].isRunning():
            self._log_line(f"[T{i+1}] Already running, skip.")
            return

        prompt     = t.get("prompt", "").strip() or t.get("hint", "").strip() or "cinematic motion"
        n_prev     = len(t.get("clip_paths", []))
        dest       = str(out_dir / (f"clip_{i+1:03d}.mp4" if n_prev == 0
                                    else f"clip_{i+1:03d}_v{n_prev + 1:02d}.mp4"))
        model_name = t.get("model", list(MODELS.keys())[0])
        model_id   = MODELS[model_name]["id"]
        model_info = MODELS[model_name]

        g_ar      = self._global_ar_combo.currentText()
        g_res     = self._global_res_combo.currentText()
        valid_ars = model_info.get("aspect_ratios", ["16:9"])
        valid_res = model_info.get("resolutions",   ["720p"])
        ar  = g_ar  if g_ar  in valid_ars else valid_ars[0]
        res = g_res if g_res in valid_res else valid_res[0]

        gen_kwargs: Dict = {
            "model_id":     model_id,
            "duration":     t.get("duration", 5),
            "aspect_ratio": ar,
            "resolution":   res,
            "camera_fixed": t.get("camera_fixed", False),
            "seed":         t.get("seed", -1),
        }
        if model_info.get("has_audio"):
            gen_kwargs["generate_audio"] = t.get("audio", True)
        if model_info.get("kling_mode"):
            gen_kwargs["kling_mode"] = model_info["kling_mode"]
        last_for_api = last if model_info.get("has_end_frame") else None

        t["status"] = "generating"
        self._strip.set_transition_status(i, "generating")

        provider = model_info.get("provider", "higgsfield")
        client = self._clients.get(provider)
        if not client:
            try:
                client = load_client(provider)
                self._clients[provider] = client
            except Exception as exc:
                self._log_line(f"[T{i+1}] ✗ {provider} key error: {exc}")
                t["status"] = "error"
                t["error_msg"] = str(exc)
                self._strip.set_transition_status(i, "error")
                return

        w = ClipWorker(i, client, first, last_for_api, prompt, dest, **gen_kwargs)
        w.progress.connect(self._on_clip_progress)
        w.finished.connect(self._on_clip_done)
        w.error.connect(self._on_clip_error)
        self._clip_workers[i] = w
        w.start()
        self._log_line(f"[T{i+1}] Generating with {model_name}…")

    def _on_generate_single(self, idx: int):
        """Render only the clip for transition idx."""
        transitions = self._settings_panel.get_all_transitions()
        pairs = self._strip.get_transitions()
        if idx >= len(pairs):
            self._set_status(f"T{idx+1} doesn't exist.")
            return

        if not self._output_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path.home() / "koan_video" / ts
            out_dir.mkdir(parents=True, exist_ok=True)
            self._output_dir = str(out_dir)
            self._log_line(f"Output: {out_dir}")
        else:
            out_dir = Path(self._output_dir)

        model_name = transitions[idx].get("model", list(MODELS.keys())[0])
        provider = MODELS.get(model_name, {}).get("provider", "higgsfield")
        try:
            self._clients[provider] = load_client(provider)
        except Exception as exc:
            self._log_line(f"API error ({provider}): {exc}")
            self._set_status(f"API key error for {provider} — check ⚙ API Keys.")
            return

        first, last = pairs[idx]
        count = transitions[idx].get("max_renders", 1)
        self._queue_renders(idx, transitions[idx], first, last, out_dir, count)
        label = f"{count} render(s)" if count > 1 else "1 render"
        self._set_status(f"T{idx+1}: generating {label}…")



    def _on_clip_progress(self, idx: int, status: str):
        self._log_line(f"[T{idx+1}] {status}")

    def _on_clip_done(self, idx: int, dest_path: str):
        t = self._transitions[idx]
        t["status"]    = "done"
        t["clip_path"] = dest_path
        t.setdefault("clip_paths", []).append(dest_path)
        self._strip.set_transition_status(idx, "done")
        self._log_line(f"[T{idx+1}] ✓ Saved → {dest_path}")
        self._preview.load(dest_path)
        # Always update the handle's renders dropdown
        self._update_renders_selector(idx)
        # Fire next queued render if any remain
        remaining = self._pending_renders.get(idx, 1)
        if remaining > 1:
            self._pending_renders[idx] = remaining - 1
            pairs = self._strip.get_transitions()
            if idx < len(pairs) and self._output_dir:
                first, last = pairs[idx]
                self._launch_clip(idx, t, first, last, Path(self._output_dir))
                n_done  = t.get("max_renders", 1) - remaining + 1
                n_total = t.get("max_renders", 1)
                self._set_status(f"T{idx+1}: render {n_done}/{n_total}…")
                return
        self._pending_renders.pop(idx, None)
        self._set_status("Clip done.")
        self._save_state()

    def _on_clip_error(self, idx: int, msg: str):
        self._transitions[idx]["status"]    = "error"
        self._transitions[idx]["error_msg"] = msg
        self._strip.set_transition_status(idx, "error")
        self._log_line(f"[T{idx+1}] ✗ Error: {msg}")

    # ── stitch ────────────────────────────────────────────────────────────────
    def _on_stitch(self):
        clips = [t.get("clip_path", "") for t in self._transitions
                 if t.get("clip_path") and Path(t["clip_path"]).exists()]
        if len(clips) < 2:
            self._set_status("Need at least 2 finished clips to stitch.")
            return
        if not self._output_dir:
            self._set_status("No output directory — generate clips first.")
            return

        final = Path(self._output_dir) / "final.mp4"
        list_file = Path(self._output_dir) / "concat_list.txt"
        list_file.write_text(
            "\n".join(f"file '{c}'" for c in clips), encoding="utf-8"
        )
        ff = _find_ffmpeg()
        if not ff:
            self._log_line(
                "ffmpeg not found. Run launch.bat to auto-install, or download "
                "from https://ffmpeg.org and add to PATH."
            )
            self._set_status("ffmpeg not found.")
            return

        cmd = [ff, "-y", "-f", "concat", "-safe", "0",
               "-i", str(list_file), "-c", "copy", str(final)]
        self._log_line(f"Stitching {len(clips)} clips → {final.name} …")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                self._log_line(f"ffmpeg error: {result.stderr[-300:]}")
                self._set_status("ffmpeg failed.")
            else:
                self._log_line(f"✓ Stitched → {final}")
                self._set_status("Stitched!")
                self._preview.load(str(final))
        except subprocess.TimeoutExpired:
            self._log_line("ffmpeg timed out.")

    # ── export ────────────────────────────────────────────────────────────────
    def _on_export(self):
        if not self._output_dir:
            self._set_status("Nothing to export yet.")
            return
        # open the output folder in the OS file manager
        import platform
        folder = self._output_dir
        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
        self._set_status(f"Opened: {folder}")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _log_line(self, text: str):
        self._log.append(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _set_status(self, text: str):
        self._status_lbl.setText(text)

    # ── state persistence ─────────────────────────────────────────────────────
    _STATE_FILE = Path(__file__).parent / "video_state.json"

    def _save_state(self):
        try:
            # Strip session-only fields before saving
            clean_transitions = []
            for t in self._transitions:
                ct = {k: v for k, v in t.items()
                      if k not in ("status", "error_msg")}
                clean_transitions.append(ct)
            state = {
                "frame_paths":  self._strip.get_frame_paths(),
                "transitions":  clean_transitions,
                "llm_model":    self._llm_model_combo.currentText().strip(),
                "global_style": self._global_style_edit.toPlainText(),
                "global_ar":    self._global_ar_combo.currentText(),
                "global_res":   self._global_res_combo.currentText(),
                "output_dir":   self._output_dir or "",
            }
            self._STATE_FILE.write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass   # never crash on save

    def _load_state(self):
        try:
            if not self._STATE_FILE.exists():
                return
            state = json.loads(self._STATE_FILE.read_text(encoding="utf-8"))

            # Restore global style
            gs = state.get("global_style", "")
            if gs:
                self._global_style_edit.blockSignals(True)
                self._global_style_edit.setPlainText(gs)
                self._global_style_edit.blockSignals(False)

            # Restore global AR + resolution
            g_ar = state.get("global_ar", "")
            if g_ar and self._global_ar_combo.findText(g_ar) >= 0:
                self._global_ar_combo.blockSignals(True)
                self._global_ar_combo.setCurrentText(g_ar)
                self._global_ar_combo.blockSignals(False)
            g_res = state.get("global_res", "")
            if g_res and self._global_res_combo.findText(g_res) >= 0:
                self._global_res_combo.blockSignals(True)
                self._global_res_combo.setCurrentText(g_res)
                self._global_res_combo.blockSignals(False)

            # Restore LLM model
            llm = state.get("llm_model", "")
            if llm:
                idx = self._llm_model_combo.findText(llm)
                if idx >= 0:
                    self._llm_model_combo.setCurrentIndex(idx)
                else:
                    self._llm_model_combo.setCurrentText(llm)

            # Restore output dir
            self._output_dir = state.get("output_dir") or None

            # Restore frames (only paths that still exist on disk)
            paths = [p for p in state.get("frame_paths", []) if Path(p).exists()]
            if paths:
                self._strip.add_frames(paths)

            # Restore transitions (sync count to actual frame count)
            saved_t = state.get("transitions", [])
            needed  = max(0, len(paths) - 1)
            for i in range(needed):
                if i < len(saved_t):
                    # Merge saved over defaults
                    self._transitions[i].update(saved_t[i])
                    # Filter clip_paths to only files still on disk
                    valid_clips = [p for p in self._transitions[i].get("clip_paths", [])
                                   if Path(p).exists()]
                    self._transitions[i]["clip_paths"] = valid_clips
                    # Ensure active clip_path is still valid; fall back to latest
                    cur = self._transitions[i].get("clip_path", "")
                    if not cur or not Path(cur).exists():
                        self._transitions[i]["clip_path"] = valid_clips[-1] if valid_clips else ""

            # Populate render selectors in handles for transitions with saved history
            for i in range(len(self._transitions)):
                if self._transitions[i].get("clip_paths"):
                    self._update_renders_selector(i)
        except Exception:
            pass   # corrupt file → start fresh

    def _clear_all(self):
        # Cancel any running workers
        for w in list(self._clip_workers.values()):
            w.cancel()
        for w in list(self._enhance_workers.values()):
            w.quit()
        self._clip_workers.clear()
        self._enhance_workers.clear()

        # Reset strip and transitions
        self._strip._frame_paths.clear()
        self._strip._selected_t = -1
        self._strip._rebuild()
        self._transitions.clear()
        self._settings_panel.unbind()
        self._preview.stop()
        self._preview.setText("No clip loaded")
        self._log.clear()
        self._output_dir = None
        self._set_status("Cleared.")
        self._save_state()

    def save_state(self):
        """Public — called by MainWindow on close."""
        self._save_state()
