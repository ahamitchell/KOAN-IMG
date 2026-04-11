"""narrative_tab.py — NARRATIVE tab for KOAN.img

Iterative visual curation: each generation's selections seed the next.
Top: selections grid (the narrative). Bottom: results grid (current search output).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from ui_app import (
    SearchWorker, ImageGrid, SeedTile,
    _load_thumb, _attach_thumb,
    THUMB_W, THUMB_H, VIDEO_EXTS,
    _slugify, _browse_dir,
)

# ── constants ────────────────────────────────────────────────────────────────
GREEN  = "#40ff6b"
BG     = "#050607"
BG2    = "#070a08"
BORDER = "rgba(64,255,107,0.35)"
DIM    = "#1a2e1e"

SEL_THUMB_W = THUMB_W    # same as PICK cards
SEL_THUMB_H = THUMB_H

_STATE_FILE = Path(__file__).parent / "narrative_state.json"


def _gen_label(gen_idx: int, sel_idx: int) -> str:
    return f"{gen_idx + 1}{chr(65 + sel_idx)}"


# ── SelectionCard — card in the selections grid ─────────────────────────────

class SelectionCard(QFrame):
    """A confirmed selection in the narrative grid. Same size as PICK's ImageCard."""
    weight_changed = pyqtSignal(int, int, float)  # gen_idx, sel_idx, weight

    def __init__(self, gen_idx: int, sel_idx: int, path: str,
                 weight: float = 1.0, show_weight: bool = True,
                 parent=None):
        super().__init__(parent)
        self.gen_idx = gen_idx
        self.sel_idx = sel_idx
        self.path = path

        self.setObjectName("card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(SEL_THUMB_W + 18)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(4)

        # label (same position as rank label in ImageCard)
        lbl = QLabel(_gen_label(gen_idx, sel_idx), objectName="rank")
        lbl.setWordWrap(True)
        vl.addWidget(lbl)

        # thumbnail (same size as PICK)
        thumb = QLabel()
        thumb.setFixedSize(SEL_THUMB_W, SEL_THUMB_H)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setStyleSheet("background:#070a08; border-radius:4px;")
        self._movie = _attach_thumb(thumb, path, SEL_THUMB_W, SEL_THUMB_H)
        vl.addWidget(thumb)

        # weight slider (only on latest generation)
        self._slider = None
        if show_weight:
            wrow = QHBoxLayout()
            wrow.addWidget(QLabel("W:"))
            self._slider = QSlider(Qt.Orientation.Horizontal)
            self._slider.setRange(1, 30)
            self._slider.setValue(max(1, min(30, int(round(weight * 10)))))
            self._slider.setFixedWidth(80)
            wrow.addWidget(self._slider)
            self._wval = QLabel(f"{weight:.1f}")
            self._wval.setStyleSheet("font-size: 10px; min-width: 22px;")
            wrow.addWidget(self._wval)
            vl.addLayout(wrow)
            self._slider.valueChanged.connect(self._on_slider)

    def _on_slider(self, v: int) -> None:
        w = v / 10.0
        self._wval.setText(f"{w:.1f}")
        self.weight_changed.emit(self.gen_idx, self.sel_idx, w)

    def get_weight(self) -> float:
        if self._slider:
            return self._slider.value() / 10.0
        return 1.0

    def remove_weight_slider(self) -> None:
        """Remove the weight slider (called when this gen is no longer the latest)."""
        if self._slider is not None:
            # find and remove the weight row layout
            vl = self.layout()
            for i in range(vl.count()):
                item = vl.itemAt(i)
                if item and item.layout():
                    # check if this layout contains the slider
                    lay = item.layout()
                    for j in range(lay.count()):
                        w = lay.itemAt(j).widget()
                        if w is self._slider:
                            # remove entire layout
                            while lay.count():
                                child = lay.takeAt(0)
                                if child.widget():
                                    child.widget().deleteLater()
                            vl.removeItem(item)
                            self._slider = None
                            return


# ── ReRunButton — sits at the end of each generation group ───────────────────

class ReRunButton(QPushButton):
    rerun_requested = pyqtSignal(int)

    def __init__(self, gen_idx: int, parent=None):
        super().__init__("↻", parent)
        self.gen_idx = gen_idx
        self.setFixedSize(SEL_THUMB_W + 18, 30)
        self.setToolTip(f"Re-run from Generation {gen_idx + 1}")
        self.setStyleSheet(
            f"QPushButton {{ color: {GREEN}; background: {DIM}; "
            f"border: 1px solid {BORDER}; border-radius: 3px; "
            f"font-size: 10px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #2a5a32; }}"
        )
        self.clicked.connect(lambda: self.rerun_requested.emit(self.gen_idx))


# ── NarrativeTab ─────────────────────────────────────────────────────────────

class NarrativeTab(QWidget):
    push_to_video_signal = pyqtSignal(list)

    def __init__(self, state: Dict, parent=None):
        super().__init__(parent)
        self._state = state
        self._generations: List[Dict] = []
        self._seeds: List[Dict] = []
        self._worker: Optional[SearchWorker] = None
        self._sel_cards: List[SelectionCard] = []
        self._rerun_buttons: List[ReRunButton] = []
        self._active_gen_idx: int = -1

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        # ── LEFT PANEL ───────────────────────────────────────────────────────
        left_outer = QScrollArea()
        left_outer.setWidgetResizable(True)
        left_outer.setFixedWidth(360)
        left_outer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_w = QWidget()
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(10)

        # SETTINGS
        grp_set = QGroupBox("SETTINGS")
        gs = QVBoxLayout(grp_set)
        gs.setSpacing(6)

        r = QHBoxLayout()
        r.addWidget(QLabel("Index folder:"))
        self._idx_dir = QLineEdit(state.get("narr_index_dir",
                                             state.get("pick_index_dir", "")))
        self._idx_dir.setPlaceholderText("catalog.sqlite folder…")
        r.addWidget(self._idx_dir, 1)
        b = QPushButton("…"); b.setFixedWidth(28)
        b.clicked.connect(lambda: _browse_dir(self, self._idx_dir))
        r.addWidget(b)
        gs.addLayout(r)

        r = QHBoxLayout()
        r.addWidget(QLabel("Export folder:"))
        self._exp_dir = QLineEdit(state.get("narr_export_root",
                                             state.get("pick_export_root", "")))
        self._exp_dir.setPlaceholderText("Where to export…")
        r.addWidget(self._exp_dir, 1)
        b = QPushButton("…"); b.setFixedWidth(28)
        b.clicked.connect(lambda: _browse_dir(self, self._exp_dir))
        r.addWidget(b)
        gs.addLayout(r)

        r = QHBoxLayout()
        r.addWidget(QLabel("Results:"))
        self._n_results = QSpinBox()
        self._n_results.setRange(1, 500)
        self._n_results.setValue(int(state.get("narr_n_results", 20)))
        r.addWidget(self._n_results)
        r.addSpacing(12)
        r.addWidget(QLabel("Pool:"))
        self._top_k = QSpinBox()
        self._top_k.setRange(1, 999999)
        self._top_k.setFixedWidth(100)
        self._top_k.setValue(int(state.get("narr_top_k", 200)))
        r.addWidget(self._top_k)
        r.addStretch()
        gs.addLayout(r)

        r = QHBoxLayout()
        r.addWidget(QLabel("Concept:"))
        self._w_clip = QSlider(Qt.Orientation.Horizontal)
        self._w_clip.setRange(0, 100)
        self._w_clip.setValue(int(state.get("narr_w_clip", 75)))
        r.addWidget(self._w_clip, 1)
        self._w_clip_lbl = QLabel(f"{self._w_clip.value()/100:.2f}")
        self._w_clip_lbl.setFixedWidth(36)
        self._w_clip.valueChanged.connect(
            lambda v: self._w_clip_lbl.setText(f"{v/100:.2f}"))
        r.addWidget(self._w_clip_lbl)
        gs.addLayout(r)

        r = QHBoxLayout()
        self._chk_dedupe = QCheckBox("Dedupe")
        self._chk_dedupe.setChecked(bool(state.get("narr_dedupe", True)))
        r.addWidget(self._chk_dedupe)
        self._dd_slider = QSlider(Qt.Orientation.Horizontal)
        self._dd_slider.setRange(50, 99)
        self._dd_slider.setValue(int(float(state.get("narr_dedupe_thr", 0.95)) * 100))
        r.addWidget(self._dd_slider, 1)
        self._dd_lbl = QLabel(f"{self._dd_slider.value()/100:.2f}")
        self._dd_lbl.setFixedWidth(36)
        self._dd_slider.valueChanged.connect(
            lambda v: self._dd_lbl.setText(f"{v/100:.2f}"))
        r.addWidget(self._dd_lbl)
        gs.addLayout(r)

        left_lay.addWidget(grp_set)

        # INITIAL SEEDS
        grp_seeds = QGroupBox("INITIAL SEEDS")
        gseed = QVBoxLayout(grp_seeds)
        gseed.setSpacing(6)

        gseed.addWidget(QLabel("POSITIVE PROMPT"))
        self._pos_prompt = QLineEdit(state.get("narr_text_prompt", ""))
        self._pos_prompt.setPlaceholderText("e.g. golden hour portrait")
        gseed.addWidget(self._pos_prompt)

        gseed.addWidget(QLabel("NEGATIVE PROMPT"))
        self._neg_prompt = QLineEdit(state.get("narr_neg_prompt", ""))
        self._neg_prompt.setPlaceholderText("e.g. blurry, dark")
        gseed.addWidget(self._neg_prompt)

        r = QHBoxLayout()
        r.addWidget(QLabel("Text influence:"))
        self._w_text = QSlider(Qt.Orientation.Horizontal)
        self._w_text.setRange(0, 100)
        self._w_text.setValue(int(float(state.get("narr_w_text", 0.5)) * 100))
        r.addWidget(self._w_text, 1)
        self._w_text_lbl = QLabel(f"{self._w_text.value()/100:.2f}")
        self._w_text_lbl.setFixedWidth(36)
        self._w_text.valueChanged.connect(
            lambda v: self._w_text_lbl.setText(f"{v/100:.2f}"))
        r.addWidget(self._w_text_lbl)
        gseed.addLayout(r)

        self._seeds_container = QWidget()
        self._seeds_lay = QVBoxLayout(self._seeds_container)
        self._seeds_lay.setContentsMargins(0, 0, 0, 0)
        self._seeds_lay.setSpacing(6)
        gseed.addWidget(self._seeds_container)

        btn_add = QPushButton("＋  Add Seed Image")
        btn_add.clicked.connect(self._add_seed_dialog)
        gseed.addWidget(btn_add)

        left_lay.addWidget(grp_seeds)
        left_lay.addStretch()
        left_outer.setWidget(left_w)

        # left wrap + start button
        left_wrap = QWidget()
        lw_lay = QVBoxLayout(left_wrap)
        lw_lay.setContentsMargins(0, 0, 0, 0)
        lw_lay.addWidget(left_outer, 1)

        self._btn_start = QPushButton("▶  START NARRATIVE")
        self._btn_start.setFixedHeight(44)
        self._btn_start.setStyleSheet(
            "font-size:14px; font-weight:800; background:#061408;"
            "border:2px solid rgba(64,255,107,0.8);")
        self._btn_start.clicked.connect(self._on_start)
        lw_lay.addWidget(self._btn_start)

        splitter.addWidget(left_wrap)
        splitter.setStretchFactor(0, 0)

        # ── RIGHT PANEL ──────────────────────────────────────────────────────
        right_w = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(4, 4, 4, 4)
        right_lay.setSpacing(6)

        # toolbar
        tb = QHBoxLayout()
        btn_reset = QPushButton("✕  RESET")
        btn_reset.clicked.connect(self._reset)
        tb.addWidget(btn_reset)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("font-size:11px; color:rgba(64,255,107,0.7);")
        tb.addWidget(self._status_lbl)
        tb.addStretch()
        btn_exp = QPushButton("⬇  EXPORT ALL")
        btn_exp.setStyleSheet("font-weight:700;")
        btn_exp.clicked.connect(self._export_all)
        tb.addWidget(btn_exp)
        btn_vid = QPushButton("→ VIDEO")
        btn_vid.setStyleSheet(
            f"QPushButton {{ color: {BG}; background: {GREEN}; font-weight: 700; "
            f"border: none; border-radius: 4px; padding: 4px 12px; }}"
            f"QPushButton:hover {{ background: #60ff8b; }}")
        btn_vid.clicked.connect(self._push_to_video)
        tb.addWidget(btn_vid)
        right_lay.addLayout(tb)

        # ── TOP: selections grid (no scroll — expands naturally) ────────────
        self._sel_widget = QWidget()
        self._sel_widget.setStyleSheet(
            f"background: {BG2}; border: 1px solid rgba(64,255,107,0.15); "
            f"border-radius: 4px;")
        self._sel_grid = QGridLayout(self._sel_widget)
        self._sel_grid.setSpacing(8)
        self._sel_grid.setContentsMargins(8, 8, 8, 8)

        self._sel_empty = QLabel("Selections will appear here")
        self._sel_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sel_empty.setStyleSheet(
            "color: rgba(64,255,107,0.25); font-size: 12px; padding: 16px;")
        self._sel_grid.addWidget(self._sel_empty, 0, 0)

        # ── scrollable content area (selections + controls + results) ────────
        self._scroll_content = QWidget()
        self._scroll_lay = QVBoxLayout(self._scroll_content)
        self._scroll_lay.setContentsMargins(0, 0, 0, 0)
        self._scroll_lay.setSpacing(6)

        self._scroll_lay.addWidget(self._sel_widget)

        # ── controls bar (between selections and results) ────────────────────
        self._ctrl_bar = QWidget()
        self._ctrl_bar.setStyleSheet(
            f"background: {BG}; border: 1px solid rgba(64,255,107,0.2); "
            f"border-radius: 4px;")
        cb = QHBoxLayout(self._ctrl_bar)
        cb.setContentsMargins(10, 6, 10, 6)
        cb.setSpacing(6)

        cb.addWidget(self._mklbl("Concept:"))
        self._next_w_clip = QSlider(Qt.Orientation.Horizontal)
        self._next_w_clip.setRange(0, 100)
        self._next_w_clip.setValue(75)
        self._next_w_clip.setFixedWidth(80)
        cb.addWidget(self._next_w_clip)
        self._next_w_clip_lbl = QLabel("0.75")
        self._next_w_clip_lbl.setFixedWidth(30)
        self._next_w_clip.valueChanged.connect(
            lambda v: self._next_w_clip_lbl.setText(f"{v/100:.2f}"))
        cb.addWidget(self._next_w_clip_lbl)

        cb.addWidget(self._mklbl(" +"))
        self._next_pos = QLineEdit()
        self._next_pos.setPlaceholderText("positive")
        self._next_pos.setFixedWidth(140)
        cb.addWidget(self._next_pos)

        cb.addWidget(self._mklbl(" −"))
        self._next_neg = QLineEdit()
        self._next_neg.setPlaceholderText("negative")
        self._next_neg.setFixedWidth(110)
        cb.addWidget(self._next_neg)

        cb.addWidget(self._mklbl(" Text:"))
        self._next_w_text = QSlider(Qt.Orientation.Horizontal)
        self._next_w_text.setRange(0, 100)
        self._next_w_text.setValue(50)
        self._next_w_text.setFixedWidth(60)
        cb.addWidget(self._next_w_text)
        self._next_w_text_lbl = QLabel("0.50")
        self._next_w_text_lbl.setFixedWidth(30)
        self._next_w_text.valueChanged.connect(
            lambda v: self._next_w_text_lbl.setText(f"{v/100:.2f}"))
        cb.addWidget(self._next_w_text_lbl)

        cb.addStretch()

        self._btn_run_next = QPushButton("▶  RUN NEXT")
        self._btn_run_next.setStyleSheet(
            f"QPushButton {{ color: {BG}; background: {GREEN}; border: none; "
            f"border-radius: 4px; padding: 6px 16px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #60ff8b; }}"
            f"QPushButton:disabled {{ background: #1a4a22; color: #446644; }}")
        self._btn_run_next.clicked.connect(self._on_run_next)
        cb.addWidget(self._btn_run_next)

        self._ctrl_bar.hide()
        self._scroll_lay.addWidget(self._ctrl_bar)

        # ── results grid (no own scroll — expands to fit) ───────────────────
        self._results_grid = ImageGrid(show_seed_buttons=False)
        self._results_grid.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._results_grid.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._results_grid.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._results_grid.selection_changed.connect(self._on_results_sel_changed)
        self._scroll_lay.addWidget(self._results_grid)
        self._scroll_lay.addStretch()

        # wrap in single scroll area
        self._right_scroll = QScrollArea()
        self._right_scroll.setWidgetResizable(True)
        self._right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._right_scroll.setWidget(self._scroll_content)
        right_lay.addWidget(self._right_scroll, 1)

        # ── confirm bar (sticky at bottom) ───────────────────────────────────
        self._confirm_bar = QWidget()
        self._confirm_bar.setStyleSheet(
            f"background: {BG}; border-top: 1px solid rgba(64,255,107,0.3);")
        cfb = QHBoxLayout(self._confirm_bar)
        cfb.setContentsMargins(10, 6, 10, 6)
        self._sel_count_lbl = QLabel("0 selected (pick 1–5)")
        self._sel_count_lbl.setStyleSheet(
            "font-size:11px; color:rgba(64,255,107,0.7);")
        cfb.addWidget(self._sel_count_lbl)
        cfb.addStretch()
        self._btn_confirm = QPushButton("✓  CONFIRM SELECTION")
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.setStyleSheet(
            f"QPushButton {{ color: {BG}; background: {GREEN}; border: none; "
            f"border-radius: 4px; padding: 6px 16px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #60ff8b; }}"
            f"QPushButton:disabled {{ background: #1a4a22; color: #446644; }}")
        self._btn_confirm.clicked.connect(self._on_confirm)
        cfb.addWidget(self._btn_confirm)
        self._confirm_bar.hide()
        right_lay.addWidget(self._confirm_bar)

        splitter.addWidget(right_w)
        splitter.setStretchFactor(1, 1)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        self._load_state()

    @staticmethod
    def _mklbl(text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(f"color: {GREEN}; font-size: 11px;")
        return l

    # ── seed management ──────────────────────────────────────────────────────

    def _rebuild_seed_tiles(self) -> None:
        while self._seeds_lay.count():
            item = self._seeds_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, s in enumerate(self._seeds):
            tile = SeedTile(index=i, path=s["path"], weight=s["weight"])
            tile.remove_clicked.connect(self._remove_seed)
            tile.weight_changed.connect(self._update_weight)
            self._seeds_lay.addWidget(tile)

    def _add_seed_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Seed Image(s)", "",
            "Images (*.jpg *.jpeg *.png *.gif *.webp *.bmp *.tif *.tiff)")
        for p in paths:
            if not any(s["path"] == p for s in self._seeds):
                self._seeds.append({"path": p, "weight": 1.0})
        self._rebuild_seed_tiles()

    def push_seeds(self, paths: List[str]) -> None:
        for p in paths:
            if not any(s["path"] == p for s in self._seeds):
                self._seeds.append({"path": p, "weight": 1.0})
        self._rebuild_seed_tiles()

    @pyqtSlot(int)
    def _remove_seed(self, idx: int) -> None:
        if 0 <= idx < len(self._seeds):
            self._seeds.pop(idx)
        self._rebuild_seed_tiles()

    @pyqtSlot(int, float)
    def _update_weight(self, idx: int, w: float) -> None:
        if 0 <= idx < len(self._seeds):
            self._seeds[idx]["weight"] = w

    # ── selections grid rebuild ──────────────────────────────────────────────

    def _rebuild_sel_grid(self) -> None:
        """Rebuild the top selections grid from generations data. Wraps into rows."""
        # clear
        for c in self._sel_cards:
            self._sel_grid.removeWidget(c)
            c.deleteLater()
        self._sel_cards.clear()
        for b in self._rerun_buttons:
            self._sel_grid.removeWidget(b)
            b.deleteLater()
        self._rerun_buttons.clear()
        while self._sel_grid.count():
            item = self._sel_grid.takeAt(0)
            if item.widget() and item.widget() is not self._sel_empty:
                item.widget().deleteLater()

        collapsed_gens = [g for g in self._generations if g.get("collapsed")]
        if not collapsed_gens:
            self._sel_empty.show()
            self._sel_grid.addWidget(self._sel_empty, 0, 0)
            return

        self._sel_empty.hide()

        # compute columns based on available width
        card_w = SEL_THUMB_W + 18 + self._sel_grid.spacing()
        avail_w = max(1, self._sel_widget.width() - 20)
        cols = max(1, avail_w // card_w)

        latest_collapsed = max(g["gen_index"] for g in collapsed_gens)
        idx = 0   # flat index across all cards

        # Prepend the initial seed image(s) as "generation 0" (labels 0A,
        # 0B, 0C…). No weight slider here (seeds are managed in the INITIAL
        # SEEDS panel) and no re-run button (can't re-run the seed step).
        for si, seed in enumerate(self._seeds):
            card = SelectionCard(
                gen_idx=-1, sel_idx=si,
                path=seed["path"],
                weight=seed.get("weight", 1.0),
                show_weight=False,
            )
            self._sel_cards.append(card)
            row, col = divmod(idx, cols)
            self._sel_grid.addWidget(card, row * 2, col)
            idx += 1

        for gen in sorted(collapsed_gens, key=lambda g: g["gen_index"]):
            gi = gen["gen_index"]
            is_latest = (gi == latest_collapsed)
            last_card_idx = idx + len(gen.get("selections", [])) - 1

            for si, sel in enumerate(gen.get("selections", [])):
                card = SelectionCard(
                    gen_idx=gi, sel_idx=si,
                    path=sel["path"],
                    weight=sel.get("weight", 1.0),
                    show_weight=is_latest,
                )
                if is_latest:
                    card.weight_changed.connect(self._on_weight_changed)
                self._sel_cards.append(card)
                row, col = divmod(idx, cols)
                self._sel_grid.addWidget(card, row * 2, col)
                idx += 1

            # re-run button under the last card of this generation
            last_row, last_col = divmod(last_card_idx, cols)
            btn = ReRunButton(gi)
            btn.rerun_requested.connect(self._on_rerun)
            self._rerun_buttons.append(btn)
            self._sel_grid.addWidget(btn, last_row * 2 + 1, last_col)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._sel_cards:
            self._rebuild_sel_grid()

    def _on_weight_changed(self, gen_idx: int, sel_idx: int, weight: float) -> None:
        for gen in self._generations:
            if gen["gen_index"] == gen_idx:
                sels = gen.get("selections", [])
                if 0 <= sel_idx < len(sels):
                    sels[sel_idx]["weight"] = weight
                break

    # ── narrative flow ───────────────────────────────────────────────────────

    def _on_start(self) -> None:
        """Start the first generation from initial seeds/text."""
        if self._worker and self._worker.isRunning():
            return
        if self._generations:
            return  # already started — use RUN NEXT

        idx_dir = self._idx_dir.text().strip()
        if not idx_dir or not Path(idx_dir).is_dir():
            QMessageBox.warning(self, "KOAN.img", "Index folder not found.")
            return

        if not self._seeds and not self._pos_prompt.text().strip():
            QMessageBox.warning(self, "KOAN.img",
                                "Add at least one seed image or a text prompt.")
            return

        gen = {
            "gen_index": 0,
            "seeds": list(self._seeds),
            "text_prompt": self._pos_prompt.text().strip(),
            "neg_prompt": self._neg_prompt.text().strip(),
            "w_text": self._w_text.value() / 100.0,
            "w_clip": self._w_clip.value() / 100.0,
            "selections": [],
            "collapsed": False,
        }
        self._generations.append(gen)
        self._active_gen_idx = 0
        self._fire_search(gen)

    def _on_confirm(self) -> None:
        """Confirm selection from results grid — move to selections grid."""
        selected = self._results_grid.get_selected()
        if not (1 <= len(selected) <= 5):
            return

        gen = self._generations[self._active_gen_idx]
        gen["selections"] = [{"path": p, "weight": 1.0} for p in selected]
        gen["collapsed"] = True

        # remove weight sliders from previous generations' cards
        for card in self._sel_cards:
            card.remove_weight_slider()

        # clear results grid and its selection state
        self._results_grid.selected_paths.clear()
        self._results_grid.populate([], set())
        self._confirm_bar.hide()

        # rebuild selections grid (now includes this gen)
        self._rebuild_sel_grid()

        # show controls bar
        self._ctrl_bar.show()

        self._btn_start.setText("▶  RUN NEXT")
        self._status_lbl.setText(
            f"{sum(len(g.get('selections',[])) for g in self._generations if g.get('collapsed'))} selections")

    def _on_run_next(self) -> None:
        """Run next generation using the latest selections as seeds."""
        if self._worker and self._worker.isRunning():
            return

        idx_dir = self._idx_dir.text().strip()
        if not idx_dir or not Path(idx_dir).is_dir():
            QMessageBox.warning(self, "KOAN.img", "Index folder not found.")
            return

        # get seeds from latest collapsed generation with current weights
        latest = None
        for gen in self._generations:
            if gen.get("collapsed"):
                if latest is None or gen["gen_index"] > latest["gen_index"]:
                    latest = gen

        if latest is None:
            return

        # read weights from the cards
        seeds = []
        for card in self._sel_cards:
            if card.gen_idx == latest["gen_index"]:
                seeds.append({"path": card.path, "weight": card.get_weight()})

        if not seeds:
            seeds = [{"path": s["path"], "weight": s.get("weight", 1.0)}
                     for s in latest["selections"]]

        new_idx = latest["gen_index"] + 1
        gen = {
            "gen_index": new_idx,
            "seeds": seeds,
            "text_prompt": self._next_pos.text().strip(),
            "neg_prompt":  self._next_neg.text().strip(),
            "w_text":      self._next_w_text.value() / 100.0,
            "w_clip":      self._next_w_clip.value() / 100.0,
            "selections": [],
            "collapsed": False,
        }

        # trim anything after latest collapsed gen (for re-run cases)
        self._generations = [g for g in self._generations
                             if g["gen_index"] <= latest["gen_index"]]
        self._generations.append(gen)
        self._active_gen_idx = new_idx

        self._ctrl_bar.hide()
        self._fire_search(gen)

    def _fire_search(self, gen: Dict) -> None:
        self._btn_start.setEnabled(False)
        self._btn_start.setText("⏳  Running…")
        self._btn_run_next.setEnabled(False)

        self._worker = SearchWorker(
            index_dir   = self._idx_dir.text().strip(),
            seeds       = gen["seeds"],
            text_prompt = gen.get("text_prompt", ""),
            neg_prompt  = gen.get("neg_prompt", ""),
            w_text      = gen.get("w_text", 0.5),
            n_results   = self._n_results.value(),
            top_k       = self._top_k.value(),
            w_clip      = gen.get("w_clip", 0.75),
            dedupe      = self._chk_dedupe.isChecked(),
            dedupe_thr  = self._dd_slider.value() / 100.0,
        )
        self._worker.finished.connect(self._on_search_done)
        self._worker.error.connect(self._on_search_error)
        self._worker.start()

    @pyqtSlot(object)
    def _on_search_done(self, items: List[Dict]) -> None:
        self._btn_start.setEnabled(True)
        self._btn_start.setText("▶  RUN NEXT")
        self._btn_run_next.setEnabled(True)

        self._results_grid.selected_paths.clear()
        self._results_grid.populate(items, set())
        self._confirm_bar.show()
        self._status_lbl.setText(f"{len(items)} results")

    @pyqtSlot(str)
    def _on_search_error(self, msg: str) -> None:
        self._btn_start.setEnabled(True)
        self._btn_start.setText(
            "▶  RUN NEXT" if self._generations else "▶  START NARRATIVE")
        self._btn_run_next.setEnabled(True)
        QMessageBox.critical(self, "Search error", msg)

    def _on_results_sel_changed(self, n: int) -> None:
        self._sel_count_lbl.setText(f"{n} selected (pick 1–5)")
        self._btn_confirm.setEnabled(1 <= n <= 5)

    # ── re-run ───────────────────────────────────────────────────────────────

    def _on_rerun(self, gen_idx: int) -> None:
        """Re-run from a generation. Keep its cards in the selection grid,
        ditch everything after, show controls so user can tweak and re-run."""
        if self._worker and self._worker.isRunning():
            return

        gen = next((g for g in self._generations if g["gen_index"] == gen_idx), None)
        if gen is None:
            return

        # trim everything after this generation
        self._generations = [g for g in self._generations
                             if g["gen_index"] <= gen_idx]

        # clear results
        self._results_grid.populate([], set())
        self._confirm_bar.hide()

        # rebuild selections grid (only gens up to this one)
        self._rebuild_sel_grid()

        # pre-fill controls with this gen's settings
        self._next_pos.setText(gen.get("text_prompt", ""))
        self._next_neg.setText(gen.get("neg_prompt", ""))
        self._next_w_text.setValue(int(gen.get("w_text", 0.5) * 100))
        self._next_w_clip.setValue(int(gen.get("w_clip", 0.75) * 100))

        # show controls — user tweaks, hits RUN NEXT
        self._ctrl_bar.show()
        self._active_gen_idx = gen_idx

    # ── reset ────────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._generations.clear()
        self._active_gen_idx = -1
        self._results_grid.populate([], set())
        self._results_grid.selected_paths.clear()
        self._confirm_bar.hide()
        self._ctrl_bar.hide()
        self._rebuild_sel_grid()
        self._btn_start.setText("▶  START NARRATIVE")
        self._status_lbl.setText("")

    # ── export ───────────────────────────────────────────────────────────────

    def _get_all_selections(self) -> List[tuple]:
        """Return [(label, path), …] for the full narrative sequence.

        The initial seed image(s) are prepended as "generation 0" with labels
        "0A", "0B", "0C", … so the seeds are the starting point of the
        narrative wherever this is consumed — export, push-to-video, etc.
        Text-only narratives (no seeds) are unchanged.
        """
        result = [
            (_gen_label(-1, i), s["path"]) for i, s in enumerate(self._seeds)
        ]
        for gen in self._generations:
            gi = gen["gen_index"]
            for si, sel in enumerate(gen.get("selections", [])):
                result.append((_gen_label(gi, si), sel["path"]))
        return result

    def _export_all(self) -> None:
        selections = self._get_all_selections()
        if not selections:
            QMessageBox.information(self, "KOAN.img", "No selections to export.")
            return
        export_root = self._exp_dir.text().strip()
        if not export_root:
            QMessageBox.warning(self, "KOAN.img", "Export folder is not set.")
            return

        pos = self._pos_prompt.text().strip()
        if pos:
            subfolder = _slugify(pos)
        else:
            from datetime import datetime
            subfolder = datetime.now().strftime("narrative-%Y%m%d-%H%M%S")

        dest = Path(export_root) / subfolder
        # ensure unique folder — never overwrite a previous export
        if dest.exists():
            n = 2
            while True:
                candidate = Path(export_root) / f"{subfolder}-{n}"
                if not candidate.exists():
                    dest = candidate
                    break
                n += 1
        dest.mkdir(parents=True, exist_ok=False)
        errors = []
        for label, src in selections:
            try:
                src_p = Path(src)
                ext = src_p.suffix.lower()
                if ext == ".webp":
                    img = Image.open(str(src_p))
                    if getattr(img, "is_animated", False):
                        dst = dest / f"{label}_{src_p.stem}.gif"
                        frames, durs = [], []
                        for fi in range(img.n_frames):
                            img.seek(fi)
                            frames.append(img.convert("RGBA").copy())
                            durs.append(img.info.get("duration", 100))
                        frames[0].save(str(dst), save_all=True,
                                       append_images=frames[1:],
                                       duration=durs, loop=0)
                    else:
                        dst = dest / f"{label}_{src_p.stem}.jpg"
                        img.convert("RGB").save(str(dst), "JPEG", quality=95)
                else:
                    shutil.copy2(str(src_p), str(dest / f"{label}_{src_p.name}"))
            except Exception as exc:
                errors.append(f"{src}: {exc}")

        msg = f"Exported {len(selections) - len(errors)} file(s) to:\n{dest}"
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "Export complete", msg)

    def _push_to_video(self) -> None:
        sels = self._get_all_selections()
        if sels:
            self.push_to_video_signal.emit([s[1] for s in sels])

    # ── state ────────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict:
        return {
            "narr_index_dir":   self._idx_dir.text(),
            "narr_export_root": self._exp_dir.text(),
            "narr_n_results":   self._n_results.value(),
            "narr_top_k":       self._top_k.value(),
            "narr_w_clip":      self._w_clip.value(),
            "narr_text_prompt": self._pos_prompt.text(),
            "narr_neg_prompt":  self._neg_prompt.text(),
            "narr_w_text":      self._w_text.value() / 100.0,
            "narr_dedupe":      self._chk_dedupe.isChecked(),
            "narr_dedupe_thr":  self._dd_slider.value() / 100.0,
        }

    def save_state(self) -> None:
        try:
            gens = []
            for g in self._generations:
                if g.get("selections"):
                    gens.append({
                        "gen_index":  g["gen_index"],
                        "seeds":      g.get("seeds", []),
                        "text_prompt": g.get("text_prompt", ""),
                        "neg_prompt": g.get("neg_prompt", ""),
                        "w_text":     g.get("w_text", 0.5),
                        "w_clip":     g.get("w_clip", 0.75),
                        "selections": g["selections"],
                    })
            _STATE_FILE.write_text(
                json.dumps({"generations": gens, "initial_seeds": self._seeds},
                           indent=2, ensure_ascii=False),
                encoding="utf-8")
        except Exception:
            pass

    def _load_state(self) -> None:
        try:
            if not _STATE_FILE.exists():
                return
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            for s in state.get("initial_seeds", []):
                if Path(s.get("path", "")).exists():
                    self._seeds.append(s)
            self._rebuild_seed_tiles()

            for sg in state.get("generations", []):
                sels = [s for s in sg.get("selections", [])
                        if Path(s.get("path", "")).exists()]
                if not sels:
                    continue
                sg["selections"] = sels
                sg["collapsed"] = True
                self._generations.append(sg)

            if self._generations:
                self._active_gen_idx = max(g["gen_index"] for g in self._generations)
                self._rebuild_sel_grid()
                self._ctrl_bar.show()
                self._btn_start.setText("▶  RUN NEXT")
        except Exception:
            pass
