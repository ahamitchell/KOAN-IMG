"""
build_shape_index.py — build the shape feature index for COLLAGE tab.

Reads image paths from the existing ai_index/catalog.sqlite and writes
shape features to mix_index/shapes.sqlite. Resumable.

Usage:
    python build_shape_index.py <ai_index_dir> <mix_index_dir>
                                [--chunk_size N] [--reset_progress]
                                [--one_chunk]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow importing shape_features whether run directly or as a module
_HERE = Path(__file__).resolve().parent
_APP = _HERE.parent
sys.path.insert(0, str(_APP))

from collage.shape_features import extract_all


# ── SQLite helpers ────────────────────────────────────────────────────────────

def _db_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def _db_init(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS shape_edges (
            id   TEXT PRIMARY KEY,
            vec  BLOB
        );
        CREATE TABLE IF NOT EXISTS shape_circles (
            id   TEXT PRIMARY KEY,
            vec  BLOB
        );
        CREATE TABLE IF NOT EXISTS shape_flow (
            id   TEXT PRIMARY KEY,
            vec  BLOB
        );
        CREATE TABLE IF NOT EXISTS shape_color_patches (
            id   TEXT PRIMARY KEY,
            vec  BLOB
        );
        CREATE TABLE IF NOT EXISTS progress (
            k TEXT PRIMARY KEY,
            v TEXT
        );
        """
    )
    conn.commit()


def _get_progress(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT v FROM progress WHERE k='offset';").fetchone()
    return int(row[0]) if row else 0


def _set_progress(conn: sqlite3.Connection, offset: int) -> None:
    conn.execute("INSERT OR REPLACE INTO progress(k,v) VALUES('offset',?);", (str(int(offset)),))
    conn.commit()


def _np_to_blob(x: np.ndarray) -> bytes:
    return np.asarray(x, dtype=np.float32).tobytes(order="C")


# ── Image loading (mirrors common.py, no import needed) ──────────────────────

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def _open_image(path: Path) -> Image.Image | None:
    ext = path.suffix.lower()
    try:
        if ext in SUPPORTED_IMAGE_EXTS:
            with Image.open(path) as im:
                try:
                    im.seek(0)
                except Exception:
                    pass
                return im.convert("RGB").copy()
        if ext in SUPPORTED_VIDEO_EXTS:
            import cv2
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return None
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame).convert("RGB")
    except Exception:
        return None
    return None


# ── Summary line (matches KOAN convention) ───────────────────────────────────

def _print_summary(status: str, end_offset: int, total: int, indexed_new: int,
                   skipped_existing: int, skipped_unreadable: int) -> None:
    st = (status or "").strip() or "more"
    print(
        f"KOAN_SUMMARY status={st} end_offset={int(end_offset)} "
        f"total_paths={int(total)} indexed_new={int(indexed_new)} "
        f"skipped_existing={int(skipped_existing)} "
        f"skipped_unreadable={int(skipped_unreadable)}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ai_index_dir")
    ap.add_argument("mix_index_dir")
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--reset_progress", action="store_true")
    ap.add_argument("--one_chunk", action="store_true")
    args = ap.parse_args()

    ai_dir = Path(args.ai_index_dir).expanduser().resolve()
    mix_dir = Path(args.mix_index_dir).expanduser().resolve()
    mix_dir.mkdir(parents=True, exist_ok=True)

    ai_db = ai_dir / "catalog.sqlite"
    mix_db = mix_dir / "shapes.sqlite"

    if not ai_db.exists():
        print(f"ERROR: catalog.sqlite not found at {ai_db}")
        sys.exit(1)

    ai_conn = _db_connect(ai_db)
    mix_conn = _db_connect(mix_db)
    _db_init(mix_conn)

    if args.reset_progress:
        _set_progress(mix_conn, 0)

    # Read all (id, path) from catalog — use whichever schema exists
    try:
        rows = ai_conn.execute("SELECT id, path FROM images ORDER BY rowid;").fetchall()
    except sqlite3.OperationalError:
        rows = ai_conn.execute("SELECT idx, path FROM images ORDER BY idx;").fetchall()
    ai_conn.close()

    all_items = [(str(r[0]), str(r[1])) for r in rows]
    total = len(all_items)

    if total == 0:
        _print_summary("done", 0, 0, 0, 0, 0)
        print("done")
        return

    try:
        while True:
            offset = _get_progress(mix_conn)
            if offset >= total:
                _print_summary("done", offset, total, 0, 0, 0)
                print("done")
                return

            end = min(total, offset + args.chunk_size)
            chunk = all_items[offset:end]

            print(f"Chunk: {offset} -> {end} (total {total})")

            indexed_new = 0
            skipped_existing = 0
            skipped_unreadable = 0
            pending = 0

            cur = mix_conn.cursor()

            for img_id, img_path in tqdm(chunk, desc=f"Shape-indexing {offset}..{end}"):
                # Skip if already indexed
                row = mix_conn.execute(
                    "SELECT 1 FROM shape_edges WHERE id=?;", (img_id,)
                ).fetchone()
                if row is not None:
                    skipped_existing += 1
                    continue

                p = Path(img_path)
                if not p.exists():
                    skipped_unreadable += 1
                    continue

                img = _open_image(p)
                if img is None:
                    skipped_unreadable += 1
                    continue

                try:
                    feats = extract_all(img)
                except Exception:
                    skipped_unreadable += 1
                    continue

                cur.execute(
                    "INSERT OR REPLACE INTO shape_edges(id, vec) VALUES(?,?);",
                    (img_id, _np_to_blob(feats["edges"])),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO shape_circles(id, vec) VALUES(?,?);",
                    (img_id, _np_to_blob(feats["circle"])),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO shape_flow(id, vec) VALUES(?,?);",
                    (img_id, _np_to_blob(feats["flow"])),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO shape_color_patches(id, vec) VALUES(?,?);",
                    (img_id, _np_to_blob(feats["color_patches"])),
                )
                indexed_new += 1
                pending += 1

                if pending >= 100:
                    mix_conn.commit()
                    pending = 0

            if pending:
                mix_conn.commit()

            _set_progress(mix_conn, end)

            status = "done" if end >= total else "more"
            _print_summary(status, end, total, indexed_new, skipped_existing, skipped_unreadable)
            print("done" if end >= total else "chunk_done")

            if args.one_chunk or end >= total:
                return

    except KeyboardInterrupt:
        mix_conn.commit()
        print("stopped")


if __name__ == "__main__":
    main()
