from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        n = np.linalg.norm(x) + eps
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if not max_side or max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m > max_side and m > 0:
        s = max_side / float(m)
        nw = max(1, int(w * s))
        nh = max(1, int(h * s))
        img = img.resize((nw, nh), Image.BICUBIC)
    return img


def open_image_rgb(path: Path, max_side: int = 1024) -> Image.Image:
    ext = path.suffix.lower()

    with Image.open(path) as im:
        if ext in {".gif", ".webp"}:
            try:
                im.seek(0)
            except Exception:
                pass

        img = im.convert("RGB").copy()

    img = _resize_max_side(img, max_side=max_side)
    return img


def open_video_first_frame_rgb(path: Path, max_side: int = 1024) -> Image.Image:
    import cv2

    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to read first frame")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame).convert("RGB").copy()

    img = _resize_max_side(img, max_side=max_side)
    return img


def open_media_rgb(path: Path, max_side: int = 1024, allow_video: bool = False) -> Image.Image:
    ext = path.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTS:
        return open_image_rgb(path, max_side=max_side)
    if allow_video and ext in SUPPORTED_VIDEO_EXTS:
        return open_video_first_frame_rgb(path, max_side=max_side)
    raise ValueError("Unsupported file type")


def iter_media_paths(root: Path, recursive: bool, include_videos: bool) -> Iterable[Path]:
    root = root.expanduser().resolve()

    exts = set(SUPPORTED_IMAGE_EXTS)
    if include_videos:
        exts |= set(SUPPORTED_VIDEO_EXTS)

    if recursive:
        for dirpath, _, filenames in os.walk(root):
            dp = Path(dirpath)
            for fn in filenames:
                p = dp / fn
                if p.suffix.lower() in exts:
                    yield p
        return

    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def db_connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            idx INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            width INTEGER,
            height INTEGER,
            mtime INTEGER,
            kind TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS colors (
            idx INTEGER PRIMARY KEY,
            vec BLOB
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS captions (
            idx INTEGER PRIMARY KEY,
            caption TEXT
        );
        """
    )
    conn.commit()


def np_to_blob(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32)
    return x.tobytes(order="C")


def blob_to_np(b: bytes, dim: int) -> np.ndarray:
    x = np.frombuffer(b, dtype=np.float32)
    if x.size != dim:
        raise ValueError(f"Color vector dim mismatch. expected={dim} got={x.size}")
    return x


def file_mtime(path: Path) -> int:
    try:
        return int(path.stat().st_mtime)
    except Exception:
        return 0


def media_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTS:
        return "image"
    if ext in SUPPORTED_VIDEO_EXTS:
        return "video"
    return "other"
