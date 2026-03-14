from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np
from tqdm import tqdm

from captioner import Captioner
from common import ensure_dir, file_mtime, open_image_rgb
from embedder import Embedder
from features import color_signature_from_path, color_signature_lab_hist

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def iter_media_paths(root: Path, recursive: bool, include_videos: bool) -> Iterable[Path]:
    root = root.expanduser().resolve()
    for dirpath, _, filenames in os.walk(root):
        dp = Path(dirpath)
        for fn in filenames:
            p = dp / fn
            ext = p.suffix.lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                yield p
            elif include_videos and ext in SUPPORTED_VIDEO_EXTS:
                yield p
        if not recursive:
            break


def read_video_first_frame_rgb(path: Path):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _is_animated_image(path: Path) -> bool:
    from PIL import Image

    try:
        im = Image.open(path)
        try:
            if getattr(im, "is_animated", False) and int(getattr(im, "n_frames", 1)) > 1:
                return True
            return False
        finally:
            try:
                im.close()
            except Exception:
                pass
    except Exception:
        return False


def read_image_first_frame_rgb(path: Path, max_side: int = 1024):
    from PIL import Image

    im = Image.open(path)
    try:
        try:
            im.seek(0)
        except Exception:
            pass
        frame = im.convert("RGB")
    finally:
        try:
            im.close()
        except Exception:
            pass

    if max_side and max_side > 0:
        w, h = frame.size
        if w > 0 and h > 0:
            scale = float(max_side) / float(max(w, h))
            if scale < 1.0:
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                frame = frame.resize((nw, nh), resample=Image.BICUBIC)
    return frame


def stable_id_for_path(p: Path) -> str:
    s = str(p.as_posix()).lower().encode("utf-8", errors="ignore")
    return hashlib.sha1(s).hexdigest()


def db_connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            faiss_idx INTEGER UNIQUE,
            path TEXT UNIQUE,
            width INTEGER,
            height INTEGER,
            mtime INTEGER
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS colors (
            id TEXT PRIMARY KEY,
            vec BLOB
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS captions (
            id TEXT PRIMARY KEY,
            caption TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS progress (
            k TEXT PRIMARY KEY,
            v TEXT
        );
        """
    )
    conn.commit()


def get_progress(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT v FROM progress WHERE k='offset';").fetchone()
    if not row:
        return 0
    try:
        return int(row[0])
    except Exception:
        return 0


def set_progress(conn: sqlite3.Connection, offset: int) -> None:
    conn.execute("INSERT OR REPLACE INTO progress(k,v) VALUES('offset', ?);", (str(int(offset)),))
    conn.commit()


def np_to_blob(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32)
    return x.tobytes(order="C")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)


def _clean_model_name(s: str) -> str:
    v = (s or "").strip()
    return v if v else "ViT-B-32"


def _clean_pretrained(s: str) -> str:
    v = (s or "").strip()
    return v if v else "laion2b_s34b_b79k"


def _load_faiss(index_path: Path) -> Optional[faiss.Index]:
    if not index_path.exists():
        return None
    try:
        return faiss.read_index(str(index_path))
    except Exception:
        return None


def _infer_dim_from_paths(emb: Embedder, paths: list[Path], probe: int) -> Optional[int]:
    for p in paths[: max(0, int(probe))]:
        try:
            ext = p.suffix.lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                if ext in {".gif", ".webp"}:
                    img = read_image_first_frame_rgb(p, max_side=1024)
                else:
                    img = open_image_rgb(p, max_side=1024)
            else:
                img = read_video_first_frame_rgb(p)
                if img is None:
                    continue
            feat = emb.embed_pil(img)
            return int(feat.shape[0])
        except Exception:
            continue
    return None


def _print_summary(
    status: str,
    end_offset: int,
    total_paths: int,
    indexed_new: int,
    skipped_existing: int,
    skipped_unreadable: int,
) -> None:
    st = (status or "").strip() or "more"
    print(
        "KOAN_SUMMARY "
        f"status={st} "
        f"end_offset={int(end_offset)} "
        f"total_paths={int(total_paths)} "
        f"indexed_new={int(indexed_new)} "
        f"skipped_existing={int(skipped_existing)} "
        f"skipped_unreadable={int(skipped_unreadable)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media_root", type=str)
    ap.add_argument("out_dir", type=str)

    ap.add_argument("--chunk_size", type=int, default=2000)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--reset_progress", action="store_true")
    ap.add_argument("--include_videos", action="store_true")

    ap.add_argument("--model_name", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--batch_commit", type=int, default=200)

    ap.add_argument("--dim_probe_limit", type=int, default=800)

    ap.add_argument("--one_chunk", action="store_true")

    args = ap.parse_args()

    args.model_name = _clean_model_name(args.model_name)
    args.pretrained = _clean_pretrained(args.pretrained)

    media_root = Path(args.media_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    db_path = out_dir / "catalog.sqlite"
    index_path = out_dir / "clip.faiss"
    meta_path = out_dir / "meta.npz"

    conn = db_connect(db_path)
    db_init(conn)

    if args.reset_progress:
        set_progress(conn, 0)

    all_paths = list(
        iter_media_paths(
            media_root,
            recursive=bool(args.recursive),
            include_videos=bool(args.include_videos),
        )
    )
    total = len(all_paths)
    if total == 0:
        _print_summary("done", 0, 0, 0, 0, 0)
        print("done")
        return

    emb = Embedder(model_name=args.model_name, pretrained=args.pretrained)
    cap = Captioner()

    index = _load_faiss(index_path)
    if index is None:
        dim = _infer_dim_from_paths(emb, all_paths, probe=int(args.dim_probe_limit))
        if dim is None:
            _print_summary("done", 0, total, 0, 0, total)
            raise SystemExit("Could not infer CLIP dim from any readable media in the folder.")
        index = faiss.IndexFlatIP(int(dim))

    try:
        while True:
            offset = get_progress(conn)
            if offset >= total:
                np.savez(str(meta_path), total_files=total, last_offset=offset, dim=int(index.d))
                _print_summary("done", offset, total, 0, 0, 0)
                print("done")
                return

            end = min(total, offset + int(args.chunk_size))
            chunk = all_paths[offset:end]

            print(f"Chunk: {offset} -> {end} (total {total})")

            cur = conn.cursor()
            pending = 0

            indexed_new = 0
            skipped_existing = 0
            skipped_unreadable = 0

            for p in tqdm(chunk, desc=f"Indexing {offset} to {end}"):
                p = p.resolve()
                sid = stable_id_for_path(p)

                row = conn.execute("SELECT faiss_idx FROM images WHERE id=?;", (sid,)).fetchone()
                if row is not None:
                    skipped_existing += 1
                    continue

                try:
                    ext = p.suffix.lower()

                    if ext in SUPPORTED_IMAGE_EXTS:
                        if ext in {".gif", ".webp"}:
                            img = read_image_first_frame_rgb(p, max_side=1024)
                            if _is_animated_image(p):
                                col = color_signature_lab_hist(img)
                            else:
                                col = color_signature_from_path(p)
                        else:
                            img = open_image_rgb(p, max_side=1024)
                            col = color_signature_from_path(p)
                    else:
                        img = read_video_first_frame_rgb(p)
                        if img is None:
                            skipped_unreadable += 1
                            continue
                        col = color_signature_lab_hist(img)

                    w, h = img.size
                    feat = emb.embed_pil(img).astype(np.float32).reshape(-1)

                    if int(feat.shape[0]) != int(index.d):
                        skipped_unreadable += 1
                        continue

                    feat = l2_normalize(feat)
                    mt = file_mtime(p)

                    caption_text = ""
                    try:
                        caption_text = cap.caption(img)
                    except Exception:
                        caption_text = ""
                except Exception:
                    skipped_unreadable += 1
                    continue

                faiss_idx = int(index.ntotal)
                index.add(feat.reshape(1, -1))

                cur.execute(
                    "INSERT OR REPLACE INTO images(id, faiss_idx, path, width, height, mtime) VALUES(?,?,?,?,?,?);",
                    (sid, int(faiss_idx), str(p), int(w), int(h), int(mt)),
                )
                cur.execute("INSERT OR REPLACE INTO colors(id, vec) VALUES(?,?);", (sid, np_to_blob(col)))
                cur.execute("INSERT OR REPLACE INTO captions(id, caption) VALUES(?,?);", (sid, str(caption_text)))

                indexed_new += 1

                pending += 1
                if pending >= int(args.batch_commit):
                    conn.commit()
                    pending = 0

            if pending:
                conn.commit()

            faiss.write_index(index, str(index_path))
            set_progress(conn, end)
            np.savez(str(meta_path), total_files=total, last_offset=end, dim=int(index.d))

            status = "done" if end >= total else "more"
            _print_summary(status, end, total, indexed_new, skipped_existing, skipped_unreadable)
            print("done")

            if bool(args.one_chunk):
                return

    except KeyboardInterrupt:
        faiss.write_index(index, str(index_path))
        np.savez(str(meta_path), total_files=total, last_offset=get_progress(conn), dim=int(index.d))
        print("stopped")


if __name__ == "__main__":
    main()
