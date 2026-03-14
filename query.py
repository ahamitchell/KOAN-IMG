from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, List

import faiss
import numpy as np

from common import blob_to_np, db_connect
from embedder import Embedder
from features import color_signature_from_path


STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "with", "for", "from", "by",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "there", "here", "over", "under", "into", "out", "up", "down", "near", "far",
    "person", "people", "man", "woman", "boy", "girl", "photo", "image", "picture",
}


def tokenize(text: str) -> list[str]:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [x for x in t.split() if len(x) >= 3 and x not in STOP]


def score_mix(clip_sim: float, col_sim: float, w_clip: float) -> float:
    w_clip = float(w_clip)
    w_col = 1.0 - w_clip
    return (w_clip * clip_sim) + (w_col * col_sim)


def _has_table(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {str(r[1]) for r in rows}


def _detect_images_schema(conn) -> tuple[str, str, bool]:
    """
    Returns:
      images_key: "idx" or "id"
      path_col: "path"
      has_faiss_idx: whether images has a faiss_idx column (chunked schema)
    """
    if not _has_table(conn, "images"):
        raise RuntimeError("Missing images table. Rebuild the index.")

    cols = _table_columns(conn, "images")

    if "idx" in cols:
        return "idx", "path", ("faiss_idx" in cols)

    if "id" in cols:
        return "id", "path", ("faiss_idx" in cols)

    raise RuntimeError("Images table has no idx or id column. Rebuild the index.")


def _detect_key_col(conn, table: str) -> Optional[str]:
    if not _has_table(conn, table):
        return None
    cols = _table_columns(conn, table)
    if "idx" in cols:
        return "idx"
    if "id" in cols:
        return "id"
    return None


def load_artifacts_cacheable(index_dir: Path) -> tuple:
    """Returns (db_path, index, images_key, images_has_faiss_idx, colors_key, captions_key, dim).

    Excludes the SQLite connection so the result is safe to cache across threads.
    Call open_db(db_path) to get a fresh connection when needed.
    """
    index_dir = index_dir.expanduser().resolve()
    db_path = index_dir / "catalog.sqlite"
    index_path = index_dir / "clip.faiss"
    meta_path = index_dir / "meta.npz"
    if not db_path.exists() or not index_path.exists() or not meta_path.exists():
        raise RuntimeError("Missing index artifacts. Run indexing first.")

    meta = np.load(str(meta_path), allow_pickle=True)
    dim = int(meta["dim"]) if "dim" in meta else None

    index = faiss.read_index(str(index_path))

    # Open a temporary connection just to detect schema, then close it
    conn_tmp = db_connect(db_path)
    images_key, _, images_has_faiss_idx = _detect_images_schema(conn_tmp)
    colors_key = _detect_key_col(conn_tmp, "colors")
    captions_key = _detect_key_col(conn_tmp, "captions")
    conn_tmp.close()

    return db_path, index, images_key, images_has_faiss_idx, colors_key, captions_key, dim


def open_db(db_path: Path):
    """Open a fresh SQLite connection. Call this per-request, never cache the result."""
    return db_connect(db_path)


def _fetch_one(conn, sql: str, params: tuple) -> Optional[tuple]:
    row = conn.execute(sql, params).fetchone()
    return None if row is None else row


KeyVal = Union[int, str]


def _key_for_path(conn, images_key: str, path: Path) -> Optional[KeyVal]:
    row = _fetch_one(conn, f"SELECT {images_key} FROM images WHERE path=?;", (str(path),))
    if row is None:
        return None
    return row[0]


def _key_for_faiss_row(conn, images_key: str, images_has_faiss_idx: bool, faiss_row: int) -> Optional[KeyVal]:
    """
    Maps a FAISS row id to the images primary key value.

    Works with:
      A) legacy schema: images(idx INTEGER PRIMARY KEY)
      B) chunked schema: images(id TEXT PRIMARY KEY, faiss_idx INTEGER UNIQUE)
    """
    if images_has_faiss_idx:
        row = _fetch_one(conn, f"SELECT {images_key} FROM images WHERE faiss_idx=?;", (int(faiss_row),))
        if row is not None:
            return row[0]
        return None

    # Legacy: idx equals FAISS insertion order
    row = _fetch_one(conn, f"SELECT {images_key} FROM images WHERE {images_key}=?;", (int(faiss_row),))
    if row is not None:
        return row[0]

    # Common offset fallback
    row = _fetch_one(conn, f"SELECT {images_key} FROM images WHERE {images_key}=?;", (int(faiss_row) + 1,))
    if row is not None:
        return row[0]

    if faiss_row > 0:
        row = _fetch_one(conn, f"SELECT {images_key} FROM images WHERE {images_key}=?;", (int(faiss_row) - 1,))
        if row is not None:
            return row[0]

    return None


def fetch_path(conn, images_key: str, image_key_val: KeyVal) -> str:
    row = _fetch_one(conn, f"SELECT path FROM images WHERE {images_key}=?;", (image_key_val,))
    return "" if row is None else str(row[0] or "")


def fetch_color(conn, colors_key: Optional[str], image_key_val: KeyVal, color_dim: int) -> np.ndarray:
    if colors_key is None or not _has_table(conn, "colors"):
        return np.zeros((color_dim,), dtype=np.float32)

    row = _fetch_one(conn, f"SELECT vec FROM colors WHERE {colors_key}=?;", (image_key_val,))
    if row is None:
        return np.zeros((color_dim,), dtype=np.float32)

    try:
        return blob_to_np(row[0], color_dim)
    except Exception:
        return np.zeros((color_dim,), dtype=np.float32)


def fetch_caption(conn, captions_key: Optional[str], image_key_val: KeyVal) -> str:
    if captions_key is None or not _has_table(conn, "captions"):
        return ""
    row = _fetch_one(conn, f"SELECT caption FROM captions WHERE {captions_key}=?;", (image_key_val,))
    return "" if row is None else str(row[0] or "")


@dataclass
class PickResult:
    rank: int
    score: float
    path: str
    caption: str
    overlap_tokens: list[str]


@dataclass
class PickReport:
    seed_path: str
    seed_caption: str
    common_tokens: list[str]
    results: list[PickResult]


def pick_similar_cached(
    artifacts: tuple,
    emb: "Embedder",
    seed_path: Optional[Path] = None,
    n_results: int = 19,
    top_k: int = 500,
    w_clip: float = 0.75,
    text_prompt: str = "",
    w_text: float = 0.0,
    neg_prompt: str = "",
    dedupe: bool = True,
    dedupe_threshold: float = 0.97,
    seed_override_vec: Optional[np.ndarray] = None,
    seeds: Optional[List[Dict]] = None,
) -> PickReport:
    """Like pick_similar but accepts pre-loaded artifacts and embedder (for caching).

    artifacts must be the tuple from load_artifacts_cacheable (db_path, index, ...).
    A fresh DB connection is opened per call so this is thread-safe.

    Query modes (in priority order):
      - Multi-seed list:  seeds=[{"path": str, "w_concept": float}, ...] — weighted average
      - Override vec:     seed_override_vec set (pre-averaged multi-seed vector, legacy)
      - Image only:       seed_path set, text_prompt empty
      - Image + text:     seed_path set, text_prompt set, w_text controls blend
      - Text only:        seed_path None, text_prompt set

    For multi-seed mode each entry in `seeds` must have:
      "path"      — absolute path to the image file
      "w_concept" — relative weight for CLIP concept blending (will be normalised)
    Colour blending also uses w_concept as the per-seed colour weight.
    Text influence (w_text, 0-1) is applied on top of the blended image vector.
    """
    db_path, index, images_key, images_has_faiss_idx, colors_key, captions_key, _ = artifacts

    conn = open_db(db_path)
    try:
        text_prompt = (text_prompt or "").strip()
        has_text = bool(text_prompt)

        # ── Multi-seed mode ──────────────────────────────────────────────────
        if seeds and len(seeds) > 0:
            valid_seeds = [s for s in seeds if s.get("path") and Path(s["path"]).exists()]
            if not valid_seeds and not has_text:
                raise ValueError("No valid seed images found and no text prompt provided.")

            if valid_seeds:
                # Build per-seed CLIP vectors
                raw_weights = [max(float(s.get("w_concept", 1.0)), 1e-6) for s in valid_seeds]
                total_w = sum(raw_weights)
                norm_weights = [w / total_w for w in raw_weights]

                clip_vecs = []
                col_vecs: List[Optional[np.ndarray]] = []
                col_weights = []
                seed_paths_for_exclusion = []

                for s, nw in zip(valid_seeds, norm_weights):
                    sp = Path(s["path"]).expanduser().resolve()
                    seed_paths_for_exclusion.append(sp)
                    v = emb.embed_path(sp).astype(np.float32).flatten()
                    clip_vecs.append(nw * v)

                    raw_col = color_signature_from_path(sp, allow_video=False)
                    if raw_col is not None:
                        cn = np.linalg.norm(raw_col) + 1e-12
                        col_vecs.append((raw_col / cn).astype(np.float32))
                        col_weights.append(nw)
                    else:
                        col_vecs.append(None)

                # Weighted-average CLIP vector
                blended_clip = np.sum(clip_vecs, axis=0).astype(np.float32)
                cn = np.linalg.norm(blended_clip) + 1e-12
                blended_clip = blended_clip / cn

                # Blend with text if present
                if has_text:
                    txt_vec = emb.embed_text(text_prompt).astype(np.float32).flatten()
                    w_i = max(0.0, 1.0 - float(w_text))
                    combined = w_i * blended_clip + float(w_text) * txt_vec
                    norm_c = np.linalg.norm(combined) + 1e-12
                    seed_feat = (combined / norm_c).reshape(1, -1)
                else:
                    seed_feat = blended_clip.reshape(1, -1)

                # Weighted-average colour vector
                valid_col = [(cv, cw) for cv, cw in zip(col_vecs, col_weights) if cv is not None]
                if valid_col:
                    col_total_w = sum(cw for _, cw in valid_col)
                    seed_col_arr = np.sum([cv * (cw / col_total_w) for cv, cw in valid_col], axis=0).astype(np.float32)
                    col_n = np.linalg.norm(seed_col_arr) + 1e-12
                    seed_col: Optional[np.ndarray] = (seed_col_arr / col_n).astype(np.float32)
                    color_dim = int(seed_col.shape[0])
                else:
                    seed_col = None
                    color_dim = 4096

                has_image = True  # colour scoring active
                seed_path = None  # no single seed path to exclude (handled below)

            else:
                # Only text (no valid images)
                seed_feat = emb.embed_text(text_prompt).astype(np.float32).reshape(1, -1)
                seed_col = None
                color_dim = 4096
                has_image = False
                seed_paths_for_exclusion = []

            seed_caption = ""
            seed_toks: set = set()

        # ── Legacy override vec mode ─────────────────────────────────────────
        elif seed_override_vec is not None:
            vec = seed_override_vec.astype(np.float32).flatten()
            if has_text:
                txt_vec = emb.embed_text(text_prompt).astype(np.float32).flatten()
                w_i = max(0.0, 1.0 - float(w_text))
                combined = w_i * vec + float(w_text) * txt_vec
                norm = np.linalg.norm(combined) + 1e-12
                seed_feat = (combined / norm).reshape(1, -1)
            else:
                norm = np.linalg.norm(vec) + 1e-12
                seed_feat = (vec / norm).reshape(1, -1)
            has_image = False  # no single seed image for colour scoring
            seed_col = None
            color_dim = 4096
            seed_caption = ""
            seed_toks = set()
            seed_paths_for_exclusion = []

        # ── Single seed / text-only mode ─────────────────────────────────────
        else:
            has_image = seed_path is not None
            seed_paths_for_exclusion = [seed_path.expanduser().resolve()] if has_image else []

            if not has_image and not has_text:
                raise ValueError("Provide at least an image or a text prompt.")

            # Build query vector
            if has_image and has_text:
                seed_path = seed_path.expanduser().resolve()
                img_vec = emb.embed_path(seed_path).astype(np.float32)
                txt_vec = emb.embed_text(text_prompt).astype(np.float32)
                w_i = max(0.0, 1.0 - float(w_text))
                w_t = float(w_text)
                combined = w_i * img_vec + w_t * txt_vec
                norm = np.linalg.norm(combined) + 1e-12
                seed_feat = (combined / norm).reshape(1, -1)
            elif has_image:
                seed_path = seed_path.expanduser().resolve()
                seed_feat = emb.embed_path(seed_path).astype(np.float32).reshape(1, -1)
            else:
                seed_feat = emb.embed_text(text_prompt).astype(np.float32).reshape(1, -1)
                seed_path = None

            _raw_col = color_signature_from_path(seed_path, allow_video=False) if has_image else None
            if _raw_col is not None:
                _col_norm = np.linalg.norm(_raw_col) + 1e-12
                seed_col = (_raw_col / _col_norm).astype(np.float32)
            else:
                seed_col = None

            # Color baseline — only available when we have a seed image
            color_dim = 4096  # default LAB 16^3 bins
            if seed_col is not None:
                color_dim = int(seed_col.shape[0])

            seed_key = _key_for_path(conn, images_key, seed_path) if seed_path is not None else None
            seed_caption = fetch_caption(conn, captions_key, seed_key) if seed_key is not None else ""
            seed_toks = set(tokenize(seed_caption)) if seed_caption else set()

        # ── Negative prompt subtraction ──────────────────────────────────────
        # Embed the negative text and push the query vector away from it.
        # Strength is fixed at 0.5 — enough to steer without overwhelming the seed.
        neg_prompt = (neg_prompt or "").strip()
        if neg_prompt:
            neg_vec = emb.embed_text(neg_prompt).astype(np.float32).flatten()
            neg_vec /= (np.linalg.norm(neg_vec) + 1e-12)
            q = seed_feat.flatten()
            q = q - 0.5 * neg_vec
            norm_q = np.linalg.norm(q) + 1e-12
            seed_feat = (q / norm_q).reshape(1, -1)

        D, I = index.search(seed_feat, int(top_k))

        # Build set of resolved seed paths to exclude from results
        _excl_paths: set = set()
        if seed_path is not None:
            _excl_paths.add(seed_path.resolve() if hasattr(seed_path, "resolve") else Path(seed_path).resolve())
        for _sp in (seed_paths_for_exclusion if "seed_paths_for_exclusion" in dir() else []):
            if _sp is not None:
                _excl_paths.add(Path(_sp).resolve())

        scored: List[Tuple[float, str, str, list[str]]] = []
        for sim, faiss_row in zip(D[0].tolist(), I[0].tolist()):
            if faiss_row < 0:
                continue

            img_key = _key_for_faiss_row(conn, images_key, images_has_faiss_idx, int(faiss_row))
            if img_key is None:
                continue

            p = fetch_path(conn, images_key, img_key)
            if not p:
                continue

            if _excl_paths and Path(p).resolve() in _excl_paths:
                continue

            if seed_col is not None:
                col_raw = fetch_color(conn, colors_key, img_key, color_dim)
                col_norm = np.linalg.norm(col_raw) + 1e-12
                col = (col_raw / col_norm).astype(np.float32)
                col_sim = float(np.dot(seed_col, col))
            else:
                col_sim = 0.0

            clip_sim = float(sim)

            cap_text = fetch_caption(conn, captions_key, img_key)
            cand_toks = set(tokenize(cap_text))
            overlap = sorted(list(seed_toks.intersection(cand_toks)))[:12]

            # In text-only mode use clip_sim directly (no colour to blend)
            s = score_mix(clip_sim, col_sim, w_clip) if seed_col is not None else clip_sim
            scored.append((s, p, cap_text, overlap))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Near-duplicate filtering: for each candidate in rank order, skip it if
        # its CLIP vector is too similar to any already-accepted result.
        # Uses index.reconstruct() to retrieve stored vectors — no extra model calls.
        if dedupe and index.ntotal > 0:
            accepted_vecs: List[np.ndarray] = []
            top: List[Tuple[float, str, str, list[str]]] = []
            for entry in scored:
                if len(top) >= int(n_results):
                    break
                _, p, cap_text, overlap = entry
                score_val = entry[0]
                # Look up the FAISS row for this path to retrieve its stored vector
                img_key = _key_for_path(conn, images_key, Path(p))
                faiss_row: Optional[int] = None
                if img_key is not None:
                    if images_has_faiss_idx:
                        row = _fetch_one(conn, f"SELECT faiss_idx FROM images WHERE {images_key}=?;", (img_key,))
                        if row is not None and row[0] is not None:
                            faiss_row = int(row[0])
                    else:
                        # Legacy: key value is the FAISS row
                        try:
                            faiss_row = int(img_key)
                        except (TypeError, ValueError):
                            faiss_row = None

                is_dup = False
                if faiss_row is not None and 0 <= faiss_row < index.ntotal:
                    try:
                        vec = index.reconstruct(faiss_row).astype(np.float32)
                        vec_norm = np.linalg.norm(vec) + 1e-12
                        vec = vec / vec_norm
                        for av in accepted_vecs:
                            if float(np.dot(vec, av)) >= float(dedupe_threshold):
                                is_dup = True
                                break
                        if not is_dup:
                            accepted_vecs.append(vec)
                    except Exception:
                        pass  # reconstruct failed — accept the candidate

                if not is_dup:
                    top.append(entry)
        else:
            top = scored[: int(n_results)]

        common_tokens: list[str] = []
        all_tokens: list[str] = []
        for _, _, cap_text, _ in top:
            all_tokens.extend(tokenize(cap_text))
        if all_tokens:
            common_tokens = [w for w, _ in Counter(all_tokens).most_common(15)]

        results: list[PickResult] = []
        for i, (s, p, cap, overlap) in enumerate(top, start=1):
            results.append(PickResult(rank=i, score=float(s), path=str(p), caption=cap, overlap_tokens=overlap))

        return PickReport(
            seed_path=str(seed_path) if seed_path else "",
            seed_caption=seed_caption,
            common_tokens=common_tokens,
            results=results,
        )
    finally:
        conn.close()
