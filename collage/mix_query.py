"""
mix_query.py — shape-based image matching engine for the COLLAGE tab.

Combines shape feature similarity (from mix_index/shapes.sqlite) with
optional CLIP semantic similarity (from ai_index/clip.faiss via existing query.py).
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_APP = _HERE.parent
sys.path.insert(0, str(_APP))

from collage.shape_features import extract_all, shape_similarity
from collage.mix_compose import seam_continuity_score


# ── DB helpers ────────────────────────────────────────────────────────────────

def open_shape_db(mix_index_dir: Path) -> sqlite3.Connection:
    db = mix_index_dir / "shapes.sqlite"
    if not db.exists():
        raise RuntimeError(f"Shape index not found at {db}. Run BUILD SHAPE INDEX first.")
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def open_catalog_db(ai_index_dir: Path) -> sqlite3.Connection:
    db = ai_index_dir / "catalog.sqlite"
    if not db.exists():
        raise RuntimeError(f"catalog.sqlite not found at {db}.")
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def count_shape_indexed(mix_index_dir: Path) -> int:
    db = mix_index_dir / "shapes.sqlite"
    if not db.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT COUNT(*) FROM shape_edges;").fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _blob_to_np(b: bytes, dim: int) -> np.ndarray:
    x = np.frombuffer(b, dtype=np.float32)
    if x.size != dim:
        raise ValueError(f"dim mismatch: expected {dim}, got {x.size}")
    return x.copy()


def _fetch_shape_features(shape_conn: sqlite3.Connection, img_id: str) -> Optional[dict]:
    """Fetch all shape feature blobs for an image id. Returns None if missing."""
    e = shape_conn.execute("SELECT vec FROM shape_edges WHERE id=?;", (img_id,)).fetchone()
    if e is None:
        return None
    ci = shape_conn.execute("SELECT vec FROM shape_circles WHERE id=?;", (img_id,)).fetchone()
    fl = shape_conn.execute("SELECT vec FROM shape_flow WHERE id=?;", (img_id,)).fetchone()
    cp = shape_conn.execute("SELECT vec FROM shape_color_patches WHERE id=?;", (img_id,)).fetchone()

    try:
        return {
            "edges":         _blob_to_np(e[0], 256),
            "circle":        _blob_to_np(ci[0], 4) if ci else np.zeros(4, dtype=np.float32),
            "flow":          _blob_to_np(fl[0], 512) if fl else np.zeros(512, dtype=np.float32),
            "color_patches": _blob_to_np(cp[0], 27) if cp else np.zeros(27, dtype=np.float32),
        }
    except Exception:
        return None


# ── Catalog path lookup ───────────────────────────────────────────────────────

def _fetch_path_for_id(catalog_conn: sqlite3.Connection, img_id: str) -> Optional[str]:
    """Works with both 'id' (chunked) and 'idx' (legacy) schema."""
    row = catalog_conn.execute("SELECT path FROM images WHERE id=?;", (img_id,)).fetchone()
    if row:
        return str(row[0])
    # Legacy schema fallback — idx is integer, won't match a text sha1 id, safe to skip
    return None


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ShapeMatchResult:
    rank: int
    score: float
    path: str
    image_id: str
    shape_scores: dict  # {"edges": float, "circles": float, "flow": float, "color_patches": float}
    clip_score: float
    seam_score: float   # how well the match continues the seed across the seam


# ── Main matching function ────────────────────────────────────────────────────

def find_shape_matches(
    source_img: Image.Image,
    ai_index_dir: Path,
    mix_index_dir: Path,
    n_results: int = 20,
    weights: Optional[dict] = None,
    clip_weight: float = 0.3,
    seam_weight: float = 0.3,
    seam_angle_deg: float = 0.0,
    seam_position: float = 0.5,
    use_clip_prefilter: bool = True,
    clip_prefilter_k: int = 2000,
    artifacts: Optional[tuple] = None,
    emb=None,
) -> list[ShapeMatchResult]:
    """
    Find images in the shape index that best match source_img.

    Parameters
    ----------
    source_img       : PIL RGB image (the query)
    ai_index_dir     : path to ai_index (catalog.sqlite + clip.faiss)
    mix_index_dir    : path to mix_index (shapes.sqlite)
    n_results        : number of results to return
    weights          : dict with keys "edges", "circles", "flow", "color_patches"
                       values are floats 0-1 (feature weights for shape scoring)
    clip_weight      : how much CLIP semantic score contributes (0 = shape only)
    seam_weight      : how much seam continuity score contributes (0 = ignore seam)
    seam_angle_deg   : split line angle used for seam continuity scoring
    seam_position    : split line position used for seam continuity scoring
    use_clip_prefilter: if True and artifacts+emb provided, first restrict candidates
                        to top clip_prefilter_k by CLIP similarity
    artifacts        : result of load_artifacts_cacheable (for CLIP scores)
    emb              : Embedder instance

    Returns
    -------
    List of ShapeMatchResult sorted by descending score.
    """
    if weights is None:
        weights = {"edges": 0.4, "circles": 0.3, "flow": 0.2, "color_patches": 0.1}

    source_img = source_img.convert("RGB")
    source_feats = extract_all(source_img)

    shape_conn = open_shape_db(mix_index_dir)
    catalog_conn = open_catalog_db(ai_index_dir)

    # ── Optional CLIP pre-filter ──────────────────────────────────────────────
    clip_scores: dict[str, float] = {}

    if use_clip_prefilter and artifacts is not None and emb is not None and clip_weight > 0:
        try:
            import faiss
            db_path, index, images_key, images_has_faiss_idx, *_ = artifacts

            src_vec = emb.embed_pil(source_img).astype(np.float32).reshape(1, -1)
            D, I = index.search(src_vec, min(int(clip_prefilter_k), int(index.ntotal)))

            # Build id→clip_score map using catalog
            for sim, faiss_row in zip(D[0].tolist(), I[0].tolist()):
                if faiss_row < 0:
                    continue
                if images_has_faiss_idx:
                    row = catalog_conn.execute(
                        f"SELECT {images_key} FROM images WHERE faiss_idx=?;",
                        (int(faiss_row),),
                    ).fetchone()
                else:
                    row = catalog_conn.execute(
                        f"SELECT {images_key} FROM images WHERE {images_key}=?;",
                        (int(faiss_row),),
                    ).fetchone()
                if row:
                    clip_scores[str(row[0])] = float(sim)

            candidate_ids = list(clip_scores.keys())
        except Exception:
            candidate_ids = None
    else:
        candidate_ids = None

    # ── Load candidates from shape index ─────────────────────────────────────
    if candidate_ids is not None:
        # Only score CLIP pre-filtered candidates
        rows = [(cid,) for cid in candidate_ids]
    else:
        # Score everything in the shape index
        rows = shape_conn.execute("SELECT id FROM shape_edges;").fetchall()

    # ── Score each candidate ──────────────────────────────────────────────────
    _sw = float(seam_weight)
    _cw = float(clip_weight)
    w_shape = max(0.0, 1.0 - _cw - _sw)
    source_arr = np.array(source_img.convert("RGB"), dtype=np.uint8)
    scored: list[tuple[float, str, dict, float, float]] = []

    for (img_id,) in rows:
        cand_feats = _fetch_shape_features(shape_conn, str(img_id))
        if cand_feats is None:
            continue

        # Per-feature scores for transparency
        individual = {}
        for feat_key, w_key in [
            ("edges", "edges"),
            ("circle", "circles"),
            ("flow", "flow"),
            ("color_patches", "color_patches"),
        ]:
            a = source_feats.get(feat_key)
            b = cand_feats.get(feat_key)
            if a is not None and b is not None:
                na = float(np.linalg.norm(a)) + 1e-12
                nb = float(np.linalg.norm(b)) + 1e-12
                individual[w_key] = float(np.dot(a / na, b / nb))
            else:
                individual[w_key] = 0.0

        shape_score = shape_similarity(source_feats, cand_feats, weights)
        clip_score = clip_scores.get(str(img_id), 0.0)

        # Seam continuity — load the candidate image and score at the seam
        s_score = 0.0
        if _sw > 0:
            cand_path = _fetch_path_for_id(catalog_conn, str(img_id))
            if cand_path:
                try:
                    cand_img = Image.open(cand_path).convert("RGB")
                    cand_arr = np.array(cand_img.resize(
                        (source_arr.shape[1], source_arr.shape[0]), Image.BILINEAR
                    ), dtype=np.uint8)
                    s_score = seam_continuity_score(
                        source_arr, cand_arr,
                        angle_deg=seam_angle_deg,
                        position=seam_position,
                    )
                except Exception:
                    s_score = 0.0

        total = w_shape * shape_score + _cw * clip_score + _sw * s_score
        scored.append((total, str(img_id), individual, clip_score, s_score))

    shape_conn.close()

    # Sort descending
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n_results]

    results: list[ShapeMatchResult] = []
    for rank, (score, img_id, shape_scores, clip_score, seam_score) in enumerate(top, start=1):
        path = _fetch_path_for_id(catalog_conn, img_id) or ""
        results.append(ShapeMatchResult(
            rank=rank,
            score=float(score),
            path=path,
            image_id=img_id,
            shape_scores=shape_scores,
            clip_score=clip_score,
            seam_score=float(seam_score),
        ))

    catalog_conn.close()
    return results
