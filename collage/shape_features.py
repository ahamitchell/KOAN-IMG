from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


def _to_gray_arr(img: Image.Image, size: int) -> np.ndarray:
    """Resize to size x size and convert to uint8 grayscale array."""
    img = img.resize((size, size), Image.BILINEAR).convert("L")
    return np.array(img, dtype=np.uint8)


def _to_rgb_arr(img: Image.Image, size: int) -> np.ndarray:
    img = img.resize((size, size), Image.BILINEAR).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x)) + eps
    return (x / n).astype(np.float32)


# ── Edge density vector ───────────────────────────────────────────────────────

def edge_density_vec(img: Image.Image, grid: int = 16) -> np.ndarray:
    """
    Canny edge map divided into grid x grid cells.
    Returns float32[grid*grid] L2-normalised edge density per cell.
    """
    arr = _to_gray_arr(img, grid * 16)  # 256x256
    blurred = cv2.GaussianBlur(arr, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100).astype(np.float32) / 255.0

    cell = edges.shape[0] // grid
    vec = np.zeros(grid * grid, dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            patch = edges[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            vec[r * grid + c] = float(patch.mean())

    return _l2(vec)


# ── Circle feature ────────────────────────────────────────────────────────────

def circle_feature(img: Image.Image) -> np.ndarray:
    """
    HoughCircles — returns best circle as [cx_norm, cy_norm, r_norm, conf] float32[4].
    All values normalised 0-1 relative to image size.
    Returns zeros if no circle found.
    """
    arr = _to_gray_arr(img, 256)
    blurred = cv2.GaussianBlur(arr, (5, 5), 1)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=60,
        param2=30,
        minRadius=10,
        maxRadius=120,
    )
    if circles is None:
        return np.zeros(4, dtype=np.float32)

    # Pick the circle with largest radius (most dominant)
    c = circles[0]
    best = c[np.argmax(c[:, 2])]
    cx, cy, r = best[0], best[1], best[2]
    conf = min(1.0, float(r) / 128.0)  # larger circle = higher conf
    return np.array(
        [cx / 256.0, cy / 256.0, r / 128.0, conf],
        dtype=np.float32,
    )


# ── Flow / gradient orientation vector ───────────────────────────────────────

def flow_vec(img: Image.Image, grid: int = 8) -> np.ndarray:
    """
    Sobel gradient orientations binned into 8 directions, computed per cell.
    Returns float32[grid*grid*8] = float32[512] → actually we return grid*grid dominant bin
    as float32[grid*grid] for simplicity, or the full 8-bin histogram per cell.

    We return the full 8-bin histogram per cell → float32[grid*grid*8] L2-normalised.
    """
    arr = _to_gray_arr(img, grid * 16)  # 128x128
    sobelx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(sobely, sobelx)  # -pi..pi
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Bin angles into 8 orientations (0..7)
    bins = ((angle + np.pi) / (2 * np.pi) * 8).astype(np.int32) % 8

    cell = arr.shape[0] // grid
    n_bins = 8
    vec = np.zeros(grid * grid * n_bins, dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            ms = mag[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            bs = bins[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            hist = np.zeros(n_bins, dtype=np.float32)
            for b in range(n_bins):
                hist[b] = float(ms[bs == b].sum())
            cell_idx = r * grid + c
            vec[cell_idx * n_bins:(cell_idx + 1) * n_bins] = hist

    return _l2(vec)


# ── Color patch vector ────────────────────────────────────────────────────────

def color_patch_vec(img: Image.Image, grid: int = 3) -> np.ndarray:
    """
    Divide image into grid x grid cells, compute mean LAB [L, a, b] per cell.
    Returns float32[grid*grid*3] L2-normalised.
    """
    arr = _to_rgb_arr(img, grid * 32)  # 96x96
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)

    cell = arr.shape[0] // grid
    vec = np.zeros(grid * grid * 3, dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            patch = lab[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            mean_lab = patch.reshape(-1, 3).mean(axis=0)
            idx = (r * grid + c) * 3
            vec[idx:idx + 3] = mean_lab

    return _l2(vec)


# ── Contour keypoints ─────────────────────────────────────────────────────────

def dominant_contour_points(img: Image.Image, n_points: int = 20) -> np.ndarray | None:
    """
    Find the largest contour in the image and return n_points evenly-sampled
    points along it as float32[n_points, 2] in 0-1 normalised coordinates.
    Returns None if no contour found.
    """
    arr = _to_gray_arr(img, 512)
    blurred = cv2.GaussianBlur(arr, (5, 5), 1)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None

    # Evenly sample n_points along the contour
    n = len(largest)
    indices = np.linspace(0, n - 1, n_points, dtype=int)
    pts = largest[indices].reshape(-1, 2).astype(np.float32)
    pts[:, 0] /= 512.0
    pts[:, 1] /= 512.0
    return pts


def dominant_centroid(img: Image.Image) -> tuple[float, float]:
    """
    Returns (cx_norm, cy_norm) of the largest contour centroid, normalised 0-1.
    Falls back to image centre if no contour found.
    """
    arr = _to_gray_arr(img, 256)
    blurred = cv2.GaussianBlur(arr, (5, 5), 1)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"] / 256.0
            cy = M["m01"] / M["m00"] / 256.0
            return float(cx), float(cy)

    return 0.5, 0.5


# ── Combined extraction ───────────────────────────────────────────────────────

def extract_all(img: Image.Image) -> dict:
    """
    Extract all shape features from a PIL RGB image.
    Returns:
      {
        "edges":         np.float32[256]  — edge density grid
        "circle":        np.float32[4]    — best Hough circle [cx,cy,r,conf]
        "flow":          np.float32[512]  — gradient orientation histograms
        "color_patches": np.float32[27]   — mean LAB per 3x3 cell
      }
    """
    img = img.convert("RGB")
    return {
        "edges":         edge_density_vec(img),
        "circle":        circle_feature(img),
        "flow":          flow_vec(img),
        "color_patches": color_patch_vec(img),
    }


# ── Similarity ────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) + 1e-12
    nb = float(np.linalg.norm(b)) + 1e-12
    return float(np.dot(a / na, b / nb))


def shape_similarity(feat_a: dict, feat_b: dict, weights: dict) -> float:
    """
    Weighted cosine similarity between two feature dicts.
    weights keys: "edges", "circles", "flow", "color_patches"
    Any key not present in weights (or weight=0) is skipped.
    Returns float in [-1, 1].
    """
    total_w = 0.0
    score = 0.0

    for key, wkey in [("edges", "edges"), ("circle", "circles"), ("flow", "flow"), ("color_patches", "color_patches")]:
        w = float(weights.get(wkey, 0.0))
        if w <= 0.0:
            continue
        a = feat_a.get(key)
        b = feat_b.get(key)
        if a is None or b is None:
            continue
        score += w * _cosine(a, b)
        total_w += w

    if total_w <= 0.0:
        return 0.0
    return score / total_w
