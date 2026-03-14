"""
mix_compose.py — seam-split collage compositor.

Core concept:
  - A split line (angle + position) divides both images
  - Seed fills one side, match fills the other
  - Match is scaled + rotated so its content aligns at the seam
  - Seam can be hard cut or softly feathered (user-controlled)
  - Output is always seed's native size

Split line is defined by:
  angle_deg  — 0 = vertical split, 90 = horizontal, 45 = diagonal
  position   — 0.0..1.0 where the line crosses the image centre axis
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# ── File type constants ───────────────────────────────────────────────────────

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


# ── Frame extraction ──────────────────────────────────────────────────────────

def is_animated(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in SUPPORTED_VIDEO_EXTS:
        return True
    try:
        im = Image.open(path)
        return bool(getattr(im, "is_animated", False)) and int(getattr(im, "n_frames", 1)) > 1
    except Exception:
        return False


def get_frame_duration_ms(path: Path) -> int:
    ext = path.suffix.lower()
    if ext in SUPPORTED_VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        cap.release()
        return max(33, int(1000.0 / fps))
    try:
        im = Image.open(path)
        return max(33, int(im.info.get("duration", 100)))
    except Exception:
        return 100


def extract_frames(path: Path, max_frames: int = 48) -> list[Image.Image]:
    ext = path.suffix.lower()
    if ext in SUPPORTED_VIDEO_EXTS:
        frames: list[Image.Image] = []
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, total // max_frames)
        idx = 0
        while len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB"))
            idx += step
            if idx >= total:
                break
        cap.release()
        return frames if frames else [_blank()]
    try:
        im = Image.open(path)
        frames = []
        try:
            while True:
                frames.append(im.copy().convert("RGB"))
                if len(frames) >= max_frames:
                    break
                im.seek(im.tell() + 1)
        except EOFError:
            pass
        return frames if frames else [_blank()]
    except Exception:
        return [_blank()]


def _blank(size: tuple[int, int] = (256, 256)) -> Image.Image:
    return Image.new("RGB", size, (20, 20, 20))


# ── Output assembly ───────────────────────────────────────────────────────────

def frames_to_gif(frames: list[Image.Image], duration_ms: int = 100) -> bytes:
    buf = io.BytesIO()
    if not frames:
        return b""
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)
    return buf.getvalue()


def frames_to_mp4(frames: list[Image.Image], fps: float = 10.0) -> bytes:
    if not frames:
        return b""
    w, h = frames[0].size
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2BGR))
    writer.release()
    with open(tmp.name, "rb") as fh:
        data = fh.read()
    Path(tmp.name).unlink(missing_ok=True)
    return data


# ── Split line helpers ────────────────────────────────────────────────────────

def suggest_split_angle(img: Image.Image) -> float:
    """
    Analyse the seed image and suggest the most natural split angle in degrees.
    0   = vertical split   (left | right)
    90  = horizontal split (top | bottom)
    Uses dominant edge orientation via Hough line detection.
    Falls back to 0 (vertical) if nothing clear found.
    """
    arr = np.array(img.convert("L").resize((512, 512), Image.BILINEAR), dtype=np.uint8)
    blurred = cv2.GaussianBlur(arr, (5, 5), 1)
    edges = cv2.Canny(blurred, 30, 100)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)
    if lines is None or len(lines) == 0:
        return 0.0

    # Collect angles of all detected lines
    angles = [line[0][1] for line in lines]  # theta in radians
    # Convert to degrees, map to 0..180
    degs = [float(np.degrees(a)) % 180 for a in angles]

    # Find the most common angle cluster (bin into 10-degree buckets)
    bins = np.zeros(18, dtype=int)
    for d in degs:
        bins[int(d // 10) % 18] += 1
    dominant_bin = int(np.argmax(bins))
    dominant_angle = dominant_bin * 10.0 + 5.0  # centre of bin

    # Convert from Hough angle convention to split angle:
    # Hough theta=0 means vertical line → split_angle=0 (vertical split)
    # Hough theta=90 means horizontal line → split_angle=90 (horizontal split)
    # We want perpendicular to the dominant edge direction
    split_angle = (dominant_angle + 90.0) % 180.0
    return round(split_angle, 1)


def build_split_mask(
    width: int,
    height: int,
    angle_deg: float,
    position: float,
    feather_px: int = 0,
) -> np.ndarray:
    """
    Build a float32 mask (h, w) where:
      1.0 = seed side  (keep seed pixel)
      0.0 = match side (keep match pixel)

    angle_deg: 0 = vertical line (seed left, match right)
               90 = horizontal line (seed top, match bottom)
               any angle in between
    position: 0.0..1.0 — where the line crosses the image
               0.5 = dead centre, <0.5 shifts toward top/left, >0.5 toward bottom/right
    feather_px: pixels of crossfade either side of the line (0 = hard cut)
    """
    # Build coordinate grids
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    # Normalise to -0.5..0.5
    xn = xs / width - 0.5
    yn = ys / height - 0.5

    # Rotate coordinate system by angle
    rad = np.deg2rad(float(angle_deg))
    # Project onto the axis perpendicular to the split line
    proj = xn * np.cos(rad) + yn * np.sin(rad)

    # Position offset: map 0..1 → -0.5..0.5
    offset = float(position) - 0.5

    # Signed distance from the split line
    dist = proj - offset

    if feather_px <= 0:
        # Hard cut
        mask = (dist <= 0).astype(np.float32)
    else:
        # Soft feather: linear ramp across feather_px pixels
        # Scale dist to pixels (approximate)
        px_scale = max(width, height)
        dist_px = dist * px_scale
        mask = np.clip(0.5 - dist_px / (feather_px + 1e-6) * 0.5, 0.0, 1.0)

    return mask.astype(np.float32)


# ── Seam-region feature extraction ───────────────────────────────────────────

def extract_seam_strip(
    arr: np.ndarray,
    angle_deg: float,
    position: float,
    strip_width: float = 0.15,
) -> np.ndarray:
    """
    Extract a normalised feature vector from the strip of pixels near the seam line
    on a given side. Used for seam-continuity scoring.

    Returns float32[48] — 3x4 grid of mean LAB colours in the strip.
    """
    h, w = arr.shape[:2]
    mask = build_split_mask(w, h, angle_deg, position, feather_px=0)

    # Strip: pixels within strip_width of the seam
    # Thin the mask to just the band near the boundary
    strip_half = max(2, int(strip_width * min(w, h) / 2))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xs / w - 0.5
    yn = ys / h - 0.5
    rad = np.deg2rad(float(angle_deg))
    proj = xn * np.cos(rad) + yn * np.sin(rad)
    offset = float(position) - 0.5
    dist_px = np.abs((proj - offset) * max(w, h))
    strip_mask = (dist_px < strip_half)

    if strip_mask.sum() < 10:
        return np.zeros(48, dtype=np.float32)

    # Divide the strip into a 3x4 grid along the seam
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    grid_rows, grid_cols = 3, 4
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    vec = np.zeros(grid_rows * grid_cols * 3, dtype=np.float32)
    for r in range(grid_rows):
        for c in range(grid_cols):
            r0, r1 = r * cell_h, (r + 1) * cell_h
            c0, c1 = c * cell_w, (c + 1) * cell_w
            cell_strip = strip_mask[r0:r1, c0:c1]
            if cell_strip.sum() < 3:
                continue
            cell_lab = lab[r0:r1, c0:c1]
            mean_lab = cell_lab[cell_strip].mean(axis=0)
            idx = (r * grid_cols + c) * 3
            vec[idx:idx + 3] = mean_lab

    n = float(np.linalg.norm(vec)) + 1e-12
    return (vec / n).astype(np.float32)


def seam_continuity_score(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    angle_deg: float,
    position: float,
) -> float:
    """
    Score how well arr_b continues arr_a across the seam.
    Compares the colour/texture strip on the seed side with the
    corresponding strip on the match side (mirror side of the seam).
    Returns cosine similarity in [-1, 1].
    """
    # Seed strip: the side that will show (seed side of seam)
    vec_a = extract_seam_strip(arr_a, angle_deg, position, strip_width=0.12)
    # Match strip: the opposite side (what the match brings to the seam)
    # Mirror the position for the match side
    mirror_pos = 1.0 - float(position)
    vec_b = extract_seam_strip(arr_b, angle_deg, mirror_pos, strip_width=0.12)

    na = float(np.linalg.norm(vec_a)) + 1e-12
    nb = float(np.linalg.norm(vec_b)) + 1e-12
    return float(np.dot(vec_a / na, vec_b / nb))


# ── Match alignment transform ─────────────────────────────────────────────────

def align_match_to_seed(
    arr_seed: np.ndarray,
    arr_match: np.ndarray,
    angle_deg: float,
    position: float,
) -> np.ndarray:
    """
    Scale and rotate arr_match so that its content aligns with arr_seed
    across the seam line. Returns the transformed match array at seed size.

    Strategy:
    1. Both images are brought to seed size
    2. We compute the dominant edge orientation on each side of the seam
    3. Apply affine transform (scale + rotation) to register match onto seed
    """
    out_h, out_w = arr_seed.shape[:2]

    # Resize match to seed size as starting point
    match_resized = cv2.resize(arr_match, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    # Build masks for each side
    mask_seed_side = build_split_mask(out_w, out_h, angle_deg, position, feather_px=0)
    mask_match_side = 1.0 - mask_seed_side

    # Find keypoints on each image using ORB, restricted to the relevant side
    gray_seed = cv2.cvtColor(arr_seed, cv2.COLOR_RGB2GRAY)
    gray_match = cv2.cvtColor(match_resized, cv2.COLOR_RGB2GRAY)

    # Mask the keypoint regions to the seam strip (20% either side)
    strip_half = int(0.20 * min(out_w, out_h))
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    xn = xs / out_w - 0.5
    yn = ys / out_h - 0.5
    rad = np.deg2rad(float(angle_deg))
    proj = xn * np.cos(rad) + yn * np.sin(rad)
    offset = float(position) - 0.5
    dist_px = np.abs((proj - offset) * max(out_w, out_h))
    seam_strip = (dist_px < strip_half).astype(np.uint8) * 255

    orb = cv2.ORB_create(nfeatures=500)
    kp_a, des_a = orb.detectAndCompute(gray_seed, seam_strip)
    kp_b, des_b = orb.detectAndCompute(gray_match, seam_strip)

    if (des_a is not None and des_b is not None
            and len(kp_a) >= 4 and len(kp_b) >= 4):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_a, des_b)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:min(40, len(matches))]

        if len(good) >= 4:
            pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
            pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
            # Estimate similarity transform (scale + rotation + translation)
            M, inliers = cv2.estimateAffinePartial2D(
                pts_b, pts_a,
                method=cv2.RANSAC,
                ransacReprojThreshold=8.0,
            )
            if M is not None and inliers is not None and inliers.sum() >= 3:
                aligned = cv2.warpAffine(
                    match_resized, M, (out_w, out_h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT,
                )
                return aligned

    # Fallback: return resized match as-is
    return match_resized


# ── Core compositor ───────────────────────────────────────────────────────────

def compose_seam_split(
    img_a: Image.Image,      # seed — defines output size and one side
    img_b: Image.Image,      # match — fills the other side
    angle_deg: float = 0.0,  # split line angle: 0=vertical, 90=horizontal
    position: float = 0.5,   # 0.0..1.0 position of split line
    feather_px: int = 0,     # 0=hard cut, >0=soft blend at seam
    align: bool = True,      # whether to scale+rotate match to align at seam
) -> Image.Image:
    """
    Seam-split compositor.
    Seed fills one side of the split line, match fills the other.
    Output is seed's native size.
    """
    img_a = img_a.convert("RGB")
    img_b = img_b.convert("RGB")
    out_w, out_h = img_a.size

    arr_a = np.array(img_a, dtype=np.uint8)

    # Resize match to seed dimensions
    arr_b_raw = np.array(img_b.resize((out_w, out_h), Image.LANCZOS), dtype=np.uint8)

    # Optionally align match to seed at the seam
    if align:
        arr_b = align_match_to_seed(arr_a, arr_b_raw, angle_deg, position)
    else:
        arr_b = arr_b_raw

    # Build split mask
    mask = build_split_mask(out_w, out_h, angle_deg, position, feather_px=feather_px)

    # Composite: seed where mask=1, match where mask=0
    m = mask[..., np.newaxis].astype(np.float32)
    a = arr_a.astype(np.float32)
    b = arr_b.astype(np.float32)
    result = (a * m + b * (1.0 - m)).clip(0, 255).astype(np.uint8)

    return Image.fromarray(result)


# ── Ellipse / circle mask ─────────────────────────────────────────────────────

def build_ellipse_mask(
    width: int,
    height: int,
    cx: float,       # centre x, 0..1
    cy: float,       # centre y, 0..1
    rx: float,       # x radius, 0..1 relative to width
    ry: float,       # y radius, 0..1 relative to height
    feather_px: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """
    Build float32 mask (h, w):
      1.0 inside ellipse (seed shows here) if invert=False
      0.0 inside ellipse (match shows here) if invert=True
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = (xs / width  - cx) / max(rx, 1e-6)
    yn = (ys / height - cy) / max(ry, 1e-6)
    dist = np.sqrt(xn ** 2 + yn ** 2)  # 0=centre, 1=edge, >1=outside

    if feather_px <= 0:
        inside = (dist <= 1.0).astype(np.float32)
    else:
        feather_norm = feather_px / max(width, height)
        inside = np.clip(1.0 - (dist - 1.0) / (feather_norm + 1e-6), 0.0, 1.0)

    return (1.0 - inside) if invert else inside


def build_contour_mask(
    arr: np.ndarray,          # source image to extract contour from
    canvas_w: int,
    canvas_h: int,
    cx: float = 0.5,          # target centre x 0..1
    cy: float = 0.5,          # target centre y 0..1
    scale: float = 1.0,       # scale the contour
    feather_px: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """
    Extract dominant contour from arr, place it centred at (cx,cy) on a
    canvas_w x canvas_h canvas, fill and optionally feather it.
    Falls back to ellipse mask if no contour found.
    """
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 25, 80)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return build_ellipse_mask(canvas_w, canvas_h, cx, cy, 0.35, 0.35,
                                  feather_px=feather_px, invert=invert)

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return build_ellipse_mask(canvas_w, canvas_h, cx, cy, 0.35, 0.35,
                                  feather_px=feather_px, invert=invert)

    # Contour centroid
    orig_cx = M["m10"] / M["m00"]
    orig_cy = M["m01"] / M["m00"]

    # Scale to canvas and shift centroid to (cx, cy)
    target_cx = cx * canvas_w
    target_cy = cy * canvas_h
    sx = canvas_w / w * scale
    sy = canvas_h / h * scale

    pts = largest.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = (pts[:, 0] - orig_cx) * sx + target_cx
    pts[:, 1] = (pts[:, 1] - orig_cy) * sy + target_cy
    pts = pts.astype(np.int32).reshape(-1, 1, 2)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.drawContours(canvas, [pts], -1, 255, thickness=cv2.FILLED)

    if feather_px > 0:
        k = feather_px * 2 + 1
        canvas_f = cv2.GaussianBlur(canvas.astype(np.float32), (k, k), feather_px / 2.0)
    else:
        canvas_f = canvas.astype(np.float32)

    mask = (canvas_f / 255.0).clip(0.0, 1.0)
    return (1.0 - mask) if invert else mask


# ── Mask suggestion ───────────────────────────────────────────────────────────

def suggest_mask(
    img_a: Image.Image,    # seed
    img_b: Image.Image,    # match candidate
) -> dict:
    """
    Analyse both images and suggest the best mask type and parameters.

    Returns a dict:
    {
        "type":     "line" | "circle" | "ellipse" | "contour"
        "angle":    float   (line only)
        "position": float   (line only, 0..1)
        "cx":       float   (shape masks, 0..1)
        "cy":       float   (shape masks, 0..1)
        "rx":       float   (ellipse x-radius 0..1)
        "ry":       float   (ellipse y-radius 0..1)
        "reason":   str     (human-readable explanation)
    }
    """
    arr_a = np.array(img_a.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)
    arr_b = np.array(img_b.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)

    gray_a = cv2.cvtColor(arr_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(arr_b, cv2.COLOR_RGB2GRAY)

    # ── Check for dominant circle in either image ─────────────────────────────
    def _best_circle(gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=30, param1=60, param2=28,
                                   minRadius=20, maxRadius=110)
        if circles is not None:
            c = circles[0]
            best = c[np.argmax(c[:, 2])]
            return best[0] / 256.0, best[1] / 256.0, best[2] / 128.0  # cx,cy,r norm
        return None

    ca = _best_circle(gray_a)
    cb = _best_circle(gray_b)

    if ca and ca[2] > 0.15:
        # Strong circle in seed — use circle mask centred on it
        return {"type": "circle", "cx": ca[0], "cy": ca[1],
                "rx": ca[2] * 0.9, "ry": ca[2] * 0.9,
                "angle": 0.0, "position": 0.5,
                "reason": "Strong circular shape detected in seed"}

    if cb and cb[2] > 0.15:
        # Strong circle in match — use circle mask centred on seed's centre
        return {"type": "circle", "cx": 0.5, "cy": 0.5,
                "rx": cb[2] * 0.9, "ry": cb[2] * 0.9,
                "angle": 0.0, "position": 0.5,
                "reason": "Strong circular shape detected in match"}

    # ── Check for dominant contour shape ──────────────────────────────────────
    def _contour_roundness(gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        edges = cv2.Canny(blurred, 25, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 200:
            return 0.0, None
        perimeter = cv2.arcLength(c, True)
        roundness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        return float(roundness), c

    rnd_a, cnt_a = _contour_roundness(gray_a)
    rnd_b, cnt_b = _contour_roundness(gray_b)

    # Roundness: 1.0=perfect circle, lower=elongated/complex
    if rnd_b > 0.5 and cnt_b is not None:
        # Fairly round shape in match → suggest ellipse centred on seed's main shape
        x, y, w, h = cv2.boundingRect(cnt_b)
        return {"type": "ellipse",
                "cx": 0.5, "cy": 0.5,
                "rx": (w / 256.0) * 0.5, "ry": (h / 256.0) * 0.5,
                "angle": 0.0, "position": 0.5,
                "reason": "Rounded shape in match — using ellipse mask"}

    if rnd_b > 0.2 and cnt_b is not None:
        # Irregular but clear shape → contour mask
        return {"type": "contour",
                "cx": 0.5, "cy": 0.5,
                "rx": 0.4, "ry": 0.4,
                "angle": 0.0, "position": 0.5,
                "reason": "Distinct irregular shape in match — using contour mask"}

    # ── Default: line split along dominant edge direction ────────────────────
    angle = suggest_split_angle(img_a)
    return {"type": "line",
            "angle": angle, "position": 0.5,
            "cx": 0.5, "cy": 0.5, "rx": 0.4, "ry": 0.4,
            "reason": f"Line split at {angle:.0f}° following dominant edges"}


# ── Mask thumbnail for results grid ──────────────────────────────────────────

def render_mask_thumbnail(
    img_a: Image.Image,
    img_b: Image.Image,
    mask_params: dict,
    size: int = 120,
) -> Image.Image:
    """
    Render a thumbnail of the match image (img_b) with the mask boundary
    drawn on top as a KOAN-green line. The seed (img_a) does NOT appear —
    the boundary shows where the cut will be made on the match.
    Returns a square PIL RGB image of `size` x `size`.
    """
    w = h = size
    feather = max(1, size // 30)

    mtype = mask_params.get("type", "line")

    if mtype == "line":
        mask = build_split_mask(w, h,
                                angle_deg=float(mask_params.get("angle", 0.0)),
                                position=float(mask_params.get("position", 0.5)),
                                feather_px=feather)
    elif mtype in ("circle", "ellipse"):
        mask = build_ellipse_mask(w, h,
                                  cx=float(mask_params.get("cx", 0.5)),
                                  cy=float(mask_params.get("cy", 0.5)),
                                  rx=float(mask_params.get("rx", 0.35)),
                                  ry=float(mask_params.get("ry", 0.35)),
                                  feather_px=feather,
                                  invert=False)
    else:  # contour
        arr_b = np.array(img_b.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.uint8)
        mask = build_contour_mask(arr_b, w, h,
                                  cx=float(mask_params.get("cx", 0.5)),
                                  cy=float(mask_params.get("cy", 0.5)),
                                  feather_px=feather)

    # Base: match image only, dimmed slightly outside the mask area
    thumb_b = np.array(img_b.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32)
    m = mask[..., np.newaxis]
    # Brighten the "match" side (where match will show), dim the "seed" side
    composite = (thumb_b * (1.0 - m) + thumb_b * m * 0.45).clip(0, 255).astype(np.uint8)

    # Draw the mask boundary as a bright KOAN-green line
    result = Image.fromarray(composite)
    mask_u8 = (mask * 255).astype(np.uint8)
    boundary = cv2.Canny(mask_u8, 50, 150)
    # Dilate boundary by 1px so it's visible at small sizes
    boundary = cv2.dilate(boundary, np.ones((2, 2), dtype=np.uint8), iterations=1)
    boundary_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    boundary_rgb[boundary > 0] = [64, 255, 120]  # KOAN green
    boundary_img = Image.fromarray(boundary_rgb)
    boundary_mask = Image.fromarray(boundary).convert("L")
    result.paste(boundary_img, mask=boundary_mask)

    return result


# ── compose_with_mask — unified compositor supporting all mask types ──────────

def compose_with_mask(
    img_a: Image.Image,
    img_b: Image.Image,
    mask_params: dict,
    feather_px: int = 0,
    align: bool = True,
) -> Image.Image:
    """
    Composite seed (img_a) and match (img_b) using mask_params.
    mask_params keys: type, angle, position, cx, cy, rx, ry
    """
    img_a = img_a.convert("RGB")
    img_b = img_b.convert("RGB")
    out_w, out_h = img_a.size
    arr_a = np.array(img_a, dtype=np.uint8)

    mtype = mask_params.get("type", "line")

    if mtype == "line":
        angle = float(mask_params.get("angle", 0.0))
        pos   = float(mask_params.get("position", 0.5))
        arr_b_raw = np.array(img_b.resize((out_w, out_h), Image.LANCZOS), dtype=np.uint8)
        arr_b = align_match_to_seed(arr_a, arr_b_raw, angle, pos) if align else arr_b_raw
        mask = build_split_mask(out_w, out_h, angle, pos, feather_px=feather_px)

    elif mtype in ("circle", "ellipse"):
        arr_b = np.array(img_b.resize((out_w, out_h), Image.LANCZOS), dtype=np.uint8)
        mask = build_ellipse_mask(out_w, out_h,
                                  cx=float(mask_params.get("cx", 0.5)),
                                  cy=float(mask_params.get("cy", 0.5)),
                                  rx=float(mask_params.get("rx", 0.35)),
                                  ry=float(mask_params.get("ry", 0.35)),
                                  feather_px=feather_px,
                                  invert=True)  # match shows inside shape

    else:  # contour
        arr_b = np.array(img_b.resize((out_w, out_h), Image.LANCZOS), dtype=np.uint8)
        mask = build_contour_mask(arr_b, out_w, out_h,
                                  cx=float(mask_params.get("cx", 0.5)),
                                  cy=float(mask_params.get("cy", 0.5)),
                                  feather_px=feather_px,
                                  invert=True)

    m = mask[..., np.newaxis].astype(np.float32)
    result = (arr_a.astype(np.float32) * m +
              arr_b.astype(np.float32) * (1.0 - m)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ── Dispatch ──────────────────────────────────────────────────────────────────

def compose_single_frame(
    img_a: Image.Image,
    img_b: Image.Image,
    mask_params: Optional[dict] = None,
    feather_px: int = 0,
    align: bool = True,
) -> Image.Image:
    """
    Compose one pair of frames using mask_params.
    Falls back to vertical line split if mask_params is None.
    Returns PIL Image at img_a's native size.
    """
    if mask_params is None:
        mask_params = {"type": "line", "angle": 0.0, "position": 0.5}
    return compose_with_mask(img_a, img_b, mask_params,
                             feather_px=feather_px, align=align)


def compose_animated(
    path_a: Path,
    path_b: Path,
    mask_params: Optional[dict] = None,
    feather_px: int = 0,
    align: bool = True,
    output_format: str = "gif",
    max_frames: int = 48,
    fps: float = 10.0,
) -> bytes:
    """Compose an animated collage. Output is GIF or MP4 bytes."""
    if mask_params is None:
        mask_params = {"type": "line", "angle": 0.0, "position": 0.5}

    frames_a = extract_frames(path_a, max_frames=max_frames)
    frames_b = extract_frames(path_b, max_frames=max_frames)
    n = max(len(frames_a), len(frames_b))

    def _loop(lst: list, count: int) -> list:
        if not lst:
            return [_blank()] * count
        return [lst[i % len(lst)] for i in range(count)]

    frames_a = _loop(frames_a, n)
    frames_b = _loop(frames_b, n)

    composed = [
        compose_single_frame(fa, fb, mask_params=mask_params,
                             feather_px=feather_px, align=align)
        for fa, fb in zip(frames_a, frames_b)
    ]

    duration_ms = min(get_frame_duration_ms(path_a), get_frame_duration_ms(path_b))
    return frames_to_mp4(composed, fps=fps) if output_format == "mp4" else frames_to_gif(composed, duration_ms=duration_ms)
