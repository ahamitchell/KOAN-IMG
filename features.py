from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from common import l2_normalize, open_media_rgb


def color_signature_lab_hist(
    img: Image.Image,
    bins_l: int = 16,
    bins_a: int = 16,
    bins_b: int = 16,
) -> np.ndarray:
    arr = np.array(img, dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)

    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    hist = cv2.calcHist(
        [l, a, b],
        [0, 1, 2],
        None,
        [bins_l, bins_a, bins_b],
        [0, 256, 0, 256, 0, 256],
    )
    hist = hist.astype(np.float32).reshape(-1)
    hist = hist / (hist.sum() + 1e-12)
    return l2_normalize(hist)


def color_signature_from_path(path: Path, allow_video: bool = False) -> np.ndarray:
    img = open_media_rgb(path, max_side=1024, allow_video=allow_video)
    return color_signature_lab_hist(img)
