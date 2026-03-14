from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import open_clip

from common import l2_normalize, open_media_rgb


@dataclass
class Embedder:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    @torch.inference_mode()
    def embed_pil(self, img: Image.Image) -> np.ndarray:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat.float().cpu().numpy().reshape(-1).astype(np.float32)
        feat = l2_normalize(feat)
        return feat

    @torch.inference_mode()
    def embed_text(self, text: str) -> np.ndarray:
        tokens = open_clip.tokenize([text]).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat.float().cpu().numpy().reshape(-1).astype(np.float32)
        feat = l2_normalize(feat)
        return feat

    def embed_path(self, path: Path, allow_video: bool = False) -> np.ndarray:
        img = open_media_rgb(Path(path), max_side=1024, allow_video=allow_video)
        return self.embed_pil(img)
