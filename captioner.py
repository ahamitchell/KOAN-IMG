from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


@dataclass
class Captioner:
    model_id: str = "Salesforce/blip-image-captioning-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def caption(self, img: Image.Image, max_new_tokens: int = 30) -> str:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return (text or "").strip()
