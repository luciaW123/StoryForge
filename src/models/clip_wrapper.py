"""
CLIP-L/14 wrapper for consistency scoring.

Used by the Consistency Agent to compute cosine similarity between keyframe
embeddings and per-character reference embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.utils import get_logger

log = get_logger("clip")


class CLIPWrapper:
    """open_clip ViT-L/14 image encoder."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import open_clip
        import torch

        log.info("Loading CLIP: %s / %s", self.model_name, self.pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model.eval()
        if torch.cuda.is_available():
            model = model.to(self.device)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        log.info("CLIP loaded.")

    def encode_image(self, image: Image.Image | Path | str) -> np.ndarray:
        """Return a unit-normalized embedding vector."""
        self._load()
        import torch

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.to(self.device)
        with torch.no_grad():
            emb = self._model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu().numpy()[0].astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Return a unit-normalized text embedding (for CLIP-T metric)."""
        self._load()
        import torch

        # CLIP-L/14 has a 77-token context window
        tokens = self._tokenizer([text])
        if torch.cuda.is_available():
            tokens = tokens.to(self.device)
        with torch.no_grad():
            emb = self._model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu().numpy()[0].astype(np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


_singleton: Optional[CLIPWrapper] = None


def get_clip() -> CLIPWrapper:
    global _singleton
    if _singleton is None:
        _singleton = CLIPWrapper()
    return _singleton
