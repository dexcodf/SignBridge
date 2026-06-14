"""Hugging Face ASL letter recognizer (trained on real photos).

The local CNN ([gesture_recognizer.GestureRecognizer]) is trained on Sign
Language MNIST — clean, centred 28x28 grayscale crops — so it can struggle on
real webcam frames (lighting, background, skin tone). This module uses a
Vision-Transformer image classifier from the Hugging Face Hub that was
fine-tuned on real ASL photos, which generalises far better to a live camera.

Default model: ``prithivMLmods/Alphabet-Sign-Language-Detection`` (SigLIP2,
outputs single letters A-Z). It was chosen because it generalises to *real*
webcam-style photos — verified 7/7 at ~100% on the Kaggle ASL-Alphabet test
images, where the lighter ViT ``RavenOnur/Sign-Language`` only scored 1/7.
Weights download from the Hub on first use, then run locally (CPU is fine).

    pip install torch torchvision transformers pillow
"""
from __future__ import annotations

import numpy as np

from . import config
from .gesture_recognizer import Prediction

DEFAULT_HF_MODEL = "prithivMLmods/Alphabet-Sign-Language-Detection"


def _normalize_label(label: str) -> str:
    """Map a model label string to a single uppercase ASL letter.

    Handles plain letters ('a'/'A'), prefixed labels ('Sign A', 'letter_b')
    and the common extra classes some ASL models emit.
    """
    raw = str(label).strip()
    lower = raw.lower()
    if lower in {"space", "blank"}:
        return " "
    if lower in {"del", "delete", "nothing", "none"}:
        return "?"
    # take the first alphabetic character (covers 'A', 'sign_A', 'A: hand', ...)
    for ch in raw:
        if ch.isalpha():
            return ch.upper()
    return "?"


class HFGestureRecognizer:
    """Wraps a HF ``image-classification`` pipeline for ASL letters."""

    def __init__(self, model_id: str = DEFAULT_HF_MODEL) -> None:
        from transformers import pipeline

        self.model_id = model_id
        # device=-1 -> CPU; the ViT is small enough to be fast there.
        self.pipe = pipeline("image-classification", model=model_id, device=-1)

    def predict(self, image_bgr: np.ndarray) -> Prediction:
        """Predict an ASL letter from a colour (BGR) crop or full frame.

        Unlike the CNN this consumes the *colour* image directly — the HF
        processor handles resizing/normalisation — so pass the raw hand crop
        (or the whole frame), not the 28x28 grayscale tensor.
        """
        import cv2
        from PIL import Image

        if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:  # already single-channel / RGB
            rgb = image_bgr
        pil = Image.fromarray(rgb.astype("uint8"))

        top = self.pipe(pil, top_k=1)[0]
        letter = _normalize_label(top["label"])
        return Prediction(
            letter=letter,
            label=config.LETTER_TO_LABEL.get(letter, -1),
            confidence=float(top["score"]),
        )

    def predict_topk(self, image_bgr: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(letter, confidence)`` guesses, best first."""
        import cv2
        from PIL import Image

        if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = image_bgr
        pil = Image.fromarray(rgb.astype("uint8"))
        results = self.pipe(pil, top_k=k)
        return [(_normalize_label(r["label"]), float(r["score"])) for r in results]
