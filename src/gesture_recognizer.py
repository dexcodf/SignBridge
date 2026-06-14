"""Turn a pre-processed hand image into a stable ASL letter.

This module owns two responsibilities:

1. Loading the trained Keras model and predicting a letter + confidence.
2. Temporal smoothing — a webcam jitters, so a single noisy frame should not
   commit a letter. :class:`SentenceBuilder` only accepts a letter once the
   classifier has agreed on it for several consecutive frames.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import config


@dataclass
class Prediction:
    letter: str
    label: int
    confidence: float


class GestureRecognizer:
    """Loads ``models/sign_cnn.keras`` and predicts letters from crops."""

    def __init__(self, model_path: str | Path = config.MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {self.model_path}. Train one first:\n"
                "  python -m src.train_model"
            )
        from tensorflow import keras

        self.model = keras.models.load_model(self.model_path)

    def predict(self, image: np.ndarray) -> Prediction:
        """Predict the letter for a single (28,28,1) [0,1] image."""
        batch = np.expand_dims(image, axis=0)
        probs = self.model.predict(batch, verbose=0)[0]
        label = int(np.argmax(probs))
        return Prediction(
            letter=config.label_to_letter(label),
            label=label,
            confidence=float(probs[label]),
        )


@dataclass
class SentenceBuilder:
    """Accumulate stable predictions into words and sentences.

    A letter is committed only after it is the top prediction for
    ``stable_frames`` consecutive frames above ``min_confidence``. This turns a
    noisy per-frame stream into deliberate fingerspelling.
    """

    stable_frames: int = 12
    min_confidence: float = 0.70
    text: str = ""
    _recent: deque = field(default_factory=lambda: deque(maxlen=12))
    _last_committed: Optional[str] = None

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.stable_frames)

    def update(self, pred: Optional[Prediction]) -> Optional[str]:
        """Feed one prediction; return a letter when one commits.

        A low-confidence/None prediction (hand lost or moving) counts as a
        "release": it clears the smoothing window and re-arms the committer so
        the next sign can be accepted — even if it's the same letter. This stops
        a single held sign from being typed over and over.
        """
        if pred is None or pred.confidence < self.min_confidence:
            self._recent.clear()
            self._last_committed = None  # released -> same letter may commit again
            return None

        self._recent.append(pred.letter)
        if (
            len(self._recent) == self._recent.maxlen
            and len(set(self._recent)) == 1
        ):
            letter = self._recent[0]
            self._recent.clear()
            if letter == self._last_committed:
                return None  # still holding the letter we just typed; wait for release
            self._last_committed = letter
            self.text += letter
            return letter
        return None

    # -- manual editing helpers, handy for the UI ---------------------------- #
    def add_char(self, c: str) -> None:
        """Append an arbitrary character (space, punctuation, …)."""
        self.text += c
        self._last_committed = None

    def add_space(self) -> None:
        self.add_char(" ")

    def backspace(self) -> None:
        self.text = self.text[:-1]
        self._last_committed = None

    def clear(self) -> None:
        self.text = ""
        self._recent.clear()
        self._last_committed = None

    def set_text(self, s: str) -> None:
        """Replace the whole sentence (e.g. after a manual correction)."""
        self.text = s
        self._recent.clear()
        self._last_committed = None
