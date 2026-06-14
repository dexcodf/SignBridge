"""Landmark-based ASL letter recognizer (appearance-invariant, fast).

Instead of classifying pixels (which ties a model to its training lighting,
skin tone and background), this classifies the *geometry* of the hand from
MediaPipe's 21 landmarks. The features are translation-, scale- and
handedness-normalised, so the same sign produces the same vector on any camera.

A small scikit-learn MLP is trained offline (``src.train_landmarks``) on
landmarks extracted from real ASL photos and saved to
``models/landmark_mlp.joblib``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .gesture_recognizer import Prediction

MODEL_PATH = config.MODELS_DIR / "landmark_mlp.joblib"


def landmarks_to_features(landmarks, handedness: str | None = None) -> np.ndarray:
    """Turn 21 hand landmarks into a 63-d pose-normalised feature vector.

    ``landmarks`` is an iterable of ``(x, y, z)`` tuples or objects with
    ``.x/.y/.z``. Normalisation makes the same sign map to (nearly) the same
    vector on any camera, hand or orientation:

    1. translate so the wrist sits at the origin,
    2. mirror left hands to a canonical right hand,
    3. **rotate** in the image plane so the wrist→middle-finger-MCP axis points
       up — this removes hand/camera tilt, the biggest source of error across
       different setups,
    4. scale by the largest landmark distance (size-invariant).
    """
    pts = np.array(
        [[p.x, p.y, p.z] if hasattr(p, "x") else p for p in landmarks],
        dtype=np.float32,
    )  # (21, 3)
    pts = pts - pts[0]  # wrist -> origin
    if handedness and str(handedness).lower().startswith("left"):
        pts[:, 0] = -pts[:, 0]  # mirror to a canonical right hand

    # Rotate the x-y plane so landmark 9 (middle-finger MCP) points straight up.
    v = pts[9, :2]
    if np.linalg.norm(v) > 1e-6:
        delta = (-np.pi / 2.0) - np.arctan2(v[1], v[0])
        c, s = np.cos(delta), np.sin(delta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts[:, :2] = pts[:, :2] @ rot.T

    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale > 0:
        pts = pts / scale
    return pts.flatten()


class LandmarkRecognizer:
    """Loads the trained MLP and classifies a hand's landmarks into a letter."""

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        import joblib

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No landmark model at {model_path}. Train one:\n"
                "  python -m src.train_landmarks"
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes = [str(c) for c in bundle["classes"]]
        self.model_id = "landmark-mlp"

    def _probs(self, landmarks, handedness):
        feat = landmarks_to_features(landmarks, handedness).reshape(1, -1)
        return self.model.predict_proba(feat)[0]

    def predict(self, landmarks, handedness: str | None = None) -> Prediction:
        probs = self._probs(landmarks, handedness)
        i = int(np.argmax(probs))
        letter = self.classes[i]
        return Prediction(
            letter=letter,
            label=config.LETTER_TO_LABEL.get(letter, -1),
            confidence=float(probs[i]),
        )

    def predict_topk(self, landmarks, handedness: str | None = None, k: int = 3):
        probs = self._probs(landmarks, handedness)
        idx = np.argsort(probs)[::-1][:k]
        return [(self.classes[i], float(probs[i])) for i in idx]
