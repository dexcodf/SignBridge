"""Train the landmark MLP: ASL photos -> MediaPipe landmarks -> letter.

For each image in ``--data-dir`` (one sub-folder per letter), MediaPipe extracts
21 hand landmarks. The raw landmarks are cached to ``models/landmarks_raw.npz``
so feature/augmentation tweaks can be re-trained in seconds without re-running
MediaPipe. Features are pose-normalised (see ``landmarks_to_features``) and the
training set is augmented with small jitter so the MLP is robust to hand-shape
and landmark-estimation variation across people and cameras.

    python -m src.train_landmarks --data-dir .../asl_alphabet_train --per-class 300

Saves ``models/landmark_mlp.joblib``. Re-run with ``--use-cache`` to retrain
from the cached landmarks without touching the images.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import config
from .hand_detector import HandDetector
from .landmark_recognizer import MODEL_PATH, landmarks_to_features

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
RAW_CACHE = config.MODELS_DIR / "landmarks_raw.npz"


def extract_raw(data_dir: Path, per_class: int):
    """Run MediaPipe over the images, return raw landmarks (N,21,3), labels, handedness."""
    import cv2

    detector = HandDetector(max_hands=1, detection_confidence=0.4)
    pts_all: list[np.ndarray] = []
    y: list[str] = []
    handed: list[str] = []

    letter_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and len(d.name) == 1 and d.name.isalpha()
    )
    if not letter_dirs:
        raise SystemExit(f"No single-letter sub-folders found in {data_dir}")

    print(f"Found {len(letter_dirs)} letter folders in {data_dir}")
    for d in letter_dirs:
        letter = d.name.upper()
        files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in IMG_EXTS][:per_class]
        kept = 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            crop = detector.detect(img, draw=False)
            if crop is None or not crop.landmarks:
                continue
            pts_all.append(np.array(crop.landmarks, dtype=np.float32))
            y.append(letter)
            handed.append(crop.handedness or "Right")
            kept += 1
        print(f"  {letter}: {kept}/{len(files)} hands detected")
    detector.close()
    return np.array(pts_all, dtype=np.float32), np.array(y), np.array(handed)


def _augment(raw: np.ndarray, n: int, sigma: float, rng) -> list[np.ndarray]:
    """Return ``n`` jittered copies of a raw (21,3) landmark set."""
    return [raw + rng.normal(0, sigma, raw.shape).astype(np.float32) for _ in range(n)]


def train(
    data_dir: str | None,
    per_class: int = 300,
    out: Path = MODEL_PATH,
    use_cache: bool = False,
    n_aug: int = 4,
    sigma: float = 0.008,
) -> None:
    import joblib
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    t0 = time.time()
    if use_cache and RAW_CACHE.exists():
        print(f"Loading cached landmarks from {RAW_CACHE}")
        z = np.load(RAW_CACHE, allow_pickle=True)
        raw, y, handed = z["pts"], z["y"], z["handed"]
    else:
        if not data_dir:
            raise SystemExit("--data-dir is required (no cache to use).")
        raw, y, handed = extract_raw(Path(data_dir), per_class)
        np.savez_compressed(RAW_CACHE, pts=raw, y=y, handed=handed)
        print(f"Cached {len(raw)} raw landmark sets -> {RAW_CACHE}")
    print(f"{len(raw)} samples in {time.time() - t0:.0f}s")
    if len(raw) < 50:
        raise SystemExit("Too few detected hands to train.")

    # split on raw samples so augmented copies never leak across the split
    idx = np.arange(len(raw))
    tr_idx, te_idx = train_test_split(idx, test_size=0.15, random_state=0, stratify=y)
    rng = np.random.default_rng(0)

    def build(indices, augment):
        X, yy = [], []
        for i in indices:
            X.append(landmarks_to_features(raw[i], handed[i]))
            yy.append(y[i])
            if augment:
                for a in _augment(raw[i], n_aug, sigma, rng):
                    X.append(landmarks_to_features(a, handed[i]))
                    yy.append(y[i])
        return np.array(X, dtype=np.float32), np.array(yy)

    X_tr, y_tr = build(tr_idx, augment=True)
    X_te, y_te = build(te_idx, augment=False)
    print(f"train {X_tr.shape} (with x{n_aug} aug) | test {X_te.shape}")

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=600,
        early_stopping=True,
        random_state=0,
    )
    print("Training MLP…")
    clf.fit(X_tr, y_tr)

    print(f"\nHeld-out accuracy: {clf.score(X_te, y_te):.3f}\n")
    print(classification_report(y_te, clf.predict(X_te)))

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": clf.classes_}, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the landmark MLP")
    parser.add_argument("--data-dir", default=None, help="root with one sub-folder per letter")
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--use-cache", action="store_true", help="retrain from cached landmarks")
    parser.add_argument("--n-aug", type=int, default=4)
    parser.add_argument("--out", default=str(MODEL_PATH))
    args = parser.parse_args()
    train(args.data_dir, args.per_class, Path(args.out), args.use_cache, args.n_aug)
