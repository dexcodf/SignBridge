"""Central configuration: paths, image geometry and label mapping.

Keeping every magic number in one place means the notebook, the training
script, the real-time demo and the Streamlit app all agree on the same
contract (28x28 grayscale, 25 logical classes, A-Z minus J and Z).
"""
from __future__ import annotations

import string
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all relative to the project root, so the code is machine independent) #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRAIN_CSV = DATA_DIR / "sign_mnist_train.csv"
TEST_CSV = DATA_DIR / "sign_mnist_test.csv"
MODEL_PATH = MODELS_DIR / "sign_cnn.keras"

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Image / model geometry                                                       #
# --------------------------------------------------------------------------- #
IMG_SIZE = 28            # Sign Language MNIST images are 28x28
NUM_CLASSES = 25         # labels run 0..24 (J=9 and Z=25 need motion, absent)

# --------------------------------------------------------------------------- #
# Label <-> letter mapping                                                     #
# Labels are 0-indexed against the alphabet: 0->A, 1->B, ... 24->Y.            #
# J (9) and Z (25) are dynamic gestures and are not in the static dataset.     #
# --------------------------------------------------------------------------- #
LABEL_TO_LETTER = {i: string.ascii_uppercase[i] for i in range(26)}
LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}
MISSING_LETTERS = {"J", "Z"}  # documented gaps in Sign Language MNIST


def label_to_letter(label: int) -> str:
    """Map a model class index to its ASL letter."""
    return LABEL_TO_LETTER.get(int(label), "?")
