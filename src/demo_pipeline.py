"""End-to-end demo: gesture images -> recognized text -> Supertonic speech.

This walks the full pipeline the live app uses, minus the webcam: for each
letter of a target word it grabs a real Sign Language MNIST test image, reads
it with the trained CNN, assembles the recognized word, then speaks it with
Supertonic (Hugging Face) TTS.

    python -m src.demo_pipeline --word HELLO
"""
from __future__ import annotations

import argparse

import numpy as np

from . import config
from .data_loader import clean, load_raw, to_arrays
from .gesture_recognizer import GestureRecognizer
from .tts_engine import TTSEngine


def run(word: str, seed: int = 0) -> None:
    word = word.upper()
    bad = [c for c in word if c in config.MISSING_LETTERS or c not in config.LETTER_TO_LABEL]
    if bad:
        raise SystemExit(f"Letters not in the static dataset: {bad} (J and Z are excluded).")

    test = clean(load_raw(config.TEST_CSV))
    X, y = to_arrays(test)
    rng = np.random.RandomState(seed)
    recognizer = GestureRecognizer()

    print(f"Target word: {word}\n--- reading one gesture image per letter ---")
    recognized = ""
    for ch in word:
        label = config.LETTER_TO_LABEL[ch]
        candidates = np.where(y == label)[0]
        img = X[rng.choice(candidates)]
        pred = recognizer.predict(img)
        recognized += pred.letter
        ok = "OK " if pred.letter == ch else "XX "
        print(f"  {ok}signed '{ch}'  ->  read '{pred.letter}'  @ {pred.confidence:.0%}")

    print(f"\nRecognized text: {recognized!r}")

    tts = TTSEngine()
    out = config.OUTPUTS_DIR / "demo_speech.wav"
    print(f"Speaking with backend: {tts.backend} (downloads the model on first run)…")
    tts.synthesize(recognized, out)
    print(f"Saved spoken audio -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture -> text -> Supertonic speech demo")
    parser.add_argument("--word", default="HELLO", help="word to fingerspell and speak")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.word, args.seed)
