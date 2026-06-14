"""Train a small CNN to classify ASL letters from 28x28 images.

Run from the project root:

    python -m src.train_model --epochs 15

The trained model is saved to ``models/sign_cnn.keras`` and a training-history
plot to ``outputs/training_history.png``. Keras/TensorFlow is imported lazily
so the rest of the package (data loading, EDA) works even without it installed.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import config
from .data_loader import load_dataset


def build_model():
    """A compact CNN sized for 28x28 single-channel sign images.

    Augmentation is baked in as Keras 3 preprocessing *layers* (active only at
    training time, bypassed during predict/evaluate). Signs survive small
    shifts/rotations/zoom but not flips — a mirrored hand is a different
    gesture — so there is no RandomFlip.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1)),
            layers.RandomRotation(0.04, fill_mode="constant"),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
            layers.RandomZoom(0.1, fill_mode="constant"),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(config.NUM_CLASSES, activation="softmax"),
        ],
        name="sign_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs: int = 15, batch_size: int = 128) -> None:
    from tensorflow import keras

    X_train, y_train, X_test, y_test = load_dataset()
    print(f"train={X_train.shape}  test={X_test.shape}")

    model = build_model()  # augmentation layers are baked into the model
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=4, restore_best_weights=True, monitor="val_accuracy"
        ),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        epochs=epochs,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  (loss {loss:.4f})")

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH}")

    _save_history_plot(history)


def _save_history_plot(history) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "training_history.png"
    fig.savefig(out, dpi=120)
    print(f"Saved history plot -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ASL letter CNN")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)
