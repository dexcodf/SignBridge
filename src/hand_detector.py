"""Locate a hand in a frame with MediaPipe's HandLandmarker (Tasks API).

Newer MediaPipe builds ship only the **Tasks** API (``mediapipe.tasks``) and
drop the legacy ``mediapipe.solutions.hands`` module, so this uses
``HandLandmarker`` directly. The ``.task`` model is downloaded once into
``models/`` on first use.

MediaPipe gives us 21 normalised hand landmarks; we take their bounding box,
pad it, crop the region from the original frame, then convert to the 28x28
grayscale contract the CNN was trained on (the HF ViT consumes the colour crop).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config

# Google-hosted HandLandmarker bundle (downloaded once, then cached locally).
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = config.MODELS_DIR / "hand_landmarker.task"

# Standard MediaPipe 21-point hand topology, for drawing the skeleton with cv2.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def ensure_landmarker_model(path: Path = HAND_LANDMARKER_PATH) -> Path:
    """Download the hand_landmarker.task bundle if it isn't present yet."""
    path = Path(path)
    if not path.exists():
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(HAND_LANDMARKER_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
    return path


@dataclass
class HandCrop:
    """Result of detecting and pre-processing one hand."""
    image: np.ndarray            # (IMG_SIZE, IMG_SIZE, 1) float32 in [0, 1]
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in original frame coords
    landmarks: list | None = None    # 21 (x, y, z) normalised hand landmarks
    handedness: str | None = None    # "Left" / "Right" (as seen in the image)


class HandDetector:
    """Thin wrapper around MediaPipe ``HandLandmarker`` (IMAGE running mode).

    Parameters
    ----------
    max_hands:
        How many hands to track (1 is enough for fingerspelling).
    detection_confidence / presence_confidence / tracking_confidence:
        MediaPipe thresholds; lower them if detection is flaky under poor light.
    padding:
        Fraction of the bounding-box size added on every side before cropping,
        so the whole hand (not just the landmarks) reaches the classifier.
    """

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        padding: float = 0.2,
        running_mode: str = "image",
        process_width: int | None = None,
        model_path: Path = HAND_LANDMARKER_PATH,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        self._video = running_mode.lower() == "video"
        # Run detection on a downscaled copy for speed; landmarks are normalised
        # so they still map back onto the full-resolution frame.
        self.process_width = process_width
        model_path = ensure_landmarker_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if self._video else vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.padding = padding

    def detect(
        self,
        frame_bgr: np.ndarray,
        draw: bool = True,
        timestamp_ms: int | None = None,
    ) -> Optional[HandCrop]:
        """Return the pre-processed crop for the most prominent hand, or None.

        ``frame_bgr`` is a standard OpenCV BGR frame. When ``draw`` is True the
        skeleton and bounding box are drawn onto ``frame_bgr`` in place. In VIDEO
        mode a strictly increasing ``timestamp_ms`` must be supplied; MediaPipe
        then tracks the hand between frames instead of re-running full palm
        detection every frame, which is much faster for a live stream.
        """
        h, w = frame_bgr.shape[:2]
        src = frame_bgr
        if self.process_width and w > self.process_width:
            new_h = int(h * self.process_width / w)
            src = cv2.resize(frame_bgr, (self.process_width, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        if self._video:
            if timestamp_ms is None:
                raise ValueError("VIDEO running_mode requires a timestamp_ms.")
            result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        # Pad RELATIVE TO THE HAND, not the frame, so the crop stays tight and the
        # hand fills it (the classifiers were trained on hand-filling images).
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = (max_x - min_x) * self.padding
        pad_y = (max_y - min_y) * self.padding
        x1 = int(max(0, (min_x - pad_x) * w))
        y1 = int(max(0, (min_y - pad_y) * h))
        x2 = int(min(w, (max_x + pad_x) * w))
        y2 = int(min(h, (max_y + pad_y) * h))
        if x2 <= x1 or y2 <= y1:
            return None

        # Square crop (computed before drawing) avoids the aspect-ratio
        # distortion that hurts classification when a tall hand box is squashed
        # into the model's square input.
        processed = self.preprocess(self.square_region(frame_bgr, (x1, y1, x2, y2)))

        handed = None
        if getattr(result, "handedness", None):
            try:
                handed = result.handedness[0][0].category_name
            except (IndexError, AttributeError):
                handed = None
        lm_list = [(lm.x, lm.y, lm.z) for lm in landmarks]

        if draw:
            self._draw(frame_bgr, landmarks, (x1, y1, x2, y2))
        return HandCrop(
            image=processed, bbox=(x1, y1, x2, y2),
            landmarks=lm_list, handedness=handed,
        )

    @staticmethod
    def _draw(frame_bgr: np.ndarray, landmarks, bbox) -> None:
        """Draw the hand skeleton (cv2) and bounding box in place."""
        h, w = frame_bgr.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[a], pts[b], (255, 255, 255), 2)
        for p in pts:
            cv2.circle(frame_bgr, p, 3, (0, 200, 0), -1)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)

    @staticmethod
    def square_region(frame_bgr: np.ndarray, bbox, margin: float = 1.0) -> np.ndarray:
        """Return a square crop centred on ``bbox`` (the model expects square input).

        The box is grown to its longer side times ``margin`` and centred on the
        hand; any part outside the frame is filled by replicating the edge, so
        the result is always exactly square (no resize distortion later).
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = int(max(x2 - x1, y2 - y1) * margin / 2) or 1

        sx1, sy1, sx2, sy2 = cx - half, cy - half, cx + half, cy + half
        region = frame_bgr[max(0, sy1):min(h, sy2), max(0, sx1):min(w, sx2)]
        pad_t, pad_b = max(0, -sy1), max(0, sy2 - h)
        pad_l, pad_r = max(0, -sx1), max(0, sx2 - w)
        if pad_t or pad_b or pad_l or pad_r:
            region = cv2.copyMakeBorder(
                region, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
            )
        return region

    @staticmethod
    def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
        """Convert a colour hand crop into the model's 28x28x1 [0,1] tensor."""
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        arr = resized.astype("float32") / 255.0
        return arr.reshape(config.IMG_SIZE, config.IMG_SIZE, 1)

    @staticmethod
    def center_square(frame_bgr: np.ndarray) -> np.ndarray:
        """Crop the largest centred square of a frame.

        Used as a fallback when MediaPipe can't lock onto a hand: if the user
        holds their hand filling the frame on a plain background, classifying
        the centre square still works.
        """
        h, w = frame_bgr.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        return frame_bgr[y0 : y0 + side, x0 : x0 + side]

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
