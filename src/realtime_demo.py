"""Standalone real-time webcam demo (true live video via OpenCV).

Run from the project root:

    python -m src.realtime_demo

Controls (focus the video window):
    SPACE  insert a space between words
    B      backspace (delete last character)
    C      clear the whole sentence
    S      speak the sentence with Supertonic TTS
    Q/ESC  quit

This is the "full" experience; the Streamlit app offers a snapshot-based
version that works without an OpenCV display window.
"""
from __future__ import annotations

import cv2

from .gesture_recognizer import GestureRecognizer, SentenceBuilder
from .hand_detector import HandDetector
from .tts_engine import TTSEngine


def _draw_hud(frame, sentence: str, current: str, conf: float) -> None:
    """Overlay the live letter, confidence and accumulated sentence."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(
        frame, f"Letter: {current}  ({conf:0.2f})", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, f"> {sentence}", (10, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )


def main(camera_index: int = 0) -> None:
    detector = HandDetector(max_hands=1)
    recognizer = GestureRecognizer()
    builder = SentenceBuilder()
    tts = TTSEngine()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}.")

    print(__doc__)
    current_letter, current_conf = "-", 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror -> feels natural to the signer

            crop = detector.detect(frame, draw=True)
            if crop is not None:
                pred = recognizer.predict(crop.image)
                current_letter, current_conf = pred.letter, pred.confidence
                builder.update(pred)
            else:
                current_letter, current_conf = "-", 0.0

            _draw_hud(frame, builder.text, current_letter, current_conf)
            cv2.imshow("Sign Language -> Text -> Speech", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):       # q or ESC
                break
            elif key == ord(" "):
                builder.add_space()
            elif key == ord("b"):
                builder.backspace()
            elif key == ord("c"):
                builder.clear()
            elif key == ord("s") and builder.text.strip():
                print(f"Speaking: {builder.text!r}")
                tts.speak(builder.text)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
