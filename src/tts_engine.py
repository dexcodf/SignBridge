"""Text-to-speech using Supertonic (Hugging Face), with a graceful fallback.

Primary engine: Supertonic 3 (https://huggingface.co/Supertone/supertonic-3),
a lightweight on-device TTS that runs on CPU via ONNX Runtime.

    pip install supertonic

The first call downloads the ONNX assets from Hugging Face automatically.
If ``supertonic`` is not installed we fall back to ``pyttsx3`` (offline,
OS voices) so the rest of the app still produces audio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config


class TTSEngine:
    """Unified ``speak``/``synthesize`` interface over Supertonic or pyttsx3."""

    def __init__(self, voice_name: str = "M1", lang: str = "en") -> None:
        self.voice_name = voice_name
        self.lang = lang
        self.backend = None          # "supertonic" | "pyttsx3" | None
        self._tts = None             # Supertonic TTS instance
        self._style = None           # Supertonic voice style
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from supertonic import TTS

            self._tts = TTS(auto_download=True)
            self._style = self._tts.get_voice_style(voice_name=self.voice_name)
            self.backend = "supertonic"
            return
        except Exception as exc:  # noqa: BLE001 - any failure -> try fallback
            print(f"[tts] Supertonic unavailable ({exc}); trying pyttsx3.")

        try:
            import pyttsx3  # noqa: F401

            self.backend = "pyttsx3"
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] No TTS backend available ({exc}).")
            self.backend = None

    @property
    def available(self) -> bool:
        return self.backend is not None

    def synthesize(self, text: str, out_path: Optional[str | Path] = None) -> Path:
        """Render ``text`` to a WAV file and return its path."""
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot synthesize empty text.")

        out_path = Path(out_path) if out_path else config.OUTPUTS_DIR / "speech.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "supertonic":
            wav, _duration = self._tts.synthesize(
                text, voice_style=self._style, lang=self.lang
            )
            self._tts.save_audio(wav, str(out_path))
        elif self.backend == "pyttsx3":
            import pyttsx3

            engine = pyttsx3.init()
            engine.save_to_file(text, str(out_path))
            engine.runAndWait()
        else:
            raise RuntimeError(
                "No TTS backend installed. Run: pip install supertonic"
            )
        return out_path

    def speak(self, text: str) -> Path:
        """Synthesize ``text`` and play it through the default audio device."""
        path = self.synthesize(text)
        self._play(path)
        return path

    @staticmethod
    def _play(path: Path) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf

            data, samplerate = sf.read(str(path))
            sd.play(data, samplerate)
            sd.wait()
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] Saved {path} but could not auto-play ({exc}).")
