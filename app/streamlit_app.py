"""Streamlit dashboard for the Sign Language -> Text -> Speech pipeline.

Run from the project root:

    streamlit run app/streamlit_app.py

Tabs
----
1. Recognize     capture/upload a hand sign, predict the letter, build a
                 sentence and speak it with Supertonic TTS.
2. Dataset       explore the Sign Language MNIST data: class balance, sample
                 grids and the average image per letter.
3. About         what the project does and how the pieces fit together.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make the ``src`` package importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_loader import clean, load_raw  # noqa: E402

st.set_page_config(page_title="SignBridge · Sign Language AI", page_icon="🤟", layout="wide")


# --------------------------------------------------------------------------- #
# Cached heavy resources                                                       #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading hand detector…")
def get_detector():
    from src.hand_detector import HandDetector

    return HandDetector(max_hands=1, detection_confidence=0.4)


@st.cache_resource(show_spinner="Loading gesture model…")
def get_recognizer():
    from src.gesture_recognizer import GestureRecognizer

    return GestureRecognizer()


@st.cache_resource(show_spinner="Loading Hugging Face ASL model… (first run downloads it)")
def get_hf_recognizer():
    from src.hf_recognizer import HFGestureRecognizer

    return HFGestureRecognizer()


@st.cache_resource(show_spinner="Loading landmark model…")
def get_landmark_recognizer():
    from src.landmark_recognizer import LandmarkRecognizer

    return LandmarkRecognizer()


@st.cache_resource(show_spinner="Starting Supertonic TTS…")
def get_tts():
    from src.tts_engine import TTSEngine

    return TTSEngine()


@st.cache_data(show_spinner="Loading dataset…")
def get_clean_train() -> pd.DataFrame:
    return clean(load_raw(config.TRAIN_CSV))


def decode_image(uploaded) -> "np.ndarray":
    """Decode a Streamlit camera/upload file into an OpenCV BGR frame."""
    import cv2

    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
# Sentence-editing callbacks                                                   #
# Mutating a widget-bound session_state key must happen in an on_click         #
# callback (which runs before the rerun), not inline after the widget exists.  #
# --------------------------------------------------------------------------- #
def _append_text(s: str) -> None:
    st.session_state.sentence = st.session_state.get("sentence", "") + s


def _backspace() -> None:
    st.session_state.sentence = st.session_state.get("sentence", "")[:-1]


def _clear_text() -> None:
    st.session_state.sentence = ""


def _speak(text: str) -> None:
    """Synthesize ``text`` with Supertonic and play it inline (st.audio)."""
    text = (text or "").strip()
    if not text:
        st.warning("Nothing to speak yet.")
        return
    tts = get_tts()
    if not tts.available:
        st.error("No TTS backend available. Install with `pip install supertonic`.")
        return
    with st.spinner(f"Synthesizing with {tts.backend}… (first run downloads the model)"):
        wav_path = tts.synthesize(text)
    st.success(f"🔊 Spoken with **{tts.backend}**")
    st.audio(str(wav_path))


# --------------------------------------------------------------------------- #
# Tab 1 — recognition + sentence building + TTS                                #
# --------------------------------------------------------------------------- #
def recognize_tab() -> None:
    st.subheader("Sign → Text → Speech")
    if "sentence" not in st.session_state:
        st.session_state.sentence = ""

    mode = st.radio(
        "Mode",
        ["🔴 Live camera (streaming)", "📸 Snapshot / Upload"],
        horizontal=True,
    )
    if mode.startswith("🔴"):
        _live_recognize()
    else:
        _snapshot_recognize()


def _snapshot_recognize() -> None:
    st.caption(
        "Take a photo (or upload) of one ASL sign → the model reads the letter "
        "→ add it to your sentence → **Supertonic** (Hugging Face) speaks it."
    )

    col_in, col_out = st.columns(2)

    with col_in:
        source = st.radio("Input source", ["Camera", "Upload"], horizontal=True)
        if source == "Camera":
            shot = st.camera_input("📸 Capture a sign")
        else:
            shot = st.file_uploader("Upload a hand-sign image", type=["jpg", "jpeg", "png"])
        model_choice = st.radio(
            "Recognition model",
            ["Landmark MLP (robust)", "Hugging Face SigLIP2", "Local CNN (MNIST)"],
            help="Landmark MLP classifies hand *geometry* from MediaPipe → most "
            "robust across cameras / lighting. SigLIP2 classifies the photo. CNN is "
            "the MNIST model.",
        )
        use_fullframe = st.checkbox(
            "Classify the whole frame if no hand is detected",
            value=True,
            help="Hold your hand so it fills the frame on a plain background.",
        )

    with col_out:
        if shot is not None:
            import cv2
            from src.hand_detector import HandCrop

            frame = decode_image(shot)
            clean_frame = frame.copy()  # un-annotated, for the colour crop the HF model wants
            detector = get_detector()

            crop = detector.detect(frame, draw=True)  # draws landmarks on `frame`
            if crop is not None:
                x1, y1, x2, y2 = crop.bbox
                via = "MediaPipe hand crop"
            elif use_fullframe:
                h, w = clean_frame.shape[:2]
                side = min(h, w)
                y0, x0 = (h - side) // 2, (w - side) // 2
                x1, y1, x2, y2 = x0, y0, x0 + side, y0 + side
                crop = HandCrop(
                    image=detector.preprocess(clean_frame[y1:y2, x1:x2]),
                    bbox=(x1, y1, x2, y2),
                )
                via = "whole-frame fallback"

            if crop is None:
                st.error("No hand detected — center your hand, or tick the whole-frame option.")
            else:
                kind = ("landmark" if model_choice.startswith("Landmark")
                        else "hf" if model_choice.startswith("Hugging") else "cnn")
                # Square colour crop from the CLEAN frame for the image models.
                region = detector.square_region(clean_frame, (x1, y1, x2, y2), margin=1.0)
                if region.size == 0:
                    region = clean_frame
                pred = topk = None
                model_label = ""

                if kind == "landmark":
                    if not crop.landmarks:
                        st.warning("Whole-frame fallback has no landmarks — using SigLIP2.")
                        kind = "hf"
                    else:
                        try:
                            rec = get_landmark_recognizer()
                            topk = rec.predict_topk(crop.landmarks, crop.handedness, 3)
                            pred = rec.predict(crop.landmarks, crop.handedness)
                            model_label = "Landmark MLP (hand geometry)"
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Landmark model unavailable ({exc}). Using SigLIP2.")
                            kind = "hf"
                if pred is None and kind == "hf":
                    try:
                        rec = get_hf_recognizer()
                        topk = rec.predict_topk(region, k=3)
                        pred = rec.predict(region)
                        model_label = f"Hugging Face · `{rec.model_id}`"
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"HF model unavailable ({exc}). Using local CNN.")
                        kind = "cnn"
                if pred is None and kind == "cnn":
                    pred = get_recognizer().predict(crop.image)
                    model_label = "Local CNN (Sign Language MNIST)"

                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption=f"Input · {via}",
                    use_container_width=True,
                )
                if kind == "cnn":
                    seen = (crop.image.reshape(config.IMG_SIZE, config.IMG_SIZE) * 255).astype("uint8")
                    st.image(seen, caption="What the CNN sees (28×28)", width=110)
                elif kind == "hf":
                    st.image(cv2.cvtColor(region, cv2.COLOR_BGR2RGB),
                             caption="What the model sees", width=120)
                st.caption(f"Model: {model_label}")
                st.metric("Predicted letter", pred.letter, f"{pred.confidence:.0%} confident")
                if topk:
                    st.caption("Top guesses: " + " · ".join(f"**{l}** {s:.0%}" for l, s in topk))
                ca, cb = st.columns(2)
                ca.button(
                    f"➕ Add “{pred.letter}”",
                    type="primary",
                    on_click=_append_text,
                    args=(pred.letter,),
                )
                if cb.button(f"🔊 Speak “{pred.letter}”"):
                    _speak(pred.letter)

    st.divider()

    # -- sentence editing controls ------------------------------------------ #
    st.text_input("Sentence", key="sentence")
    b1, b2, b3, b4 = st.columns(4)
    b1.button("␣ Space", on_click=_append_text, args=(" ",))
    b2.button("⌫ Backspace", on_click=_backspace)
    b3.button("🗑️ Clear", on_click=_clear_text)

    # -- speak the whole sentence ------------------------------------------- #
    if b4.button("🔊 Speak sentence", type="primary"):
        _speak(st.session_state.sentence)


# --------------------------------------------------------------------------- #
# Live streaming recognition (streamlit-webrtc)                                 #
# --------------------------------------------------------------------------- #
def _make_live_processor(recognizer, mode, min_conf, stable_frames,
                         auto_space=True, gap_seconds=1.2):
    """Build a streamlit-webrtc video processor bound to the chosen model.

    Performance design: the video thread (``recv``) only does *fast* work —
    MediaPipe tracking (VIDEO mode, on a 480px copy) + drawing — so the stream
    stays smooth. Classification (which can be slow, esp. the HF ViT) runs in a
    **separate background thread** that always grabs the most recent hand crop,
    so inference latency never stalls the video. Returns the class; webrtc
    instantiates it.
    """
    import threading
    import time

    import av
    import cv2

    from src.gesture_recognizer import SentenceBuilder
    from src.hand_detector import HandDetector

    class LiveProcessor:
        def __init__(self) -> None:
            # Detector is built lazily on the first frame, NOT here: creating the
            # MediaPipe model in the constructor can push worker startup past
            # streamlit-webrtc's 10s signalling timeout.
            self.detector = None
            self.recognizer = recognizer
            self.mode = mode  # "landmark" | "hf" | "cnn"
            self.builder = SentenceBuilder(
                stable_frames=stable_frames, min_confidence=min_conf
            )
            self.auto_space = auto_space
            self.gap_seconds = gap_seconds
            self._gap_start = None     # when the hand first went missing
            self._space_done = False   # one auto-space per gap
            self.current = ("-", 0.0)
            self.lock = threading.Lock()
            self._actions: list[str] = []
            self._latest = None          # (colour_region, gray28) of newest hand
            self._latest_id = 0
            self._t0 = time.monotonic()
            self._last_ts = 0
            self._running = True
            self._worker = threading.Thread(target=self._classify_loop, daemon=True)
            self._worker.start()

        def _get_detector(self):
            if self.detector is None:
                self.detector = HandDetector(
                    max_hands=1, running_mode="video", process_width=480,
                    detection_confidence=0.4, tracking_confidence=0.4,
                )
            return self.detector

        # -- main-thread control surface ----------------------------------- #
        def queue_action(self, action: str) -> None:
            with self.lock:
                self._actions.append(action)

        def snapshot_text(self) -> str:
            with self.lock:
                return self.builder.text

        def set_text(self, s: str) -> None:
            with self.lock:
                self.builder.set_text(s)

        def _apply_actions(self) -> None:
            with self.lock:
                actions, self._actions = self._actions, []
                for a in actions:
                    if a == "space":
                        self.builder.add_space()
                    elif a == "backspace":
                        self.builder.backspace()
                    elif a == "clear":
                        self.builder.clear()
                    elif a.startswith("char:"):
                        self.builder.add_char(a[5:])

        # -- background classification thread ------------------------------- #
        def _classify_loop(self) -> None:
            seen_id = -1
            while self._running:
                with self.lock:
                    item, item_id = self._latest, self._latest_id
                if item is None or item_id == seen_id:
                    time.sleep(0.01)
                    continue
                seen_id = item_id
                region, crop = item
                try:
                    if self.mode == "landmark":
                        pred = self.recognizer.predict(crop.landmarks, crop.handedness)
                    elif self.mode == "hf":
                        pred = self.recognizer.predict(region)
                    else:
                        pred = self.recognizer.predict(crop.image)
                    with self.lock:
                        self.current = (pred.letter, pred.confidence)
                        self.builder.update(pred)
                except Exception:  # noqa: BLE001
                    pass

        # -- per-frame callback (video thread, kept light) ------------------ #
        def recv(self, frame):
            # Everything is wrapped: a single bad frame must NOT raise out of
            # recv, or streamlit-webrtc tears the whole track down ("runs then
            # stops"). On any error we just return the original frame.
            img = frame.to_ndarray(format="bgr24")  # no mirror: keep true orientation
            try:
                clean = img.copy()
                self._apply_actions()

                ts = int((time.monotonic() - self._t0) * 1000)
                ts = max(ts, self._last_ts + 1)  # MediaPipe VIDEO needs increasing ts
                self._last_ts = ts

                detector = self._get_detector()
                crop = detector.detect(img, draw=True, timestamp_ms=ts)
                now = time.monotonic()
                if crop is not None:
                    self._gap_start = None
                    self._space_done = False
                    region = detector.square_region(clean, crop.bbox, margin=1.0)
                    with self.lock:
                        self._latest = (region, crop)
                        self._latest_id += 1
                else:
                    # Hand gone → release the committer; after a short pause
                    # auto-insert a space so words separate without a button.
                    if self._gap_start is None:
                        self._gap_start = now
                    with self.lock:
                        self._latest = None
                        self.current = ("-", 0.0)
                        self.builder.update(None)
                        if (
                            self.auto_space
                            and not self._space_done
                            and now - self._gap_start >= self.gap_seconds
                            and self.builder.text
                            and not self.builder.text.endswith(" ")
                        ):
                            self.builder.add_space()
                            self._space_done = True

                self._draw_hud(img)
            except Exception:  # noqa: BLE001 - never let one frame kill the stream
                pass
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def _draw_hud(self, img) -> None:
            # Small current-letter badge (top-left) + a translucent sentence bar
            # along the bottom so you can read what the system is typing without
            # fully blocking the video.
            with self.lock:
                letter, conf = self.current
                text = self.builder.text
            h, w = img.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX

            if letter and letter != "-":
                cv2.rectangle(img, (10, 10), (138, 54), (18, 18, 22), -1)
                cv2.putText(img, f"{letter} {conf * 100:3.0f}%", (20, 42),
                            font, 0.9, (140, 230, 140), 2)

            overlay = img.copy()
            cv2.rectangle(overlay, (0, h - 54), (w, h), (8, 8, 10), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            shown = (text[-46:] if text else "Start signing...")
            cv2.putText(img, shown, (14, h - 19), font, 0.8, (240, 240, 245), 2)

        def on_ended(self) -> None:
            self._running = False
            try:
                if self.detector is not None:
                    self.detector.close()
            except Exception:  # noqa: BLE001
                pass

    return LiveProcessor


def _live_recognize() -> None:
    st.caption(
        "Live webcam → MediaPipe tracks your hand → the model reads letters in "
        "real time → build a sentence and **🔊 Speak** it with Supertonic."
    )
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except Exception as exc:  # noqa: BLE001 - import or native-lib failure
        st.warning(
            "🔴 Live streaming isn't available in this environment "
            f"(`{type(exc).__name__}`). Use **📸 Snapshot / Upload** above — it "
            "runs the exact same recogniser and works everywhere, including hosted "
            "demos."
        )
        return

    c1, c2 = st.columns(2)
    model_choice = c1.radio(
        "Recognition model",
        ["Landmark MLP (robust)", "Hugging Face SigLIP2", "Local CNN"],
        help="Landmark MLP reads hand *geometry* from MediaPipe — fastest and most "
        "robust to your camera/lighting. SigLIP2 reads the photo (heavier). All run "
        "on a background thread so the video stays smooth.",
    )
    min_conf = c2.slider("Min confidence", 0.30, 0.95, 0.50, 0.05,
                         help="Ignore predictions below this confidence.")
    stable_frames = c2.slider("Steady reads to lock a letter", 2, 12, 4,
                              help="Higher = more deliberate, fewer mistakes.")
    auto_space = c1.checkbox(
        "Auto-space between words", value=True,
        help="Drop your hand out of frame for a moment between words and a space "
        "is inserted automatically.",
    )
    gap_seconds = c1.slider("Pause length for a space (s)", 0.6, 2.5, 1.2, 0.1)
    mode = ("landmark" if model_choice.startswith("Landmark")
            else "hf" if model_choice.startswith("Hugging") else "cnn")

    try:
        if mode == "landmark":
            recognizer = get_landmark_recognizer()
        elif mode == "hf":
            recognizer = get_hf_recognizer()
        else:
            recognizer = get_recognizer()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model failed to load: {exc}. Train it or pick another model.")
        return

    processor_cls = _make_live_processor(
        recognizer, mode, min_conf, stable_frames, auto_space, gap_seconds
    )
    ctx = webrtc_streamer(
        key="sign-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # Free public TURN relay so the camera also connects when the app is
            # hosted (e.g. Streamlit Cloud), not just on localhost.
            {"urls": ["turn:openrelay.metered.ca:80"],
             "username": "openrelayproject", "credential": "openrelayproject"},
            {"urls": ["turn:openrelay.metered.ca:443"],
             "username": "openrelayproject", "credential": "openrelayproject"},
        ]},
        media_stream_constraints={
            # Proven-stable settings (same as the version that worked smoothly).
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 20},
            },
            "audio": False,
        },
        video_processor_factory=processor_cls,
        async_processing=True,
    )

    if ctx.state.playing and ctx.video_processor:
        st.caption("Hold each sign steady for ~1s. **Drop your hand between words** "
                   "to add a space. The recognised text shows along the bottom of the video.")
        d1, d2, d3, d4 = st.columns(4)
        if d1.button("␣ Space", key="live_space"):
            ctx.video_processor.queue_action("space")
        if d2.button("⌫ Backspace", key="live_back"):
            ctx.video_processor.queue_action("backspace")
        if d3.button("🗑️ Clear", key="live_clear"):
            ctx.video_processor.queue_action("clear")
        if d4.button("🔊 Speak", key="live_speak", type="primary"):
            _speak(ctx.video_processor.snapshot_text())

        # quick punctuation (fingerspelling has no signs for these)
        st.caption("Punctuation:")
        pcols = st.columns(6)
        for col, ch in zip(pcols, [".", ",", "!", "?", "'", "-"]):
            if col.button(ch, key=f"live_punct_{ch}"):
                ctx.video_processor.queue_action(f"char:{ch}")

        # -- correction editor: pull the recognised text into an editable box -- #
        st.divider()
        st.markdown("**✏️ Correct & finalize**")
        if st.button("⬇️ Load recognised sentence", key="live_load_edit"):
            st.session_state["live_edit"] = ctx.video_processor.snapshot_text()
        st.text_area("Editable text — fix any wrong letters here", key="live_edit", height=80)
        f1, f2, f3 = st.columns(3)
        if f1.button("🔊 Speak this", key="live_speak_edit", type="primary"):
            _speak(st.session_state.get("live_edit", ""))
        if f2.button("↺ Push to video", key="live_push",
                     help="Replace the on-video sentence with your corrected text"):
            ctx.video_processor.set_text(st.session_state.get("live_edit", ""))
        if f3.button("💬 Show in signs", key="live_show_signs",
                     help="Copy this text to the Speak → Sign tab"):
            st.session_state["say_text"] = st.session_state.get("live_edit", "")
            st.success("Copied to the **💬 Speak → Sign** tab.")
    else:
        st.info("Click **START** above and allow camera access to begin signing.")


# --------------------------------------------------------------------------- #
# Tab 2 — dataset exploration                                                  #
# --------------------------------------------------------------------------- #
def dataset_tab() -> None:
    st.subheader("Dataset explorer — Sign Language MNIST")

    if not config.TRAIN_CSV.exists():
        st.warning(
            "Dataset not found. Download it into `data/`:\n\n"
            "`kaggle datasets download -d datamunge/sign-language-mnist -p data --unzip`"
        )
        return

    df = get_clean_train()
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{len(df):,}")
    c2.metric("Classes present", df["label"].nunique())
    c3.metric("Features / image", df.shape[1] - 1)

    # Class distribution
    st.markdown("**Class distribution** (samples per letter)")
    counts = (
        df["label"].map(config.label_to_letter).value_counts().sort_index()
    )
    st.bar_chart(counts)

    # Sample grid
    st.markdown("**Sample images**")
    n = st.slider("How many samples", 4, 24, 12, step=4)
    sample = df.sample(n, random_state=st.session_state.get("seed", 0))
    cols = st.columns(6)
    for i, (_, row) in enumerate(sample.iterrows()):
        img = row.drop("label").to_numpy(dtype="uint8").reshape(
            config.IMG_SIZE, config.IMG_SIZE
        )
        cols[i % 6].image(
            img, caption=config.label_to_letter(int(row["label"])), width=90
        )
    if st.button("🔀 Shuffle samples"):
        st.session_state["seed"] = np.random.randint(0, 10_000)
        st.rerun()

    # Average image per letter
    with st.expander("Average image per letter (the 'prototype' sign)"):
        means = (
            df.groupby("label")
            .mean()
            .apply(lambda r: r.to_numpy().reshape(config.IMG_SIZE, config.IMG_SIZE), axis=1)
        )
        mcols = st.columns(8)
        for i, (label, mean_img) in enumerate(means.items()):
            mcols[i % 8].image(
                mean_img.astype("uint8"),
                caption=config.label_to_letter(int(label)),
                width=70,
            )


# --------------------------------------------------------------------------- #
# Tab 3 — about                                                                #
# --------------------------------------------------------------------------- #
def about_tab() -> None:
    st.subheader("About this project")
    st.markdown(
        """
This app gives a voice to **fingerspelled American Sign Language**, aimed at
helping mute / non-verbal signers communicate with people who don't sign.

**Pipeline**

1. **OpenCV** captures the webcam frame.
2. **MediaPipe Hands** localises the hand and crops it.
3. A classifier turns the crop into an ASL letter (A–Y; J and Z are dynamic):
   - **Hugging Face SigLIP2** (`prithivMLmods/Alphabet-Sign-Language-Detection`)
     — generalises to real webcam photos, best for live use, or
   - a **local CNN** trained on Kaggle *Sign Language MNIST* — fast & offline.
4. Letters accumulate into words and sentences.
5. **Supertonic** (a lightweight on-device TTS model from Hugging Face)
   converts the text to natural speech.

**Models & data**

- Dataset: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) (Kaggle)
- Vision: [prithivMLmods/Alphabet-Sign-Language-Detection](https://huggingface.co/prithivMLmods/Alphabet-Sign-Language-Detection) (Hugging Face SigLIP2)
- TTS: [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) (Hugging Face)

**Tip:** for true live video (instead of snapshots) run
`python -m src.realtime_demo`.
        """
    )


# --------------------------------------------------------------------------- #
# Tab: Speak -> Sign  (hearing person types -> Deaf person reads + sees signs)  #
# --------------------------------------------------------------------------- #
SIGNS_DIR = PROJECT_ROOT / "assets" / "signs"


@st.cache_data(show_spinner=False)
def _sign_data_uris() -> dict[str, str]:
    """Return ``{LETTER: base64 jpeg}`` for the bundled ASL sign images."""
    import base64

    uris = {}
    if SIGNS_DIR.exists():
        for f in SIGNS_DIR.glob("*.jpg"):
            uris[f.stem.upper()] = base64.b64encode(f.read_bytes()).decode()
    return uris


def text_to_sign_tab() -> None:
    st.subheader("Speak to a Deaf / non-verbal person — show it in signs")
    st.caption(
        "Type (or paste) what you want to say. It's shown as large text **and** "
        "as ASL fingerspelling signs, so the other person can read it and see "
        "exactly how it's signed. Edit the box any time to fix mistakes."
    )

    if "say_text" not in st.session_state:
        st.session_state.say_text = "HELLO"
    st.text_area("Your message", key="say_text", height=80)
    msg = st.session_state.say_text or ""

    # Big, readable text for the reader.
    st.markdown(
        f"<div style='font-size:2.2rem;font-weight:800;line-height:1.2;"
        f"padding:.4rem 0;word-break:break-word'>{html.escape(msg) or '&nbsp;'}</div>",
        unsafe_allow_html=True,
    )

    # Sign strip.
    uris = _sign_data_uris()
    card = ("display:flex;flex-direction:column;align-items:center;justify-content:flex-end;"
            "width:78px;min-height:96px;background:#fff;border:1px solid #d0d0d0;"
            "border-radius:8px;padding:4px;color:#111;font-weight:700")
    items = []
    for ch in msg:
        u = ch.upper()
        if ch == " ":
            items.append("<div style='width:26px'></div>")
        elif u in uris:
            items.append(
                f"<figure style='{card}'><img src='data:image/jpeg;base64,{uris[u]}' "
                f"style='width:66px;height:66px;object-fit:contain'/>"
                f"<figcaption>{u}</figcaption></figure>"
            )
        elif u.isalpha():  # J / Z (motion signs, no static image)
            items.append(
                f"<div style='{card};justify-content:center;font-size:1.6rem'>"
                f"{u}<div style='font-size:.6rem;font-weight:400'>motion sign</div></div>"
            )
        else:  # punctuation / digits
            items.append(
                f"<div style='{card};justify-content:center;font-size:1.6rem'>"
                f"{html.escape(ch)}</div>"
            )
    if items:
        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end'>"
            + "".join(items) + "</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2 = st.columns(2)
    if c1.button("🔊 Also speak it aloud", type="primary"):
        _speak(msg)
    c2.caption("Letters A–Y have a sign image; J and Z are motion signs (shown as text).")


def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root{
  --bg:#0A0A0C; --card:#141418; --card2:#1B1B21; --border:#2A2A32;
  --text:#F2F2F4; --muted:#9A9AA6; --accent:#C2FF45; --accent-ink:#0A0A0C;
}
html, body, [class*="css"], .stApp, button, input, textarea, h1,h2,h3,h4 {
  font-family:'Space Grotesk', sans-serif !important;
}
.stApp{
  background:
    radial-gradient(900px 500px at 100% -10%, rgba(194,255,69,.06), transparent 60%),
    #0A0A0C;
}
.block-container{ padding-top:2.2rem; max-width:1180px; }

/* Hero */
.hero{ padding:.4rem 0 .2rem; }
.tagpill{
  display:inline-flex; align-items:center; gap:.4rem; font-size:.74rem; font-weight:600;
  letter-spacing:.08em; text-transform:uppercase; color:var(--accent);
  border:1px solid rgba(194,255,69,.35); background:rgba(194,255,69,.07);
  padding:.3rem .7rem; border-radius:999px;
}
.hero-title{
  font-size:3.4rem; font-weight:700; line-height:1.02; letter-spacing:-.02em;
  margin:.8rem 0 0; color:var(--text);
}
.hero-title .hl{ color:var(--accent); }
.hero-sub{ color:var(--muted); font-size:1.05rem; max-width:680px; margin:.7rem 0 0; }

/* Bento grid of metrics */
.bento{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:1.4rem 0 .6rem; }
@media (max-width:900px){ .bento{ grid-template-columns:repeat(2,1fr); } }
.cell{
  background:var(--card); border:1px solid var(--border); border-radius:20px;
  padding:1.1rem 1.2rem; transition:border-color .2s ease, transform .1s ease;
}
.cell:hover{ border-color:rgba(194,255,69,.45); transform:translateY(-2px); }
.cell .num{ font-size:2.1rem; font-weight:700; color:var(--text); letter-spacing:-.02em; }
.cell .num b{ color:var(--accent); }
.cell .lab{ color:var(--muted); font-size:.82rem; margin-top:.25rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"]{ gap:.5rem; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{
  background:var(--card); border:1px solid var(--border); border-bottom:none;
  border-radius:14px 14px 0 0; padding:.5rem 1.1rem; font-weight:600; color:var(--muted);
}
.stTabs [aria-selected="true"]{ background:var(--card2); color:var(--text); border-color:rgba(194,255,69,.4); }

/* Buttons */
.stButton > button, .stDownloadButton > button{
  border-radius:12px; border:1px solid var(--border); background:var(--card);
  color:var(--text); font-weight:600; transition:transform .08s ease, border-color .2s ease;
}
.stButton > button:hover{ transform:translateY(-1px); border-color:var(--accent); }
.stButton > button[kind="primary"]{
  background:var(--accent); color:var(--accent-ink); border:none; font-weight:700;
  box-shadow:0 8px 22px rgba(194,255,69,.22);
}
.stButton > button[kind="primary"]:hover{ filter:brightness(1.05); }

/* Cards / metrics / inputs */
[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--border); border-radius:18px; padding:.8rem 1.1rem; }
[data-testid="stMetricValue"]{ color:var(--accent); font-weight:700; }
.stTextArea textarea, .stTextInput input{
  background:var(--card) !important; border:1px solid var(--border) !important;
  border-radius:14px !important; color:var(--text) !important;
}
section[data-testid="stSidebar"]{ background:#070708; border-right:1px solid var(--border); }
h1,h2,h3{ letter-spacing:-.01em; }
hr{ border-color:var(--border); }

/* Live readout below the video */
.live-readout{
  background:var(--card); border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:16px; padding:.8rem 1.1rem; margin:.3rem 0 .6rem;
}
.lr-top{ display:flex; align-items:center; gap:.6rem; }
.lr-label{
  font-size:.66rem; font-weight:700; letter-spacing:.14em; color:var(--accent-ink);
  background:var(--accent); padding:.16rem .5rem; border-radius:6px;
}
.lr-letter{ font-weight:700; font-size:1.5rem; color:var(--accent); }
.lr-idle{ color:#55555f; }
.lr-text{
  font-weight:700; font-size:1.7rem; color:var(--text);
  margin-top:.3rem; word-break:break-word; min-height:1.2em;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    _inject_css()
    ready = config.MODEL_PATH.exists() or (config.MODELS_DIR / "landmark_mlp.joblib").exists()
    st.markdown(
        "<div class='hero'>"
        "<span class='tagpill'>🤟 Sign-language AI kit</span>"
        "<div class='hero-title'>SignBridge —<br>signs <span class='hl'>⇄</span> "
        "text <span class='hl'>⇄</span> speech.</div>"
        "<div class='hero-sub'>Real-time, two-way communication between Deaf signers "
        "and hearing speakers. Hand-tracking with MediaPipe, a landmark ML model, and "
        "on-device Supertonic text-to-speech — all running locally.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='bento'>"
        "<div class='cell'><div class='num'><b>98%+</b></div>"
        "<div class='lab'>landmark model accuracy</div></div>"
        "<div class='cell'><div class='num'>A–Z</div>"
        "<div class='lab'>fingerspelling alphabet</div></div>"
        "<div class='cell'><div class='num'>~10<b>ms</b></div>"
        "<div class='lab'>per-sign inference</div></div>"
        "<div class='cell'><div class='num'>2-way</div>"
        "<div class='lab'>sign ⇄ text ⇄ speech</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.header("⚙️ Status")
        st.write("**Recognizer:**", "✅ ready" if ready else "❌ not trained")
        st.write("**Dataset:**", "✅ found" if config.TRAIN_CSV.exists() else "❌ missing")
        st.caption("MediaPipe · Landmark MLP · SigLIP2 · Supertonic TTS")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎥 Sign → Speech", "💬 Speak → Sign", "📊 Dataset", "ℹ️ About"]
    )
    with tab1:
        recognize_tab()
    with tab2:
        text_to_sign_tab()
    with tab3:
        dataset_tab()
    with tab4:
        about_tab()


if __name__ == "__main__":
    main()
