# 🤟 Sign Language → Text → Speech

Give a voice to fingerspelled **American Sign Language**. The app reads ASL hand
signs from a webcam, turns them into text, and speaks the text aloud — built to
help mute / non-verbal signers communicate with people who don't sign.

```
Webcam ──▶ OpenCV ──▶ MediaPipe Hands ──▶ CNN classifier ──▶ letters
                                                                │
                                            sentence builder ◀──┘
                                                                │
                                          Supertonic TTS ──▶ 🔊 speech
```

## What's inside

| Piece | Tech | File |
|-------|------|------|
| Hand localisation & crop | OpenCV + MediaPipe Hands | `src/hand_detector.py` |
| Letter classifier (robust, default) | **Landmark MLP** on MediaPipe hand geometry | `src/landmark_recognizer.py`, `src/train_landmarks.py` |
| Letter classifier (real photos) | **HF SigLIP2** `prithivMLmods/Alphabet-Sign-Language-Detection` | `src/hf_recognizer.py` |
| Letter classifier (offline) | Keras CNN on **Sign Language MNIST** (Kaggle) | `src/train_model.py`, `src/gesture_recognizer.py` |
| Text → speech | **Supertonic** (Hugging Face, ONNX, on-device) | `src/tts_engine.py` |
| Live demo | OpenCV window | `src/realtime_demo.py` |
| Dashboard | Streamlit | `app/streamlit_app.py` |
| Data cleaning + EDA + training | Jupyter | `notebooks/sign_language_eda.ipynb` |

Three interchangeable recognizers (pick in the dashboard's **Recognize** tab):
the **Landmark MLP** (default) classifies the *geometry* of MediaPipe's 21 hand
landmarks — appearance-invariant, so it's the most robust across cameras and
lighting (99.5% held-out, incl. the hard fist-family letters); the **HF SigLIP2**
model classifies the photo; the **local CNN** is fast/offline but tuned to clean
dataset crops.

Retrain the landmark model on any folder-per-letter image set:

```bash
kaggle datasets download -d grassknoted/asl-alphabet -p /tmp/asl --unzip
python -m src.train_landmarks --data-dir /tmp/asl/asl_alphabet_train/asl_alphabet_train --per-class 300
# -> models/landmark_mlp.joblib
```

- **Dataset:** [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) — 28×28 grayscale ASL letters (A–Y, excluding the motion-based J and Z).
- **Vision model:** [prithivMLmods/Alphabet-Sign-Language-Detection](https://huggingface.co/prithivMLmods/Alphabet-Sign-Language-Detection) — a SigLIP2 image classifier that generalises to real ASL photos (7/7 on the Kaggle ASL-Alphabet test set).
- **TTS model:** [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) — a ~99M-param multilingual TTS that runs entirely on CPU.

## Setup

> Python **3.10–3.12** (MediaPipe + TensorFlow wheels). Tested on 3.12.

```bash
cd sign_language_tts
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

> **Windows + OneDrive note:** this project folder lives under OneDrive. Do
> **not** put the `.venv` here — OneDrive tries to sync the ~10k files
> TensorFlow installs, which stalls `pip` for a very long time, and the deep
> paths can break Windows' 260-char limit. Create the venv on a **local, short
> path** instead:
>
> ```powershell
> py -3.12 -m venv C:\Users\%USERNAME%\venvs\signtts
> C:\Users\%USERNAME%\venvs\signtts\Scripts\Activate.ps1
> pip install -r requirements.txt
> ```
>
> Requirements also use `ipykernel` rather than the full `jupyter` metapackage,
> whose deeply-nested JupyterLab widget files are the worst long-path offender.
> The notebook still runs fine in **VS Code**. If you specifically want
> JupyterLab, enable long paths first (admin PowerShell):
> `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`

### 1. Get the data

Configure the Kaggle CLI (`~/.kaggle/kaggle.json`), then:

```bash
python -m src.download_data
# or manually:
kaggle datasets download -d datamunge/sign-language-mnist -p data --unzip
```

You should end up with `data/sign_mnist_train.csv` and `data/sign_mnist_test.csv`.

### 2. Explore the data & train (notebook)

Open `notebooks/sign_language_eda.ipynb` in **VS Code** (or any Jupyter UI) and
select the `.venv` interpreter as the kernel. The notebook covers data cleaning,
EDA / visualisations, and trains the CNN.
Prefer the command line? Train directly:

```bash
python -m src.train_model --epochs 15      # saves models/sign_cnn.keras
```

### 3. Run it

**Streamlit dashboard** (recognition + sentence builder + TTS + dataset explorer):

```bash
streamlit run app/streamlit_app.py
```

The **Recognize** tab has two modes: **🔴 Live camera** (real-time webcam
streaming via `streamlit-webrtc` — click *START*, allow camera, sign, then
*Speak*) and **📸 Snapshot / Upload** (one photo at a time).

**Live webcam demo** (standalone OpenCV window, no browser):

```bash
python -m src.realtime_demo
# SPACE=space  B=backspace  C=clear  S=speak  Q=quit
```

## Project layout

```
sign_language_tts/
├── app/
│   └── streamlit_app.py        # dashboard (3 tabs)
├── notebooks/
│   └── sign_language_eda.ipynb # cleaning, EDA, visualisation, training
├── src/
│   ├── config.py               # paths, geometry, label mapping
│   ├── data_loader.py          # load + clean Sign Language MNIST
│   ├── download_data.py        # Kaggle download helper
│   ├── train_model.py          # CNN training
│   ├── hand_detector.py        # OpenCV + MediaPipe
│   ├── gesture_recognizer.py   # model inference + sentence smoothing
│   ├── tts_engine.py           # Supertonic TTS (+ pyttsx3 fallback)
│   └── realtime_demo.py        # live OpenCV loop
├── data/    models/    outputs/
└── requirements.txt
```

## How recognition stays stable

A webcam jitters, so a single noisy frame shouldn't commit a letter.
`SentenceBuilder` only accepts a letter once the CNN has agreed on it for
several consecutive frames above a confidence threshold — turning a noisy
per-frame stream into deliberate fingerspelling.

## Notes & limitations

- Sign Language MNIST is **static fingerspelling**, so dynamic letters **J** and
  **Z** are out of scope, as are full word-level signs.
- The model is trained on clean, centred crops; good lighting and a plain
  background improve webcam accuracy.
- If `supertonic` isn't installed, TTS automatically falls back to the offline
  `pyttsx3` OS voice so the app still talks.
