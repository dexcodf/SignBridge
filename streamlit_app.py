"""Deployment entry point (kept next to requirements.txt / packages.txt).

Streamlit Community Cloud installs the dependency files that sit beside the main
app file, so this thin launcher lives in the project root and simply runs the
real dashboard in ``app/streamlit_app.py``.

Deploy on share.streamlit.io with:
    Main file path:  sign_language_tts/streamlit_app.py
"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "app" / "streamlit_app.py"),
    run_name="__main__",
)
