"""Download the Sign Language MNIST dataset from Kaggle into ``data/``.

Prerequisites: the Kaggle CLI configured with an API token at
``~/.kaggle/kaggle.json`` (Account -> Create New API Token on kaggle.com).

    python -m src.download_data

If the CLI/token is missing, follow the printed manual instructions.
"""
from __future__ import annotations

import subprocess
import sys

from . import config

DATASET = "datamunge/sign-language-mnist"


def main() -> int:
    if config.TRAIN_CSV.exists() and config.TEST_CSV.exists():
        print(f"Dataset already present in {config.DATA_DIR}")
        return 0

    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", DATASET, "-p", str(config.DATA_DIR), "--unzip",
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(
            f"\nAutomatic download failed ({exc}).\n"
            "Manual steps:\n"
            "  1. pip install kaggle\n"
            "  2. Put your kaggle.json token in ~/.kaggle/ (chmod 600)\n"
            f"  3. kaggle datasets download -d {DATASET} -p data --unzip\n"
            "Or download the CSVs from the dataset page and drop them in data/."
        )
        return 1

    print(f"Done. CSVs in {config.DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
