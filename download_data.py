"""
Downloads and unpacks the official DialogueRNN feature pickles
(IEMOCAP, MELD, AVEC) from the DeCLaRe Lab repo.

Usage:
    python download_data.py
"""

import os
import zipfile
import shutil
import urllib.request

DATA_DIR = "data"
ZIP_URL = "https://github.com/declare-lab/conv-emotion/raw/master/DialogueRNN/DialogueRNN_features.zip"
ZIP_PATH = os.path.join(DATA_DIR, "DialogueRNN_features.zip")

IEMOCAP_PKL = os.path.join(
    DATA_DIR, "DialogueRNN_features", "IEMOCAP_features", "IEMOCAP_features_raw.pkl"
)


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(IEMOCAP_PKL):
        print(f"Already present: {IEMOCAP_PKL}")
        return

    print("Downloading DialogueRNN_features.zip ...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)

    size_kb = os.path.getsize(ZIP_PATH) / 1024
    if size_kb < 100:
        raise RuntimeError(
            f"Downloaded file is only {size_kb:.1f} KB — likely a 404 page, "
            f"not the real zip. Check the URL: {ZIP_URL}"
        )

    print(f"Downloaded {size_kb:.1f} KB. Unzipping ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    # clean up macOS metadata junk and the zip itself
    macosx_junk = os.path.join(DATA_DIR, "__MACOSX")
    if os.path.exists(macosx_junk):
        shutil.rmtree(macosx_junk)
    os.remove(ZIP_PATH)

    if not os.path.exists(IEMOCAP_PKL):
        raise RuntimeError(
            f"Extraction finished but {IEMOCAP_PKL} not found — "
            f"the repo's internal folder structure may have changed."
        )

    print(f"Done. IEMOCAP features at: {IEMOCAP_PKL}")


if __name__ == "__main__":
    download()