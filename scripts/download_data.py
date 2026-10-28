"""Download the FinTech Credit Risk Dataset (ref. [33]) from Google Drive.

The dataset is shared as an .xlsx file in Google Drive.  If the automatic
download fails (Drive sometimes asks for confirmation on large files), open
the link in a browser, choose File → Download → CSV, and save it as
``data/fintech_credit_risk.csv``.
"""
import argparse
import urllib.request
from pathlib import Path

FILE_ID = "1mx1Oin1HUaaqQx8_rSoCK6oJbJsenZs1"
URLS = [
    f"https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=xlsx",
    f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm=t",
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/fintech_credit_risk.xlsx")
args = ap.parse_args()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

for url in URLS:
    try:
        print(f"trying {url}")
        urllib.request.urlretrieve(url, out)
        if out.stat().st_size > 10_000 and out.read_bytes()[:2] == b"PK":  # xlsx = zip
            print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
            print("set data.path in configs/default.yaml to this file")
            break
    except Exception as exc:  # noqa: BLE001
        print(f"  failed: {exc}")
else:
    print("Automatic download failed — please download the sheet manually (see data/README.md).")
