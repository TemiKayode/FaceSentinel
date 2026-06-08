"""
00_download_test_faces.py
=========================
Downloads three royalty-free, public-domain portrait PHOTOGRAPHS into
./test_faces/ for use in pipeline verification tests.

These are actual photographs (not paintings) so ArcFace can detect faces.

Usage
-----
    .\\venv\\Scripts\\python.exe scripts/00_download_test_faces.py
"""
import sys
import urllib.request
from pathlib import Path

DEST = Path("./test_faces")
DEST.mkdir(exist_ok=True)

# Direct full-resolution files (no thumbnail sizing restrictions).
# All are public-domain photographs from Wikimedia Commons.
SAMPLES = {
    "alice.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/d/d3/"
        "Albert_Einstein_Head.jpg"
    ),
    "bob.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/1/1c/"
        "Charles_Darwin_by_Julia_Margaret_Cameron_2.jpg"
    ),
    "carol.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/a/ab/"
        "Abraham_Lincoln_O-77_matte_collodion_print.jpg"
    ),
}

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def download(filename: str, url: str) -> bool:
    dest = DEST / filename
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True
    print(f"  Downloading {filename} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            data = resp.read()
            f.write(data)
        size = dest.stat().st_size
        print(f"  OK  {filename}  ({size:,} bytes)")
        return True
    except Exception as exc:
        print(f"  FAIL  {filename}: {exc}")
        return False


def main() -> None:
    print(f"Saving test images to {DEST.resolve()}\n")
    results = {name: download(name, url) for name, url in SAMPLES.items()}
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        print("Place your own JPEG photos in test_faces/ and retry.")
        sys.exit(1)
    print(
        "\nDone. 3 public-domain portrait photos saved."
        "\nFor reliable biometric testing, replace with clear frontal"
        "\nJPEG photos of real people in test_faces/."
    )


if __name__ == "__main__":
    main()
