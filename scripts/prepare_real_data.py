"""Download author-published TrashNet and official PyTorch weights with checksums.

Run from the repository root: python scripts/prepare_real_data.py
Files are cached locally; source images and checkpoints stay out of Git.
"""

import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download(url, destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        subprocess.run(
            ["curl", "-fL", "--retry", "2", "--max-time", "180", url, "-o", str(temporary)],
            check=True,
        )
        temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if expected and not digest.startswith(expected):
        raise ValueError(f"Checksum mismatch for {destination}; remove it and retry")
    print(destination.relative_to(ROOT), digest)


def main():
    archive = ROOT / "artifacts/downloads/trashnet.zip"
    download(
        "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip",
        archive,
        "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2",
    )
    root = ROOT / "data/raw/trashnet"
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if not (root / info.filename).resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive path")
        bundle.extractall(root)
    download(
        "https://raw.githubusercontent.com/garythung/trashnet/master/LICENSE", root / "LICENSE"
    )
    download(
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        ROOT / "artifacts/downloads/resnet18-f37072fd.pth",
        "f37072fd",
    )
    print("Ready: train-classifier data/raw/trashnet/dataset-resized")


if __name__ == "__main__":
    main()
