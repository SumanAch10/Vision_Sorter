"""Create a portable model bundle, checksums, and public model evidence."""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_sorter.pipeline import Sorter
from vision_sorter.server import model_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/real-model")
    parser.add_argument("--segmenter", default="artifacts/segmentation-model")
    parser.add_argument("--output", default="artifacts/releases")
    args = parser.parse_args()
    sorter = Sorter(args.model, segmenter_dir=args.segmenter)
    if sorter.manifest.get("demo") or sorter.segmenter.manifest.get("demo"):
        raise ValueError("Release requires real-data model artifacts")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "vision-sorter-models-0.3.0.tar.gz"
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        for name, source in [
            ("real-model", Path(args.model)),
            ("segmentation-model", Path(args.segmenter)),
        ]:
            destination = base / name
            destination.mkdir()
            for file in ("weights.pt", "manifest.json"):
                shutil.copy2(source / file, destination / file)
        (base / "model-card.json").write_text(json.dumps(model_summary(sorter), indent=2))
        (base / "MODEL_BUNDLE_README.txt").write_text(
            "Extract into artifacts/. Contains model weights and manifests, not source photographs. Both models are unapproved for robot use. See docs/DEPLOYMENT.md for runtime configuration.\n"
        )
        for filename in ("TrashNet-LICENSE.txt", "TACO-LICENSE.txt", "DATA_SOURCES.md"):
            shutil.copy2(ROOT / "docs" / filename, base / filename)
        with tarfile.open(archive, "w:gz") as bundle:
            for file in sorted(base.rglob("*")):
                if file.is_file():
                    bundle.add(file, arcname=str(file.relative_to(base)), recursive=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n")
    (out / "model-card.json").write_text(json.dumps(model_summary(sorter), indent=2))
    print(
        json.dumps(
            dict(archive=str(archive), sha256=digest, bytes=archive.stat().st_size), indent=2
        )
    )


if __name__ == "__main__":
    main()
