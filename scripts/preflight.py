"""Verify model artifacts, response contracts, and deployment prerequisites."""

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_sorter.cli import default_model, default_segmenter
from vision_sorter.pipeline import Sorter
from vision_sorter.schemas import Decision
from vision_sorter.settings import Settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--require-segmentation", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    Settings.from_env().validate_binding(args.host)
    sorter = Sorter(default_model(), segmenter_dir=default_segmenter())
    if args.require_segmentation and sorter.segmenter is None:
        raise ValueError("A real semantic segmentation artifact is required")
    image = Image.new("RGB", (320, 240), (45, 55, 50))
    ImageDraw.Draw(image).rectangle((100, 60, 200, 180), fill=(180, 180, 180))
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    result = sorter.infer(raw.getvalue())
    Decision.model_validate(result)
    if result["simulation"]["executed"]:
        raise ValueError("Unreviewed preflight input unexpectedly authorized a pick")
    print(
        json.dumps(
            dict(
                status="passed",
                model_version=sorter.manifest["model_version"],
                segmentation_model_version=sorter.segmenter.manifest["model_version"]
                if sorter.segmenter
                else None,
                mask_source=result.get("mask_source"),
                response_schema="1.0",
                hardware_enabled=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
