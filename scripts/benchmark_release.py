"""Measure local CPU pipeline latency; never confound it with task accuracy."""

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_sorter.pipeline import Sorter
from vision_sorter.schemas import Decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    sorter = Sorter("artifacts/real-model", segmenter_dir="artifacts/segmentation-model")
    manifest = sorter.segmenter.manifest
    ids = manifest["splits"]["test"].copy()
    random.Random(42).shuffle(ids)
    ids = ids[: args.samples]
    root = Path("data/processed/taco-all")
    raw = [(root / manifest["records"][i]["image"]).read_bytes() for i in ids]
    sorter.infer(raw[0])  # excluded warmup
    latencies = []
    for image in raw:
        start = time.perf_counter()
        result = sorter.infer(image)
        Decision.model_validate(result)
        latencies.append((time.perf_counter() - start) * 1000)
        if result["simulation"]["executed"]:
            raise ValueError("Unreviewed benchmark image authorized a pick")
    report = dict(
        device="CPU",
        platform=platform.platform(),
        torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),
        sample_count=len(ids),
        seed=42,
        classifier=sorter.manifest["model_version"],
        segmenter=manifest["model_version"],
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        min_ms=min(latencies),
        max_ms=max(latencies),
        measurement="Sequential full pipeline plus schema validation; disk read and HTTP transfer excluded.",
        image_source="Seeded sample of held-out TACO author-hosted thumbnails",
        limitations=[
            "Not a throughput or concurrent-load benchmark; target deployment hardware will differ."
        ],
    )
    Path("docs/latency.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
