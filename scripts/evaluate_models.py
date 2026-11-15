"""Compare the synthetic baseline with a real-photo classifier on shared classes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_sorter.classification import collect_records
from vision_sorter.pipeline import Sorter
from vision_sorter.training import tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/raw/trashnet/dataset-resized")
    parser.add_argument("--model", default="artifacts/real-model")
    parser.add_argument("--baseline", default="artifacts/demo-model")
    parser.add_argument("--output", default="docs/evaluation.json")
    args = parser.parse_args()
    root = Path(args.dataset)
    candidate = Sorter(args.model)
    m = candidate.manifest
    records = collect_records(root)
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if fingerprint != m["dataset_sha256"]:
        raise ValueError("Evaluation dataset does not match recorded training/test records")
    rows = [
        m["records"][i]
        for i in m["splits"]["test"]
        if m["classes"][m["records"][i]["class_id"]] in ("metal", "plastic", "paper")
    ]
    model = Sorter(args.baseline).model
    guesses = []
    with torch.inference_mode():
        for start in range(0, len(rows), 24):
            batch = []
            for row in rows[start : start + 24]:
                with Image.open(root / row["image"]) as im:
                    batch.append(tensor(im.convert("RGB")))
            labels = model(torch.stack(batch)).argmax(1)
            counts = torch.stack([(labels == i).sum((1, 2)) for i in range(1, 5)], 1)
            ids = counts.argmax(1).tolist()
            guesses.extend([("metal", "plastic", "paper", "trash")[i] for i in ids])
    correct = sum(p == m["classes"][r["class_id"]] for p, r in zip(guesses, rows))
    confusion = m["test"]["confusion_matrix"]
    subset = [m["classes"].index(x) for x in ("metal", "plastic", "paper")]
    real_correct = sum(confusion[i][i] for i in subset)
    report = dict(
        comparison="Same held-out metal/plastic/paper images; synthetic can/bottle labels mapped to metal/plastic",
        sample_count=len(rows),
        synthetic_baseline_accuracy=correct / len(rows),
        real_classifier_accuracy=real_correct / len(rows),
        six_class_test=m["test"],
        uncalibrated_six_class_test=m["test_uncalibrated"],
        model_version=m["model_version"],
        seed=m["seed"],
        limitations=m["limitations"],
    )
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("six_class_test", "uncalibrated_six_class_test", "limitations")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
