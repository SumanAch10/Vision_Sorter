"""CLI entry point for the complete local workflow."""

import argparse
import json
import os
from pathlib import Path

from .pipeline import Sorter
from .training import generate_demo, train


def default_model():
    configured = os.getenv("SORTER_MODEL")
    if configured:
        return configured
    if Path("artifacts/real-model/manifest.json").is_file():
        return "artifacts/real-model"
    return "artifacts/demo-model"


def default_segmenter():
    if "SORTER_SEGMENTER" in os.environ:
        return os.environ["SORTER_SEGMENTER"] or None
    if Path("artifacts/segmentation-model/manifest.json").is_file():
        return "artifacts/segmentation-model"
    return None


def main():
    parser = argparse.ArgumentParser(prog="vision-sorter")
    subs = parser.add_subparsers(dest="command", required=True)
    demo = subs.add_parser(
        "demo", help="Generate synthetic data, train a demo model, and launch the UI"
    )
    demo.add_argument("--epochs", type=int, default=12)
    demo.add_argument("--no-serve", action="store_true")
    tr = subs.add_parser("train")
    tr.add_argument("dataset")
    tr.add_argument("--output", default="artifacts/model")
    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--seed", type=int, default=42)
    classifier = subs.add_parser(
        "train-classifier", help="Train real-photo material classification from class folders"
    )
    classifier.add_argument("dataset")
    classifier.add_argument("--output", default="artifacts/real-model")
    classifier.add_argument("--pretrained", default="artifacts/downloads/resnet18-f37072fd.pth")
    classifier.add_argument("--epochs", type=int, default=150)
    classifier.add_argument("--seed", type=int, default=42)
    fine = subs.add_parser(
        "fine-tune", help="Fine-tune a real-photo model using its recorded train/validation splits"
    )
    fine.add_argument("dataset")
    fine.add_argument("--model", default="artifacts/real-model")
    fine.add_argument("--output", default="artifacts/tuned-model")
    fine.add_argument("--epochs", type=int, default=4)
    fine.add_argument("--seed", type=int, default=42)
    segmentation = subs.add_parser(
        "train-segmenter", help="Train real binary litter semantic segmentation"
    )
    segmentation.add_argument("dataset")
    segmentation.add_argument("--output", default="artifacts/segmentation-model")
    segmentation.add_argument("--pretrained", default="artifacts/downloads/resnet18-f37072fd.pth")
    segmentation.add_argument("--epochs", type=int, default=20)
    segmentation.add_argument("--seed", type=int, default=42)
    segmentation.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    for name in ("serve", "infer"):
        p = subs.add_parser(name)
        p.add_argument("--model", default=default_model())
        p.add_argument(
            "--segmenter",
            default=default_segmenter(),
            help="Real segmentation model; use an empty string to disable",
        )
        p.add_argument("--calibration", default=os.getenv("SORTER_CALIBRATION"))
        p.add_argument(
            "--threshold", type=float, default=float(os.getenv("SORTER_THRESHOLD", ".85"))
        )
        if name == "infer":
            p.add_argument("image")
            p.add_argument("--output")
        else:
            p.add_argument("--host", default=os.getenv("SORTER_HOST", "127.0.0.1"))
            p.add_argument("--port", type=int, default=int(os.getenv("SORTER_PORT", "8000")))
    cal = subs.add_parser("calibrate")
    cal.add_argument("points", help="JSON with pixels, millimeters, image_size, workspace_mm")
    cal.add_argument("--output", default="artifacts/calibration.json")
    args = parser.parse_args()
    if args.command == "train-segmenter":
        from .litter_segmentation import train_segmenter

        train_segmenter(
            args.dataset, args.output, args.pretrained, args.epochs, args.seed, args.device
        )
    elif args.command == "fine-tune":
        from .classification import fine_tune_classifier

        fine_tune_classifier(args.dataset, args.model, args.output, args.epochs, args.seed)
    elif args.command == "train-classifier":
        from .classification import train_classifier

        train_classifier(args.dataset, args.output, args.pretrained, args.epochs, args.seed)
    elif args.command == "train":
        train(args.dataset, args.output, args.epochs, args.seed)
    elif args.command == "calibrate":
        from .calibration import fit_calibration

        data = json.loads(Path(args.points).read_text())
        result = fit_calibration(
            data["pixels"], data["millimeters"], data["image_size"], data["workspace_mm"]
        )
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(
            "Calibration written with validated=false. Verify against independent reference points before approving."
        )
    elif args.command == "demo":
        dataset = generate_demo("data/processed/demo")
        train(dataset, "artifacts/demo-model", args.epochs, demo=True)
        print("Demo image: data/processed/demo/0.png; all decisions require review.")
        if not args.no_serve:
            from .server import serve

            serve(Sorter("artifacts/demo-model"))
    else:
        architecture = json.loads((Path(args.model) / "manifest.json").read_text()).get(
            "architecture"
        )
        segmenter = args.segmenter if architecture == "resnet18-material" else None
        sorter = Sorter(args.model, args.calibration, args.threshold, segmenter)
        if args.command == "serve":
            from .server import serve

            serve(sorter, args.host, args.port)
        else:
            result = json.dumps(sorter.infer(Path(args.image).read_bytes()), indent=2)
            if args.output:
                Path(args.output).write_text(result)
            else:
                print(result)


if __name__ == "__main__":
    main()
