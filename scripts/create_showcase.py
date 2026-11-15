"""Generate public project visuals from measured results, without dataset photos."""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_sorter.pipeline import Sorter
from vision_sorter.server import model_summary

BG = "#0d1714"
PANEL = "#172b20"
INK = "#edf6ee"
MUTED = "#a6bdad"
GREEN = "#b6f28f"
GOLD = "#eac78a"


def base(title, subtitle):
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.text(0.06, 0.93, "VISION SORTER / PROJECT NOTES", color=GREEN, fontsize=13, weight="bold")
    fig.text(0.06, 0.84, title, color=INK, fontsize=33, weight="bold")
    fig.text(0.06, 0.79, subtitle, color=MUTED, fontsize=15)
    fig.text(
        0.06,
        0.045,
        "Research prototype · simulation only · classification and segmentation evaluated separately",
        color=MUTED,
        fontsize=11,
    )
    return fig


def box(fig, x, y, width, height, color=PANEL):
    fig.add_artist(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color,
            edgecolor="#38543f",
            zorder=0,
        )
    )


def main():
    out = ROOT / "artifacts/showcase"
    out.mkdir(parents=True, exist_ok=True)
    sorter = Sorter(
        ROOT / "artifacts/real-model", segmenter_dir=ROOT / "artifacts/segmentation-model"
    )
    classifier = sorter.manifest
    segmenter = sorter.segmenter.manifest
    evidence = model_summary(sorter)
    tests = (ROOT / "artifacts/test-results.log").read_text()
    match = re.search(r"Ran (\d+) tests", tests)
    test_count = int(match.group(1)) if match else None
    if not re.search(r"^OK$", tests, re.MULTILINE):
        raise ValueError("Expected a successful local test run")
    report = dict(
        version="0.3.0",
        models=evidence,
        classifier_dataset_counts={k: len(v) for k, v in classifier["splits"].items()},
        segmenter_dataset_counts={k: len(v) for k, v in segmenter["splits"].items()},
        verification=dict(
            local_tests_passed=test_count,
            dependency_lock=True,
            wheel_includes_ui=True,
            local_http_verified=True,
            docker_runtime_verified=False,
            cloud_deployed=False,
            physical_robot_verified=False,
        ),
        limitations=[
            "Separate tasks and datasets; no measured joint pick accuracy.",
            "Binary litter/background masks, not per-material or instance segmentation.",
            "The silver-can regression is still misclassified and rejected.",
            "No external untouched benchmark or industrial safety validation.",
        ],
    )
    latency = ROOT / "docs/latency.json"
    if latency.exists():
        report["local_latency"] = json.loads(latency.read_text())
    (ROOT / "docs/RELEASE_REPORT.json").write_text(json.dumps(report, indent=2))
    accuracy = classifier["test"]["accuracy"]
    iou = segmenter["test"]["foreground_iou_macro"]
    dice = segmenter["test"]["foreground_dice_macro"]
    nclass = classifier["test"]["sample_count"]
    nseg = segmenter["test"]["sample_count"]
    fig = base(
        "From a photo to a reviewed sorting decision.",
        "Material classification + real binary semantic segmentation + a typed inference API",
    )
    values = [
        (f"{accuracy * 100:.1f}%", "MATERIAL ACCURACY", f"{nclass} held-out TrashNet photos"),
        (f"{iou:.2f}", "FOREGROUND IoU", f"{nseg} held-out TACO photos"),
        (f"{dice:.2f}", "FOREGROUND DICE", "Litter versus background · 256px evaluation"),
    ]
    for x, (value, label, note) in zip([0.07, 0.37, 0.67], values):
        box(fig, x, 0.51, 0.25, 0.20)
        fig.text(x + 0.015, 0.62, value, color=GREEN, fontsize=45, weight="bold")
        fig.text(x + 0.015, 0.568, label, color=INK, fontsize=12, weight="bold")
        fig.text(x + 0.015, 0.53, note, color=MUTED, fontsize=10)
    labels = ["VALIDATE IMAGE", "CLASSIFY + SEGMENT", "CHECK THE EVIDENCE", "REVIEW / SIMULATE"]
    for idx, label in enumerate(labels):
        x = 0.07 + idx * 0.225
        box(fig, x, 0.33, 0.19, 0.08)
        fig.text(x + 0.095, 0.362, label, ha="center", color=INK, fontsize=10, weight="bold")
        if idx < 3:
            fig.add_artist(
                FancyArrowPatch(
                    (x + 0.195, 0.37),
                    (x + 0.22, 0.37),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=15,
                    color=GREEN,
                )
            )
    fig.text(
        0.07,
        0.225,
        f"{test_count} passing tests  •  FastAPI / OpenAPI  •  locked CPU dependencies  •  model checksums",
        color=INK,
        fontsize=15,
    )
    fig.text(
        0.07,
        0.17,
        "Docker configuration included; container runtime and public cloud rollout remain unverified.",
        color=MUTED,
        fontsize=12,
    )
    fig.savefig(out / "linkedin-project-card.png", dpi=140, facecolor=BG)
    plt.close(fig)

    fig = base(
        "Two tasks. Two honest measurements.",
        "Whole-image material labels are separate from litter/background pixel labels.",
    )
    ax = fig.add_axes([0.08, 0.22, 0.39, 0.48], facecolor=BG)
    classes = classifier["classes"]
    f1 = [classifier["test"]["per_class"][name]["f1"] for name in classes]
    ax.barh(classes, f1, color=GREEN, height=0.58)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title(
        "Material classification · per-class F1", color=INK, loc="left", pad=18, fontsize=15
    )
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.set_xlabel("F1 score", color=MUTED)
    for i, value in enumerate(f1):
        ax.text(value + 0.015, i, f"{value:.2f}", color=INK, va="center", fontsize=11)
    ax2 = fig.add_axes([0.58, 0.22, 0.34, 0.48], facecolor=BG)
    ax2.hist(segmenter["test"]["per_image_iou"], bins=10, range=(0, 1), color=GREEN, edgecolor=BG)
    ax2.set_title(
        "Litter segmentation · held-out distribution", color=INK, loc="left", pad=18, fontsize=14
    )
    ax2.tick_params(colors=MUTED)
    ax2.set_xlabel("Foreground IoU per image", color=MUTED)
    ax2.set_ylabel("Number of images", color=MUTED)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            a.spines[side].set_color("#47604d")
    fig.text(
        0.08,
        0.13,
        "Source-batch-disjoint segmentation split. Tiny objects, clutter, and unfamiliar appearances remain difficult.",
        color=MUTED,
        fontsize=12,
    )
    fig.savefig(out / "linkedin-evaluation.png", dpi=140, facecolor=BG)
    plt.close(fig)

    fig = base(
        "A high score is not a verified grasp.",
        "Known regression: a silver-can photo is still misclassified. The decision is rejected.",
    )
    decision = json.loads((ROOT / "artifacts/deployed-segmentation-response.json").read_text())
    ax = fig.add_axes([0.09, 0.34, 0.52, 0.35], facecolor=BG)
    candidates = decision["candidates"]
    names = [row["object_class"] for row in candidates]
    scores = [row["confidence"] for row in candidates]
    ax.barh(names, scores, color=[GOLD, GREEN, MUTED], height=0.55)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    for i, value in enumerate(scores):
        ax.text(value + 0.02, i, f"{value * 100:.1f}%", color=INK, va="center", fontsize=16)
    ax.tick_params(colors=MUTED, labelsize=15)
    ax.set_xlabel("Top three material class scores", color=MUTED)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#47604d")
    box(fig, 0.69, 0.37, 0.22, 0.25)
    fig.text(0.715, 0.55, "NEEDS REVIEW", color=GOLD, fontsize=18, weight="bold")
    fig.text(
        0.715,
        0.50,
        "Low class confidence\nAmbiguous alternatives\nNo pick issued",
        color=INK,
        fontsize=14,
        linespacing=1.8,
        va="top",
    )
    fig.text(
        0.09,
        0.20,
        "The learned mask can locate part of the litter while the material label is wrong.",
        color=INK,
        fontsize=16,
    )
    fig.text(
        0.09,
        0.145,
        "Confidence, mask quality, calibration, and robot readiness need independent checks.",
        color=MUTED,
        fontsize=13,
    )
    fig.savefig(out / "linkedin-failure-analysis.png", dpi=140, facecolor=BG)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                output=str(out),
                tests=test_count,
                classifier_accuracy=accuracy,
                segmentation_iou=iou,
                segmentation_dice=dice,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
