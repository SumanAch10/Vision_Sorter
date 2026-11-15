"""Prepare real TACO polygon masks; split source batches before sampling.

Downloads author-hosted 640px images, scales original COCO polygons to their
actual dimensions, unions all litter categories, and writes aligned PNG masks.
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw

URL = "https://raw.githubusercontent.com/pedropro/TACO/3844bcd3535cd12ec0218494d1d79f1372f553cc/data/annotations.json"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rasterize(polygons, original_size, output_size):
    mask = Image.new("L", output_size)
    draw = ImageDraw.Draw(mask)
    sx, sy = output_size[0] / original_size[0], output_size[1] / original_size[1]
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError("Invalid TACO polygon")
        draw.polygon(
            [(polygon[i] * sx, polygon[i + 1] * sy) for i in range(0, len(polygon), 2)], fill=1
        )
    return mask


def prepare(count=600, seed=42, root=None):
    root = Path(root or ROOT / "data/processed/taco")
    root.mkdir(parents=True, exist_ok=True)
    annotation_path = ROOT / "data/raw/taco/annotations.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    if not annotation_path.exists():
        subprocess.run(["curl", "-fL", "--retry", "2", URL, "-o", str(annotation_path)], check=True)
    if (
        digest(annotation_path)
        != "812c2e3f0d336b437f276597715de2abb08bb8a641599c45f593d233ec66457b"
    ):
        raise ValueError("TACO annotation snapshot checksum mismatch")
    data = json.loads(annotation_path.read_text())
    by_image = {}
    for annotation in data["annotations"]:
        if annotation.get("iscrowd") or not isinstance(annotation["segmentation"], list):
            raise ValueError(
                "RLE/crowd annotations need an explicit decoder; refusing incomplete masks"
            )
        by_image.setdefault(annotation["image_id"], []).extend(annotation["segmentation"])
    groups = sorted({Path(im["file_name"]).parts[0] for im in data["images"]})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    group_sets = dict(
        train=groups[: int(n * 0.7)],
        validation=groups[int(n * 0.7) : int(n * 0.85)],
        test=groups[int(n * 0.85) :],
    )
    wanted = dict(
        train=int(count * 0.7),
        validation=int(count * 0.15),
        test=count - int(count * 0.7) - int(count * 0.15),
    )
    selected = []
    for split, allowed in group_sets.items():
        candidates = [
            im
            for im in data["images"]
            if Path(im["file_name"]).parts[0] in allowed
            and im["id"] in by_image
            and (im.get("flickr_640_url") or "").startswith("https://img.tacodataset.org/")
        ]
        rng.shuffle(candidates)
        for im in candidates if count == 0 else candidates[: wanted[split]]:
            selected.append((split, im))
    if count and len(selected) != count:
        raise ValueError("Not enough images in source groups for requested subset")

    def process(entry):
        split, info = entry
        image_path = root / "images" / f"{info['id']}.jpg"
        mask_path = root / "masks" / f"{info['id']}.png"
        image_path.parent.mkdir(exist_ok=True)
        mask_path.parent.mkdir(exist_ok=True)
        url = info.get("flickr_640_url") or info.get("flickr_url")
        if not url or not url.startswith("https://img.tacodataset.org/"):
            raise ValueError("Unexpected image host")
        if not image_path.exists():
            temporary = image_path.with_suffix(".partial")
            subprocess.run(
                ["curl", "-fsSL", "--retry", "2", "--max-time", "45", url, "-o", str(temporary)],
                check=True,
            )
            temporary.replace(image_path)
        with Image.open(image_path) as image:
            image.load()
            size = image.size
        if abs((size[0] / size[1]) / (info["width"] / info["height"]) - 1) > 0.02:
            raise ValueError("Thumbnail aspect ratio differs from annotated image")
        mask = rasterize(by_image[info["id"]], (info["width"], info["height"]), size)
        mask.save(mask_path)
        return dict(
            image=str(image_path.relative_to(root)),
            mask=str(mask_path.relative_to(root)),
            source_id=info["id"],
            source_group=Path(info["file_name"]).parts[0],
            split=split,
            image_sha256=digest(image_path),
            mask_sha256=digest(mask_path),
            source_url=url,
            source_original_url=info.get("flickr_url"),
            source_license=info.get("license"),
        )

    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process, entry): entry for entry in selected}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                split, info = futures[future]
                failures.append(dict(source_id=info["id"], split=split, error=str(exc)))
            if (len(records) + len(failures)) % 50 == 0:
                print(
                    json.dumps(dict(prepared=len(records), failed=len(failures), requested=count)),
                    flush=True,
                )
    records.sort(key=lambda r: r["source_id"])
    # Exclude every copy of duplicate content before training; record the source IDs.
    from collections import Counter

    counts = Counter(r["image_sha256"] for r in records)
    duplicates = {digest for digest, n in counts.items() if n > 1}
    audit = [
        dict(
            reason="duplicate_image_content",
            source_ids=[r["source_id"] for r in records if r["image_sha256"] == digest],
            sha256=digest,
        )
        for digest in sorted(duplicates)
    ]
    records = [r for r in records if r["image_sha256"] not in duplicates]
    result = dict(
        schema_version="1.0",
        classes=["background", "litter"],
        seed=seed,
        source="https://github.com/pedropro/TACO",
        annotation_sha256=digest(annotation_path),
        requested_count=count,
        data_audit=audit,
        group_sets=group_sets,
        records=records,
        download_failures=failures,
        selection="Seeded sample from author-hosted thumbnails only; external-host images excluded before sampling.",
        eligible_image_count=sum(
            (im.get("flickr_640_url") or "").startswith("https://img.tacodataset.org/")
            for im in data["images"]
        ),
        limitations=[
            "Binary union of all annotated litter categories, not per-material segmentation.",
            "Source batch-disjoint split; batches are not guaranteed independent acquisition sessions.",
            "Image license fields may be unspecified; do not redistribute source photographs without checking rights.",
        ],
    )
    path = root / "dataset.json"
    path.write_text(json.dumps(result, indent=2))
    if failures:
        print(
            f"WARNING: {len(failures)} images unavailable; exclusions recorded. Actual count: {len(records)}",
            file=sys.stderr,
        )
    print(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()
    prepare(args.count, args.seed, args.output)
