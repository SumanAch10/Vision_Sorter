"""Seeded paired image/mask learning with recorded splits and held-out metrics."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import CLASS_NAMES
from .segmentation import SorterUNet

SIZE = 128


def tensor(image):
    return (
        torch.from_numpy(
            np.asarray(image.resize((SIZE, SIZE)), dtype=np.float32).copy().transpose(2, 0, 1)
        )
        / 127.5
        - 1
    )


def generate_demo(directory, count=120, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []
    colors = [(155, 160, 170), (45, 140, 215), (225, 205, 145), (190, 70, 100)]
    for i in range(count):
        label = i % 4
        image = Image.new("RGB", (128, 128), tuple(rng.randint(15, 35) for _ in range(3)))
        mask = Image.new("L", image.size)
        x, y = rng.randint(12, 38), rng.randint(12, 30)
        w, h = rng.randint(35, 65), rng.randint(50, 85)
        box = (x, y, min(x + w, 116), min(y + h, 116))
        for canvas, color in [(image, colors[label]), (mask, label + 1)]:
            draw = ImageDraw.Draw(canvas)
            if label in (0, 1):
                draw.rounded_rectangle(box, radius=8 if label == 0 else 16, fill=color)
            elif label == 2:
                draw.polygon(
                    [(x, y + 10), (box[2], y), (box[2] - 5, box[3]), (x + 5, box[3] - 5)],
                    fill=color,
                )
            else:
                draw.ellipse(box, fill=color)
        image.save(directory / f"{i}.png")
        mask.save(directory / f"{i}-mask.png")
        records.append(dict(image=f"{i}.png", mask=f"{i}-mask.png", class_id=label))
    path = directory / "dataset.json"
    path.write_text(json.dumps(records, indent=2))
    return path


def load_dataset(path):
    path = Path(path)
    records = json.loads(path.read_text())
    if len(records) < 12:
        raise ValueError("At least 12 paired examples are required")
    images, masks = [], []
    for row in records:
        with Image.open(path.parent / row["image"]) as image:
            original_size = image.size
            images.append(tensor(image.convert("RGB")))
        with Image.open(path.parent / row["mask"]) as mask:
            if mask.size != original_size:
                raise ValueError("Mask and image dimensions must match")
            arr = np.array(mask.resize((SIZE, SIZE), Image.Resampling.NEAREST), dtype=np.int64)
            if arr.ndim != 2 or arr.min() < 0 or arr.max() > 4:
                raise ValueError("Mask IDs must be 0=background, 1..4=classes")
            masks.append(torch.from_numpy(arr))
    return torch.stack(images), torch.stack(masks), records


def metrics(model, x, y):
    import time

    start = time.perf_counter()
    with torch.inference_mode():
        pred = model(x).argmax(1)
    latency = (time.perf_counter() - start) * 1000 / len(x)
    per_class = {}
    for i, name in enumerate(("background",) + CLASS_NAMES):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        per_class[name] = dict(
            iou=tp / max(1, tp + fp + fn),
            dice=2 * tp / max(1, 2 * tp + fp + fn),
            precision=tp / max(1, tp + fp),
            recall=tp / max(1, tp + fn),
        )

    def dominant(mask):
        return torch.stack([(mask == i).sum((1, 2)) for i in range(1, 5)], 1).argmax(1)

    errors = []
    for p, t in zip(pred, y):
        pc = (p > 0).nonzero().float()
        tc = (t > 0).nonzero().float()
        if len(pc) and len(tc):
            errors.append(float(torch.linalg.vector_norm(pc.mean(0) - tc.mean(0))))
    return dict(
        pixel_accuracy=float((pred == y).float().mean()),
        classification_accuracy=float((dominant(pred) == dominant(y)).float().mean()),
        per_class=per_class,
        localization_error_px=float(np.mean(errors)) if errors else None,
        latency_ms_per_image=latency,
        false_pick_rate=None,
        limitations=["False-pick rate requires independent robot trials; not measured."],
    )


def train(dataset, output, epochs=12, seed=42, demo=False):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    x, y, records = load_dataset(dataset)
    order = torch.randperm(len(x)).tolist()
    n = len(order)
    splits = dict(
        train=order[: int(n * 0.7)],
        validation=order[int(n * 0.7) : int(n * 0.85)],
        test=order[int(n * 0.85) :],
    )
    model = SorterUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = torch.nn.CrossEntropyLoss()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best = -1
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        ids = splits["train"].copy()
        random.shuffle(ids)
        for offset in range(0, len(ids), 8):
            batch = ids[offset : offset + 8]
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        val = metrics(model, x[splits["validation"]], y[splits["validation"]])
        score = np.mean([v["iou"] for k, v in val["per_class"].items() if k != "background"])
        history.append(
            dict(epoch=epoch + 1, loss=float(np.mean(losses)), validation_miou=float(score))
        )
        if score > best:
            best = score
            torch.save(model.state_dict(), output / "weights.pt")
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(torch.load(output / "weights.pt", weights_only=True))
    weights_hash = hashlib.sha256((output / "weights.pt").read_bytes()).hexdigest()
    manifest = dict(
        schema_version="1.0",
        model_version=f"unet-{'demo' if demo else 'dataset'}-{weights_hash[:10]}",
        architecture="sorter-unet",
        classes=list(CLASS_NAMES),
        demo=demo,
        approved_for_simulation=False,
        seed=seed,
        epochs=epochs,
        preprocessing=dict(size=[128, 128], normalization="RGB / 127.5 - 1"),
        dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        weights_sha256=weights_hash,
        splits=splits,
        records=records,
        validation=metrics(model, x[splits["validation"]], y[splits["validation"]]),
        test=metrics(model, x[splits["test"]], y[splits["test"]]),
        history=history,
        limitations=[
            "Single isolated object on fixed tabletop. No industrial validation.",
            "Synthetic demonstration only."
            if demo
            else "Review held-out results and representative unknowns before approval.",
        ],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
