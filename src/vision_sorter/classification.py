"""Real-photo transfer learning, independent of the synthetic segmentation model.

ResNet-18 supplies frozen ImageNet features. A six-way material head is learned
from TrashNet; validation selects the head and temperature, never the test set.
"""

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

MATERIALS = ("cardboard", "glass", "metal", "paper", "plastic", "trash")


def photo_tensor(image):
    return ResNet18_Weights.IMAGENET1K_V1.transforms()(
        ImageOps.exif_transpose(image).convert("RGB")
    )


def build_classifier(num_classes, pretrained_path=None):
    model = resnet18(weights=None)
    if pretrained_path:
        raw = Path(pretrained_path).read_bytes()
        if not hashlib.sha256(raw).hexdigest().startswith("f37072fd"):
            raise ValueError("Official ResNet-18 checkpoint checksum mismatch")
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def collect_records(root, audit=None):
    """Deduplicate content and exclude conflicting labels before splitting."""
    root = Path(root)
    groups = {}
    for cid, name in enumerate(MATERIALS):
        for path in sorted((root / name).glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append(
                dict(image=str(path.relative_to(root)), class_id=cid, sha256=digest)
            )
    records = []
    for digest, group in groups.items():
        if len({r["class_id"] for r in group}) > 1:
            if audit is not None:
                audit.append(
                    dict(
                        reason="conflicting_labels",
                        images=[r["image"] for r in group],
                        sha256=digest,
                    )
                )
            continue
        records.append(group[0])
        if len(group) > 1 and audit is not None:
            audit.append(
                dict(
                    reason="duplicate_content",
                    kept=group[0]["image"],
                    excluded=[r["image"] for r in group[1:]],
                    sha256=digest,
                )
            )
    if any(sum(r["class_id"] == cid for r in records) < 10 for cid in range(len(MATERIALS))):
        raise ValueError("Need at least 10 unique images in each of: " + ", ".join(MATERIALS))
    return records


def stratified_split(records, seed):
    rng = random.Random(seed)
    split = dict(train=[], validation=[], test=[])
    for cid in sorted({r["class_id"] for r in records}):
        ids = [i for i, r in enumerate(records) if r["class_id"] == cid]
        rng.shuffle(ids)
        n = len(ids)
        split["train"].extend(ids[: int(n * 0.7)])
        split["validation"].extend(ids[int(n * 0.7) : int(n * 0.85)])
        split["test"].extend(ids[int(n * 0.85) :])
    return split


def classification_metrics(logits, targets, temperature=1.0, threshold=0.85, margin=0.15):
    probs = (logits / temperature).softmax(1)
    conf, pred = probs.max(1)
    correct = pred.eq(targets)
    confusion = torch.zeros((len(MATERIALS), len(MATERIALS)), dtype=torch.int64)
    for truth, guess in zip(targets.tolist(), pred.tolist()):
        confusion[truth, guess] += 1
    per_class = {}
    for cid, name in enumerate(MATERIALS):
        tp = int(confusion[cid, cid])
        precision = tp / max(1, int(confusion[:, cid].sum()))
        recall = tp / max(1, int(confusion[cid].sum()))
        per_class[name] = dict(
            precision=precision,
            recall=recall,
            f1=2 * precision * recall / max(1e-12, precision + recall),
            support=int(confusion[cid].sum()),
        )
    ece = 0.0
    for i in range(10):
        selected = (conf > i / 10) & (conf <= (i + 1) / 10)
        if selected.any():
            ece += float(
                selected.float().mean()
                * (correct[selected].float().mean() - conf[selected].mean()).abs()
            )
    top = probs.topk(2, dim=1).values
    accepted = (conf >= threshold) & ((top[:, 0] - top[:, 1]) >= margin)
    return dict(
        accuracy=float(correct.float().mean()),
        macro_f1=float(np.mean([v["f1"] for v in per_class.values()])),
        nll=float(nn.functional.cross_entropy(logits / temperature, targets)),
        expected_calibration_error=ece,
        brier_score=float(
            ((probs - nn.functional.one_hot(targets, len(MATERIALS))) ** 2).sum(1).mean()
        ),
        confidence_threshold=threshold,
        margin_threshold=margin,
        accepted_count=int(accepted.sum()),
        sample_count=len(targets),
        coverage=float(accepted.float().mean()),
        accepted_accuracy=float(correct[accepted].float().mean()) if accepted.any() else None,
        confusion_matrix=confusion.tolist(),
        per_class=per_class,
    )


def fit_temperature(logits, targets):
    # Bounded one-dimensional validation NLL search. Test examples never enter it.
    candidates = torch.logspace(math.log10(0.25), math.log10(5.0), 161)
    losses = torch.stack([nn.functional.cross_entropy(logits / t, targets) for t in candidates])
    return float(candidates[losses.argmin()])


def train_classifier(dataset, output, pretrained, epochs=150, seed=42):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root, out = Path(dataset), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    audit = []
    records = collect_records(root, audit)
    splits = stratified_split(records, seed)
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    model = build_classifier(len(MATERIALS), pretrained)
    head = model.fc
    model.fc = nn.Identity()
    model.eval()
    chunks = []
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(records), 24):
            batch = []
            for row in records[offset : offset + 24]:
                with Image.open(root / row["image"]) as image:
                    batch.append(photo_tensor(image))
            chunks.append(model(torch.stack(batch)))
            if offset % 240 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="features",
                            processed=min(offset + 24, len(records)),
                            total=len(records),
                        )
                    ),
                    flush=True,
                )
    features = torch.cat(chunks).clone()
    targets = torch.tensor([r["class_id"] for r in records])
    train_ids, val_ids = splits["train"], splits["validation"]
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.1)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        head.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(features[train_ids]), targets[train_ids])
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.cross_entropy(head(features[val_ids]), targets[val_ids]))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        history.append(
            dict(epoch=epoch + 1, train_loss=float(loss.detach()), validation_loss=val_loss)
        )
    head.load_state_dict(best_state)
    with torch.no_grad():
        val_logits = head(features[val_ids])
        temperature = fit_temperature(val_logits, targets[val_ids])
        test_logits = head(features[splits["test"]])
    model.fc = head
    torch.save(model.state_dict(), out / "weights.pt")
    digest = hashlib.sha256((out / "weights.pt").read_bytes()).hexdigest()
    manifest = dict(
        schema_version="1.0",
        architecture="resnet18-material",
        model_version=f"resnet18-trashnet-{digest[:10]}",
        classes=list(MATERIALS),
        demo=False,
        approved_for_simulation=False,
        temperature=temperature,
        preprocessing=dict(recipe="ResNet18_Weights.IMAGENET1K_V1", size=[224, 224]),
        weights_sha256=digest,
        dataset_sha256=fingerprint,
        records=records,
        splits=splits,
        seed=seed,
        epochs=epochs,
        history=history,
        data_audit=audit,
        source=dict(
            dataset="https://github.com/garythung/trashnet",
            authors=["Gary Thung", "Mindy Yang"],
            license="MIT",
            pretrained="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        ),
        training_seconds=round(time.perf_counter() - start, 2),
        validation=classification_metrics(val_logits, targets[val_ids], temperature),
        test_uncalibrated=classification_metrics(test_logits, targets[splits["test"]]),
        test=classification_metrics(test_logits, targets[splits["test"]], temperature),
        limitations=[
            "Material classification only; metal does not prove can, plastic does not prove bottle.",
            "No real-data segmentation, localization, robot approval, or industrial validation.",
            "Random stratified content-deduplicated split; near-duplicate and acquisition-session leakage remain possible.",
            "Confidence calibrated on TrashNet validation images, not arbitrary web photos or unknown objects.",
        ],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(dict(model=manifest["model_version"], test=manifest["test"]), indent=2),
        flush=True,
    )
    return manifest


def fine_tune_classifier(dataset, model_dir, output, epochs=4, seed=42):
    """Fine-tune only layer4 and the head; choose exclusively by validation NLL."""
    from torchvision import transforms as T

    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    root, source, out = Path(dataset), Path(model_dir), Path(output)
    manifest = json.loads((source / "manifest.json").read_text())
    records = collect_records(root)
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest["dataset_sha256"]:
        raise ValueError("Dataset contents differ from recorded training split")
    if (
        hashlib.sha256((source / "weights.pt").read_bytes()).hexdigest()
        != manifest["weights_sha256"]
    ):
        raise ValueError("Source model checksum mismatch")
    model = build_classifier(len(MATERIALS))
    model.load_state_dict(torch.load(source / "weights.pt", weights_only=True, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.layer4, model.fc):
        for p in module.parameters():
            p.requires_grad_(True)
    augment = T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15, fill=255),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    splits = manifest["splits"]
    targets = torch.tensor([r["class_id"] for r in records])

    def logits_for(ids):
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(ids), 24):
                images = []
                for idx in ids[start : start + 24]:
                    with Image.open(root / records[idx]["image"]) as im:
                        images.append(photo_tensor(im))
                chunks.append(model(torch.stack(images)))
        return torch.cat(chunks)

    val_ids = splits["validation"]
    initial_val = logits_for(val_ids)
    best_loss = float(nn.functional.cross_entropy(initial_val, targets[val_ids]))
    best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.layer4.parameters(), lr=0.0001),
            dict(params=model.fc.parameters(), lr=0.0003),
        ],
        weight_decay=0.01,
    )
    history = []
    for epoch in range(epochs):
        ids = splits["train"].copy()
        random.shuffle(ids)
        losses = []
        model.eval()
        model.layer4.train()
        model.fc.train()
        for offset in range(0, len(ids), 16):
            batch_ids = ids[offset : offset + 16]
            images = []
            for idx in batch_ids:
                with Image.open(root / records[idx]["image"]) as im:
                    images.append(augment(ImageOps.exif_transpose(im).convert("RGB")))
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(torch.stack(images)), targets[batch_ids])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if offset % 512 == 0:
                print(
                    json.dumps(
                        dict(stage="fine_tuning", epoch=epoch + 1, processed=offset, total=len(ids))
                    ),
                    flush=True,
                )
        val = logits_for(val_ids)
        val_loss = float(nn.functional.cross_entropy(val, targets[val_ids]))
        if val_loss < best_loss:
            best_loss = val_loss
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(
            dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=val_loss)
        )
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best)
    val_logits = logits_for(val_ids)
    temperature = fit_temperature(val_logits, targets[val_ids])
    test_ids = splits["test"]
    test_logits = logits_for(test_ids)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "weights.pt")
    digest = hashlib.sha256((out / "weights.pt").read_bytes()).hexdigest()
    manifest.update(
        model_version=f"resnet18-trashnet-tuned-{digest[:10]}",
        weights_sha256=digest,
        temperature=temperature,
        fine_tuning=dict(
            epochs=epochs,
            seed=seed,
            history=history,
            layers=["layer4", "fc"],
            augmentation="random crop, horizontal flip, rotation 15 degrees, color jitter 0.1",
            selection="lowest validation cross-entropy including frozen baseline",
        ),
        validation=classification_metrics(val_logits, targets[val_ids], temperature),
        test_uncalibrated=classification_metrics(test_logits, targets[test_ids]),
        test=classification_metrics(test_logits, targets[test_ids], temperature),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(dict(model=manifest["model_version"], test=manifest["test"]), indent=2),
        flush=True,
    )
    return manifest
