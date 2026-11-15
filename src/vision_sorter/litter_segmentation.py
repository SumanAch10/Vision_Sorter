"""ResNet-18 U-Net trained on real human-annotated litter masks.

Binary semantic labels: background=0, litter=1. Material classification is a
separate task; this module never assigns material identities to pixels.
"""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet18

CLASSES = ["background", "litter"]
SIZE = 256
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def geometry(size, side=SIZE):
    w, h = size
    scale = min(side / w, side / h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    return rw, rh, (side - rw) // 2, (side - rh) // 2


def prepare_image(image, side=SIZE):
    image = ImageOps.exif_transpose(image).convert("RGB")
    rw, rh, left, top = geometry(image.size, side)
    square = Image.new("RGB", (side, side), (128, 128, 128))
    square.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (left, top))
    x = torch.from_numpy(np.array(square, dtype=np.float32).transpose(2, 0, 1).copy()) / 255
    return (x - MEAN) / STD


def prepare_mask(mask, side=SIZE):
    rw, rh, left, top = geometry(mask.size, side)
    square = Image.new("L", (side, side), 255)
    square.paste(mask.resize((rw, rh), Image.Resampling.NEAREST), (left, top))
    return torch.from_numpy(np.array(square, dtype=np.int64))


def restore_probability(probability, size):
    """Undo letterboxing before restoring original image coordinates."""
    side = probability.shape[-1]
    rw, rh, left, top = geometry(size, side)
    cropped = probability[top : top + rh, left : left + rw]
    return np.asarray(
        Image.fromarray(np.asarray(cropped, dtype=np.float32)).resize(
            size, Image.Resampling.BILINEAR
        )
    ).copy()


def conv_block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1, bias=False),
        nn.GroupNorm(8, b),
        nn.ReLU(),
        nn.Conv2d(b, b, 3, padding=1, bias=False),
        nn.GroupNorm(8, b),
        nn.ReLU(),
    )


class LitterUNet(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.encoder = resnet18(weights=None)
        if pretrained:
            path = Path(pretrained)
            if not hashlib.sha256(path.read_bytes()).hexdigest().startswith("f37072fd"):
                raise ValueError("Pretrained encoder checksum mismatch")
            self.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.encoder.fc = nn.Identity()
        self.d3 = conv_block(512 + 256, 128)
        self.d2 = conv_block(128 + 128, 64)
        self.d1 = conv_block(64 + 64, 32)
        self.d0 = conv_block(32 + 64, 16)
        self.head = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        e = self.encoder
        s0 = e.relu(e.bn1(e.conv1(x)))
        s1 = e.layer1(e.maxpool(s0))
        s2 = e.layer2(s1)
        s3 = e.layer3(s2)
        s4 = e.layer4(s3)

        def up(value, skip, block):
            return block(
                torch.cat(
                    [
                        nn.functional.interpolate(
                            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        skip,
                    ],
                    1,
                )
            )

        y = up(s4, s3, self.d3)
        y = up(y, s2, self.d2)
        y = up(y, s1, self.d1)
        y = up(y, s0, self.d0)
        return nn.functional.interpolate(
            self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def binary_metrics(probabilities, targets, threshold=0.5):
    valid = targets != 255
    prediction = (probabilities >= threshold) & valid
    truth = (targets == 1) & valid
    dims = tuple(range(1, targets.ndim))
    tp = (prediction & truth).sum(dims).double()
    fp = (prediction & ~truth).sum(dims).double()
    fn = (~prediction & truth).sum(dims).double()
    union = tp + fp + fn
    iou = torch.where(union > 0, tp / union, torch.ones_like(union))
    dice_den = 2 * tp + fp + fn
    dice = torch.where(dice_den > 0, 2 * tp / dice_den, torch.ones_like(dice_den))
    return dict(
        foreground_iou_macro=float(iou.mean()),
        foreground_dice_macro=float(dice.mean()),
        foreground_iou_micro=float(tp.sum() / (tp + fp + fn).sum().clamp_min(1)),
        precision=float(tp.sum() / (tp + fp).sum().clamp_min(1)),
        recall=float(tp.sum() / (tp + fn).sum().clamp_min(1)),
        sample_count=len(targets),
        mask_threshold=threshold,
        evaluation_resolution=[SIZE, SIZE],
        ignored_label=255,
        per_image_iou=iou.tolist(),
    )


def segmentation_loss(logits, target):
    ce = nn.functional.cross_entropy(
        logits, target, weight=logits.new_tensor([1.0, 3.0]), ignore_index=255
    )
    probability = logits.softmax(1)[:, 1]
    valid = (target != 255).float()
    truth = (target == 1).float()
    probability = probability * valid
    dice = (2 * (probability * truth).sum((1, 2)) + 1) / (
        probability.sum((1, 2)) + truth.sum((1, 2)) + 1
    )
    return ce + (1 - dice).mean()


def load_pairs(dataset):
    path = Path(dataset)
    data = json.loads(path.read_text())
    xs = []
    ys = []
    records = data["records"]
    if data["classes"] != CLASSES:
        raise ValueError("Expected binary background/litter labels")
    groups = {name: set() for name in ("train", "validation", "test")}
    hashes = set()
    for row in records:
        if row["split"] not in groups:
            raise ValueError("Invalid split name")
        groups[row["split"]].add(row["source_group"])
        ip, mp = path.parent / row["image"], path.parent / row["mask"]
        for file, key in ((ip, "image_sha256"), (mp, "mask_sha256")):
            if hashlib.sha256(file.read_bytes()).hexdigest() != row[key]:
                raise ValueError("Dataset content hash mismatch")
        if row["image_sha256"] in hashes:
            raise ValueError("Duplicate image content")
        hashes.add(row["image_sha256"])
        with Image.open(ip) as image, Image.open(mp) as mask:
            if image.size != mask.size or not set(np.unique(np.asarray(mask))).issubset({0, 1}):
                raise ValueError("Masks must be aligned, binary integer images")
            xs.append(prepare_image(image))
            ys.append(prepare_mask(mask))
    if any(
        groups[a] & groups[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Source groups overlap across splits")
    splits = {name: [i for i, r in enumerate(records) if r["split"] == name] for name in groups}
    if any(len(ids) < 5 for ids in splits.values()):
        raise ValueError("Need at least five examples per split")
    return torch.stack(xs), torch.stack(ys), data, splits


def train_segmenter(dataset, output, pretrained, epochs=20, seed=42, device="auto"):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    torch_device = torch.device(device)
    x, y, data, splits = load_pairs(dataset)
    model = LitterUNet(pretrained).to(torch_device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for module in (model.encoder.layer3, model.encoder.layer4):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=0.0001),
            dict(
                params=[
                    p for name, p in model.named_parameters() if not name.startswith("encoder.")
                ],
                lr=0.001,
            ),
        ],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    stale = 0
    start = time.perf_counter()

    def evaluate(ids):
        model.eval()
        chunks = []
        with torch.inference_mode():
            for offset in range(0, len(ids), 8):
                chunks.append(
                    model(x[ids[offset : offset + 8]].to(torch_device)).softmax(1)[:, 1].cpu()
                )
        return torch.cat(chunks)

    for epoch in range(epochs):
        model.train()
        model.encoder.eval()  # retain pretrained BN statistics on small batches
        ids = splits["train"].copy()
        random.shuffle(ids)
        losses = []
        for offset in range(0, len(ids), 8):
            batch = ids[offset : offset + 8]
            xx = x[batch].clone()
            yy = y[batch].clone()
            if random.random() < 0.5:
                xx = xx.flip(-1)
                yy = yy.flip(-1)
            if random.random() < 0.25:
                xx = xx.flip(-2)
                yy = yy.flip(-2)
            if random.random() < 0.5:
                k = random.randint(1, 3)
                xx = torch.rot90(xx, k, (-2, -1))
                yy = torch.rot90(yy, k, (-2, -1))
            xx = (xx * random.uniform(0.9, 1.1)).to(torch_device)
            yy = yy.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx)
            loss = segmentation_loss(logits, yy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(splits["validation"])
        metrics = binary_metrics(validation, y[splits["validation"]])
        score = metrics["foreground_iou_macro"]
        history.append(
            dict(
                epoch=epoch + 1,
                loss=float(np.mean(losses)),
                validation_iou=score,
                validation_dice=metrics["foreground_dice_macro"],
            )
        )
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            stale = 0
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "weights.pt"
            )
        else:
            stale += 1
        scheduler.step()
        if stale >= 10:
            break
    model.load_state_dict(
        torch.load(out / "weights.pt", weights_only=True, map_location=torch_device)
    )
    validation = evaluate(splits["validation"])
    thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    threshold = max(
        thresholds,
        key=lambda t: binary_metrics(validation, y[splits["validation"]], t)[
            "foreground_iou_macro"
        ],
    )
    test_start = time.perf_counter()
    test = evaluate(splits["test"])
    test_ms = (time.perf_counter() - test_start) * 1000 / len(test)
    digest = hashlib.sha256((out / "weights.pt").read_bytes()).hexdigest()
    manifest = dict(
        schema_version="1.0",
        architecture="resnet18-litter-unet",
        model_version=f"litter-unet-taco-{digest[:10]}",
        classes=CLASSES,
        semantic_task="binary_litter_segmentation",
        demo=False,
        approved_for_simulation=False,
        input_size=SIZE,
        mask_threshold=threshold,
        weights_sha256=digest,
        dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        annotation_sha256=data["annotation_sha256"],
        records=data["records"],
        splits=splits,
        group_sets=data["group_sets"],
        source=data["source"],
        download_failures=data["download_failures"],
        data_audit=data.get("data_audit", []),
        selection=data.get("selection"),
        seed=seed,
        device=str(torch_device),
        epochs_completed=len(history),
        history=history,
        training_seconds=round(time.perf_counter() - start, 2),
        validation=binary_metrics(validation, y[splits["validation"]], threshold),
        test=binary_metrics(test, y[splits["test"]], threshold),
        batch_latency_ms_per_image=test_ms,
        preprocessing=dict(
            size=SIZE,
            resize="aspect-preserving letterbox",
            padding_rgb=[128, 128, 128],
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            mask_ignore_label=255,
        ),
        limitations=data["limitations"]
        + [
            "Pilot model on a subset of TACO; not industrially validated.",
            "No per-material semantic labels or instance identities; touching litter may merge.",
            "Masks, centroids, and scores are predictions, not verified physical grasp points.",
            "MPS training is seeded but bitwise reproducibility across devices is not guaranteed.",
        ],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(dict(model_version=manifest["model_version"], test=manifest["test"]), indent=2),
        flush=True,
    )
    return manifest


class LitterSegmenter:
    def __init__(self, model_dir):
        root = Path(model_dir)
        self.manifest = json.loads((root / "manifest.json").read_text())
        m = self.manifest
        if (
            m.get("architecture") != "resnet18-litter-unet"
            or m.get("classes") != CLASSES
            or m.get("input_size") != SIZE
        ):
            raise ValueError("Unsupported litter segmentation manifest")
        threshold = m.get("mask_threshold")
        if (
            not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or not 0 < threshold < 1
        ):
            raise ValueError("Invalid mask threshold")
        if hashlib.sha256((root / "weights.pt").read_bytes()).hexdigest() != m["weights_sha256"]:
            raise ValueError("Segmentation model checksum mismatch")
        self.model = LitterUNet()
        self.model.load_state_dict(
            torch.load(root / "weights.pt", weights_only=True, map_location="cpu")
        )
        self.model.eval()

    def predict(self, image):
        with torch.inference_mode():
            probability = self.model(prepare_image(image).unsqueeze(0)).softmax(1)[0, 1].numpy()
        restored = restore_probability(probability, image.size)
        foreground = restored >= self.manifest["mask_threshold"]
        confidence = float(restored[foreground].mean()) if foreground.any() else 0.0
        small = probability >= self.manifest["mask_threshold"]
        rw, rh, left, top = geometry(image.size, SIZE)
        valid = np.zeros_like(small)
        valid[top : top + rh, left : left + rw] = True
        return foreground, confidence, small & valid
