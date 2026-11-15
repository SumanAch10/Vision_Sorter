"""Validated inference and conservative decision gates; never controls hardware."""

import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .calibration import Calibration
from .data import CLASS_NAMES
from .segmentation import SorterUNet
from .training import tensor

MAX_BYTES = 10 * 1024 * 1024
DESTINATIONS = dict(can="metal_bin", plastic_bottle="plastic_bin", paper="paper_bin")


class InputError(ValueError):
    pass


class Sorter:
    def __init__(self, model_dir, calibration=None, threshold=0.85, segmenter_dir=None):
        if not 0.5 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0.5 and 1")
        self.threshold = threshold
        self.calibration = Calibration.load(calibration) if calibration else None
        path = Path(model_dir)
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.architecture = self.manifest.get("architecture")
        if self.architecture == "sorter-unet":
            if self.manifest.get("classes") != list(CLASS_NAMES):
                raise ValueError("Unsupported model classes")
        elif self.architecture == "resnet18-material":
            from .classification import MATERIALS

            if self.manifest.get("classes") != list(MATERIALS):
                raise ValueError("Unsupported material classes")
            temperature = self.manifest.get("temperature")
            if (
                not isinstance(temperature, (int, float))
                or not np.isfinite(temperature)
                or not 0.25 <= temperature <= 5
            ):
                raise ValueError("Invalid calibration temperature")
        else:
            raise ValueError("Unsupported model manifest")
        if (
            hashlib.sha256((path / "weights.pt").read_bytes()).hexdigest()
            != self.manifest["weights_sha256"]
        ):
            raise ValueError("Model checksum mismatch")
        torch.set_num_threads(2)
        if self.architecture == "resnet18-material":
            from .classification import build_classifier

            self.model = build_classifier(len(self.manifest["classes"]))
        else:
            self.model = SorterUNet()
        self.model.load_state_dict(
            torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        )
        self.model.eval()
        self.segmenter = None
        if segmenter_dir:
            if self.architecture != "resnet18-material":
                raise ValueError("Real litter segmentation requires the material classifier")
            from .litter_segmentation import LitterSegmenter

            self.segmenter = LitterSegmenter(segmenter_dir)

    def infer(self, raw):
        start = time.perf_counter()
        if not raw or len(raw) > MAX_BYTES:
            raise InputError("Image must contain between 1 byte and 10 MiB")
        try:
            with Image.open(io.BytesIO(raw)) as source:
                if source.width * source.height > 16_000_000 or min(source.size) < 64:
                    raise InputError(
                        "Image must be at least 64px on each side and at most 16 megapixels"
                    )
                if source.mode not in ("RGB", "RGBA", "L"):
                    raise InputError("Supported color formats: RGB, RGBA, grayscale")
                if getattr(source, "n_frames", 1) != 1:
                    raise InputError("Animated images are unsupported")
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InputError("Invalid or corrupted image") from exc
        reasons = []
        gray = np.asarray(image.convert("L"), dtype=float)
        if gray.mean() < 15 or gray.mean() > 245 or gray.std() < 8:
            reasons.append("insufficient_image_quality")
        if self.architecture == "resnet18-material":
            return self._classify(image, reasons, start)
        with torch.inference_mode():
            probs = self.model(tensor(image).unsqueeze(0)).softmax(1)[0]
        labels = probs.argmax(0).numpy().astype("uint8")
        mask = np.asarray(Image.fromarray(labels).resize(image.size, Image.Resampling.NEAREST))
        counts = np.bincount(mask.ravel(), minlength=5)
        cid = int(counts[1:].argmax() + 1)
        foreground = mask == cid
        ys, xs = np.nonzero(foreground)
        confidence = (
            float(probs[cid][torch.from_numpy(labels == cid)].mean())
            if np.any(labels == cid)
            else 0.0
        )
        center = (
            dict(x=round(float(xs.mean()), 2), y=round(float(ys.mean()), 2)) if len(xs) else None
        )
        area = len(xs) / mask.size
        if area < 0.01 or area > 0.75:
            reasons.append("invalid_mask_area")
        if len(xs) and (
            foreground[0].any()
            or foreground[-1].any()
            or foreground[:, 0].any()
            or foreground[:, -1].any()
        ):
            reasons.append("object_touches_frame")
        if sum(counts[i] > mask.size * 0.01 for i in range(1, 5)) > 1:
            reasons.append("multiple_classes_present")
        # Reject disconnected dominant regions, including same-class multiple objects.
        small = labels == cid
        unseen = set(map(tuple, np.argwhere(small)))
        components = []
        while unseen:
            stack = [unseen.pop()]
            count = 0
            while stack:
                y, x = stack.pop()
                count += 1
                for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
            components.append(count)
        if sum(c >= max(8, small.size * 0.005) for c in components) > 1:
            reasons.append("multiple_objects_or_fragmented_mask")
        if confidence < self.threshold:
            reasons.append("low_confidence")
        name = CLASS_NAMES[cid - 1] if len(xs) else "unknown"
        if name not in DESTINATIONS:
            reasons.append("unsupported_class")
        point = None
        if self.calibration and center:
            try:
                point = self.calibration.project(center["x"], center["y"], image.size)
            except ValueError as exc:
                reasons.append(str(exc))
        else:
            reasons.append("calibration_missing" if not self.calibration else "object_not_found")
        if self.manifest.get("demo"):
            reasons.append("demo_model")
        if self.manifest.get("approved_for_simulation") is not True:
            reasons.append("model_not_approved")
        # Run-length mask preserves alignment at original image resolution.
        flat = foreground.ravel().astype("uint8")
        changes = np.flatnonzero(np.diff(np.r_[0, flat, 0]))
        result = dict(
            schema_version="1.0",
            request_id=str(uuid.uuid4()),
            status="needs_review" if reasons else "ready",
            object_class=name,
            confidence=round(confidence, 6),
            confidence_kind="uncalibrated_mean_foreground_probability",
            confidence_checks_passed=False
            if self.manifest.get("demo")
            else confidence >= self.threshold,
            warnings=["Synthetic model: predictions on real photos are unreliable."]
            if self.manifest.get("demo")
            else [],
            center_px=center,
            pick_point_mm=point if not reasons else None,
            destination=DESTINATIONS.get(name) if not reasons else None,
            model_version=self.manifest["model_version"],
            demo=self.manifest.get("demo", False),
            reasons=reasons,
            image_size=list(image.size),
            mask=dict(
                encoding="rle-start-length",
                size=list(image.size),
                runs=[[int(a), int(b - a)] for a, b in zip(changes[::2], changes[1::2])],
            ),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        result["simulation"] = simulate(result)
        return result

    def _classify(self, image, reasons, start):
        from .classification import photo_tensor

        with torch.inference_mode():
            logits = self.model(photo_tensor(image).unsqueeze(0))[0]
            probs = (logits / self.manifest["temperature"]).softmax(0)
        top = probs.topk(3)
        confidence = float(top.values[0])
        margin = float(top.values[0] - top.values[1])
        candidates = [
            dict(object_class=self.manifest["classes"][int(idx)], confidence=round(float(score), 6))
            for score, idx in zip(top.values, top.indices)
        ]
        uncertain = bool(reasons) or confidence < self.threshold or margin < 0.15
        if confidence < self.threshold:
            reasons.append("low_confidence")
        if margin < 0.15:
            reasons.append("ambiguous_prediction")
        if self.segmenter is None:
            reasons.extend(["classification_only", "localization_unavailable"])
        if self.manifest.get("approved_for_simulation") is not True:
            reasons.append("model_not_approved")
        result = dict(
            schema_version="1.0",
            request_id=str(uuid.uuid4()),
            status="needs_review",
            object_class=candidates[0]["object_class"],
            confidence=round(confidence, 6),
            confidence_kind="validation_temperature_scaled_class_probability",
            confidence_checks_passed=not uncertain,
            classification_status="uncertain" if uncertain else "confident",
            candidates=candidates,
            confidence_margin=round(margin, 6),
            warnings=[
                "Material label only; metal does not prove can and plastic does not prove bottle.",
                "Confidence is calibrated on held-out dataset images; unfamiliar photos can still be confidently wrong.",
            ],
            center_px=None,
            pick_point_mm=None,
            destination=None,
            model_version=self.manifest["model_version"],
            demo=False,
            reasons=reasons,
            image_size=list(image.size),
            mask=dict(encoding="rle-start-length", size=list(image.size), runs=[]),
            mask_source="unavailable",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        if self.segmenter is not None:
            self._localize(result, image)
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        result["simulation"] = simulate(result)
        return result

    def _localize(self, result, image):
        foreground, score, small = self.segmenter.predict(image)
        ys, xs = np.nonzero(foreground)
        area = float(foreground.mean())
        reasons = result["reasons"]
        if not len(xs):
            reasons.append("no_litter_detected")
        if not 0.005 <= area <= 0.85:
            reasons.append("invalid_mask_area")
        if score < 0.7:
            reasons.append("low_segmentation_score")
        if len(xs) and (
            foreground[0].any()
            or foreground[-1].any()
            or foreground[:, 0].any()
            or foreground[:, -1].any()
        ):
            reasons.append("object_touches_frame")
        unseen = set(map(tuple, np.argwhere(small)))
        components = []
        while unseen:
            stack = [unseen.pop()]
            count = 0
            while stack:
                y, x = stack.pop()
                count += 1
                for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
            if count >= max(12, small.size * 0.005):
                components.append(count)
        if len(components) != 1:
            reasons.append(
                "multiple_objects_or_fragmented_mask" if components else "no_valid_component"
            )
        center = (
            dict(x=round(float(xs.mean()), 2), y=round(float(ys.mean()), 2)) if len(xs) else None
        )
        point = None
        if self.calibration and center:
            try:
                point = self.calibration.project(center["x"], center["y"], image.size)
            except ValueError as exc:
                reasons.append(str(exc))
        else:
            reasons.append("calibration_missing" if not self.calibration else "object_not_found")
        if self.segmenter.manifest.get("approved_for_simulation") is not True:
            reasons.append("segmentation_not_approved")
        destinations = dict(
            metal="metal_bin",
            plastic="plastic_bin",
            paper="paper_bin",
            cardboard="paper_bin",
            glass="glass_bin",
        )
        if result["object_class"] not in destinations:
            reasons.append("unsupported_class")
        changes = np.flatnonzero(np.diff(np.r_[0, foreground.ravel().astype("uint8"), 0]))
        result.update(
            mask=dict(
                encoding="rle-start-length",
                size=list(image.size),
                runs=[[int(a), int(b - a)] for a, b in zip(changes[::2], changes[1::2])],
            ),
            mask_source="learned_binary_litter_segmentation",
            center_px=center,
            bbox_px=dict(
                x=int(xs.min()),
                y=int(ys.min()),
                width=int(xs.max() - xs.min() + 1),
                height=int(ys.max() - ys.min() + 1),
            )
            if len(xs)
            else None,
            segmentation=dict(
                model_version=self.segmenter.manifest["model_version"],
                semantic_classes=["background", "litter"],
                score=round(score, 6),
                score_kind="uncalibrated_mean_foreground_probability",
                foreground_area_fraction=round(area, 6),
                component_count=len(components),
            ),
            status="needs_review" if reasons else "ready",
            pick_point_mm=point if not reasons else None,
            destination=destinations.get(result["object_class"]) if not reasons else None,
        )
        if "no_litter_detected" in reasons or "invalid_mask_area" in reasons:
            result["classification_status"] = "uncertain"
            result["confidence_checks_passed"] = False
        result["warnings"].extend(
            [
                "Binary semantic mask labels litter versus background, not per-pixel material.",
                "The image-level material score is not a calibrated probability of a correct mask or grasp.",
            ]
        )


def simulate(decision):
    if decision["status"] != "ready":
        return dict(executed=False, mode="simulation", steps=[], reason="Decision requires review")
    p = decision["pick_point_mm"]
    if not p or decision["destination"] not in set(DESTINATIONS.values()) | {"glass_bin"}:
        raise ValueError("Invalid pick command")
    return dict(
        executed=True,
        mode="simulation",
        steps=[
            dict(action="approach", position_mm=p),
            dict(action="pick", position_mm=p),
            dict(action="place", destination=decision["destination"]),
            dict(action="return_home"),
        ],
    )
