"""Geometry and metrics invariants for real semantic segmentation."""

import io
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image, ImageDraw

from vision_sorter.litter_segmentation import (
    LitterSegmenter,
    LitterUNet,
    binary_metrics,
    prepare_image,
    prepare_mask,
    restore_probability,
    segmentation_loss,
)
from vision_sorter.pipeline import Sorter
from vision_sorter.schemas import Decision


class SegmentationTests(unittest.TestCase):
    def test_letterbox_restores_original_geometry(self):
        mask = Image.new("L", (400, 200))
        ImageDraw.Draw(mask).rectangle((100, 50, 299, 149), fill=1)
        target = prepare_mask(mask)
        self.assertEqual(tuple(target.shape), (256, 256))
        self.assertEqual(int(target[0, 0]), 255)
        probability = (target == 1).numpy().astype(np.float32)
        restored = restore_probability(probability, mask.size) >= 0.5
        expected = np.asarray(mask) > 0
        intersection = (restored & expected).sum()
        union = (restored | expected).sum()
        self.assertGreater(intersection / union, 0.98)
        ys, xs = np.nonzero(restored)
        self.assertAlmostEqual(xs.mean(), 199.5, delta=2)
        self.assertAlmostEqual(ys.mean(), 99.5, delta=2)
        self.assertEqual(tuple(prepare_image(Image.new("RGB", (400, 200))).shape), (3, 256, 256))

    def test_coco_polygons_scale_to_thumbnail_coordinates(self):
        from scripts.prepare_segmentation_data import rasterize

        mask = rasterize([[50, 25, 150, 25, 150, 75, 50, 75]], (200, 100), (100, 50))
        self.assertEqual(mask.size, (100, 50))
        self.assertEqual(mask.getpixel((50, 25)), 1)
        self.assertEqual(mask.getpixel((10, 10)), 0)
        self.assertEqual(mask.getbbox(), (25, 12, 76, 38))

    def test_metric_ignore_pixels_and_perfect_prediction(self):
        target = torch.tensor([[[255, 0, 0], [0, 1, 1]]])
        probability = torch.tensor([[[1.0, 0, 0], [0, 1, 1]]])
        metrics = binary_metrics(probability, target)
        self.assertEqual(metrics["foreground_iou_macro"], 1)
        self.assertEqual(metrics["foreground_dice_macro"], 1)
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 1)

    def test_loss_has_finite_gradients(self):
        logits = torch.randn(2, 2, 32, 32, requires_grad=True)
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, 8:16, 8:16] = 1
        target[:, :2] = 255
        loss = segmentation_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_model_preserves_spatial_shape(self):
        model = LitterUNet().eval()
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.randn(1, 3, 64, 96)).shape), (1, 2, 64, 96))

    def test_no_mask_and_multiple_regions_never_pick(self):
        sorter = Sorter("artifacts/demo-model")
        segmenter = Mock()
        segmenter.manifest = dict(model_version="fixture-segmenter", approved_for_simulation=True)
        sorter.manifest = dict(sorter.manifest, approved_for_simulation=True)
        sorter.segmenter = segmenter
        raw = io.BytesIO()
        Image.new("RGB", (128, 128), "white").save(raw, format="PNG")
        for many in (False, True):
            mask = np.zeros((128, 128), dtype=bool)
            if many:
                mask[20:45, 20:45] = True
                mask[80:105, 80:105] = True
            segmenter.predict.return_value = (mask, 0.95, mask)
            result = sorter.infer(raw.getvalue())
            sorter._localize(result, Image.new("RGB", (128, 128), "white"))
            Decision.model_validate(result)
            self.assertFalse(result["simulation"]["executed"])
            self.assertIsNone(result["pick_point_mm"])
            self.assertIn(
                "multiple_objects_or_fragmented_mask" if many else "no_litter_detected",
                result["reasons"],
            )

    def test_valid_mask_with_reviewed_models_can_simulate(self):
        from vision_sorter.calibration import Calibration

        sorter = Sorter("artifacts/demo-model")
        sorter.calibration = Calibration(
            dict(
                homography=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]],
                image_size=[128, 128],
                workspace_mm=[0, 0, 64, 64],
                validated=True,
                rms_error_mm=0.1,
            )
        )
        segmenter = Mock()
        segmenter.manifest = dict(model_version="reviewed-fixture", approved_for_simulation=True)
        mask = np.zeros((128, 128), dtype=bool)
        mask[40:80, 40:80] = True
        segmenter.predict.return_value = (mask, 0.95, mask)
        sorter.segmenter = segmenter
        raw = io.BytesIO()
        Image.new("RGB", (128, 128), (100, 120, 130)).save(raw, format="PNG")
        result = sorter.infer(raw.getvalue())
        result.update(object_class="metal", demo=False, reasons=[], confidence_checks_passed=True)
        sorter._localize(result, Image.new("RGB", (128, 128)))
        from vision_sorter.pipeline import simulate

        result["simulation"] = simulate(result)
        Decision.model_validate(result)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["simulation"]["executed"])
        self.assertEqual(result["pick_point_mm"], dict(x=29.75, y=29.75))
        self.assertEqual(result["destination"], "metal_bin")

    def test_response_schema_rejects_unsafe_or_misaligned_output(self):
        result = Sorter("artifacts/demo-model").infer(
            Path("data/processed/demo/0.png").read_bytes()
        )
        result["mask"]["runs"] = [[128 * 128, 1]]
        with self.assertRaises(ValueError):
            Decision.model_validate(result)
        result["mask"]["runs"] = []
        result["pick_point_mm"] = {"x": 1, "y": 2}
        with self.assertRaises(ValueError):
            Decision.model_validate(result)

    @unittest.skipUnless(
        Path("artifacts/segmentation-model/manifest.json").exists(),
        "Train real segmenter first",
    )
    def test_real_masks_and_source_group_isolation(self):
        root = Path("artifacts/segmentation-model")
        segmenter = LitterSegmenter(root)
        m = segmenter.manifest
        groups = {
            split: {m["records"][i]["source_group"] for i in ids}
            for split, ids in m["splits"].items()
        }
        self.assertFalse(
            groups["train"] & groups["test"]
            or groups["validation"] & groups["test"]
            or groups["train"] & groups["validation"]
        )
        row = m["records"][m["splits"]["test"][0]]
        path = Path("data/processed/taco-all") / row["image"]
        sorter = Sorter("artifacts/real-model", segmenter_dir=root)
        result = sorter.infer(path.read_bytes())
        Decision.model_validate(result)
        self.assertEqual(result["mask_source"], "learned_binary_litter_segmentation")
        self.assertEqual(result["segmentation"]["semantic_classes"], ["background", "litter"])
        self.assertIn("segmentation_not_approved", result["reasons"])
        self.assertNotIn("classification_only", result["reasons"])
        self.assertFalse(result["simulation"]["executed"])
