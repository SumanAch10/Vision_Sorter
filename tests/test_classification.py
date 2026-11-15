"""Regression tests for calibration, held-out splits, and classification-only safety."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from vision_sorter.classification import (
    MATERIALS,
    classification_metrics,
    collect_records,
    fit_temperature,
    photo_tensor,
    stratified_split,
)
from vision_sorter.pipeline import Sorter


class ClassificationTests(unittest.TestCase):
    def test_split_is_stratified_reproducible_and_disjoint(self):
        records = [dict(class_id=cid, sha256=f"{cid}-{i}") for cid in range(6) for i in range(20)]
        split = stratified_split(records, 42)
        self.assertEqual(split, stratified_split(records, 42))
        self.assertNotEqual(split, stratified_split(records, 1))
        sets = [set(ids) for ids in split.values()]
        self.assertEqual(len(set.union(*sets)), len(records))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for ids in split.values():
            self.assertEqual({records[i]["class_id"] for i in ids}, set(range(6)))

    def test_temperature_reduces_overconfidence_without_changing_label(self):
        # Identical high logits with mixed true labels create overconfidence.
        logits = torch.tensor([[12.0, 0, 0, 0, 0, 0]] * 10)
        labels = torch.tensor([0] * 7 + [1] * 3)
        temperature = fit_temperature(logits, labels)
        self.assertGreater(temperature, 1)
        before = classification_metrics(logits, labels)
        after = classification_metrics(logits, labels, temperature)
        self.assertLess(after["nll"], before["nll"])
        self.assertEqual(after["accuracy"], before["accuracy"])
        self.assertLess(after["expected_calibration_error"], before["expected_calibration_error"])

    def test_ambiguity_abstains_even_above_score_threshold(self):
        logits = torch.tensor([[1.0, 0.99, -10, -10, -10, -10]])
        result = classification_metrics(logits, torch.tensor([0]), threshold=0.5, margin=0.15)
        self.assertGreater(float(logits.softmax(1).max()), 0.5)
        self.assertEqual(result["accepted_count"], 0)
        self.assertIsNone(result["accepted_accuracy"])

    def test_duplicates_and_conflicting_labels_are_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for cid, name in enumerate(MATERIALS):
                (root / name).mkdir()
                for i in range(12):
                    Image.new("RGB", (64, 64), (cid * 30, i * 15, 80)).save(
                        root / name / f"{i}.png"
                    )
            same = (root / "metal/0.png").read_bytes()
            (root / "metal/duplicate.png").write_bytes(same)
            (root / "paper/conflict.png").write_bytes((root / "glass/0.png").read_bytes())
            audit = []
            records = collect_records(root, audit)
            self.assertEqual(len(records), 71)
            self.assertEqual(len({r["sha256"] for r in records}), len(records))
            self.assertEqual(
                {a["reason"] for a in audit}, {"conflicting_labels", "duplicate_content"}
            )

    def test_preprocessing_matches_manifest_recipe(self):
        t = photo_tensor(Image.new("RGB", (400, 200), (255, 0, 0)))
        self.assertEqual(tuple(t.shape), (3, 224, 224))
        self.assertAlmostEqual(float(t[0, 0, 0]), (1 - 0.485) / 0.229, places=5)

    @unittest.skipUnless(
        Path("artifacts/real-model/manifest.json").exists(), "Train real-photo model first"
    )
    def test_real_model_has_no_synthetic_localization(self):
        sorter = Sorter("artifacts/real-model")
        row = sorter.manifest["records"][sorter.manifest["splits"]["test"][0]]
        result = sorter.infer(
            (Path("data/raw/trashnet/dataset-resized") / row["image"]).read_bytes()
        )
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(result["candidates"][0]["confidence"], result["confidence"])
        self.assertEqual(
            result["confidence_kind"], "validation_temperature_scaled_class_probability"
        )
        self.assertEqual(result["status"], "needs_review")
        self.assertFalse(result["simulation"]["executed"])
        self.assertIsNone(result["center_px"])
        self.assertIsNone(result["pick_point_mm"])
        self.assertEqual(result["mask"]["runs"], [])
        self.assertIn("localization_unavailable", result["reasons"])
        self.assertIn(result["object_class"], MATERIALS)
        # Even an accidentally flipped approval flag must not authorize this path.
        sorter.manifest["approved_for_simulation"] = True
        retry = sorter.infer(
            (Path("data/raw/trashnet/dataset-resized") / row["image"]).read_bytes()
        )
        self.assertFalse(retry["simulation"]["executed"])


if __name__ == "__main__":
    unittest.main()
