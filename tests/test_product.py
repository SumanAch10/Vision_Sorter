import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vision_sorter.calibration import Calibration, fit_calibration
from vision_sorter.pipeline import InputError, Sorter, simulate
from vision_sorter.segmentation import SorterUNet
from vision_sorter.training import tensor


class ProductTests(unittest.TestCase):
    def test_tensor_and_mask_alignment(self):
        image = Image.new("RGB", (240, 180), (255, 0, 0))
        t = tensor(image)
        self.assertEqual(tuple(t.shape), (3, 128, 128))
        self.assertEqual(float(t[0, 0, 0]), 1)
        self.assertEqual(float(t[1, 0, 0]), -1)
        self.assertEqual(tuple(SorterUNet()(t[None]).shape), (1, 5, 128, 128))

    def test_calibration(self):
        value = fit_calibration(
            [[0, 0], [640, 0], [0, 480], [640, 480]],
            [[0, 0], [320, 0], [0, 240], [320, 240]],
            [640, 480],
            [0, 0, 320, 240],
        )
        c = Calibration(value)
        with self.assertRaisesRegex(ValueError, "not_validated"):
            c.project(100, 100, (640, 480))
        value["validated"] = True
        c = Calibration(value)
        self.assertEqual(c.project(100, 100, (640, 480)), dict(x=50.0, y=50.0))
        with self.assertRaisesRegex(ValueError, "resolution"):
            c.project(100, 100, (128, 128))
        with self.assertRaisesRegex(ValueError, "outside"):
            c.project(1000, 1000, (640, 480))
        with self.assertRaises(ValueError):
            fit_calibration([[0, 0]] * 4, [[0, 0]] * 4, [640, 480], [0, 0, 320, 240])

    def test_simulator_refuses_review(self):
        self.assertFalse(simulate(dict(status="needs_review"))["executed"])
        result = simulate(
            dict(status="ready", pick_point_mm=dict(x=10, y=20), destination="metal_bin")
        )
        self.assertTrue(result["executed"])
        self.assertEqual(
            [s["action"] for s in result["steps"]], ["approach", "pick", "place", "return_home"]
        )

    def test_inference_and_rle(self):
        sorter = Sorter("artifacts/demo-model")
        with self.assertRaises(InputError):
            sorter.infer(b"not an image")
        out = sorter.infer(Path("data/processed/demo/0.png").read_bytes())
        self.assertEqual(out["schema_version"], "1.0")
        self.assertEqual(out["status"], "needs_review")
        self.assertIn("demo_model", out["reasons"])
        self.assertIsNone(out["pick_point_mm"])
        self.assertFalse(out["simulation"]["executed"])
        mask = np.zeros(128 * 128, dtype=np.uint8)
        for start, length in out["mask"]["runs"]:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start + length, len(mask))
            mask[start : start + length] = 1
        y, x = np.nonzero(mask.reshape(128, 128))
        self.assertAlmostEqual(x.mean(), out["center_px"]["x"], places=1)
        self.assertAlmostEqual(y.mean(), out["center_px"]["y"], places=1)
        raw = io.BytesIO()
        Image.new("RGB", (128, 128)).save(raw, format="PNG")
        self.assertIn("insufficient_image_quality", sorter.infer(raw.getvalue())["reasons"])

    def test_approved_fixture_reaches_simulation(self):
        # In-memory fixture exercises acceptance; on-disk demo approval is unchanged.
        sorter = Sorter("artifacts/demo-model")
        sorter.manifest = dict(sorter.manifest, demo=False, approved_for_simulation=True)
        sorter.calibration = Calibration(
            dict(
                homography=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]],
                image_size=[128, 128],
                workspace_mm=[0, 0, 64, 64],
                validated=True,
                rms_error_mm=0.1,
            )
        )
        out = sorter.infer(Path("data/processed/demo/0.png").read_bytes())
        self.assertEqual(out["status"], "ready", out["reasons"])
        self.assertEqual(out["destination"], "metal_bin")
        self.assertTrue(out["simulation"]["executed"])
        self.assertIsNotNone(out["pick_point_mm"])

    def test_bad_manifest_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            manifest = json.loads(Path("artifacts/demo-model/manifest.json").read_text())
            (p / "manifest.json").write_text(json.dumps(manifest))
            (p / "weights.pt").write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "checksum"):
                Sorter(p)


if __name__ == "__main__":
    unittest.main()
