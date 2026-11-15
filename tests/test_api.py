"""Exercise deployed ASGI boundaries with real models and bounded test fixtures."""

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from vision_sorter.pipeline import Sorter
from vision_sorter.server import create_app
from vision_sorter.settings import Settings


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sorter = Sorter("artifacts/demo-model")

    def setUp(self):
        self.client = TestClient(create_app(self.sorter, Settings()))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_ready_ui_and_openapi(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        ready = self.client.get("/ready").json()
        self.assertFalse(ready["hardware_enabled"])
        self.assertEqual(ready["architecture"], "sorter-unet")
        self.assertIn("Alternative predictions", self.client.get("/").text)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("Decision", schema["components"]["schemas"])
        self.assertIn("/v1/infer", schema["paths"])
        missing = self.client.get("/missing")
        self.assertEqual(missing.status_code, 404)
        self.assertIn("request_id", missing.json())

    def test_corrupt_unsupported_empty_and_oversized(self):
        for body, kind, status in [
            (b"broken", "image/png", 400),
            (b"broken", "text/plain", 415),
            (b"", "image/png", 413),
        ]:
            response = self.client.post("/v1/infer", content=body, headers={"Content-Type": kind})
            self.assertEqual(response.status_code, status)
            self.assertEqual(response.json()["request_id"], response.headers["x-request-id"])
        response = self.client.post(
            "/v1/infer",
            content=b"x",
            headers={"Content-Type": "image/png", "Content-Length": str(11 * 1024 * 1024)},
        )
        self.assertEqual(response.status_code, 413)

    def test_schema_nulls_security_headers_and_request_id(self):
        response = self.client.post(
            "/v1/infer",
            content=Path("data/processed/demo/0.png").read_bytes(),
            headers={"Content-Type": "image/png", "X-Request-ID": "test-request-123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["request_id"], "test-request-123")
        self.assertEqual(response.headers["x-request-id"], result["request_id"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(result["status"], "needs_review")
        self.assertFalse(result["simulation"]["executed"])
        self.assertIsNone(result["pick_point_mm"])
        self.assertFalse(result["confidence_checks_passed"])
        self.assertTrue(result["warnings"])
        self.assertEqual(result["confidence_kind"], "uncalibrated_mean_foreground_probability")

    def test_authentication_and_no_training_records_exposed(self):
        secret = "test-key-for-test-suite-only"
        with TestClient(create_app(self.sorter, Settings(api_key=secret))) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/").status_code, 200)
            denied = client.get("/v1/model")
            self.assertEqual(denied.status_code, 401)
            self.assertEqual(denied.headers["www-authenticate"], "Bearer")
            self.assertEqual(client.get("/metrics").status_code, 401)
            self.assertEqual(
                client.get("/v1/model", headers={"Authorization": "Bearer wrong"}).status_code, 401
            )
            response = client.get("/v1/model", headers={"Authorization": "Bearer " + secret})
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("records", response.json())
            self.assertNotIn("splits", response.json())
            self.assertNotIn(secret, response.text)

    def test_rate_limit_and_metrics(self):
        with TestClient(create_app(self.sorter, Settings(requests_per_minute=1))) as client:
            headers = {"Content-Type": "image/png"}
            self.assertEqual(
                client.post("/v1/infer", content=b"broken", headers=headers).status_code, 400
            )
            response = client.post("/v1/infer", content=b"broken", headers=headers)
            self.assertEqual(response.status_code, 429)
            self.assertIn("retry-after", response.headers)
            self.assertTrue(response.json()["retryable"])
            self.assertIn("vision_sorter_errors_total", client.get("/metrics").text)

    def test_not_ready_and_busy(self):
        app = create_app(self.sorter, Settings())
        with TestClient(app) as client:
            app.state.ready = False
            self.assertEqual(client.get("/ready").status_code, 503)
            app.state.ready = True
            client.portal.call(app.state.inference_lock.acquire)
            try:
                response = client.post(
                    "/v1/infer", content=b"x", headers={"Content-Type": "image/png"}
                )
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["error"], "busy")
            finally:
                client.portal.call(app.state.inference_lock.release)

    def test_binding_and_secret_validation(self):
        with self.assertRaises(ValueError):
            Settings(api_key="short")
        with self.assertRaises(ValueError):
            Settings().validate_binding("0.0.0.0")
        Settings(api_key="test-secret-at-least-sixteen").validate_binding("0.0.0.0")
        Settings(allow_anonymous=True).validate_binding("0.0.0.0")
