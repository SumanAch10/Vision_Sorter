"""Black-box release checks against a running HTTP/HTTPS deployment.

Credentials are read from a file and never printed. An HTTPS endpoint is tested
with certificate validation enabled; pass --ca-file for a private test CA.
"""

import argparse
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_sorter.schemas import Decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--ca-file")
    parser.add_argument("--image", default=str(ROOT / "demo1.jpeg"))
    parser.add_argument("--output", default=str(ROOT / "artifacts/deployment-test/http.json"))
    parser.add_argument("--test-rate-limit", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if not base.startswith(("http://127.0.0.1:", "https://")):
        raise ValueError("Use a loopback HTTP address or verified HTTPS endpoint")
    secret = Path(args.api_key_file).read_text().strip()
    if len(secret) < 16:
        raise ValueError("A configured deployment API key is required")
    context = ssl.create_default_context(cafile=args.ca_file)
    checks = []

    def request(path, body=None, content_type=None, authorized=True, extra=None):
        headers = {}
        if authorized:
            headers["Authorization"] = "Bearer " + secret
        if content_type:
            headers["Content-Type"] = content_type
        headers.update(extra or {})
        req = urllib.request.Request(base + path, data=body, headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=20, context=context)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read()
            value = (
                json.loads(raw)
                if "application/json" in response.headers.get("Content-Type", "")
                else raw.decode()
            )
            return response.status, {k.lower(): v for k, v in response.headers.items()}, value

    def check(name, condition, detail=None):
        checks.append(dict(name=name, passed=bool(condition), detail=detail))
        if not condition:
            raise AssertionError(name)

    try:
        status, _, value = request("/health", authorized=False)
        check("public_liveness", status == 200 and value["status"] == "ok")
        status, _, ready = request("/ready", authorized=False)
        check(
            "real_model_readiness",
            status == 200 and ready["semantic_segmentation"] and not ready["demo"],
        )
        check("hardware_disabled", ready["hardware_enabled"] is False)
        for route in ("/v1/model", "/metrics"):
            status, _, value = request(route, authorized=False)
            check("authentication_" + route, status == 401 and value["error"] == "unauthorized")
        status, _, _ = request(
            "/v1/model", extra={"Authorization": "Bearer deliberately-wrong-key"}
        )
        check("wrong_key_rejected", status == 401)
        status, _, metadata = request("/v1/model")
        check(
            "authorized_model_metadata",
            status == 200 and metadata["capabilities"]["real_semantic_segmentation"],
        )
        check(
            "training_records_not_exposed",
            "records" not in metadata and "records" not in metadata["segmentation"],
        )
        status, _, schema = request("/openapi.json", authorized=False)
        check("openapi_contract", status == 200 and "Decision" in schema["components"]["schemas"])
        raw = Path(args.image).read_bytes()
        identity = "deployment-test-" + str(uuid.uuid4())
        status, headers, value = request(
            "/v1/infer", raw, "image/jpeg", extra={"X-Request-ID": identity}
        )
        check("authenticated_real_inference", status == 200)
        decision = Decision.model_validate(value)
        check(
            "request_id_roundtrip",
            headers.get("x-request-id") == identity and decision.request_id == identity,
        )
        check(
            "response_security_headers",
            headers.get("x-content-type-options") == "nosniff"
            and headers.get("cache-control") == "no-store",
        )
        check(
            "real_semantic_mask",
            decision.mask_source == "learned_binary_litter_segmentation"
            and len(decision.mask.runs) > 0,
        )
        check(
            "mask_coordinates_restored",
            decision.center_px is not None and decision.bbox_px is not None,
        )
        check(
            "unreviewed_pick_rejected",
            decision.status == "needs_review"
            and not decision.simulation.executed
            and decision.pick_point_mm is None,
        )
        for name, body, kind, expected in [
            ("corrupt_image", b"broken", "image/png", 400),
            ("unsupported_type", b"broken", "text/plain", 415),
            ("empty_body", b"", "image/png", 413),
        ]:
            status, error_headers, error = request("/v1/infer", body, kind)
            check(
                name,
                status == expected and error["request_id"] == error_headers.get("x-request-id"),
            )
        status, _, _ = request(
            "/v1/infer", b"x", "image/png", extra={"Content-Length": str(11 * 1024 * 1024)}
        )
        check("declared_oversize_rejected", status == 413)
        barrier = threading.Barrier(6)

        def concurrent(_):
            barrier.wait(timeout=10)
            return request("/v1/infer", raw, "image/jpeg")[0]

        with ThreadPoolExecutor(max_workers=6) as pool:
            statuses = list(pool.map(concurrent, range(6)))
        check(
            "concurrent_load_bounded",
            all(s in (200, 503) for s in statuses),
            dict(success=statuses.count(200), busy=statuses.count(503)),
        )
        check("busy_rejection_observed", 503 in statuses)
        status, _, value = request("/metrics")
        check(
            "authenticated_operational_metrics",
            status == 200 and "vision_sorter_inferences_total" in value,
        )
        if args.test_rate_limit:
            statuses = [request("/v1/infer", b"broken", "image/png")[0] for _ in range(65)]
            check(
                "rate_limit_enforced",
                429 in statuses and all(s in (400, 429) for s in statuses),
                dict(rejected=statuses.count(429)),
            )
        status, _, _ = request("/ready", authorized=False)
        check("readiness_after_errors_and_load", status == 200)
        result = dict(
            status="passed",
            base_url=base,
            certificate_validation=base.startswith("https://"),
            model_version=ready["model_version"],
            segmentation_model_version=ready["segmentation_model_version"],
            check_count=len(checks),
            checks=checks,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    except Exception as exc:
        result = dict(status="failed", base_url=base, checks=checks, error=str(exc))
        raise
    finally:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
