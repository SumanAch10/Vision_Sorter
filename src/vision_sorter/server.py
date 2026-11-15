"""ASGI service: bounded input, typed responses, authentication and observability."""

import asyncio
import hmac
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect

from .pipeline import MAX_BYTES, InputError, Sorter
from .schemas import Decision, ErrorResponse
from .settings import Settings

LOG = logging.getLogger("vision_sorter")


def model_summary(sorter):
    def summary(m):
        keys = (
            "schema_version",
            "model_version",
            "architecture",
            "classes",
            "demo",
            "approved_for_simulation",
            "temperature",
            "preprocessing",
            "source",
            "limitations",
            "semantic_task",
            "input_size",
            "mask_threshold",
        )
        value = {k: m[k] for k in keys if k in m}
        for split in ("validation", "test"):
            if split in m:
                value[split] = {k: v for k, v in m[split].items() if k != "per_image_iou"}
        return value

    result = summary(sorter.manifest)
    result["capabilities"] = dict(
        classification=True,
        semantic_segmentation=sorter.segmenter is not None or sorter.architecture == "sorter-unet",
        real_semantic_segmentation=sorter.segmenter is not None,
        physical_robot_control=False,
    )
    if sorter.segmenter:
        result["segmentation"] = summary(sorter.segmenter.manifest)
    return result


def create_app(sorter=None, settings=None):
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        if app.state.sorter is None:
            from .cli import default_model, default_segmenter

            app.state.sorter = await run_in_threadpool(
                Sorter,
                default_model(),
                os.getenv("SORTER_CALIBRATION"),
                float(os.getenv("SORTER_THRESHOLD", ".85")),
                default_segmenter(),
            )
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False

    app = FastAPI(
        title="Vision Sorter API",
        version="0.3.0",
        description="Material classification and binary litter semantic segmentation. Simulation only.",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=Path(__file__).with_name("static")), name="static")
    app.state.sorter = sorter
    app.state.ready = False
    app.state.inference_lock = asyncio.Lock()
    app.state.counters = dict(requests=0, errors=0, inferences=0, inference_seconds=0.0)
    clients = OrderedDict()
    page = Path(__file__).with_name("index.html").read_text()
    protected = {"/v1/infer", "/v1/model", "/metrics"}
    bearer = HTTPBearer(auto_error=False, description="API key if configured for this station")

    def error(request, code, name, message, retryable=False):
        return JSONResponse(
            status_code=code,
            content=ErrorResponse(
                error=name,
                message=message,
                request_id=request.state.request_id,
                retryable=retryable,
            ).model_dump(),
        )

    @app.middleware("http")
    async def boundaries(request, call_next):
        candidate = request.headers.get("x-request-id", "")
        request.state.request_id = (
            candidate if re.fullmatch(r"[A-Za-z0-9-]{1,64}", candidate) else str(uuid.uuid4())
        )
        start = time.perf_counter()
        app.state.counters["requests"] += 1
        path = request.url.path
        response = None
        if path in protected:
            supplied = request.headers.get("authorization", "")
            if config.api_key and not hmac.compare_digest(
                supplied.encode(), ("Bearer " + config.api_key).encode()
            ):
                response = error(request, 401, "unauthorized", "Provide a valid bearer API key")
            if response is None and path == "/v1/infer":
                host = request.client.host if request.client else "unknown"
                now = time.monotonic()
                tokens, stamp = clients.pop(host, (float(config.requests_per_minute), now))
                tokens = min(
                    config.requests_per_minute,
                    tokens + (now - stamp) * config.requests_per_minute / 60,
                )
                clients[host] = (max(0.0, tokens - 1), now)
                if len(clients) > config.max_clients:
                    clients.popitem(last=False)
                if tokens < 1:
                    response = error(request, 429, "rate_limited", "Request rate exceeded", True)
                    response.headers["Retry-After"] = str(
                        max(1, int(60 / config.requests_per_minute))
                    )
        try:
            if response is None:
                response = await call_next(request)
        except Exception:
            LOG.exception(
                json.dumps(dict(event="unhandled_error", request_id=request.state.request_id))
            )
            response = error(request, 500, "internal_error", "The request could not be completed")
        if response.status_code >= 400:
            app.state.counters["errors"] += 1
        response.headers.update(
            {
                "X-Request-ID": request.state.request_id,
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
            }
        )
        if path == "/":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        if response.status_code == 401:
            response.headers["WWW-Authenticate"] = "Bearer"
        LOG.info(
            json.dumps(
                dict(
                    event="request",
                    request_id=request.state.request_id,
                    path=path,
                    status_code=response.status_code,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                )
            )
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error(request, exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return error(request, 422, "invalid_request", "Request does not match the API schema")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        return page

    @app.get("/health")
    async def health():
        return dict(status="ok")

    @app.get("/ready")
    async def ready(request: Request):
        current = app.state.sorter
        if not app.state.ready or current is None:
            return error(request, 503, "not_ready", "Model is not loaded", True)
        return dict(
            status="ready",
            model_version=current.manifest["model_version"],
            architecture=current.architecture,
            demo=current.manifest.get("demo", False),
            semantic_segmentation=current.segmenter is not None,
            segmentation_model_version=current.segmenter.manifest["model_version"]
            if current.segmenter
            else None,
            hardware_enabled=False,
        )

    @app.get("/v1/model", dependencies=[Security(bearer)])
    async def metadata(request: Request):
        if not app.state.ready:
            return error(request, 503, "not_ready", "Model is not loaded", True)
        return model_summary(app.state.sorter)

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    async def metrics():
        c = app.state.counters
        return "\n".join(
            [
                "# TYPE vision_sorter_requests_total counter",
                f"vision_sorter_requests_total {c['requests']}",
                "# TYPE vision_sorter_errors_total counter",
                f"vision_sorter_errors_total {c['errors']}",
                "# TYPE vision_sorter_inferences_total counter",
                f"vision_sorter_inferences_total {c['inferences']}",
                "# TYPE vision_sorter_inference_seconds_total counter",
                f"vision_sorter_inference_seconds_total {c['inference_seconds']}",
                "# TYPE vision_sorter_ready gauge",
                f"vision_sorter_ready {int(app.state.ready)}",
                "",
            ]
        )

    errors = {code: {"model": ErrorResponse} for code in (400, 401, 408, 413, 415, 429, 500, 503)}

    @app.post(
        "/v1/infer",
        response_model=Decision,
        dependencies=[Security(bearer)],
        response_model_exclude_none=False,
        responses=errors,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    kind: {"schema": {"type": "string", "format": "binary"}}
                    for kind in ("image/png", "image/jpeg", "image/webp")
                },
            }
        },
    )
    async def infer(request: Request):
        if not app.state.ready:
            return error(request, 503, "not_ready", "Model is not loaded", True)
        if request.headers.get("content-type", "").split(";")[0] not in (
            "image/png",
            "image/jpeg",
            "image/webp",
        ):
            return error(
                request, 415, "unsupported_media_type", "Use a PNG, JPEG, or WebP image body"
            )
        length = request.headers.get("content-length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError:
                return error(
                    request, 400, "invalid_content_length", "Content-Length must be an integer"
                )
            if not 0 < declared <= MAX_BYTES:
                return error(
                    request, 413, "image_size_limit", "Image must be between 1 byte and 10 MiB"
                )
        lock = app.state.inference_lock
        if lock.locked():
            return error(request, 503, "busy", "Another inference is in progress", True)
        # No await between this check and acquisition: one event loop owns this lock.
        await lock.acquire()
        try:
            body = bytearray()
            try:
                async with asyncio.timeout(config.request_timeout_s):
                    async for chunk in request.stream():
                        if len(body) + len(chunk) > MAX_BYTES:
                            return error(request, 413, "image_size_limit", "Image exceeds 10 MiB")
                        body.extend(chunk)
            except ClientDisconnect:
                return error(request, 400, "client_disconnected", "Image upload was interrupted")
            except TimeoutError:
                return error(request, 408, "request_timeout", "Image upload timed out", True)
            if not body:
                return error(request, 413, "image_size_limit", "Image body is empty")
            if length is not None and len(body) != declared:
                return error(
                    request, 400, "incomplete_body", "Body length differs from Content-Length"
                )
            try:
                start = time.perf_counter()
                result = await run_in_threadpool(app.state.sorter.infer, bytes(body))
                result["request_id"] = request.state.request_id
                validated = Decision.model_validate(result)
                app.state.counters["inferences"] += 1
                app.state.counters["inference_seconds"] += time.perf_counter() - start
                LOG.info(
                    json.dumps(
                        {
                            k: result[k]
                            for k in (
                                "request_id",
                                "model_version",
                                "confidence",
                                "latency_ms",
                                "status",
                                "reasons",
                            )
                        }
                    )
                )
                return validated
            except InputError as exc:
                return error(request, 400, "invalid_image", str(exc))
            except Exception:
                LOG.exception(
                    json.dumps(dict(event="inference_error", request_id=request.state.request_id))
                )
                return error(request, 500, "inference_failed", "Image inference failed")
        finally:
            lock.release()

    return app


def serve(sorter, host="127.0.0.1", port=8000):
    import uvicorn

    settings = Settings.from_env()
    settings.validate_binding(host)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        create_app(sorter, settings),
        host=host,
        port=port,
        workers=1,
        limit_concurrency=32,
        backlog=64,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=30,
        proxy_headers=False,
        access_log=False,
        server_header=False,
    )
