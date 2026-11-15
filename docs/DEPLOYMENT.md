# Deployment runbook

## Scope and release gate

Version 0.3.0 packages a real-photo material classifier and a real-data **binary litter/background semantic segmenter**. The runtime is an authenticated, bounded FastAPI/Uvicorn service with a typed OpenAPI contract. Neither model is approved for robot operation. Cloud account configuration, public TLS ingress, and an actual physical safety assessment are outside the local release.

The actual release image has now been built and exercised on **Linux ARM64** in an isolated Lima VM. It passed 24 HTTP checks, 23 HTTPS checks with certificate validation enabled, 14 runtime/security checks, graceful shutdown, and 28 local unit tests. See [verification results](DEPLOYMENT_VERIFICATION.md) and [machine-readable evidence](DEPLOYMENT_VERIFICATION.json).

The test used a private two-day localhost certificate authority trusted only by the test client, not a public-domain certificate. Public cloud rollout, actual domain/DNS configuration, Linux AMD64 runtime, and physical robot operation remain unverified. Python, uv, and the optional Nginx ingress use registry-verified immutable image digests.

## Install exact dependencies

Use Python 3.12 and uv 0.13.7:

```sh
uv sync --frozen --no-dev
```

`uv.lock` records transitive versions and hashes. On Linux, PyTorch/torchvision resolve from the official CPU wheel index, avoiding CUDA dependencies. macOS uses its native wheels. `requirements.lock` is an exported alternative; `uv.lock` is authoritative. If installing the export with pip, include `--extra-index-url https://download.pytorch.org/whl/cpu` so CPU local-version wheels can be found, and use `--require-hashes`.

Prepare real-data models using README instructions, or extract the locally generated `artifacts/releases/vision-sorter-models-0.3.0.tar.gz` into `artifacts/`. The archive includes checksums and attribution, not source photographs. Verify its archive hash against `SHA256SUMS` before extraction. Keep the matching application version, model bundle, and lockfile together.

```sh
uv run --frozen python scripts/preflight.py --require-segmentation
```

Preflight loads and checksum-validates both models, executes inference, validates the response, and checks that an unreviewed frame cannot produce a pick. Model readiness means successful loading, not acceptable task accuracy.

## Local service

```sh
uv run --frozen vision-sorter serve
```

Visit http://127.0.0.1:8000. For a trusted local development session, no API key is required. The interface displays the actual model identities and held-out metrics; the key field stays in page memory and is never saved in localStorage.

## Container deployment

Generate a secret once. Existing secrets are retained, not overwritten:

```sh
uv run --frozen python scripts/create_api_key.py
docker compose config
docker compose up --build -d
curl -f http://127.0.0.1:8000/ready
```

The secret is stored at `.secrets/api_key`, excluded from Git and the Docker context, and mounted by Compose. Supply it to the UI's API-key field to connect. `/v1/model`, `/v1/infer`, and `/metrics` require `Authorization: Bearer <key>`. Credentials are never accepted in URLs. Health/readiness, UI assets, and OpenAPI documentation remain public.

The image runs as UID/GID 10001 with CPU inference. Compose adds a read-only root filesystem, limited temporary storage, dropped capabilities, no-new-privileges, a 2GiB memory limit, two CPU quota, a PID limit, restart policy, and loopback-only port publishing. It does not download data or model weights at startup.

**Secret file access:** on Linux the container's UID 10001 must be able to read the mounted secret. The generator uses mode 0600 for local privacy; for deployment, use your platform's secret mechanism with UID 10001 ownership or grant the container group read access. A typical Linux host setup is `sudo chown 10001:10001 .secrets/api_key && sudo chmod 0400 .secrets/api_key`. Do not make it world-readable. The CI container smoke test explicitly handles its fixture secret ownership.

For a real public deployment, publish through a managed HTTPS load balancer or authenticated TLS reverse proxy. Keep port 8000 private; do not send API keys or photographs over plaintext public HTTP. Trust forwarded headers only from a configured proxy. The included service does not trust arbitrary `X-Forwarded-For`; behind one proxy, its per-client rate limit therefore applies to that proxy address. Apply independent per-client and body limits at ingress.

The CLI refuses non-loopback binds without an API key or explicit `SORTER_ALLOW_ANONYMOUS=true`. The anonymous override is intended for deliberate demos behind another access control. Prefer key-file secrets. Keys are loaded at startup. Rotate secrets with a container recreation (`docker compose up -d --force-recreate`), because a bind-mounted file can retain its old inode after an atomic replacement. Recreate/reload the TLS proxy as well after API container updates so its upstream address is refreshed.

## Limits, observability, and error behavior

- Accepted image types: raw PNG, JPEG, WebP. Maximum body 10MiB; maximum 16 megapixels; minimum 64 pixels on each side. Animated/multiframe images are rejected. EXIF rotation is applied.
- Upload timeout: 15 seconds by default; bounded streaming rejects oversized chunked bodies as well as declared sizes.
- One inference per worker; concurrent inference returns retryable 503. Uvicorn bounds overall connection/task concurrency at 32, backlog at 64, and keep-alive at five seconds. CPU inference itself is not forcibly interrupted by an upload timeout; use process supervision for hangs.
- Rate limit: 60 inference attempts per minute per directly connected client by default. Process-local token buckets are bounded to 1,024 clients. Multiple replicas need an ingress/shared limiter.
- Consistent JSON errors include error, message, request_id, retryable. Status codes: 400 invalid image/body, 401 unauthorized, 408 upload timeout, 413 size limit, 415 media type, 429 rate limit, 500 inference failure, 503 not ready/busy.
- Request IDs are generated or accept a bounded safe `X-Request-ID` value; responses echo that ID. Logs contain request/model/status/confidence/reasons/latency, never image bytes or API keys. Source training records are omitted from public model metadata.
- `/metrics` exposes Prometheus-compatible request, error, inference, cumulative inference-duration counters, and readiness. Metrics are per process, not a cluster aggregate.
- `/health` is process liveness. `/ready` becomes successful only after model loading and becomes false at shutdown. Docker probes readiness. SIGTERM receives a 30-second graceful shutdown window.

Environment settings are listed in `.env.example`. The CLI reads process environment variables, not `.env` automatically. Invalid model checksums, class orders, thresholds, and secret configurations fail startup.

## Test and build gate

```sh
uv sync --frozen --extra dev
uv run --frozen ruff check src scripts tests
uv run --frozen ruff format --check src scripts tests
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python scripts/preflight.py --require-segmentation
uv build --wheel
```

The GitHub Actions workflow runs static checks, creates a synthetic fixture without downloading real photos, runs tests, builds a non-root container using explicit demo artifacts, and probes its health/auth boundaries. Real-artifact integration tests run locally when their artifacts exist; they are skipped in the offline synthetic CI fixture. A passing demo container test is not a real-model accuracy evaluation. Before production rollout, run the real release image and representative requests on the target host.

## Rollback and model updates

Retain the previous application image and paired classifier/segmenter manifests and weights. To update, train outside the serving process, evaluate untouched external data, package a new bundle, run preflight, deploy a new immutable image tag, then probe readiness and sample inference. Roll back by redeploying the prior image and its matching secrets/configuration. Never replace a checkpoint while a process is loading it; restart after model/configuration changes.

A sample systemd unit is in `deploy/vision-sorter.service` for a Linux Python-service installation. It assumes a dedicated service account, installation under `/opt/vision-sorter`, and environment configuration in `/etc/vision-sorter.env`; it has not been installed automatically.
