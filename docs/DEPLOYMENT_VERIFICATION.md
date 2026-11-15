# Deployment verification

**Result: passed on Linux ARM64 using the actual release models.**

The release image was built from the Dockerfile and run in an isolated Lima VM with Docker Engine 29.9.3. The existing Python app on port 8000 was left untouched. The test containers were stopped after testing and the VM was shut down.

| Check group | Result |
|---|---|
| Real release image build | Passed, Linux ARM64 |
| HTTP API checks | 24 passed |
| HTTPS API checks | 23 passed, certificate validation enabled |
| Runtime/security checks | 14 passed |
| Graceful shutdown | Passed; Uvicorn completed lifespan cleanup before SIGTERM exit |
| Local unit/integration suite | 28 passed |

The API checks exercised readiness, real-model identities, authentication rejection, valid material classification and real semantic masks, original-image coordinates, schema validation, request IDs, invalid/corrupt/oversized input handling, concurrency rejection, rate limiting, metrics, and continued readiness after errors. In a six-request concurrent burst, one inference completed and five requests received bounded retryable busy responses.

Runtime checks confirmed UID/GID 10001, read-only root filesystem, writable bounded `/tmp`, dropped capabilities, no-new-privileges, 2GiB memory / two CPU / 128 PID limits, a read-only secret with mode 0400 owned by UID 10001, successful health probes, and failure to bind externally without a secret. No API-key value was found in deployment logs.

HTTPS was tested through the included non-root Nginx ingress. A private two-day CA signed a localhost/127.0.0.1 certificate. The client verified the chain and hostname using that CA file; no insecure verification bypass was used, and the CA was not added to system trust.

Testing found and fixed an Nginx startup failure under the read-only configuration: unused FastCGI/uWSGI/SCGI modules still initialized temporary directories under `/var/cache`. All temporary paths now use the bounded writable `/tmp` mount.

The API stopped with exit 143 after logging `Application shutdown complete` and `Finished server process`. The pinned Uvicorn replays SIGTERM after graceful cleanup; this is not a forced SIGKILL. The TLS proxy exited with code 0. Neither container was OOM-killed.

Full checks, model versions, image identity, limits, and scope are in [DEPLOYMENT_VERIFICATION.json](DEPLOYMENT_VERIFICATION.json). Build and runtime logs remain under `artifacts/deployment-test/`; test secrets and keys are ignored by Git and are never included in the image.

## Repeat on your deployment host

Build and start the application using the main runbook. To test an already-running instance:

```sh
uv run --frozen python scripts/test_deployment.py \
  --base-url https://your-actual-hostname \
  --api-key-file /path/to/private-api-key \
  --output artifacts/deployment-test/host.json
```

The test generates a short concurrent burst and deliberately invalid requests. Run it against a test/staging instance, not a shared live service. `--test-rate-limit` additionally sends enough bad images to trigger the configured limiter. For an isolated private CA, supply `--ca-file`; publicly trusted certificates need no CA override. It uses `demo1.jpeg` by default; supply `--image` if testing from another checkout.

The optional TLS stack is `docker compose -f compose.yaml -f compose.tls.yaml`. Set `SORTER_TLS_DIR` to a directory containing `server.crt` and `server.key`; the proxy's UID 101 must be able to read the key. Defaults publish HTTPS only on loopback port 8443. Supply real certificates and adapt your hosting ingress for public use. Do not deploy the temporary test certificate.

The main stack accepts `SORTER_IMAGE`, `SORTER_PUBLISHED_PORT`, and `SORTER_SECRET_FILE` to select a tested image, avoid a port collision, and use a platform-specific secret path. Recreate both the API and TLS proxy during updates and secret rotation.

This verifies software deployment on one architecture. It does not establish accuracy on arbitrary inputs, joint sorting success, an untouched external benchmark, or safety for physical robot operation. Public-domain DNS/certificates and a specific cloud account still require configuration on that host.
