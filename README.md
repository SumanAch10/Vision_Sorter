# Vision Sorter

A computer-vision inspection application that combines **six-material classification** with **real binary semantic segmentation of litter**. Upload an image to see material alternatives, a predicted litter mask, its image-coordinate centroid, and a confidence-aware simulation decision.

| Capability | Implementation | Measured held-out result |
|---|---|---|
| Material classification | ImageNet-pretrained ResNet-18, fine-tuned on TrashNet | **89.0% accuracy**, 381 images |
| Binary semantic segmentation | ResNet-18 encoder + U-Net decoder, trained on real TACO polygons | **0.51 foreground IoU / 0.61 Dice**, 313 images |
| Service | FastAPI/Uvicorn, OpenAPI, bearer authentication, limits, metrics | **28 local tests passed** |
| Packaging | Locked CPU dependencies, Python wheel, Docker/Compose, model bundle | Linux ARM64 container and verified HTTPS tested; cloud rollout pending |

These are **separate task/dataset evaluations**, not a combined sorting accuracy. Segmentation labels are **background/litter**, not per-pixel material classes or instance IDs. This is a research/portfolio application with a deployment-oriented service, not validated industrial automation. Physical robot control is disabled; model approvals remain false.

## Run the trained application

This workspace contains the generated model artifacts. From the repository root:

```sh
uv sync --frozen --extra dev
uv run --frozen vision-sorter serve
```

Open **http://127.0.0.1:8000**. Upload a PNG, JPEG, or WebP showing one isolated item. The green overlay is the learned litter mask; the white circle is its predicted centroid. Toggle the overlay to inspect errors. The material score and segmentation pixel score are distinct quantities.

```sh
uv run --frozen vision-sorter infer demo1.jpeg --output artifacts/decision.json
uv run --frozen python scripts/preflight.py --require-segmentation
```

Defaults: classifier `artifacts/real-model`, segmenter `artifacts/segmentation-model`. Use `--model`, `--segmenter`, `--calibration`, and `--threshold` to override. An empty `--segmenter ''` disables real masks. `SORTER_MODEL` and `SORTER_SEGMENTER` override automatic artifact discovery. Restart after replacing models.

A fresh clone needs either a checksum-verified model bundle extracted into `artifacts/` or the training steps below. Trained weights and source photographs are deliberately not committed to Git. The original synthetic demo remains available with `uv run --frozen vision-sorter demo` and must not be used to claim real-photo accuracy.

## Reproduce real-data training

Python 3.12 and uv 0.13.7 were used. The lockfile resolves CPU-only PyTorch wheels on Linux. Training is separate from serving; nothing downloads at service startup.

```sh
uv sync --frozen --extra dev
uv run --frozen python scripts/prepare_real_data.py
uv run --frozen vision-sorter train-classifier data/raw/trashnet/dataset-resized
uv run --frozen vision-sorter fine-tune data/raw/trashnet/dataset-resized --output artifacts/real-model
uv run --frozen python scripts/prepare_segmentation_data.py --count 0 --output data/processed/taco-all
uv run --frozen vision-sorter train-segmenter data/processed/taco-all/dataset.json --epochs 40
```

The classifier starts with frozen ResNet features, learns a six-way head, and fine-tunes the final residual stage with augmentation. Seeded stratified splits are 1,762 train / 378 validation / 381 test after excluding contradictory duplicates. Validation loss selects the checkpoint; validation NLL selects its calibration temperature.

The segmenter uses a pretrained ResNet encoder and a U-Net decoder. Human TACO polygons are scaled to author-hosted thumbnail dimensions and rasterized into a binary union of all annotated litter. Of 1,034 eligible author-hosted images, two byte-identical copies are excluded. The remaining **1,032 images** are split by source batch: **622 train / 97 validation / 313 test**. Batches are not guaranteed independent capture sessions. External-host photos are excluded before sampling; image license metadata is preserved, and source photos are not redistributed.

Images/masks are letterboxed to 256px without changing aspect ratio. Padding is ignored in cross-entropy and Dice loss. Paired flips/rotations preserve mask alignment. Validation macro foreground IoU selects the checkpoint; validation also selects the mask threshold. Reported IoU/Dice are macro foreground metrics at 256px, not native-resolution grasp accuracy. MPS training took about six minutes in this run; CPU timing will differ. Seeds do not guarantee identical weights across devices.

Content hashes, exclusions, source groups, preprocessing, splits, training history, evaluation, model checksums, and limitations are saved in manifests. Data sources and attribution are in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## What an inference decision means

The pipeline validates the image, classifies the frame, predicts binary litter pixels, restores the mask to original image coordinates, and calculates its centroid/bounding box. Connected components are a quality check, not learned instance segmentation; touching objects can merge.

The material score is a validation-temperature-scaled class probability. It can be confidently wrong on unfamiliar images. The segmentation score is an **uncalibrated** mean foreground probability. Neither establishes a correct mask, correct material, or safe grasp. Your silver-can photo remains a known material-classification failure and is rejected as uncertain.

A simulated pick requires all of these: adequate image quality, class confidence/margin, supported material, acceptable mask area/score, one significant region, no border contact, a valid resolution-matched in-bounds calibration, and explicit approval of **both** real models. Release approvals are false. Rejected decisions suppress destination and millimeter pick coordinates. Predicted image centroids remain visible for inspection. The application has no physical motor adapter.

For tabletop calibration, `vision-sorter calibrate points.json --output artifacts/calibration.json` accepts corresponding `pixels`, `millimeters`, `image_size`, and `workspace_mm: [min_x,min_y,max_x,max_y]`. At least four non-collinear pairs are needed. Verify independent points and record RMS error before setting `validated: true`; the gate requires at most 5mm. Fitting residual alone is not validation, and a planar transform does not correct lens distortion or object height.

## Deployment and API

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for exact setup, authentication, container configuration, limits, rollback, and remaining verification. Quick container preparation on a Docker host:

```sh
uv run --frozen python scripts/create_api_key.py
# On Linux, grant only container UID 10001 access to the secret; see the runbook.
docker compose up --build -d
```

The container runs as a non-root user. Compose adds a read-only filesystem, resource limits, dropped capabilities, a mounted secret, and loopback-only publishing. The GitHub workflow contains a synthetic-fixture container smoke test. **The actual release image passed Linux ARM64 container, authentication, inference, and local HTTPS tests. Public cloud rollout and Linux AMD64 remain unverified.** See [deployment verification](docs/DEPLOYMENT_VERIFICATION.md). Use TLS ingress and managed secrets before public exposure.

- `GET /health`: process health.
- `GET /ready`: model readiness and identities; never a guarantee of sorting safety.
- `GET /v1/model`: sanitized model cards, capabilities, and held-out metrics.
- `POST /v1/infer`: raw PNG/JPEG/WebP body; typed schema 1.0 response.
- `GET /metrics`: Prometheus-compatible operational counters.
- `GET /docs`: interactive OpenAPI reference.

```sh
curl -H 'Content-Type: image/jpeg' --data-binary @demo1.jpeg http://127.0.0.1:8000/v1/infer
```

When configured, send `Authorization: Bearer <key>` to inference, model metadata, and metrics. The UI accepts a key in page memory. API keys are never accepted in URLs. Non-loopback CLI binding requires a secret or deliberate anonymous override.

Masks use row-major `[start,length]` runs and `size: [width,height]`. Responses include material alternatives, confidence semantics, mask source, segmentation model/score, centroid, bounding box, rejection reasons, latency, and simulation. Images stay in memory and are not logged. Limits: 10MiB, 16 megapixels, 64px minimum per side, one inference per worker, bounded request streaming, rate limiting. Errors contain a request ID. Configuration is listed in `.env.example`; the CLI reads process environment, not `.env` automatically.

## Verify, package, and share

```sh
uv run --frozen ruff check src scripts tests
uv run --frozen ruff format --check src scripts tests
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python scripts/preflight.py --require-segmentation
uv run --frozen python scripts/benchmark_release.py
uv run --frozen python scripts/package_release.py
uv build --wheel --out-dir artifacts/releases
```

Tests cover real mask geometry, polygon scaling, ignored padding, gradients, metrics, model checksums, split isolation, ambiguity rejection, simulator acceptance/rejection, schema invariants, authentication, rate limits, bounded uploads, readiness, and API responses. Real-artifact tests require local data; the CI fixture skips those two tests explicitly.

[Release evidence](docs/RELEASE_REPORT.json) · [local CPU latency](docs/latency.json) · [code walkthrough](docs/CODE_WALKTHROUGH.md) · [deployment verification](docs/DEPLOYMENT_VERIFICATION.md)

The local 20-image sequential CPU benchmark measured about 34ms median and 36ms p95, excluding disk I/O and HTTP transfer. This is not a throughput guarantee or a target-host benchmark. Held-out data was inspected during development; a new external untouched evaluation, negative examples, broader tabletop data, real grasp validation, and checks on the intended public deployment host remain necessary before operational use.
