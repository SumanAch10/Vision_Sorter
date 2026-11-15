FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv
FROM python:3.12.16-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS runtime
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" \
    SORTER_MODEL=/app/artifacts/model SORTER_HOST=0.0.0.0
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --no-cache \
    && groupadd --gid 10001 sorter \
    && useradd --uid 10001 --gid 10001 --create-home sorter
ARG MODEL_DIR=artifacts/real-model
ARG SEGMENTER_DIR=artifacts/segmentation-model
ARG SEGMENTER_PATH=/app/artifacts/segmentation-model
COPY --chown=10001:10001 ${MODEL_DIR}/ ./artifacts/model/
COPY --chown=10001:10001 ${SEGMENTER_DIR}/ ./artifacts/segmentation-model/
COPY docs/DATA_SOURCES.md docs/TACO-LICENSE.txt docs/TrashNet-LICENSE.txt ./notices/
ENV SORTER_SEGMENTER=${SEGMENTER_PATH}
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=4)"
CMD ["vision-sorter", "serve", "--host", "0.0.0.0", "--port", "8000"]
