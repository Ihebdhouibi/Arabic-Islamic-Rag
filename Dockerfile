# syntax=docker/dockerfile:1
#
# Runtime image for the general-QA API and for one-off ingestion/migration jobs (same image, see
# docs/technical_docs/deployment.md). Embeddings and generation are hosted (OpenRouter/Together),
# so only the [rerank] extra is needed here — the cross-encoder reranker still runs in-process.
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libpq5: runtime lib for psycopg (binary wheel still needs it on slim images)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

RUN pip install -e ".[rerank]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Default: serve the API. Override the command for the ingestion/migration jobs, e.g.:
#   docker run <image> alembic upgrade head
#   docker run <image> shamela-rag build-bm25 --all
#   docker run <image> shamela-rag ingest --all --model qwen3
CMD ["uvicorn", "shamela_rag.api.app:create_app_from_settings", "--factory", "--host", "0.0.0.0", "--port", "8000"]
