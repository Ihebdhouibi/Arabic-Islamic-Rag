# Deployment

How to run the general-QA API and its data pipeline in production, and what's still missing to
call this "production-ready" beyond what's here.

## 1. Architecture

One image (`Dockerfile`), two roles:

- **API service** — `uvicorn shamela_rag.api.app:create_app_from_settings --factory`, serving
  `GET /health` and `POST /ask`. Stateless; scales horizontally.
- **One-off jobs** — same image, different command: `alembic upgrade head`,
  `shamela-rag build-bm25 --all`, `shamela-rag ingest --all --model qwen3`. These are batch jobs,
  not something the API container does on every boot.

Embeddings and generation are hosted (OpenRouter and/or Together.ai) — the image itself only needs
the `[rerank]` extra for the local cross-encoder reranker; no GPU, no multi-GB model downloads.

## 2. Why ingestion doesn't need to run on a developer machine

Ingestion is: read corpus files -> chunk -> call the embedding API -> write to Postgres/Qdrant.
None of those steps need a developer laptop specifically — they need network access to the
embedding API, network access to the production Postgres/Qdrant, and the corpus files on disk.
Run it as a one-off job (this image, `ingest`/`build-bm25` command) against the production
databases, triggered as part of deployment or manually right after infrastructure comes up. Local
machines are for iterating on a small subset (one book/category), not for running full-corpus
production ingestion.

## 3. Required environment variables

```bash
# Databases (real managed instances in production, not the dev docker-compose)
SHAMELA_POSTGRES_HOST=...
SHAMELA_POSTGRES_PORT=5432
SHAMELA_POSTGRES_USER=...
SHAMELA_POSTGRES_PASSWORD=...
SHAMELA_POSTGRES_DB=...
SHAMELA_QDRANT_URL=https://...
SHAMELA_QDRANT_COLLECTION=...
SHAMELA_QDRANT_DENSE_DIM=4096          # 4096 for qwen3, 1024 for bge-m3 — must match the ingested model

# Embeddings (hosted — see runbook §7.4)
SHAMELA_EMBEDDING_BACKEND=openrouter
SHAMELA_EMBEDDING_API_KEY=...

# Generation (hosted — see runbook §7.3)
SHAMELA_LLM_BACKEND=openai_compatible
SHAMELA_LLM_API_KEY=...
SHAMELA_LLM_API_MODEL=...

# API auth — required once this is reachable beyond localhost (see §5)
SHAMELA_API_AUTH_KEY=...
```

## 4. Build and run

```bash
docker build -t shamela-rag .

# Migrate (one-off, run once per deploy)
docker run --rm --env-file .env.production shamela-rag alembic upgrade head

# Ingest (one-off, run once — or re-run when the corpus/model changes)
docker run --rm --env-file .env.production -v /path/to/corpus:/corpus \
  -e SHAMELA_CORPUS_ROOT=/corpus \
  shamela-rag shamela-rag build-bm25 --all
docker run --rm --env-file .env.production -v /path/to/corpus:/corpus \
  -e SHAMELA_CORPUS_ROOT=/corpus \
  shamela-rag shamela-rag ingest --all --model qwen3

# Serve
docker run -d --env-file .env.production -p 8000:8000 shamela-rag
```

## 5. What's still missing for a real production deployment

This scaffolding covers the container, auth gate, and the batch-job pattern. It does **not**
cover, and someone should decide/implement before going live:

- **Managed Postgres + Qdrant.** The dev `docker-compose.yml` is for local iteration only —
  production needs persistent, backed-up, access-controlled instances (managed cloud service or
  a self-hosted equivalent with real durability guarantees).
- **Rate limiting.** `X-API-Key` auth (added here) stops anonymous access but does not stop a
  single valid caller from overwhelming the service or the hosted embedding/generation APIs.
- **Secrets management.** `.env.production` on a host is a starting point, not an end state —
  use whatever secrets manager the deployment target provides (avoid plain files in the image or
  in version control).
- **Observability.** No structured request logging, metrics, or tracing wired up yet beyond
  `logging_config.py`'s basic setup.
- **TLS termination.** Assumed to be handled by whatever sits in front of this (load balancer,
  reverse proxy, ingress) — the app itself serves plain HTTP.
- **CI/CD for the image itself.** `ci.yml` runs lint/test; nothing yet builds and pushes this
  Dockerfile or automates the migrate-then-ingest-then-serve rollout sequence.
