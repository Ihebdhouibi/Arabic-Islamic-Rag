# Running the General QA Module

How to stand up the general question-answering module on a machine with enough disk, RAM, and
(ideally) a GPU: install the model backends, ingest books, and query with cited answers.

> The corpus itself is the dataset repo (`<root>/<NN>__<category>/<id>__<book>/...`). If you have it
> checked out, `corpus_root` points at it and no separate download is needed.

## 1. Hardware guidance

| Backend | Vector dim | Approx download | Runs comfortably on |
|---------|-----------:|----------------:|---------------------|
| BGE-M3 (`--model bge-m3`, default) | 1024 | ~2.3 GB | CPU (slow) or any GPU |
| Qwen3-Embedding-8B (`--model qwen3`) | 4096 | ~16 GB | GPU with ~16 GB VRAM (fp16) |
| Qwen3 int8 / int4 (bitsandbytes) | 4096 | same weights | GPU with ~8 GB / ~4–5 GB VRAM (measure first) |
| bge-reranker-v2-m3 (cross-encoder) | — | ~2.3 GB | CPU (ok for ~100 candidates) or GPU |

- **Full corpus** (8,589 books / 7.6 M pages) requires substantial disk for Postgres + Qdrant plus
  hours-to-days of embedding time. Start with a **subset** (a category or `--limit N`) to validate.
- **Laptops** (little VRAM / disk): use `--model bge-m3` on a small subset; full fp16 Qwen3-8B needs
  a GPU box. For Qwen on smaller GPUs, run the quantization comparison below before committing to
  a production dtype.

## 2. Install

```bash
pip install -e ".[dev,bge,rerank]"     # BGE-M3 + reranker
# For Qwen3 instead of / in addition to BGE-M3:
pip install -e ".[qwen]"
# For Qwen3 int8/int4 weight quantization (CUDA + bitsandbytes):
pip install -e ".[qwen,qwen-quant]"
```

Point the Hugging Face cache at a drive with room for the weights:

```bash
# bash
export HF_HOME=/data/hf-cache
# PowerShell
$env:HF_HOME = "D:\hf-cache"
```

## 3. Start services and migrate

```bash
docker compose up -d --wait     # Postgres :5433, Qdrant :6333
alembic upgrade head
```

Configuration is environment-driven (prefix `SHAMELA_`), defaults match docker-compose.

| Variable | Default | Notes |
|----------|---------|-------|
| `SHAMELA_CORPUS_ROOT` | `.` | Dataset root |
| `SHAMELA_QDRANT_DENSE_DIM` | `1024` | **Must match the model**: 1024 for BGE-M3, 4096 for Qwen3 |
| `SHAMELA_QDRANT_COLLECTION` | `shamela_general` | Vector collection name |
| `SHAMELA_BM25_STATE_PATH` | `bm25_state.json` | Persisted surface-BM25 encoder |

> Using Qwen3? Set `SHAMELA_QDRANT_DENSE_DIM=4096` **before** ingesting, and use a fresh collection.

## 4. Fit the sparse arm (once)

The surface-BM25 encoder must be fitted over the books you will ingest so document and query sparse
vectors share one term space:

```bash
shamela-rag build-bm25 --category 3 --limit 25      # or --all
```

This writes the encoder to `SHAMELA_BM25_STATE_PATH`.

## 5. Ingest

```bash
shamela-rag ingest --category 3 --limit 25 --model bge-m3   # or --all
```

- Downloads the model on first use (into `HF_HOME`).
- Idempotent: re-running replaces a book's rows/points. Use `--dry-run` to preview counts.
- Ingestion picks up the persisted BM25 encoder automatically when the state file exists.

## 6. Query

CLI:

```bash
shamela-rag ask "ما رأي الشافعي في مالك؟" --k 5
shamela-rag ask "What did al-Shafi'i say about Malik?" --json
shamela-rag ask "..." --no-rerank        # skip the cross-encoder (faster, lexical rerank)
```

HTTP API:

```bash
uvicorn "shamela_rag.api.app:create_app_from_settings" --factory --port 8000
# then:
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H "content-type: application/json" \
  -d '{"question":"ما رأي الشافعي في مالك؟","k":5}'
```

The response is a grounded answer plus structured citations (book, author, page, category, snippet,
`is_footnote`), each resolving to a real chunk.

## 7. What you get, and generation options

- Real **retrieval**: BGE-M3 (or Qwen3) dense + surface-BM25 sparse -> RRF -> cross-encoder rerank
  -> authority boost -> parent/neighbor expansion, with citations mapped to Postgres provenance.
- **Answer text** defaults to the in-memory stub (`SHAMELA_LLM_BACKEND=memory`). For real prose,
  use a local model or a hosted API. `ask`, `/ask`, and Streamlit all read these env vars.

### 7.1 Ollama (easiest on Windows)

```bash
ollama pull qwen2.5:3b          # ~2 GB; or qwen2.5:7b ~4.7 GB
$env:SHAMELA_LLM_BACKEND = "ollama"
$env:SHAMELA_LLM_OLLAMA_MODEL = "qwen2.5:3b"
```

Ollama must be running on `http://localhost:11434`. No Python extra.

### 7.2 llama.cpp GGUF

```bash
pip install -e ".[llm]"
$env:SHAMELA_LLM_BACKEND = "llamacpp"
$env:SHAMELA_LLM_GGUF_PATH = "D:\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
$env:SHAMELA_LLM_N_GPU_LAYERS = "0"
```

Download a Q4_K_M instruct GGUF (e.g. Qwen2.5-7B-Instruct, ~4.7 GB / ~5-8 GB RAM). If
`llama-cpp-python` fails to build, use Ollama.

### 7.3 Hosted OpenAI-compatible API (Together.ai, DeepSeek, DashScope, ...)

No local weights, no GPU/RAM budget on this machine at all — a plain HTTPS call per question.
Any provider that speaks the standard `/chat/completions` schema with Bearer auth works by just
changing the base URL and model name.

```bash
$env:SHAMELA_LLM_BACKEND = "openai_compatible"
$env:SHAMELA_LLM_API_BASE_URL = "https://api.together.xyz/v1"   # default; omit to use it
$env:SHAMELA_LLM_API_KEY = "..."
$env:SHAMELA_LLM_API_MODEL = "Qwen/Qwen3.5-9B"                  # or a DeepSeek model, etc.
```

Notes from getting a Together.ai key working: only some models are pay-per-token **serverless**
(e.g. `Qwen/Qwen3.5-9B`, `deepseek-ai/DeepSeek-V4-Flash-0731` confirmed working) — larger/newer
ones (`Qwen/Qwen3-8B`, `deepseek-ai/DeepSeek-V3.1`, Turbo variants) return
`model_not_available` and require spinning up a paid dedicated endpoint first. Reasoning-style
models spend part of `max_tokens` on an internal `reasoning` field before the real answer — keep
`SHAMELA_LLM_MAX_TOKENS` generous (a few hundred) or you'll get an empty response cut off
mid-thought. Together's embeddings catalog is English-only (`bge-base-en-v1.5`) — this backend is
for generation only, it does not help with embedding/ingestion.

### 7.4 OpenRouter dense embeddings

OpenRouter hosts `qwen/qwen3-embedding-8b` and `baai/bge-m3`. Put the key in `.env` only:

```bash
# .env
SHAMELA_EMBEDDING_BACKEND=openrouter
SHAMELA_EMBEDDING_API_BASE_URL=https://openrouter.ai/api/v1
SHAMELA_EMBEDDING_API_KEY=...
```

Then `build_embedder("bge-m3")` / `build_embedder("qwen3")` use OpenRouter.

Run the three-stage model comparison:

```bash
shamela-rag compare-dense-models --stage dense-only \
  --output-dir artifacts/m6-dense-openrouter \
  --chunks artifacts/qwen-quant-golden/eval_chunks.jsonl \
  --golden docs/technical_docs/general_qa_golden_staging.jsonl

shamela-rag compare-dense-models --stage hybrid-bm25 --output-dir artifacts/m6-dense-openrouter \
  --chunks artifacts/qwen-quant-golden/eval_chunks.jsonl \
  --golden docs/technical_docs/general_qa_golden_staging.jsonl

shamela-rag compare-dense-models --stage bge-sparse --output-dir artifacts/m6-dense-openrouter \
  --chunks artifacts/qwen-quant-golden/eval_chunks.jsonl \
  --golden docs/technical_docs/general_qa_golden_staging.jsonl
```

Stage 3 requires `pip install -e ".[bge]"` for BGE-M3 learned-sparse vectors.

## 8. Evaluate (optional)

With a golden set (`docs/technical_docs/general_qa_golden_staging.jsonl` format), score retrieval:

```python
from pathlib import Path
from shamela_rag.eval import load_golden_dataset, RunConfig, run_comparison, format_comparison

dataset = load_golden_dataset(Path("golden.jsonl"))
# build a retrieve(query)->[book_id,...] callable per configuration, then:
report = run_comparison([RunConfig("bge-m3", retrieve_bge)], dataset, ks=(10, 100))
print(format_comparison(report))
```

Use this for the model comparison (Qwen3 vs BGE-M3) and the chunk-size sweep.

### Qwen3 quantization comparison (issue #135)

fp16 Qwen3-Embedding-8B is ~16 GB VRAM. Measure int8 (and optional int4 / GGUF) against that
baseline before using a quantized path for M6-03 or production:

```bash
pip install -e ".[qwen,qwen-quant]"   # bitsandbytes needs a CUDA GPU on most platforms
shamela-rag compare-qwen-quant --output-dir artifacts/qwen-quant --device cuda
# optional dense retrieval on a shared chunk sample:
#   --chunks path/to/chunks.jsonl --golden docs/technical_docs/general_qa_golden_staging.jsonl
# CPU / 16GB RAM (skip fp16 OOM + CUDA int8; official Q4_K_M GGUF):
#   pip install -e ".[qwen,llm]"
#   shamela-rag compare-qwen-quant --output-dir artifacts/qwen-quant \
#     --skip-fp16 --no-int8 --no-int4 --download-gguf
```

Writes `comparison_table.md` (load time, RSS/VRAM, ms/text, mean cosine vs fp16, recommendation)
and `metrics.json`. Quality must clear the cosine / golden-set bar — do not assume int8 is free.
