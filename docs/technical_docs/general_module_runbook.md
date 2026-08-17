# Running the General QA Module

How to stand up the general question-answering module on a machine with enough disk, RAM, and
(ideally) a GPU: install the model backends, ingest books, and query with cited answers.

> The corpus itself is the dataset repo (`<root>/<NN>__<category>/<id>__<book>/...`). If you have it
> checked out, `corpus_root` points at it and no separate download is needed.

## 1. Hardware guidance

| Backend | Vector dim | Approx download | Runs comfortably on |
|---------|-----------:|----------------:|---------------------|
| BGE-M3 (`--model bge-m3`, default) | 1024 | ~2.3 GB | CPU (slow) or any GPU |
| Qwen3-Embedding-8B (`--model qwen3`) | 4096 | ~16 GB | GPU with ~16 GB VRAM |
| bge-reranker-v2-m3 (cross-encoder) | — | ~2.3 GB | CPU (ok for ~100 candidates) or GPU |

- **Full corpus** (8,589 books / 7.6 M pages) requires substantial disk for Postgres + Qdrant plus
  hours-to-days of embedding time. Start with a **subset** (a category or `--limit N`) to validate.
- **Laptops** (little VRAM / disk): use `--model bge-m3` on a small subset; Qwen3-8B needs a GPU box.

## 2. Install

```bash
pip install -e ".[dev,bge,rerank]"     # BGE-M3 + reranker
# For Qwen3 instead of / in addition to BGE-M3:
pip install -e ".[qwen]"
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

## 7. What you get, and local generation

- Real **retrieval**: BGE-M3 (or Qwen3) dense + surface-BM25 sparse -> RRF -> cross-encoder rerank
  -> authority boost -> parent/neighbor expansion, with citations mapped to Postgres provenance.
- **Answer text** defaults to the in-memory stub (`SHAMELA_LLM_BACKEND=memory`). For real prose,
  use a local model (no external API). `ask`, `/ask`, and Streamlit read these env vars.

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
