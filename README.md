# Arabic-Islamic-Rag

Design and architecture work for a Retrieval-Augmented Generation (RAG) / GraphRAG system built
on **al-Maktaba al-Shamela (الشاملة) v4** — a full digital library of classical Islamic texts
(Quran, hadith, tafsir, fiqh across all four Sunni madhahib, aqidah, seerah, history, Arabic
language and literature, and more).

**Status: design phase.** This repository currently contains architecture research and technical
documentation, not a working implementation. See [docs/technical_docs/](docs/technical_docs/) for
the full write-up.

**Wiki:** project documentation for the general module lives in the
[GitHub Wiki](https://github.com/Ihebdhouibi/Arabic-Islamic-Rag/wiki)
(Home, Architecture, Setup, Ingestion, Retrieval, Generation/Citations, API, Contributing).

## What's here

- [`docs/technical_docs/`](docs/technical_docs/) — a documentation series covering:
  - RAG fundamentals, chunking strategies, embeddings & vector stores, retrieval strategies,
    knowledge graphs & GraphRAG, and evaluation — general technical grounding (docs 01–06).
  - Diagrams contrasting vector-store and knowledge-graph retrieval (doc 08).
  - A recommended architecture synthesized specifically for this corpus, covering four use cases:
    general Q&A, hadith takhrij (chain-of-narration tracing), tafsir lookup by verse, and fiqh
    ruling lookup across madhhabs (doc 07).
  - A survey of open-source Arabic-capable LLM candidates for the generation/routing layer
    (doc 09).
  - [`shamela4_dataset_card.md`](docs/technical_docs/shamela4_dataset_card.md) — the original
    dataset card (schema, stats, extraction notes) for the corpus this project is built on.

## The dataset

**The corpus itself is not included in this repository.** At roughly 19–31 GB across 8,589 books
and 7.6 million pages, it belongs in dedicated data storage, not a documentation/code repo — and
it's already properly hosted with Hugging Face's LFS infrastructure. Get it from:

**[huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB](https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB)**

```python
from datasets import load_dataset
ds = load_dataset("AuthenticIlm/Shamela4_Full_DB", split="full")
```

The dataset includes, alongside the raw page/table-of-contents text per book, a `_meta/` layer of
pre-curated cross-references: hadith narrators with biographical/critical (jarh wa ta'dil) data,
isnad (chain-of-transmission) links, hadith concordance across books, Quran verse-to-tafsir links,
and a 1.95M-entry Arabic root dictionary. This structured layer is a major input to the
architecture proposed in the docs above — see
[docs/technical_docs/05_knowledge_graphs_and_graphrag.md](docs/technical_docs/05_knowledge_graphs_and_graphrag.md).

## Development

The general question-answering module is implemented as the `shamela_rag` Python package under
`src/`. Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"          # editable install with dev tools
pre-commit install               # enable lint/format/commit-msg hooks
pytest                           # run the test suite
```

Planning and design live in `docs/implementation/` (plan + issue backlog) and
`docs/technical_docs/`. Contributor conventions — branch model, commit style, local setup, and the
PR checklist — are in [CONTRIBUTING.md](CONTRIBUTING.md) (see also [CLAUDE.md](CLAUDE.md) for the AI
working agreement).

### Local services

Postgres (relational/provenance + source text) and Qdrant (dense + sparse vectors) run via Docker:

```bash
docker compose up -d      # Postgres on :5433, Qdrant on :6333/:6334
docker compose down       # stop (add -v to also drop the data volumes)
```

To install the model backends, ingest books, and query with cited answers, follow
[docs/technical_docs/general_module_runbook.md](docs/technical_docs/general_module_runbook.md).

Credentials and ports default to a local dev profile and can be overridden in a `.env` file
(`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT`, `QDRANT_HTTP_PORT`,
`QDRANT_GRPC_PORT`). Requires Docker Desktop on the Linux engine.

## Citation

If you use the underlying dataset, please cite it (and the original Shamela library) as follows —
reproduced from the dataset's own card:

```bibtex
@misc{shamela4_full_2026,
  title = {Shamela 4 — Full Islamic Library Corpus},
  author = {{AuthenticIlm}},
  year = {2026},
  note = {Extracted from al-Maktaba al-Shamela v4. 8,589 books, 7.6M pages.},
  url = {https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB}
}

@online{shamela,
  title = {al-Maktaba al-Shamela},
  url = {https://shamela.ws}
}
```

## Licensing note

This repository's own contents (the documentation and, later, any code under `docs/` and
elsewhere in this repo) are MIT-licensed — see [LICENSE](LICENSE).

**That license does not extend to the underlying corpus.** Per the dataset's own card: the
original texts are in the public domain, but "the compilation and digitization rights belong to
their respective owners," and use is scoped to research/personal use with respect for the
intellectual property of muhaqqiqīn (critical-edition editors) and publishers. Treat the corpus
under those terms, independent of this repo's MIT license on its own documentation/code.
