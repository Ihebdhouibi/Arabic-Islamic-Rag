# PageIndex — Overview

> Analysis of [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex), researched from the
> project's README, official docs (docs.pageindex.ai, pageindex.ai/blog), the companion
> [Mafin2.5-FinanceBench](https://github.com/VectifyAI/Mafin2.5-FinanceBench) evaluation repo, and
> third-party technical write-ups — cited throughout, not asserted from memory, since this is an
> actively evolving external project. See [README.md](README.md) for the full document index.

## What it is

**PageIndex is an open-source (MIT-licensed), "vectorless," reasoning-based retrieval framework.**
Instead of chunking a document and embedding the chunks for similarity search, it converts a long
document into a **hierarchical tree** — essentially an LLM-generated, intelligent table of
contents — and answers questions by having an LLM **reason over that tree's structure** to decide
which section(s) to read, rather than computing vector similarity across chunks.

Core framing, from the project itself: *"similarity ≠ relevance."* PageIndex's stated goal is
retrieval that is grounded in explicit document structure and logical reasoning, and is
**traceable and explainable** — every answer comes with a citation back to a specific tree node
and page range, not just "chunk #4,281 scored 0.83."

## Key facts

| | |
|---|---|
| **Publisher** | Vectify AI |
| **License** | MIT |
| **Launched** | Open-sourced April 1, 2025 |
| **Traction** | ~34.6k GitHub stars, 3k+ forks — genuinely widely adopted, not a niche experiment |
| **Flagship result** | 98.7% accuracy on the full FinanceBench benchmark (10,231 questions) via **Mafin 2.5**, a financial-QA system built on top of PageIndex — versus a commonly cited ~30–50% (some sources: 60–80%) for traditional vector RAG on the same benchmark |
| **Deployment modes** | Self-hosted (open-source repo, standard PDF parsing) · Cloud API/MCP (enhanced OCR, hosted tree building) · Enterprise (dedicated/VPC/on-prem) |
| **Supported inputs** | PDF, Markdown (via `#` heading hierarchy); ecosystem tools extend to notebooks |
| **LLM dependency** | Requires an LLM for every stage — indexing (structure + summaries) and every retrieval query. Default model in the CLI is `gpt-4o-2024-11-20`; multi-provider support via LiteLLM |

## The one-sentence version of how it works

Build a tree once per document (title/summary/page-range per node, nested), then at query time
hand the whole tree to an LLM and let it navigate — read, select a branch, extract, judge
sufficiency, repeat or answer — the way a human expert would flip through a table of contents
rather than search an index card catalog. Full mechanics in
[01_architecture_and_how_it_works.md](01_architecture_and_how_it_works.md).

## Why this is relevant to look at right now

Two things make this worth a serious look rather than a passing mention. First, the core idea —
navigate a document's existing structure with an LLM instead of flattening it into embedded
chunks — is directly adjacent to territory this project's own docs already cover: hierarchical/
TOC-anchored chunking ([docs/technical_docs/02_chunking_strategies.md §4](../technical_docs/02_chunking_strategies.md#4-document-structure-aware--hierarchical-chunking))
and the general graph-vs-vector distinction
([docs/technical_docs/05_knowledge_graphs_and_graphrag.md](../technical_docs/05_knowledge_graphs_and_graphrag.md)).
Second, and more specifically: **this project's corpus already has a curated, hierarchical table
of contents for every book** (`toc.jsonl`) — the exact artifact PageIndex spends most of its LLM
budget generating from scratch. That's a genuinely non-obvious point of leverage, developed fully
in [03_fit_assessment.md](03_fit_assessment.md).

## Document index

- [01_architecture_and_how_it_works.md](01_architecture_and_how_it_works.md) — the indexing
  pipeline and the reasoning-based retrieval algorithm, with diagrams
- [02_strengths_limitations_and_comparison.md](02_strengths_limitations_and_comparison.md) — a
  balanced (not marketing-sourced) assessment, and how it compares structurally to vector RAG and
  GraphRAG
- [03_fit_assessment.md](03_fit_assessment.md) — where this could plausibly fit for the Shamela
  project specifically, and for other domains (financial, legal, technical documentation)
