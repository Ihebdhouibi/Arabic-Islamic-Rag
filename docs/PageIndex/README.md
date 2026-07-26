# PageIndex Analysis

Thorough evaluation of [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — a
"vectorless," reasoning-based document retrieval framework — researched for its own sake and for
fit against this project's Shamela RAG architecture. All claims are sourced (official docs, the
project's own blog, and third-party technical reviews, cited inline); nothing here is asserted
from memory about an actively-evolving external tool.

## Documents

| Doc | Covers |
|---|---|
| [00_overview.md](00_overview.md) | What it is, key facts (license, traction, flagship benchmark result), why it's worth analyzing for this project |
| [01_architecture_and_how_it_works.md](01_architecture_and_how_it_works.md) | The indexing pipeline (tree building) and reasoning-based retrieval algorithm, with diagrams; explicit about what's publicly undocumented |
| [02_strengths_limitations_and_comparison.md](02_strengths_limitations_and_comparison.md) | A balanced Good/Bad/Ugly assessment (cost, data sovereignty, single-benchmark validation) and a structural comparison to vector RAG and this project's own GraphRAG framing |
| [03_fit_assessment.md](03_fit_assessment.md) | Where this could fit the Shamela project specifically (and where it explicitly shouldn't), plus a broader fit table for other domains (financial, legal, technical docs, medical, etc.) |

## Diagrams

Same convention as [docs/technical_docs/diagrams/](../technical_docs/diagrams/README.md): each
diagram is a `.mmd` Mermaid source (editable) + a rendered `.svg` image, embedded in the docs
above.

| Diagram | Shows |
|---|---|
| [diagrams/indexing_pipeline_flow.svg](diagrams/indexing_pipeline_flow.svg) | How a document becomes a tree — structure detection, per-node summarization, recursive splitting |
| [diagrams/reasoning_retrieval_sequence.svg](diagrams/reasoning_retrieval_sequence.svg) | The query-time read-select-extract-judge loop |
| [diagrams/vector_vs_pageindex_comparison.svg](diagrams/vector_vs_pageindex_comparison.svg) | Side-by-side pipeline shapes: embed-and-search vs. tree-and-reason |
| [diagrams/shamela_integration_concept.svg](diagrams/shamela_integration_concept.svg) | How the pattern could slot into this project's existing architecture, using the corpus's already-curated `toc.jsonl` instead of LLM-inferred structure |

## One-line verdict

Adopt the *retrieval pattern* (reasoning over an existing document hierarchy) for the two use
cases that lack better structured ground truth (general Q&A, fiqh lookup), applied lazily per book
rather than as a full-corpus indexing job — and leave hadith takhrij / tafsir-by-verse exactly as
designed, since this project already has exact curated cross-references that beat anything
LLM-reasoning would rediscover. Full reasoning in
[03_fit_assessment.md §5](03_fit_assessment.md#5-bottom-line).
