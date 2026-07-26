# Embeddings — Overview

> This folder is the deep, hands-on companion to
> [docs/technical_docs/03_embeddings_and_vector_stores.md](../technical_docs/03_embeddings_and_vector_stores.md),
> which already surveys the embedding-model landscape (commercial vs. open-source vs.
> Arabic-specific), MTEB, Matryoshka embeddings, and vector database options. This folder doesn't
> repeat that survey — it goes underneath it: the actual mechanics of what an embedding is, how
> it's produced and trained, the math behind similarity search, and — the part a survey doc can't
> do — **real embeddings computed on real snippets from this project's own corpus**, not
> illustrative mockups.

## Why a separate folder, not more sections in doc 03

Doc 03 answers "which embedding model should we use." This folder answers "what is actually
happening when text becomes a vector, and what does that look like on our own data." Different
question, different depth, worth keeping separate rather than bloating a document that's already
doing its own job well.

## Document index

| Doc | Covers |
|---|---|
| [01_how_embeddings_work.md](01_how_embeddings_work.md) | Bi-encoder architecture (tokenize → encode → pool → normalize), contrastive training, and how bi-encoders/cross-encoders/late-interaction (ColBERT) differ structurally — with diagrams |
| [02_similarity_metrics_and_geometry.md](02_similarity_metrics_and_geometry.md) | Cosine similarity, dot product, and Euclidean distance explained geometrically, with a worked figure, plus what L2 normalization actually does to the geometry |
| [03_worked_example_real_corpus_embeddings.md](03_worked_example_real_corpus_embeddings.md) | **Real embeddings**, computed locally on real Shamela passages (a hadith, a tafsir excerpt, a fiqh ruling, an aqidah definition, a philological poetry commentary) — a real similarity matrix and a real 2D projection, not illustrations |
| [04_model_architectures_deep_dive.md](04_model_architectures_deep_dive.md) | Encoder-only vs. LLM-based embedding backbones, single-vector vs. multi-vector (ColBERT) tradeoffs in more depth than doc 03's survey-level treatment |
| [05_evaluation_metrics_worked_examples.md](05_evaluation_metrics_worked_examples.md) | recall@k, MRR, and NDCG computed by hand against this project's own [golden evaluation seed](../technical_docs/golden_eval_seed.jsonl), so the metrics aren't abstract |

## A note on rigor

Where this folder makes an empirical claim about *this corpus's* embeddings (doc 03), it's backed
by an actual local run of a real embedding model against real text pulled from the repository —
not a description of what would probably happen. Where it explains general theory (docs 01, 02,
04, 05), it's explained precisely enough to verify, with the same "don't assume, measure" ethos
already established across `docs/technical_docs/`.
