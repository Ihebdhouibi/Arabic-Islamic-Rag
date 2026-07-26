# Embeddings — Deep Dive and Worked Examples

The hands-on companion to
[docs/technical_docs/03_embeddings_and_vector_stores.md](../technical_docs/03_embeddings_and_vector_stores.md).
That document surveys the model/vector-store landscape; this folder explains the mechanics
underneath it and, where possible, backs claims with real computation on real Shamela text rather
than illustration. See [00_overview.md](00_overview.md) for the full rationale.

## Contents

| Doc | Covers |
|---|---|
| [00_overview.md](00_overview.md) | Why this folder exists separately from doc 03, and how to read it |
| [01_how_embeddings_work.md](01_how_embeddings_work.md) | Tokenize → encode → pool → normalize; contrastive training; bi-encoder vs. cross-encoder vs. late-interaction |
| [02_similarity_metrics_and_geometry.md](02_similarity_metrics_and_geometry.md) | Cosine similarity, dot product, Euclidean distance — the actual geometry and a hand-worked numeric example |
| [03_worked_example_real_corpus_embeddings.md](03_worked_example_real_corpus_embeddings.md) | **Real** embeddings computed locally on 8 real Shamela passages (hadith, tafsir, fiqh ×2, aqidah, philology, 2 Quran verses) — real similarity matrix, real PCA plot |
| [04_model_architectures_deep_dive.md](04_model_architectures_deep_dive.md) | Encoder-only vs. decoder-based embedding backbones; Matryoshka embeddings' actual training mechanism |
| [05_evaluation_metrics_worked_examples.md](05_evaluation_metrics_worked_examples.md) | Precision@k, recall@k, MRR, NDCG — hand-computed against a real golden-set example |

## Figures and diagrams

Same convention as the rest of the project: Mermaid `.mmd` sources + rendered `.svg` in
[diagrams/](diagrams/), hand-authored geometric/data figures in [figures/](figures/). The
similarity-matrix heatmap and PCA scatter in doc 03 are generated directly from a real local
model run (see that doc for the exact model, snippets, and reproduction script) — not mockups.
