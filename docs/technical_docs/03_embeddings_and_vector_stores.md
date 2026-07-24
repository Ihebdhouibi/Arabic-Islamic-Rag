# 03 — Embeddings and Vector Stores

> Part of a 7-document technical series on building a production RAG system over the Shamela digital library (~8,589 books, ~7.6M pages of classical Arabic Islamic scholarship). This document covers **embedding models, sparse/hybrid representations, and vector storage infrastructure** in depth. Chunking strategy, retrieval/reranking pipelines beyond the model level, knowledge graphs/GraphRAG, and evaluation methodology each have their own document — they are mentioned here only as pointers.

## 1. What a Text Embedding Actually Is

An embedding model maps a piece of text — a query, a sentence, a chunk — to a fixed-length vector of real numbers (typically a few hundred to a few thousand dimensions), trained via contrastive learning so that texts with similar *meaning* end up close together in that space regardless of shared surface words. Distance is measured with cosine similarity or dot product; most modern models normalize vectors so the two give equivalent rankings.

This is what makes semantic search possible: a query like "ruling on combining prayers while traveling" can retrieve a passage discussing *jamʿ al-taqdīm* without sharing a single English word with it. That same power is the source of embeddings' main weakness — they can miss exact-term matches (a narrator's name, a rare technical term, a verse number) that lexical search catches trivially (Section 5).

For this project, the stakes are concrete: the corpus is classical/religious-register Arabic — inconsistent diacritics, archaic and technical vocabulary, isnad-dense proper-noun chains, heavy reliance on exact-term precision (a hadith grading, a narrator's name, a specific verse). Every model below was trained overwhelmingly on modern, informal, or web-scale text; none targeted this register. Model choice here is an empirical question, not a leaderboard lookup — Section 9 gives a concrete evaluation methodology.

## 2. The Embedding Model Landscape

### 2.1 General-purpose commercial APIs

| Model | Dimensions (native → Matryoshka options) | Context | Approx. price (per 1M tokens) | Notes |
|---|---|---|---|---|
| OpenAI `text-embedding-3-small` | 1536 → 512, 256, ... | 8191 tokens | ~$0.02 | Cheapest widely-used API option; good English/multilingual baseline |
| OpenAI `text-embedding-3-large` | 3072 → 1024, 256, ... | 8191 tokens | ~$0.13 | Higher quality, especially multilingual; supports dimension truncation |
| Cohere `embed-v4.0` | up to 1536 → 1024, 512, 256 | long-context, multimodal (text + image) | ~$0.12 (text) | 100+ languages; Cohere reports notably stronger non-Latin-script performance (Arabic, Hindi, Japanese, Chinese) than OpenAI's models; also does image embeddings |
| Voyage AI `voyage-4` family | model-dependent | up to 32K (large variants) | $0.02–$0.18 depending on tier | `voyage-multilingual-2` targets multilingual retrieval specifically; strong MTEB/BEIR standing; generous free tier (200M tokens on newer models) |
| Google `gemini-embedding-001` | 3072 → 1536, 768 (Matryoshka) | 2048 tokens/input | ~$0.15 (~$0.075 batch) | Successor to the older Gecko/`text-embedding-004` line; ranks at or near the top of the MTEB multilingual leaderboard as of 2025–2026; supports 100+ languages |

Sources: [OpenAI embeddings announcement](https://openai.com/index/new-embedding-models-and-api-updates/), [Cohere Embed v4 changelog](https://docs.cohere.com/changelog/embed-multimodal-v4), [Voyage AI pricing](https://docs.voyageai.com/docs/pricing), [Google Gemini Embedding launch](https://developers.googleblog.com/gemini-embedding-available-gemini-api/).

Commercial APIs remove the operational burden of hosting a model and are the fastest path to a working prototype. Tradeoffs: per-token cost at scale (7.6M pages of chunked text is a lot of embedding calls, even at ~$0.02/1M tokens), dependency on a third party's uptime and pricing, and a data-residency question that self-hosting sidesteps entirely.

### 2.2 Open-source / self-hostable models

| Model | Params | Notes |
|---|---|---|
| BGE-M3 (BAAI) | ~560M | Multi-lingual, multi-functionality (dense + sparse + multi-vector in one model), multi-granularity (up to 8192 tokens); widely used as a strong multilingual default |
| `multilingual-e5-large` / E5 family (Microsoft) | 560M (large) | Long-standing multilingual baseline; simple prefix-based query/passage convention (`query: ...` / `passage: ...`) |
| GTE (Alibaba) / Qwen3-Embedding | 0.6B–8B | Qwen3-Embedding-8B ranked #1 on the MTEB multilingual leaderboard through much of 2025–2026 among open-weight models; strong on 100+ languages including Arabic |
| Nomic Embed v2 (MoE) | 475M total / 305M active | Fully open (weights + training code + data); Mixture-of-Experts; trained with Matryoshka representation learning; ~100 languages |
| Jina Embeddings v3 | 570M | Task-specific LoRA adapters (retrieval/clustering/classification), 8K context, Matryoshka-trained; reports beating multilingual-e5-large-instruct across multilingual tasks |

Sources: [BGE-M3 paper](https://arxiv.org/abs/2402.03216), [Nomic Embed v2 announcement](https://simonwillison.net/2025/Feb/12/nomic-embed-text-v2/), [Jina Embeddings v3 paper](https://arxiv.org/abs/2409.10173), [Qwen3-Embedding repo](https://github.com/QwenLM/Qwen3-Embedding).

Open-source models require self-hosted inference (GPU/CPU serving, batching, scaling) but eliminate per-token cost and keep every document on infrastructure you control — relevant when the marginal cost of embedding 7.6M pages via a paid API adds up, and where re-embedding the whole corpus every time chunking strategy changes is not something you want to pay for twice.

### 2.3 Arabic-specific and Arabic-capable models

Two separate questions get conflated here: "is there an Arabic-*specific* embedding model" and "do general multilingual models work well *on* Arabic." Both matter.

**Arabic-native encoder models** (AraBERT, CAMeLBERT, ARBERT/MARBERT) were built as general-purpose Arabic language encoders (classification, NER, dialect ID), not as contrastively-trained retrieval models. AraBERT trained on ~8.2B MSA tokens; CAMeLBERT ships MSA, dialectal, and *classical* Arabic variants; ARBERT/MARBERT (61GB/128GB of text) target MSA and dialectal Arabic. Using any of them directly for retrieval typically requires further contrastive fine-tuning — mean-pooling a vanilla BERT checkpoint gives mediocre retrieval quality. They are a starting point for a custom retrieval model, not a drop-in embedding API.

More recent purpose-built Arabic retrieval efforts are closer to what you want: **Swan** (Swan-Small on ARBERTv2, Swan-Large on ArMistral), released with the **ArabicMTEB** benchmark (94 datasets, 8 task types, covering cross-lingual, multi-dialectal, multi-domain Arabic) — Swan-Large reportedly outperforms `multilingual-e5-large` on most ArabicMTEB tasks ([Bhatia et al., 2024, arXiv:2411.01192](https://arxiv.org/abs/2411.01192)). **GATE** applies Matryoshka learning and hybrid-loss training specifically to Arabic STS ([arXiv:2505.24581](https://arxiv.org/abs/2505.24581)).

**The practically important finding**, directly relevant here: a 2025 systematic study of Arabic RAG pipeline components found general-purpose multilingual models — **BGE-M3** and **multilingual-e5-large** — outperformed Arabic-specialized models on real Arabic RAG retrieval, including a Qur'an Tafseer retrieval task (BGE-M3 scored highest, 82.72). The authors attribute this to multilingual training-data scale outweighing Arabic-only pretraining, and explicitly caution that **all evaluation was on Modern Standard Arabic — classical/dialectal registers remain unevaluated** ([Optimizing RAG Pipelines for Arabic, arXiv:2506.06339](https://arxiv.org/abs/2506.06339)).

Take-away for Shamela: BGE-M3 and multilingual-e5-large are the strongest currently-documented starting points — not because they are "Arabic models" but because they are large, well-trained multilingual models with good Arabic coverage. The corpus here (medieval fiqh, isnad chains, tafsir, diacritic-inconsistent classical prose) is a further, undocumented distributional shift beyond even the MSA test sets these findings rest on. Treat these rankings as a hypothesis to test on Shamela text (Section 9), not a conclusion to inherit.

### 2.4 MTEB: the standard reference point

The [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is the field's standard reference for comparing embedding models across retrieval, classification, clustering, STS, reranking, and bitext-mining, in 250+ languages, with per-language filtering including Arabic. Dedicated Arabic-centric benchmarks like ArabicMTEB have emerged precisely because MTEB's general multilingual numbers can mask register-specific weaknesses. As of 2025–2026, open-weight models (Qwen3-Embedding, BGE-M3, Nomic Embed) sit near or above leading closed-source APIs on general multilingual MTEB — "commercial API is automatically higher quality" no longer reliably holds.

**The caveat that matters more than rank:** MTEB Arabic scores are measured almost entirely on Modern Standard Arabic in contemporary registers — none of the standard tasks are built from classical fiqh or hadith corpora. A high MTEB Arabic rank shows a model handles Arabic morphology reasonably well in general; it is not evidence it handles a 12th-century fiqh manual well.

## 3. Matryoshka Embeddings

Matryoshka Representation Learning (MRL) trains an embedding model so that the *first N* dimensions of the full vector are, on their own, a useful lower-dimensional embedding — like nested Russian dolls, more important information is packed into earlier dimensions ([Kusupati et al., 2022, arXiv:2205.13147](https://arxiv.org/abs/2205.13147)). Practically, this means you can truncate a 3072-dimensional embedding down to 1024, 512, or 256 dimensions by simply slicing the vector — no retraining, no separate model — and still retain most of the retrieval quality, trading some accuracy for large reductions in storage and search latency (a 4x dimension reduction is roughly a 4x reduction in vector storage and a meaningful ANN speedup).

This is not a niche technique anymore: OpenAI's `text-embedding-3-*` models, Cohere's `embed-v4.0`, Google's `gemini-embedding-001`, and open models like Nomic Embed v2 and Jina Embeddings v3 all ship with native Matryoshka support. For a corpus the size of Shamela (millions of chunks), Matryoshka truncation is a cheap first lever to pull if storage or query latency becomes a bottleneck — before reaching for a smaller model outright, try truncating dimensions on the model you've already validated for quality.

## 4. Multi-Vector / Late-Interaction Models

Standard "dense embedding" retrieval is a **bi-encoder**: the query and each document are each compressed into a single vector, independently, and compared with one dot product. This is fast and scales well, but a single vector is a lossy summary — fine-grained token-level matches (a specific name, a specific phrase) can get averaged away.

**ColBERT** (and its successor **ColBERTv2**) instead keeps one vector *per token* for both query and document, and computes relevance via "late interaction" (MaxSim: for each query token, take its maximum similarity to any document token, then sum) ([Khattab & Zaharia, 2020, arXiv:2004.12832](https://arxiv.org/abs/2004.12832); [Santhanam et al., 2022, arXiv:2112.01488](https://arxiv.org/abs/2112.01488)). This preserves token-level precision — useful for exactly the kind of exact-term sensitivity (narrator names, technical fiqh terms) that a single pooled vector tends to blur — at the cost of storing many vectors per document instead of one, and a more expensive scoring step at query time (ColBERTv2 mitigates this with aggressive residual compression, cutting the space footprint 6–10x versus the original ColBERT). **ColPali** extends the same late-interaction idea to document *images* — encoding each page as a grid of patch embeddings via a vision-language model and applying MaxSim directly against a rendered page image, skipping OCR/text-extraction entirely ([Faysse et al., 2024, arXiv:2407.01449](https://arxiv.org/abs/2407.01449)). That approach is directly relevant if any part of the Shamela pipeline ever needs to retrieve from scanned page images rather than OCR'd/transcribed text (e.g., manuscript pages where layout, marginalia, or diagrams carry meaning OCR discards).

**Practical tradeoff:** late-interaction models are a precision upgrade over single-vector embeddings, generally used as a second-stage refinement (or with libraries purpose-built for multi-vector storage, e.g., via PLAID/Milvus/Qdrant multi-vector support) rather than as the sole first-stage retriever over a 7.6M-page corpus, because storing 100+ vectors per chunk instead of 1 has real storage and infrastructure implications at that scale.

## 5. Sparse and Hybrid Representations

Dense embeddings are not the only — or always the best — way to represent text for retrieval.

- **BM25 / TF-IDF**: classical sparse lexical retrieval — documents/queries as weighted term bags, scored by overlap and frequency statistics, no learned semantics. Cheap, interpretable, and strong at exactly what dense embeddings are weakest at: exact matches on rare terms, names, numbers, technical vocabulary.
- **SPLADE**: a *learned* sparse representation — a transformer produces a sparse, weighted bag-of-terms over the vocabulary instead of a dense vector, combining BM25-style exact-match efficiency (invertible index) with learned term expansion ([Formal et al., 2021, arXiv:2107.05720](https://arxiv.org/abs/2107.05720); [SPLADE v2, arXiv:2109.10086](https://arxiv.org/abs/2109.10086)).
- **Hybrid dense + sparse retrieval**, fused via **Reciprocal Rank Fusion (RRF)** — a simple, model-free method summing `1 / (k + rank)` across ranked lists ([Cormack, Clarke & Büttcher, 2009, CIKM](https://dl.acm.org/doi/10.1145/1571941.1572114)) — consistently outperforms either method alone in production. Dense and sparse retrieval fail on different, largely uncorrelated query types, so fusing them recovers cases either one misses.

This matters more than average for Arabic. Arabic is morphologically rich (root-and-pattern derivation, heavy affixation, hamza/alif/ta-marbuta variation, inconsistent diacritization): a naive whitespace/BM25 tokenizer fails to match `الصلاة` against `للصلاة` or `صلاته` without stemming or root normalization, while a dense embedding may recognize the relation but blur past a narrator's name or book title a lexical match would catch exactly. For a corpus dense with isnad chains, book titles, and exact Qur'anic/hadith references, **hybrid retrieval is close to a baseline requirement**, and Arabic-aware lexical normalization (root extraction, or at minimum consistent diacritic stripping) is a prerequisite for the lexical side to pull its weight.

## 6. Vector Database and Index Landscape

A brief map of where embeddings actually live at query time:

- **FAISS** (Meta): a *library*, not a server — in-process ANN search (IVF, HNSW, PQ) embedded into your own service. Fast and flexible, but you build persistence, filtering, and the API layer yourself.
- **Milvus** / Zilliz Cloud: purpose-built distributed vector database for billion-scale collections, multiple index types, hybrid dense+sparse search; operationally heavy to self-host well.
- **Qdrant**: Rust-based, strong metadata filtering and payload indexing, competitive low-latency ANN, native hybrid search; often cited as among the fastest open-source options on filtered queries.
- **Weaviate**: built-in hybrid search (vector + BM25 fused server-side) and integrated embedding modules; strong time-to-value and multi-tenancy.
- **pgvector**: a Postgres extension, not a separate system — adds vector columns and ANN indexes (IVFFlat, HNSW) directly inside Postgres. Main argument for it: embeddings, source text, and relational metadata (book, author, page, category) live in one database, queryable with ordinary SQL joins — at the cost of scaling less gracefully past tens of millions of vectors.
- **Elasticsearch / OpenSearch**: full-text search engines with added `dense_vector`/k-NN fields — a natural fit when you also need enterprise-grade full-text search and want dense + BM25 living in the same engine (an easy path to Section 5's hybrid retrieval). OpenSearch is the fully open, Apache-2.0 fork; Elasticsearch's core is open but many enterprise features require a commercial license.
- **Pinecone**: proprietary, fully managed, serverless — no infrastructure to operate, built-in hybrid search, cloud-only.
- **Chroma**: lightweight, open-source, popular for prototyping and small-to-mid scale; simplest to run locally, less proven at very large scale.

### Comparison Table 1: Vector Database / Index Options

| System | Hosting model | Hybrid search support | Metadata filtering | Approx. scale ceiling | Operational complexity | License |
|---|---|---|---|---|---|---|
| FAISS | Self-hosted library (embed in your app) | No (build it yourself) | No (build it yourself) | Tens of millions–billions, given engineering effort | High (you build the server layer) | MIT |
| Milvus / Zilliz Cloud | Self-hosted or managed (Zilliz Cloud) | Yes (dense + sparse) | Yes | Billions | High self-hosted / low managed | Apache 2.0 (core) |
| Qdrant | Self-hosted or managed (Qdrant Cloud) | Yes (dense + sparse fusion) | Yes, strong/fast | Hundreds of millions+ | Moderate | Apache 2.0 |
| Weaviate | Self-hosted or managed (WCD) | Yes, native (vector + BM25) | Yes | Hundreds of millions+ | Moderate | BSD-3-Clause |
| pgvector | Self-hosted (Postgres extension) or managed (e.g., RDS, Supabase) | Manual (pair with Postgres full-text search) | Yes, native SQL | Low tens of millions comfortably | Low (if already on Postgres) | PostgreSQL License |
| Elasticsearch | Self-hosted or managed (Elastic Cloud) | Yes (dense_vector + BM25) | Yes, rich | Hundreds of millions+ | Moderate–high | Elastic License / SSPL (core open, some features commercial) |
| OpenSearch | Self-hosted or managed (AWS OpenSearch Service) | Yes (k-NN + BM25) | Yes, rich | Hundreds of millions+ | Moderate–high | Apache 2.0 |
| Pinecone | Managed only (proprietary, serverless) | Yes, native sparse-dense | Yes | Billions (managed) | Very low | Proprietary |
| Chroma | Self-hosted or managed (Chroma Cloud) | Limited/emerging | Yes, basic | Low millions comfortably | Very low | Apache 2.0 |

For Shamela specifically (7.6M pages → likely tens of millions of chunks depending on chunking granularity): pgvector-on-Postgres is attractive if the project already needs a relational store for book/author/page metadata and doesn't yet need more than tens of millions of vectors; Qdrant or OpenSearch are the more likely production landing spots once hybrid search and filtering by book/genre/madhhab become first-class requirements at full corpus scale.

## 7. Comparison Table 2: Embedding Model Categories

| Dimension | Commercial API (OpenAI/Cohere/Voyage/Google) | Open-source, self-hosted (BGE-M3/E5/Qwen3/Nomic) | Arabic-specific (Swan/GATE/fine-tuned AraBERT-family) |
|---|---|---|---|
| Cost model | Per-token, ongoing (~$0.02–$0.18/1M tokens) | Infrastructure (GPU/CPU serving) + engineering, no per-token fee | Same as open-source, plus fine-tuning/eval effort |
| Multilingual/Arabic quality (general MSA text) | Strong, especially Cohere v4 and Gemini embedding on non-Latin scripts | Strong — BGE-M3 and multilingual-e5-large lead on documented Arabic RAG benchmarks | Competitive on ArabicMTEB tasks, but less evidence on retrieval-in-the-wild vs. the strongest multilingual models |
| Quality on classical/religious-register Arabic | Unverified — not a benchmark target for any vendor | Unverified — same gap | Unverified — even ArabicMTEB is not classical-Arabic-focused |
| Latency | Network round-trip + provider queueing; can be a bottleneck at high QPS | Local inference; can be optimized (batching, quantization, GPU) but you own it | Same as open-source |
| Data privacy | Text leaves your infrastructure to a third party | Fully on-premises/private-cloud possible | Fully on-premises/private-cloud possible |
| Ease of adoption | Very high — one API call, no infra | Moderate — requires model serving setup (vLLM/TEI/etc.) | Lower — smaller ecosystem, fewer production-hardened serving recipes |

The row that should drive the decision for this project is the third one: **no option in this table has documented performance on classical/religious-register Arabic**. Every comparison above is built on MSA or general multilingual benchmarks. This is the central argument for Section 9's evaluation approach rather than picking a row and trusting it.

## 8. Reranking: The Companion to Embeddings

Embedding-based (and hybrid) retrieval optimizes for fast recall over a large index — it returns a candidate set that is *probably* relevant, at speed. A **reranker** is a cross-encoder that scores a query and a single candidate document jointly rather than comparing two independently-computed vectors — far more expensive per pair, but meaningfully more precise since it can attend directly across both texts. Rerankers are applied only to a shortlist (e.g., top 50–100 candidates), not the full index, keeping the added cost bounded.

Common options: **Cohere Rerank** (managed API, multilingual, v3.5/v4), **BGE-reranker-v2-m3** (open-source, multilingual, pairs naturally with BGE-M3 embeddings), and **Jina Reranker** (open-source/API, multilingual, long-context; v3 reports state-of-the-art BEIR nDCG). Reranking quality is coupled to the same multilingual/register questions as embeddings — a reranker without meaningful classical Arabic exposure can misrank a promising candidate just as easily. Full pipeline mechanics (cascade strategies, latency budgets) are covered in the retrieval-strategies document.

## 9. Practical Guidance: Evaluate, Don't Assume

Every section above converges on one warning: leaderboard rank (MTEB, ArabicMTEB, or any benchmark) reflects the *distribution it was tested on* — largely MSA, modern registers, web/Wikipedia/news text. Classical/religious Arabic is a distribution shift no public benchmark measures directly. Treat leaderboard position as a shortlist filter, not a final answer.

A minimal evaluation workflow before committing to a model for Shamela:

1. **Build a small labeled retrieval eval set from the actual corpus.** Sample 50–150 realistic queries (fiqh rulings, hadith lookups, tafsir questions, isnad/biographical questions) and manually identify the correct passage(s) — book, page, chunk. It needs to be representative, including edge cases (rare narrator names, ambiguous fiqh terms, verses referenced by meaning).
2. **Measure Recall@k and NDCG@k**, not impressions, at the k your generation stage will actually use.
3. **Test a shortlist, not one model** — at minimum BGE-M3 and multilingual-e5-large (Section 2.3's evidence) against a commercial API and, if time allows, an Arabic-specific model, all on the same eval set and chunking.
4. **Test hybrid vs. dense-only** on the same set (BM25-only, dense-only, RRF-fused) to verify hybrid earns its complexity on this corpus rather than assuming it.
5. **Re-run whenever chunking strategy changes** — chunk boundaries change what a "correct" retrieval target looks like, so results don't automatically transfer across chunking schemes (see the chunking-strategy document).
6. **Keep the eval set alive** — fold real production failure cases back into it over time.

This is a days-not-weeks investment relative to the cost of discovering post-launch that the chosen model silently underperforms on the classical-register content that makes up most of the corpus.

## 10. Further Reading

- **MTEB leaderboard** (primary reference for embedding model comparison): https://huggingface.co/spaces/mteb/leaderboard
- **Swan / ArabicMTEB** — dialect-aware, Arabic-centric embedding models and benchmark: Bhatia et al., 2024, [arXiv:2411.01192](https://arxiv.org/abs/2411.01192)
- **Optimizing RAG Pipelines for Arabic** — systematic study finding multilingual models (BGE-M3, multilingual-e5-large) outperform Arabic-specialized models on Arabic RAG tasks, including Qur'an Tafseer retrieval: [arXiv:2506.06339](https://arxiv.org/abs/2506.06339)
- **GATE: General Arabic Text Embedding** — Matryoshka + hybrid loss training for Arabic STS: [arXiv:2505.24581](https://arxiv.org/abs/2505.24581)
- **BGE-M3 paper** — multi-lingual, multi-functionality, multi-granularity embeddings: [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- **Matryoshka Representation Learning** — Kusupati et al., 2022: [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)
- **ColBERT** — Khattab & Zaharia, 2020: [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)
- **ColBERTv2** — Santhanam et al., 2022: [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)
- **ColPali** — Faysse et al., 2024, vision-language document retrieval: [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)
- **SPLADE** — Formal et al., 2021: [arXiv:2107.05720](https://arxiv.org/abs/2107.05720); **SPLADE v2**: [arXiv:2109.10086](https://arxiv.org/abs/2109.10086)
- **Reciprocal Rank Fusion** — Cormack, Clarke & Büttcher, 2009, CIKM: https://dl.acm.org/doi/10.1145/1571941.1572114 (accessible overview: [OpenSearch blog](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/))
- **ARBERT & MARBERT** — Arabic-specific BERT models: https://www.researchgate.net/publication/353490320_ARBERT_MARBERT_Deep_Bidirectional_Transformers_for_Arabic
- **Jina Embeddings v3** — multilingual embeddings with task LoRA: [arXiv:2409.10173](https://arxiv.org/abs/2409.10173)
- **Qwen3-Embedding** — open-weight embedding/reranker family: https://github.com/QwenLM/Qwen3-Embedding
- Vector database docs: [Milvus](https://milvus.io/docs), [Qdrant](https://qdrant.tech/documentation/), [Weaviate](https://weaviate.io/developers/weaviate), [pgvector](https://github.com/pgvector/pgvector), [OpenSearch k-NN](https://opensearch.org/docs/latest/search-plugins/knn/index/), [Pinecone hybrid search](https://docs.pinecone.io/guides/search/hybrid-search)
