# 04 — Retrieval and Query Strategies

> Scope: this document covers what happens **between** "a user asks a question" and "a set of text chunks is handed to the LLM." It assumes chunks already exist and are already embedded (see the chunking and embeddings docs) — it explains *where* rerankers and embeddings sit in the pipeline and *how to orchestrate* retrieval around them, not how they work internally. Knowledge-graph-based retrieval is out of scope — see the dedicated GraphRAG document; it's only mentioned here as a pointer.

## Why this stage matters

A naive RAG pipeline embeds the user's raw question, runs cosine-similarity search against a vector index, and stuffs the top-k chunks into a prompt. That works for simple, well-phrased factual questions over a homogeneous corpus. It breaks down on a corpus like Shamela — 8,589 classical Arabic books spanning hadith, tafsir, fiqh across four madhhabs, aqidah, and biographical dictionaries — because:

- A contemporary-phrased Arabic query often doesn't lexically match classical morphology and madhhab-specific terminology.
- Many real questions are **comparative or multi-hop** ("what does the Hanafi school say versus the Shafi'i school?"), which one top-k pass cannot answer.
- The corpus has rich **structured metadata** (author, death year, category, madhhab) that pure vector search ignores but which is often the fastest, most precise filter available.
- Getting a ruling wrong because the right chunk was buried in position 47 of 50 is a much higher-stakes failure than in a generic chatbot.

The techniques below are levers for closing these gaps — none is mandatory everywhere. §7 gives guidance on combining them.

---

## 1. Pre-retrieval query transformation

### 1.1 Query rewriting / expansion

Pass the user's question through an LLM to fix ambiguity, expand abbreviations, and normalize terminology — e.g. mapping a colloquial phrasing to the classical term the corpus actually uses. For Arabic, this is also the natural place to normalize orthography (hamza forms, ta marbuta/ha, alif variants) and optionally inject diacritized forms of key terms if the index expects them.

### 1.2 HyDE (Hypothetical Document Embeddings)

Instead of embedding the question, ask an LLM to first generate a **hypothetical answer** (factual errors are fine — it's never shown to the user) and embed that instead. A hypothetical answer written in the target corpus's register sits closer in embedding space to real answer passages than a short question does — useful here because questions are typically contemporary in style while answers live in centuries-old prose. Cost: one extra LLM call before retrieval starts; risk: a confidently wrong hypothetical (wrong madhhab, wrong narrator) can pull retrieval in the wrong direction.

Paper: Gao, Ma, Lin, Callan, *"Precise Zero-Shot Dense Retrieval without Relevance Labels"* (2022) — https://arxiv.org/abs/2212.10496

### 1.3 Multi-query retrieval and RAG-Fusion

Generate several paraphrased variants of the question, retrieve independently for each, and fuse the result lists — typically with **Reciprocal Rank Fusion (RRF)**: score each document by `Σ 1/(k + rank_i)` across the lists it appears in (k ≈ 60), rewarding documents that show up consistently across variants without needing comparable raw similarity scores. "RAG-Fusion" is the popularized name for multi-query generation + RRF, and it's a good fit for comparative fiqh questions, where variants foregrounding different madhhab-specific terms surface passages a single phrasing would miss.

Sources: RRF — Cormack, Clarke, Büttcher, *"Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods"* (SIGIR 2009) — https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/ ; RAG-Fusion — Rackauckas (2024) — https://arxiv.org/abs/2402.03367

### 1.4 Step-back prompting

Have the LLM first generate a more abstract "step-back" question and retrieve for that alongside the original. Example: a narrow question about combining prayers on a specific flight route might retrieve nothing verbatim, but the step-back question ("what are the general conditions for combining prayers while traveling?") retrieves the governing principle — which, together with the specific case, grounds a correct answer.

Paper: Zheng et al. (Google DeepMind), *"Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"* (2023) — https://arxiv.org/abs/2310.06117

### 1.5 Query routing / intent classification

Classify the query before retrieval to decide which index, filter, or strategy to use. This is where a heterogeneous corpus pays off most: a router can distinguish a **simple factual lookup** (narrow dense retrieval, maybe scoped to a named book), a **multi-hop comparative query** (needs per-madhhab retrieval plus synthesis, §4.3), a **reference-tracing query** ("every place this hadith is cited" — a job for exact/fuzzy matching and metadata joins, not semantic similarity), or an **out-of-domain query** (skip retrieval entirely). Routing can be a small classifier, a regex layer for obvious cases, or an LLM function call; it decides *which* of the other techniques in this document apply.

---

## 2. Retrieval mechanics

**Dense vector similarity (top-k).** The baseline: embed the (possibly transformed) query, retrieve the k nearest chunk embeddings by cosine/dot-product similarity. Everything else here modifies or replaces this step. Small k (e.g. 5) minimizes noise; larger k (e.g. 50) is used when a downstream reranker will cut it back down.

**Hybrid search (dense + sparse).** Dense embeddings capture semantic similarity but can miss exact matches — proper names, narrator chains (*isnad*), exact Qur'anic wording, rare technical terms. A sparse retriever (BM25 or a learned sparse model) is complementary, and the two ranked lists are fused, usually via RRF (§1.3) or weighted score combination. (Arabic tokenization/stemming choices for BM25 are covered in the embeddings doc — the point here is only that hybrid search is usually a strict win over dense-only for a corpus this dense with proper nouns and quotations, at the cost of maintaining two indexes.)

**Metadata filtering / faceted retrieval.** Shamela's per-book metadata (author, death year, category, madhhab, edition) is a nearly-free retrieval lever, often more reliable than semantic search for narrowing scope. **Pre-filtering** — restricting the vector search to a metadata-matching subset before running ANN search — is almost always preferable to **post-filtering** (retrieve broadly, discard non-matches afterward), which wastes retrieval budget and can under-fill k. Pre-filtering is the natural integration point for query routing's output (e.g. "search only Ibn Taymiyyah's works") and for reference-tracing queries (filter to books citing a given narrator, then rank semantically).

**Parent-document / small-to-big retrieval.** Small chunks embed precisely but lack context; large chunks preserve context but dilute the embedding. Small-to-big retrieval decouples the retrieval unit from the generation unit: index small chunks, but return a larger surrounding unit at answer time.
- **Parent-document retriever** (LangChain pattern): embed small child chunks, store a pointer to a larger parent chunk, substitute the parent in at retrieval time.
- **Sentence-window retrieval** (LlamaIndex): index single sentences with a metadata "window" of N neighbors; return the window instead of the bare sentence.
- **Auto-merging retrieval** (LlamaIndex): build an explicit paragraph→section→chapter hierarchy; if enough sibling leaf nodes are retrieved, merge them up into their shared parent.

This maps naturally onto Shamela's structure — a hadith plus its surrounding commentary, or a ruling plus its reasoning paragraph — where the natural "hit" unit is smaller than what a reader needs in context.

References: LangChain, *"How to use the Parent Document Retriever"* — https://python.langchain.com/v0.2/docs/how_to/parent_document_retriever/ ; LlamaIndex, *"Auto Merging Retriever"* — https://docs.llamaindex.ai/en/stable/api_reference/packs/auto_merging_retriever/ ; LlamaIndex, *"Sentence window retriever"* — https://docs.llamaindex.ai/en/stable/api_reference/packs/sentence_window_retriever/ (RAPTOR, §4.4, extends multi-granularity retrieval further via a full summarization hierarchy.)

---

## 3. Post-retrieval optimization

First-stage retrieval optimizes for **recall** — casting a wide enough net. Post-retrieval steps optimize for **precision and usability** of what actually reaches the LLM.

**Reranking with cross-encoders.** A bi-encoder first-stage retriever scores query and document independently, which is fast but structurally less accurate than scoring them jointly. A cross-encoder reranker takes query+document together and is far more accurate but too slow to run over a whole index — so the standard pipeline retrieves a wide candidate set (commonly top-30 to top-100) and reranks it down to a narrow final set (commonly top-3 to top-10) actually sent to the LLM. Reranking ~50 candidates typically adds tens to low hundreds of milliseconds — usually worth it, since a bad ordering in the top slots has an outsized effect downstream (see below). Model choices and mechanics are covered in the embeddings doc; the point here is just the pipeline shape: cheap-and-wide, then expensive-and-narrow.

**Lost-in-the-middle mitigation.** Liu et al. showed LLMs use long contexts non-uniformly: performance is highest when relevant information sits at the very start or end of the context and degrades — sometimes sharply — when it's buried in the middle, even for long-context-specialized models. Practical fix: after reranking, reorder the final chunk list so the highest-relevance chunks sit at the start and end of the prompt, not in the middle. This costs nothing beyond a reordering step and is one of the highest-leverage, lowest-cost fixes in this whole pipeline.

Paper: Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang, *"Lost in the Middle: How Language Models Use Long Contexts"* (2023) — https://arxiv.org/abs/2307.03172

**Contextual compression / extractive filtering.** Even a reranked chunk may mix relevant and irrelevant sentences — classical prose often digresses. An extra pass (an LLM extracting only query-relevant sentences, or a lighter sentence-level similarity filter) trims each chunk before it enters the final prompt, saving context budget and shrinking what sits in the "middle." Cost is an extra call per surviving chunk, so it's typically applied only after reranking, not to the wide first-stage set.

---

## 4. Iterative and agentic retrieval

The techniques above are essentially single-pass. Questions requiring genuine multi-round reasoning — the multi-hop comparative fiqh case is the canonical example — need more.

**Self-RAG** trains/prompts a model to decide for itself when retrieval is needed, retrieve on demand (possibly multiple times mid-generation), and emit reflection tokens critiquing the relevance of retrieved passages and its own output — a departure from always-retrieve-once pipelines.
Paper: Asai, Wu, Wang, Sil, Hajishirzi, *"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"* (2023) — https://arxiv.org/abs/2310.11511

**Corrective RAG (CRAG)** adds a lightweight evaluator that grades retrieval quality (correct/ambiguous/incorrect) before generation, triggering a corrective action if it's poor — a web-search fallback in the original paper, or in a closed-corpus setting: broadening the metadata filter, switching retrieval strategy, or telling the user no confident source was found rather than generating an unsupported answer (which matters more here than in a generic assistant).
Paper: Yan, Gu, Zhu, Ling, *"Corrective Retrieval Augmented Generation"* (2024) — https://arxiv.org/abs/2401.15884

**Multi-hop / iterative retrieval and ReAct.** Multi-hop questions need retrieve → reason about what's missing → retrieve again, repeated until enough evidence is gathered. **ReAct** interleaves reasoning ("Thought"), retrieval calls ("Action"), and results ("Observation") in a loop. Applied to a comparative fiqh question: retrieve the Hanafi position, observe it, reason the Shafi'i position is still needed, retrieve again scoped to that madhhab, repeat, then synthesize. This costs multiple sequential LLM+retrieval round trips, so it's best reserved for query types flagged as multi-hop by routing (§1.5).
Paper: Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao, *"ReAct: Synergizing Reasoning and Acting in Language Models"* (ICLR 2023) — https://arxiv.org/abs/2210.03629

**RAPTOR** addresses holistic, whole-document questions that chunk-level retrieval can't answer well. It recursively embeds, clusters, and summarizes chunks bottom-up into a tree with multiple abstraction levels (leaf chunks → paragraph → chapter → book summaries); retrieval can then pull from whichever level matches the question's granularity. It's a heavier, offline-indexing-cost complement to parent-document retrieval (§2), best justified when whole-book synthesis questions are a meaningful share of usage.
Paper: Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning, *"RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"* (2024) — https://arxiv.org/abs/2401.18059

*(Graph-augmented retrieval — an explicit entity/relation graph over narrators, books, and scholars, traversed instead of embedding-searched — is another way to handle multi-hop and reference-tracing questions, covered in depth in the dedicated GraphRAG document and not duplicated here.)*

---

## 5. Citation and grounding at generation time

Retrieving the right chunks is necessary but not sufficient — the LLM must also make clear which claim is supported by which source. Common patterns: **inline citation markers** (tagging each claim, e.g. `[Sahih Muslim, hadith #123]`); **structured output with attribution** (the model returns claim + supporting chunk ID rather than free text, letting the application render citations deterministically and even validate them post hoc); and **refuse-if-unsupported** instructions, telling the model to hedge or decline rather than fill gaps from parametric knowledge — the generation-time complement of CRAG's retrieval-quality check.

This matters more in high-stakes, authoritative domains — religious, legal, and medical text are the standard examples — because an unattributed claim is a correctness failure, not just a UX gap. For a classical Islamic text corpus, precise citation down to the book and ideally the edition/page is close to a hard requirement, since scholarly claims are conventionally expected to be traceable. Citation UX and validation design are covered in the synthesis/generation doc; the point here is that the retrieval pipeline must preserve chunk-level provenance (book, author, page/hadith number) end-to-end for citation to even be possible.

---

## 6. Comparison table

| Technique | Problem it solves | Added latency/cost | Complexity | When to use |
|---|---|---|---|---|
| Query rewriting/expansion | Vocabulary/register mismatch | +1 LLM call | Low | Almost always |
| HyDE | Short question vs. long-form answer style mismatch | +1 LLM call | Low–Med | Terse/colloquial queries vs. formal corpus prose |
| Multi-query / RAG-Fusion | Single phrasing misses terminology variants | +N LLM calls, +N retrievals, fusion | Medium | Ambiguous or multi-faceted questions |
| Step-back prompting | Narrow query retrieves nothing; needs general principle | +1 LLM call, +1 retrieval | Medium | Specific cases governed by general rules (fiqh) |
| Query routing | One-size-fits-all fails on heterogeneous corpora | Small | Medium | Corpus with distinct genres/question types |
| Dense vector search | Baseline semantic retrieval | Baseline | Low | Always |
| Hybrid (dense+sparse) | Misses exact-term/proper-noun matches | +1 index + fusion | Medium | Names, quotations, technical terms |
| Metadata filtering | Ignores structured signal | ~Free if pre-filtered | Low–Med | Rich metadata, inferable scope |
| Parent-document / small-to-big | Precision vs. context tradeoff | Small (extra lookup) | Medium | Most production systems |
| Cross-encoder reranking | First-stage ranking is approximate | Tens–100s ms for ~50 candidates | Low–Med | Nearly always |
| Lost-in-the-middle reordering | Buried mid-context relevance ignored | Negligible | Low | Whenever passing >2–3 chunks |
| Contextual compression | Wastes context budget on irrelevant text | +1 pass/chunk | Medium | Long chunks, tight context, high source count |
| Self-RAG | Fixed always-retrieve is wasteful/wrong sometimes | Special training/prompting | High | Adaptive retrieval, model control available |
| CRAG | No fallback when retrieval is poor | +1 evaluator + fallback | Med–High | High-stakes domains |
| Multi-hop / ReAct | Single pass can't synthesize across sources | Multiple sequential round trips | High | Cross-madhhab comparison |
| RAPTOR | No multi-granularity retrieval (paragraph vs. book) | Heavy offline indexing | High | Holistic/summary-level Q&A |
| Graph-augmented retrieval | Relational multi-hop (citation/narrator chains) | See GraphRAG doc | See GraphRAG doc | See GraphRAG doc |

---

## 7. Decision framework by question type

**Simple factual lookup** ("what does book X say about Y", "define term Z"): query rewriting + hybrid search + metadata pre-filter (if book/author named) + cross-encoder reranking (top-30 → top-5) + lost-in-the-middle reordering. Skip multi-query fusion, step-back, and multi-hop loops — they add latency without benefit here.

**Multi-hop / comparative** ("compare Hanafi and Shafi'i positions on X", "how did consensus on Y evolve"): route to detect the pattern → decompose into sub-queries (per madhhab or era) → retrieve per sub-query (multi-query/RAG-Fusion, or an explicit ReAct-style loop if decomposition isn't obvious upfront) → rerank each subset → synthesize with per-claim citation so the reader sees which source backs which position. Consider RAPTOR-level summaries when the comparison is at the whole-book/school level rather than a specific ruling.

**"Trace this reference across sources"** (e.g. every quotation/grading of a hadith, every book citing a scholar): this is a precision/completeness task, not a similarity task — favor exact/fuzzy text matching and metadata joins over pure dense retrieval, which isn't built to guarantee exhaustive recall. Semantic search can supplement by surfacing paraphrased references, but shouldn't be primary. Graph-augmented retrieval (citation graphs, narrator chains — see GraphRAG doc) is the natural long-term fit. A CRAG-style check is valuable here too: if exact-match/graph retrieval returns nothing, surface that rather than silently falling back to a weak semantic guess.

**General guidance**: start with rewriting + hybrid + metadata filtering + reranking + lost-in-the-middle reordering — low-cost, broadly applicable, and it resolves most failure modes of a naive top-k pipeline. Add query routing once more than one question type shows up in real usage. Reserve multi-hop/agentic loops, Self-RAG-style adaptive retrieval, and RAPTOR for the specific patterns that demonstrably need them — they're the most expensive techniques here, both to build and to run.

---

## Further reading

- HyDE — Gao, Ma, Lin, Callan, *"Precise Zero-Shot Dense Retrieval without Relevance Labels"* (2022): https://arxiv.org/abs/2212.10496
- RAG-Fusion — Rackauckas, *"RAG-Fusion: a New Take on Retrieval-Augmented Generation"* (2024): https://arxiv.org/abs/2402.03367
- Reciprocal Rank Fusion (original) — Cormack, Clarke, Büttcher, *"Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods"* (SIGIR 2009): https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/
- Step-back prompting — Zheng et al., *"Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"* (2023): https://arxiv.org/abs/2310.06117
- Lost in the Middle — Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang, *"Lost in the Middle: How Language Models Use Long Contexts"* (2023): https://arxiv.org/abs/2307.03172
- Self-RAG — Asai, Wu, Wang, Sil, Hajishirzi, *"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"* (2023): https://arxiv.org/abs/2310.11511
- CRAG — Yan, Gu, Zhu, Ling, *"Corrective Retrieval Augmented Generation"* (2024): https://arxiv.org/abs/2401.15884
- ReAct — Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao, *"ReAct: Synergizing Reasoning and Acting in Language Models"* (ICLR 2023): https://arxiv.org/abs/2210.03629
- RAPTOR — Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning, *"RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"* (2024): https://arxiv.org/abs/2401.18059
- LangChain, *"How to use the Parent Document Retriever"*: https://python.langchain.com/v0.2/docs/how_to/parent_document_retriever/
- LlamaIndex, *"Auto Merging Retriever"*: https://docs.llamaindex.ai/en/stable/api_reference/packs/auto_merging_retriever/
- LlamaIndex, *"Sentence window retriever"*: https://docs.llamaindex.ai/en/stable/api_reference/packs/sentence_window_retriever/
