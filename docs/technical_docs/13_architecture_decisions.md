# 13 — Architecture Decisions

> Resolves the five open items listed in
> [07_recommended_architecture_for_shamela_rag.md §8](07_recommended_architecture_for_shamela_rag.md#8-open-decisions-before-implementation-begins),
> using what docs 08–12 learned since that list was written — most importantly, doc 10's finding
> that curated graph coverage is 19 books (tens of thousands of rows), not corpus-wide, which
> changes the scale assumption behind several of these. Written as short ADRs (Architecture
> Decision Records) — each has a status, since "decided now" doesn't mean "frozen forever." These
> are still design-phase decisions: nothing has been implemented against them yet.

## ADR-001: Graph store — PostgreSQL with recursive CTEs, not a dedicated graph database

**Status:** Decided for the pilot phase.

**Context:** [05_knowledge_graphs_and_graphrag.md §6](05_knowledge_graphs_and_graphrag.md#6-graph-databases-and-when-you-dont-need-one)
argued a dedicated graph database (Neo4j, etc.) earns its cost at millions-to-billions of
nodes/edges, and that well-indexed relational tables with recursive CTEs are sufficient below
that. Doc 10 confirmed the actual scale is smaller than doc 07 assumed: ~35,526 isnad rows across
exactly 10 books, ~37K `hadith_xrefs` rows, ~84K `tafsir_xrefs` rows, 18,989 narrators — tens of
thousands, not millions.

**Decision:** Use PostgreSQL with self-referencing foreign-key tables and `WITH RECURSIVE` queries
for isnad traversal. No Neo4j or other dedicated graph database for the pilot.

**Further consolidation:** go one step past doc 07's framing — use the **same** Postgres instance,
with the `pgvector` extension, as the vector store too, rather than standing up a separate
dedicated vector database from day one. At pilot scale (a low-millions chunk count at most, likely
far less once only the 19-book curated core plus a general sample are ingested per ADR-005),
`pgvector`'s HNSW index is adequate, and one database technology for metadata + graph + vectors is
meaningfully less operational surface than three. Lexical/BM25 search still needs its own engine —
Postgres full-text search (`tsvector`) is adequate for the pilot; a dedicated engine (OpenSearch)
is the upgrade path if lexical recall proves insufficient, not a day-one requirement.

**Consequences:** Simpler ops (one database to run, back up, and reason about) at the cost of
ceiling — `pgvector` and hand-written recursive SQL both have real scale limits.

**Revisit when:** extraction coverage is meaningfully extended (per
[11_extending_extraction_coverage.md](11_extending_extraction_coverage.md), pushing isnad edges
from 10 books toward the other 1,232), chunk count grows enough that `pgvector` ANN latency
becomes a measured problem (not a guessed one), or graph traversal patterns get complex enough
that hand-written recursive SQL becomes the maintenance bottleneck.

## ADR-002: Embedding model — no model chosen yet; the evaluation protocol is the decision

**Status:** Protocol decided; model selection deferred to measurement, on purpose.

**Context:** [03_embeddings_and_vector_stores.md §9](03_embeddings_and_vector_stores.md#9-practical-guidance-evaluate-dont-assume)
and [09_open_source_arabic_llms.md §2](09_open_source_arabic_llms.md#2-why-this-matters-general-arabic-llms-measurably-struggle-on-islamic-content-specifically)
both make the same point from different angles: general-purpose leaderboard rank (MTEB, HELM
Arabic) does not reliably predict performance on classical-register, diacritically-inconsistent,
proper-noun-dense religious Arabic — this corpus's actual register is under-represented in most
benchmarks. Picking a model now without measurement would contradict guidance already committed
to in this project's own docs.

**Decision:** The resolution of this open item is a **shortlist + a measurement protocol**, not a
model name:
- Shortlist to benchmark: **BGE-M3** and **multilingual-e5-large** (per doc 03, both outperformed
  Arabic-specialized encoder models on MSA benchmarks — worth checking whether that holds on
  classical Arabic specifically), plus one commercial API model (Cohere embed-v4 multilingual or
  OpenAI text-embedding-3-large) as a cost/quality ceiling reference.
- Protocol: retrieval metrics (recall@k, NDCG — per
  [06_evaluation_and_recent_advancements.md §2](06_evaluation_and_recent_advancements.md)) against
  the golden evaluation set, run separately per genre (hadith, tafsir, fiqh, general), since a
  model could plausibly do well on modern-register fiqh prose and poorly on diacritized hadith
  matn text — an aggregate score would hide that split.

**Consequences:** No embedding work can start until the golden set exists — this ADR is downstream
of building that set, not a substitute for it.

**Revisit when:** the golden evaluation set (doc 06) exists and the benchmark actually runs — at
that point this ADR should be replaced by one that names an actual chosen model.

## ADR-003: Query router — hybrid rule-based first pass, LLM fallback, multi-label capable

**Status:** Decided for the pilot phase.

**Context:** [07 §5](07_recommended_architecture_for_shamela_rag.md#5-query-routed-retrieval-one-path-per-use-case)
left router implementation open between rules, an LLM classifier, or a hybrid. The four use cases
have fairly distinguishable surface signals: verse references (e.g. "٢:٢٥٥" or a surah name),
isnad-style narration verbs ("حدثنا"، "أخبرنا"), explicit madhhab/fiqh terminology, or none of the
above (general). [07 §8 item 3](07_recommended_architecture_for_shamela_rag.md#8-open-decisions-before-implementation-begins)
also flagged that a single question can genuinely span more than one use case — e.g. "what's the
ruling on X, and is the hadith it's based on authentic?" spans fiqh *and* hadith takhrij.

**Decision:**
1. **Deterministic pattern matching first** — cheap, fast, unit-testable, and handles the
   unambiguous majority of queries (a verse citation is unambiguous; an isnad-marker phrase is
   unambiguous).
2. **LLM classification as fallback** only for queries the rules don't confidently match — using
   the cheap router-tier model from
   [09 §5](09_open_source_arabic_llms.md#5-a-two-model-split-is-a-legitimate-pattern-here) (e.g.
   Qwen3-8B), not the larger generation-tier model.
3. **The router emits a set of use cases, not a single label** — a compound query triggers
   multiple retrieval paths (per doc 07 §5) whose results get merged before the shared
   reranking/enrichment steps, rather than forcing an arbitrary single classification that would
   silently drop half the question.

**Consequences:** More router logic than a single LLM-classifies-everything design, but far
cheaper per-query on average and fully unit-testable for the common cases; the compound-query
handling adds real complexity to result merging that a single-label router wouldn't need.

**Revisit when:** the rule-based first pass's false-negative rate (queries it should confidently
route but doesn't) is actually measured against the golden set and found too high.

## ADR-004: Chunk sizing — 512–1024 token target for prose genres, natural-unit sizing elsewhere

**Status:** Decided as a starting point; explicitly flagged for empirical retuning.

**Context:** [02_chunking_strategies.md §10](02_chunking_strategies.md#10-chunk-size-and-overlap-tradeoffs)
covers the general tradeoff (embedding context limits, retrieval precision vs. recall, generation
context budget) without committing to numbers for this corpus specifically, since that requires
the genre-routing behavior from [07 §4](07_recommended_architecture_for_shamela_rag.md#4-chunking-mapped-per-genre)
and [10 §3 Stage 3](10_ingestion_and_indexing_pipeline.md#3-pipeline-stages) as a prerequisite,
which now exist.

**Decision, per genre:**
- **TOC-anchored prose** (general Q&A fallback, fiqh masa'il, aqidah, biography): target
  512–1024 tokens per chunk. A TOC section exceeding roughly 1,500 tokens gets recursively split
  (per [02 §2](02_chunking_strategies.md#2-recursive-characterstructure-based-splitting)) with
  10–15% overlap between resulting pieces.
- **Hadith-atomic and verse-anchored chunks:** sized to the natural unit (one hadith, one verse or
  verse-group) with no artificial ceiling — these are typically already short. The rare long
  outlier (a single tafsir passage spanning an unusually large verse group) is an edge case to
  handle if and when it's actually observed, not to design around speculatively.
- **Contextual-retrieval prefix budget:** reserve ~50–100 tokens per chunk for the breadcrumb/
  context prefix from [02 §7](02_chunking_strategies.md#7-contextual-retrieval-anthropic-september-2024),
  on top of the chunk-size target above, not counted against it.

**Consequences:** These numbers are a reasonable, literature-aligned starting point, not a result
of tuning against this corpus's actual retrieval behavior.

**Revisit when:** the golden evaluation set exists and retrieval-quality metrics can actually be
measured against different chunk-size choices — this ADR should be treated as provisional until
then.

## ADR-005: Rollout sequencing — general path first for the full corpus, graph paths validated on the 19-book curated core

**Status:** Decided, refining doc 07's original "pilot on hadith and tafsir categories."

**Context:** Doc 07 §8 suggested piloting on "categories with the richest structured data."
[10 §1](10_ingestion_and_indexing_pipeline.md#1-finding-curated-graph-coverage-is-narrower-than-category-boundaries-suggest)
found that's imprecise — the richest structured data is 19 *specific books*, not two categories
(1,241 + 273 books). [10 §4](10_ingestion_and_indexing_pipeline.md#4-revised-pilot-recommendation)
already proposed the fix; this ADR makes it the recorded decision with concrete phases.

**Decision — three phases:**
1. **Phase 1 (validation):** ingest the 19 curated books (10 hadith + 9 tafsir) plus a small
   general sample (~50–100 books across other categories) through the full pipeline — general
   hybrid path *and* graph-specific paths. This is where all four use cases from doc 07 get
   end-to-end validated against real data, including the golden-set evaluation from doc 06.
2. **Phase 2:** ingest the remainder of categories 06 and 03 (~1,232 + 264 books) through the
   general hybrid-search path only — graph-specific paths correctly fall back to general search
   for these until [doc 11](11_extending_extraction_coverage.md)'s extraction-extension work (if
   pursued) closes that gap.
3. **Phase 3:** ingest the remaining ~7,075 books across the other 38 categories, general path
   only.

**Consequences:** Phase 1 is deliberately small and cheap, front-loading validation before
spending embedding/compute cost on the full 7.6M-page corpus — if the architecture needs
correcting, it's cheaper to find out at 19 books than at 8,589.

**Revisit when:** Phase 1's golden-set evaluation results come in — a poor result there should
block Phase 2/3, not just get noted and worked around.

## Summary table

| ADR | Decision | Status |
|---|---|---|
| 001 | Postgres + recursive CTEs + pgvector; no dedicated graph DB or vector DB yet | Decided for pilot |
| 002 | Benchmark BGE-M3, multilingual-e5-large, one API model against golden set — no model chosen yet | Protocol decided, selection deferred |
| 003 | Rule-based router first pass, LLM fallback, multi-label output | Decided for pilot |
| 004 | 512–1024 tokens (prose genres), natural-unit sizing (hadith/verse), +50–100 token prefix budget | Decided as starting point |
| 005 | Phase 1: 19 curated books + ~100-book sample → Phase 2: rest of categories 06/03 → Phase 3: everything else | Decided |
