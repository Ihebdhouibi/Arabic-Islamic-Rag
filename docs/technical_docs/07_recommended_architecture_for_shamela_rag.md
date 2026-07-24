# 07 — Recommended Architecture for the Shamela RAG System

> Final document in a 7-part series on building a production RAG system over the Shamela digital
> library. Documents 01–06 covered the general theory (RAG fundamentals, chunking, embeddings,
> retrieval strategies, knowledge graphs/GraphRAG, evaluation). This document is the synthesis:
> it applies those concepts specifically to the corpus we've actually investigated, ties back to
> the architecture direction already agreed on, and lists the open decisions still ahead of
> implementation. **This is still a design document, not a build** — nothing here has been
> implemented yet.

## 1. What we're actually building on

This isn't a generic "chunk some PDFs" RAG project. The corpus at `d:\Shamela4_Full_DB` is a
fully parsed Shamela4 extraction:

- **8,589 books, 7.6M pages, 4.06M TOC entries**, across 41 subject categories (aqidah, hadith,
  tafsir, fiqh by madhhab, seerah, history, language, poetry, etc.), roughly 31 GB on disk.
- Per book: page-level text (`pages.jsonl`, with `body` and, for some books, a parallel
  `footnotes` stream carrying commentary distinct from the base text), a **hierarchical table of
  contents** mapped to exact page IDs (`toc.jsonl`), and rich metadata (`book_metadata.json`:
  author, author's death year in *hijri*, category, book type, multi-volume flag).
- A `_meta/` folder holding a **pre-curated relational knowledge graph**: ~19,000 narrators with
  biography and jarh-wa-ta'dil (narrator criticism) text, ~35,000 isnad (chain-of-narrators)
  links, a hadith-concordance table linking the same hadith across different books, a
  verse-to-tafsir-commentary link table, a canonical 6,236-verse Quran text table, and a
  1.95-million-entry Arabic root dictionary (inflected form → triliteral root).

That last point is the single most important fact shaping this architecture. See
[05_knowledge_graphs_and_graphrag.md](05_knowledge_graphs_and_graphrag.md) §5: most GraphRAG
projects spend most of their cost and error budget extracting a graph from raw text with an LLM.
This corpus's graph — narrators, isnad chains, hadith concordance, verse links — already exists,
curated by generations of hadith scholars, at essentially zero extraction cost and zero
hallucination risk. The design below is built around *using* that head start, not rebuilding it.

## 2. Confirmed direction

Two decisions were made explicitly before this design phase, and everything below follows from
them:

- **The graph layer is core infrastructure, not optional metadata** (Option C from the earlier
  brainstorm — see [05_knowledge_graphs_and_graphrag.md](05_knowledge_graphs_and_graphrag.md) §8
  for why this beats "vector-only" or "graph as filter metadata" for this corpus specifically).
- **Four use cases are in scope from the start, not one narrow one:** general Q&A across the
  whole library, hadith research/takhrij (tracing a hadith across collections with isnad/narrator
  grading), tafsir lookup by verse (comparing multiple mufassirs on one ayah), and fiqh ruling
  lookup (rulings + evidence, distinguished by madhhab).

Because these four use cases have genuinely different retrieval shapes — open-ended semantic
search, graph traversal, exact structured lookup, and filtered hybrid search, respectively — a
single flat "embed everything, retrieve top-*k*" pipeline (naive RAG, per
[01_rag_fundamentals.md](01_rag_fundamentals.md) §3) would underserve three of the four. The
architecture is a **query-routed, hybrid system** from day one.

## 3. Storage layer: three coordinated stores

| Store | Holds | Built from | Primary technique(s) |
|---|---|---|---|
| Vector index | Genre-aware chunks + embeddings + metadata | `pages.jsonl` + `toc.jsonl`, chunked per §4 | Dense embeddings ([03](03_embeddings_and_vector_stores.md)), evaluated for Arabic specifically |
| Lexical index | Same chunks, root-normalized text | `pages.jsonl` + `_meta/root_dictionary.jsonl` | BM25 / hybrid fusion ([03](03_embeddings_and_vector_stores.md) §5) |
| Relational graph tables | Narrators, isnad edges, hadith concordance, verse↔tafsir links | `_meta/*.jsonl` (already curated — load, don't extract) | Recursive queries / graph traversal ([05](05_knowledge_graphs_and_graphrag.md) §6) |

None of these is optional at this scope: the vector index handles open-ended semantic questions,
the lexical/root-normalized index compensates for the fact that classical Arabic's rich
morphology defeats naive keyword or even dense-embedding matching on inflected forms (see
[03_embeddings_and_vector_stores.md](03_embeddings_and_vector_stores.md) §5 on hybrid retrieval),
and the graph tables are what make takhrij and verse-concordance queries exact instead of
approximate.

On the graph store specifically: per
[05_knowledge_graphs_and_graphrag.md](05_knowledge_graphs_and_graphrag.md) §6, the actual scale
here (tens of thousands of narrators and isnad edges) does not require a dedicated graph
database. Well-indexed relational tables with recursive queries (e.g. Postgres `WITH RECURSIVE`
over a `narrator_id → narrated_from_id` edge table) are a reasonable starting point; a dedicated
graph engine (Neo4j, etc.) is worth revisiting only if traversal complexity or scale outgrows
that — not assumed up front.

## 4. Chunking, mapped per genre

Following [02_chunking_strategies.md](02_chunking_strategies.md) §12, chunking should be routed
by `category_id` / `book_type_label` at ingestion time rather than applying one strategy across
all 8,589 books:

| Genre (category) | Natural atomic unit | Chunking approach | Source data used |
|---|---|---|---|
| Hadith collections (كتب السنة, شروح الحديث) | One hadith (isnad + matn) | Hadith-atomic chunking | `page_isnads`, `hadith_xrefs.key_id` |
| Tafsir (التفسير) | One verse or verse-group's commentary | Verse-anchored chunking | `tafsir_xrefs`, `quran_verses` |
| Fiqh (الفقه بمذاهبه) | One مسألة (ruling + evidence) | TOC-anchored, sub-split on masa'il markers | `toc.jsonl` hierarchy |
| Aqidah / general prose with commentary | Matn vs. sharh as parallel streams | Dual-stream chunking, linked | `pages.jsonl` (`body` vs `footnotes`) |
| Poetry (الشعر ودواوينه) | One bayt / stanza | Never split mid-verse | `pages.jsonl` structure |
| Biography (التراجم والطبقات، الأنساب) | One person's entry | TOC-anchored (one heading per name) | `toc.jsonl` |
| Everything else / general Q&A fallback | TOC section | Hierarchical/TOC-anchored, per [02](02_chunking_strategies.md) §4 | `toc.jsonl` |

Two techniques from [02_chunking_strategies.md](02_chunking_strategies.md) are worth prioritizing
early rather than treating as later optimizations, specifically *because* of this corpus's
structure: **Contextual Retrieval** (§7 — prepending brief LLM-generated context to a chunk
before embedding) matters more than usual here, since a fiqh chunk or hadith commentary pulled
out of its surrounding `باب`/`فصل` context frequently loses the referent of a pronoun or the
identity of "the ruling under discussion"; and **matn/sharh dual-stream chunking** is close to
free to implement well, since the extraction already separates `body` from `footnotes` for books
that have that structure — collapsing them into one chunk would throw away a distinction the
source data already made for us.

## 5. Query-routed retrieval, one path per use case

This is where the four confirmed use cases (§2) map to concrete retrieval paths. A router
(rule-based heuristics first — e.g. presence of a verse reference like `٢:٢٥٥`, isnad markers
like "حدثنا"/"أخبرنا", or madhhab/fiqh terminology — with an LLM classifier as a fallback for
ambiguous cases) decides which path a question takes, per
[04_retrieval_and_query_strategies.md](04_retrieval_and_query_strategies.md) §1's discussion of
query routing.

**Hadith takhrij.** Resolve the query to a `key_id` (lexical/dense match against
`hadith_xrefs`-linked text); pull every `(book_id, page_id)` sharing that key; assemble each
variant's isnad from `page_isnads` + `narrators`, surfacing narrator grading (ثقة/ضعيف/etc.) from
the criticism text. This is a graph traversal, not a vector search — see
[05_knowledge_graphs_and_graphrag.md](05_knowledge_graphs_and_graphrag.md) §2 on why "trace this
chain and flag the weak link" is structurally a multi-hop graph question. Fall back to hybrid
vector+lexical search over hadith-atomic chunks only if no exact concordance match is found.

**Tafsir by verse.** Resolve the verse reference (exact citation or semantic match against
`quran_verses`) to a verse `key_id`; pull every `(book_id, page_id)` from `tafsir_xrefs` for that
verse — an exact join, not a similarity search (per
[05_knowledge_graphs_and_graphrag.md](05_knowledge_graphs_and_graphrag.md) §3 on why vector
search can't guarantee "all of them, no more, no less"). Present results as a side-by-side of
mufassirs rather than a single fused answer, since the whole value of this use case is comparison.

**Fiqh ruling lookup.** Hybrid dense+lexical retrieval (per
[04_retrieval_and_query_strategies.md](04_retrieval_and_query_strategies.md) §2) over
fiqh-category chunks, filtered/faceted by madhhab (`category_id`) and era (`death_hijri`) as
metadata filters, reranked (§6), with results grouped by madhhab so disagreement (*ikhtilaf*)
stays visible — this is a product requirement, not just a UX preference (§7 below).

**General Q&A.** The fallback path: hybrid dense+lexical retrieval over TOC-anchored chunks
across the whole corpus, reranked with a cross-encoder (per
[04_retrieval_and_query_strategies.md](04_retrieval_and_query_strategies.md) §3), with
authoritativeness signals (printed `book_type`, well-established authors) used as a ranking
boost over transcript-style content (`دروس مفرغة`) rather than treating every source as equally
weighted.

**Graph-augmented enrichment across all four paths.** Regardless of which path handled the
initial retrieval, per
[05_knowledge_graphs_and_graphrag.md](05_knowledge_graphs_and_graphrag.md) §7's "neighborhood
expansion" pattern: once a hadith passage is retrieved, automatically pull in the biography and
jarh-wa-ta'dil verdict for every narrator in its isnad before generation — context that would
never surface from vector search alone, because narrator-biography text and hadith-passage text
don't necessarily embed as "similar," even though they're tightly related.

## 6. Generation and grounding — a domain-specific requirement, not a nice-to-have

[01_rag_fundamentals.md](01_rag_fundamentals.md) §6 makes the general point that citation
grounding matters more in high-stakes domains. For this system specifically, that translates
into concrete product requirements, not just a technical preference:

- **Every answer must cite** book title, author, and page/hadith reference — never a synthesized
  claim without a traceable source, given that a wrong or uncited fiqh/hadith answer is a
  trust-and-authority failure, not merely a quality one.
- **Disagreement must be preserved, not flattened.** When madhhabs differ on a ruling, or when
  mufassirs differ on a verse's interpretation, the system should present the range of views with
  attribution rather than averaging them into one synthesized "answer" — this directly shapes the
  retrieval design in §5 (grouped-by-madhhab fiqh results, side-by-side tafsir comparison) and
  should carry through to the generation prompt design.
- **The system should not present itself as an independent authority** (e.g., issuing novel
  fatwas) — its role is retrieval and faithful presentation of existing scholarly sources, which
  is precisely what RAG's citation-grounded nature is suited for and what a purely generative
  approach is not (see [01_rag_fundamentals.md](01_rag_fundamentals.md) §1 on RAG's core
  value proposition).

## 7. Evaluation — build this before scaling, not after

Per [06_evaluation_and_recent_advancements.md](06_evaluation_and_recent_advancements.md) §5, a
golden evaluation set matters especially here because general-domain and English-centric
benchmarks (including most embedding-model leaderboards) don't transfer to classical, religious
register Arabic. Before scaling ingestion across all 8,589 books, the practical sequence is:

1. Hand-curate a modest set (50–100 examples) of representative question → expected source(s)
   pairs, spanning all four use cases — including some genuinely hard ones (a hadith with several
   book variants, a verse with well-known tafsir disagreement, a fiqh question with a
   cross-madhhab split).
2. Use retrieval metrics (precision@k, recall@k) and RAG-specific metrics (faithfulness, context
   relevance — via a framework like RAGAS, per
   [06_evaluation_and_recent_advancements.md](06_evaluation_and_recent_advancements.md) §4) against
   that set before committing to an embedding model or chunking strategy at full scale.
3. Involve subject-matter/scholarly review specifically for the fiqh and hadith-grading outputs —
   automated metrics can catch "did it retrieve something plausible" but not "did it correctly
   represent the scholarly nuance," which is the actual bar for this domain.

## 8. Open decisions before implementation begins

These are genuine open questions, not settled by the research above — worth resolving explicitly
before writing ingestion code:

1. **Graph store choice** — start with indexed Postgres/recursive-CTE tables (per
   [05_knowledge_graphs_and_graphrag.md](05_knowledge_graphs_and_graphrag.md) §6), or invest in a
   dedicated graph database from the outset. The recommendation above is to start relational and
   revisit only if traversal needs outgrow it — but that's a default, not a mandate.
2. **Embedding model selection** — needs an actual benchmark run against the golden set (§7)
   before committing, per [03_embeddings_and_vector_stores.md](03_embeddings_and_vector_stores.md)
   §9's "evaluate, don't assume" guidance — general multilingual embedding leaderboard rank does
   not necessarily predict performance on classical/religious-register Arabic.
3. **Query router implementation** — rule-based heuristics vs. a small LLM classifier vs. a hybrid
   of both, and how to handle genuinely ambiguous queries that plausibly span more than one use
   case (e.g. "what's the ruling on X, and is the hadith it's based on authentic?" spans fiqh *and*
   hadith takhrij).
4. **Chunk size and secondary-splitting policy** for TOC sections that are too large (§4) — what
   the overlap/context-window budget looks like once contextual retrieval prefixes are added.
5. **Ingestion scale and cost sequencing** — the corpus is ~31 GB / 7.6M pages before chunking
   multiplies that further; whether to pilot the full pipeline on one or two categories (e.g.
   hadith and tafsir, since those have the richest existing structured data) before scaling to all
   41 categories.

None of these need to be resolved today — they're the natural next design conversations, and each
maps to a section of the six preceding documents for deeper context when you're ready to dig in.
