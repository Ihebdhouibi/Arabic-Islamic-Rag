# 10 — Ingestion & Indexing Pipeline

> Companion to [07_recommended_architecture_for_shamela_rag.md](07_recommended_architecture_for_shamela_rag.md).
> That document describes the target state — chunked, embedded, graph-loaded. This document
> covers how raw `pages.jsonl` / `toc.jsonl` / `_meta/*` actually gets turned into that state.
> Unlike docs 01–09, most of this one is **grounded in direct inspection of the actual data**
> (via a local DuckDB session against the real parquet/JSONL files) rather than the published
> schema alone — that inspection surfaced two findings (§1–2) that materially change the pipeline
> and the scoping of doc 07's use cases, ahead of the pipeline stages themselves (§3).

## 1. Finding: curated graph coverage is narrower than category boundaries suggest

Doc 07 §8 suggested piloting on "categories with the richest structured data" (hadith, tafsir).
Direct inspection of `_meta/page_isnads.parquet` and `_meta/tafsir_xrefs.jsonl` shows the real
coverage is much more concentrated than "a category":

- **`page_isnads.parquet`** (35,526 rows, matching `extraction_manifest.json`'s `isnad_count`)
  covers exactly **10 books** — Sahih al-Bukhari, Sahih Muslim, Sunan Abi Dawud, Sunan al-Tirmidhi
  (plus its `Ilal` appendix as a separate book), Sunan al-Nasa'i, Muwatta Malik, Musnad
  al-Tayalisi, and al-Muntaqa by Ibn al-Jarud (two editions). That's the primary hadith sources —
  not the other ~1,231 books in category 06 (كتب السنة), which are commentaries, hadith-science
  treatises, and similar secondary works with no isnad extraction behind them.
- Each row's `narrator_ids` field is already an **ordered array of narrator IDs** — the isnad
  chain is fully pre-parsed structured data, not raw text requiring NER. Building `NARRATED_FROM`
  graph edges for these 10 books is a pairwise walk of that array, not an extraction task.
- **`tafsir_xrefs.jsonl`**, once correctly resolved (§2), covers **9 real tafsir books** (Zad
  al-Masir, al-Tafsir al-Munir, Tafsir al-Tustari, Tafsir Ibn Juzayy, and others) — not all ~273
  books in category 03 (التفسير).

**Implication:** the hadith-takhrij and tafsir-by-verse use cases from doc 07 work precisely for
these **19 books** today. Every other book in categories 06 and 03 falls back to the general
hybrid-search path (doc 07's "General Q&A" lane) until someone extends extraction — that's a
scoping fact to design and communicate around, not a hypothetical edge case. See
[11_extending_extraction_coverage.md](11_extending_extraction_coverage.md) for what extending
that coverage would actually involve.

## 2. Finding: `hadith_xrefs` and `tafsir_xrefs` key on a different ID space than the rest of the corpus

Every book and page in this corpus has two identifiers: this pipeline's internal, sequential
`book_id`/`page_id` (used by `pages.jsonl`, `book_metadata.json`, and `page_isnads.parquet`), and
the original Shamela numbering, `shamela_id`/`shamela_page_id`. Direct inspection shows
**`hadith_xrefs.jsonl` and `tafsir_xrefs.jsonl` both store `shamela_id`/`shamela_page_id` values
in fields literally named `book_id`/`page_id`** — inconsistent with every other table in `_meta/`.

The evidence, reproduced exactly as checked:

| Table | Field | Sample value | What it actually resolves to |
|---|---|---|---|
| `tafsir_xrefs.jsonl` | `book_id` | `23619` | Not present in `book_metadata.book_id` at all. Matches `book_metadata.shamela_id = 23619` → internal `book_id = 6104` ("زاد المسير في علم التفسير"). |
| `tafsir_xrefs.jsonl` | `page_id` | range `10–2272` for that book | Falls inside book 6104's `shamela_page_id` range (`1–2311`), nowhere near its internal `page_id` range (`5,755,145–5,757,455`). |
| `hadith_xrefs.jsonl` | `book_id` | `28107` | Matches `book_metadata.shamela_id = 28107` → internal `book_id = 6494` ("موطأ مالك – رواية يحيى"). All 10 of `hadith_xrefs`'s distinct `book_id` values resolve this way, to exactly the same 10 books as `page_isnads`. |
| `hadith_xrefs.jsonl` | `page_id` | range `421–4093` for that book | Falls inside book 6494's `shamela_page_id` range (`1–4600`), not its internal `page_id` range (`6,124,613–6,129,212`). |

`page_isnads.parquet` is the one table that got this right structurally — its `book_id` is
already the internal ID, and its page-reference column is honestly named `shamela_page_id` rather
than `page_id`, so there's no ambiguity even though it still needs the same translation step.

**Concretely, why this matters:** a naive `tafsir_xrefs.book_id = book_metadata.book_id` join
resolves only 4–5 coincidental non-tafsir books (a Maliki fiqh commentary, a Kuwaiti fatwa
collection, an ethics encyclopedia — books that happen to quote the full Quran text once each,
hence a near-6,236-row count matching total verse count) and **silently drops all 9 of the real
tafsir books** that make up the substantive part of the table. The tafsir-by-verse feature would
look broken — thin, wrong-genre results — when the actual data is present and correct; it's the
join that's wrong. The fix in both cases is a two-step resolution:

```sql
-- Resolve hadith_xrefs / tafsir_xrefs book_id (really shamela_id) to internal book_id:
SELECT x.*, m.book_id AS internal_book_id
FROM tafsir_xrefs x
JOIN book_metadata m ON m.shamela_id = x.book_id

-- Then resolve page_id (really shamela_page_id) to internal page_id, scoped to that book:
SELECT x.*, p.page_id AS internal_page_id
FROM tafsir_xrefs x
JOIN pages p ON p.book_id = <resolved internal_book_id> AND p.shamela_page_id = x.page_id
```

## 3. Pipeline stages

| Stage | What it does | Notes from §1–2 |
|---|---|---|
| **0. Schema & ID reconciliation** | For every `_meta/*` table, verify which ID space each foreign key actually uses, before any join logic is written | Not optional, per §2 — build this as an automated check (assert every `hadith_xrefs`/`tafsir_xrefs` row resolves to a real internal book/page), not a one-time manual spot check |
| **1. Metadata normalization** | Load `book_metadata.json` per book into a canonical `books` table (author, category, `death_hijri`, `book_type`, multi-volume flags) | Also verify `manifest.json`'s SHA-256 checksums against actual files here, before trusting content downstream |
| **2. Genre + coverage routing** | Decide chunking strategy per book | Route on `category_id` **and** explicit membership in the resolved 10/9 curated sets from §1 — a category-06 book outside the 10 gets TOC-anchored chunking, not hadith-atomic, since there's no isnad data behind it |
| **3. Chunking** | Per-genre chunking, per [02_chunking_strategies.md §12](02_chunking_strategies.md#12-applying-these-concepts-to-structured-multi-genre-corpora) and [07 §4](07_recommended_architecture_for_shamela_rag.md#4-chunking-mapped-per-genre) | For the 10 hadith books specifically: chunk directly on `hadeeth_key` boundaries (already delimited by `page_isnads`/`hadith_xrefs`) — no heuristic isnad-marker text parsing needed |
| **4. Contextual prefixing** | Prepend breadcrumb/context to each chunk before embedding, per [02 §7](02_chunking_strategies.md#7-contextual-retrieval-anthropic-september-2024) | Cheap structured-metadata prefix (title, author, TOC breadcrumb) by default; reserve an LLM-generated context sentence for genres where local ambiguity is highest (fiqh masa'il) to control cost across ~7.6M pages |
| **5. Embedding** | Batch-embed chunks with the model chosen per [03](03_embeddings_and_vector_stores.md)/[09](09_open_source_arabic_llms.md)'s evaluation | Content-hash each chunk (text + chunker version) so a chunker bugfix triggers re-embedding only of what changed |
| **6. Lexical / root-normalized index** | Build BM25 index plus a root-normalized field via `root_dictionary.jsonl`, per [03 §5](03_embeddings_and_vector_stores.md#5-sparse-and-hybrid-representations) | Flat token→root lookup, no ID-space issue here |
| **7. Graph loading** | Load narrators, isnad edges, hadith concordance, verse links, per [05](05_knowledge_graphs_and_graphrag.md) | Isnad edges: pairwise walk of `narrator_ids`. Hadith concordance: group resolved `hadith_xrefs` rows by `key_id`. Tafsir edges: join via `shamela_id`/`shamela_page_id` per §2 — not the naive join |
| **8. Validation / QA** | Row-count reconciliation against `manifest.json` / `extraction_manifest.json`; golden-set spot check per [06](06_evaluation_and_recent_advancements.md) | This stage, run *first* rather than last, is what would have caught §2 automatically — worth building before scaling past a pilot |

## 4. Revised pilot recommendation

Doc 07 §8 suggested piloting on "hadith and tafsir categories." Sharpened by §1: **ingest all
8,589 books through the general TOC-anchored + hybrid-search path first** — that part is
genre-independent and carries no coverage risk — **but validate the graph-specific paths (takhrij,
verse lookup) against exactly the 19 books with real curated coverage** (the 10 hadith sources +
9 tafsir works identified in §1), since that's the only place ground truth exists to check
correctness against before deciding whether extending isnad/tafsir extraction to the rest of the
corpus is worth the investment (see
[11_extending_extraction_coverage.md](11_extending_extraction_coverage.md)).

## 5. Tooling note

The findings above came from a local Python venv (`.venv/`, gitignored, not committed) with
`duckdb` installed, querying the parquet/JSONL files directly (`read_parquet(...)`,
`read_json_auto(...)`) rather than loading them into a full database first. This is a cheap,
disposable setup worth reusing for the next round of schema verification — DuckDB reads both
formats natively with no ETL step, which made spot-checking ID ranges across dozens of ad hoc
queries fast.

## 6. What was actually verified vs. still assumed

In the interest of not overstating confidence: §1–2 came from targeted spot checks (multiple
books cross-checked for `hadith_xrefs`/`tafsir_xrefs`, one book cross-checked in depth for
`page_isnads`), not an exhaustive row-by-row audit of all 35,526 isnad rows or all ~91,000
combined `hadith_xrefs`/`tafsir_xrefs` rows. The pattern was consistent enough across independent
samples (different books, both cross-reference tables) to trust as a systematic finding rather
than a one-off anomaly, but Stage 0 above should still assert it programmatically over every row
before any production ingestion run — spot checks find the shape of a problem, they don't replace
validating the whole dataset. Similarly untouched by this investigation: `authors.parquet`,
`categories.parquet`, and whether `root_dictionary.jsonl` has any analogous ID-space quirks — no
evidence of a problem there, but no verification either.
