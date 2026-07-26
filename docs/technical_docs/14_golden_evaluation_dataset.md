# 14 — Golden Evaluation Dataset

> Resolves the blocker flagged in
> [13_architecture_decisions.md ADR-002](13_architecture_decisions.md#adr-002-embedding-model--no-model-chosen-yet-the-evaluation-protocol-is-the-decision):
> embedding-model selection (and, per
> [04_retrieval_and_query_strategies.md](04_retrieval_and_query_strategies.md), retrieval-strategy
> tuning generally) can't proceed without a golden set to measure against. This document defines
> the schema, sourcing methodology per use case, and a first seed batch — pulled from the actual
> corpus via DuckDB, not invented — stored alongside this doc as
> [golden_eval_seed.jsonl](golden_eval_seed.jsonl).

## 1. Why this has to happen before model/architecture tuning

Per [06_evaluation_and_recent_advancements.md §5](06_evaluation_and_recent_advancements.md), generic
leaderboards don't transfer to this corpus's classical-register Arabic, and per
[09_open_source_arabic_llms.md §2](09_open_source_arabic_llms.md#2-why-this-matters-general-arabic-llms-measurably-struggle-on-islamic-content-specifically),
general Arabic LLMs measurably underperform on Islamic-content reasoning specifically. Both facts
say the same thing: **assume nothing about quality on this domain — measure it**, which requires
a dataset to measure against. This is the first artifact in this project that isn't a design doc —
it's structured data, sourced from the real corpus.

## 2. Schema

Each example is one JSON object:

```jsonc
{
  "id": "hadith-001",
  "use_case": "HADITH_TAKHRIJ",          // GENERAL_QA | HADITH_TAKHRIJ | TAFSIR_BY_VERSE | FIQH_LOOKUP
  "query": "...",                          // a realistic user question
  "expected_sources": [                    // ground truth the retrieval step should surface
    {
      "book_title": "...",
      "internal_book_id": 1458,            // per docs/technical_docs/diagrams/erd_data_model.md
      "shamela_page_id": 3739,             // NOT internal page_id -- see note below
      "confidence": "verified"             // verified | candidate (see §3.3)
    }
  ],
  "sourcing_method": "hadith_xrefs key_id concordance",
  "notes": "..."
}
```

**On `shamela_page_id` vs. internal `page_id`:** every example below cites `shamela_page_id`,
resolved via `book_metadata.shamela_id` per
[10_ingestion_and_indexing_pipeline.md §2](10_ingestion_and_indexing_pipeline.md#2-finding-hadith_xrefs-and-tafsir_xrefs-key-on-a-different-id-space-than-the-rest-of-the-corpus).
Once a real ingestion pipeline exists, these should be re-resolved to internal `page_id` values
against that book's actual `pages.jsonl` — not done here, since that requires the per-book file,
not just the `_meta/` tables this seed was built from.

## 3. Sourcing methodology, per use case

### 3.1 Hadith takhrij — tractable now, ground truth already exists

`hadith_xrefs.key_id` groups rows that are, by the corpus's own curation, the same underlying
hadith appearing across different collections. A `key_id` spanning many distinct books *is* a
verifiable ground-truth concordance example — no manual authoring needed, just querying and
resolving IDs correctly (per doc 10 §2's join fix). This is exactly what §4 below does.

### 3.2 Tafsir by verse — equally tractable, same mechanism

`tafsir_xrefs.key_id` = a `quran_verses.id`. Filtering to only the 9 real curated tafsir books
(per [10 §1](10_ingestion_and_indexing_pipeline.md#1-finding-curated-graph-coverage-is-narrower-than-category-boundaries-suggest) —
excluding the ~4-5 coincidental non-tafsir books that also happen to quote full verses) gives a
clean, verifiable "this verse is discussed on these exact pages of these exact books" example.

### 3.3 Fiqh lookup — no automatic ground truth; candidates only, expert review required

Unlike §3.1/3.2, there is no cross-reference table linking a fiqh *question* to the specific pages
that answer it — `category_id` only identifies a book's madhhab, not which page addresses a given
masala. §5 below includes a small number of **candidate** examples (`confidence: "candidate"`),
built by identifying foundational madhhab texts by title (e.g., al-Mudawwana for Maliki fiqh, at
book_id 566) rather than verified page references. **These need actual scholarly review before
being trusted as evaluation ground truth** — the risk of a wrong or shallow fiqh "answer key" is
exactly the kind of error [07 §6](07_recommended_architecture_for_shamela_rag.md#6-generation-and-grounding--a-domain-specific-requirement-not-a-nice-to-have)
warns against propagating.

### 3.4 General Q&A — same limitation as fiqh, broader scope

No structured ground truth exists for open-ended questions either. The practical path here is the
same as fiqh: draft candidate questions and expected-topic books, mark them `candidate`, and get
them reviewed rather than trusting them as-authored. None are included in this seed batch — worth
prioritizing once someone with subject-matter familiarity is available to review, rather than
guessing at what a "general" question and its correct sources should look like.

## 4. Seed batch summary (verified examples)

Three hadith-takhrij and three tafsir-by-verse examples, each pulled live from `_meta/` and
resolved through the `shamela_id` fix. Full detail in
[golden_eval_seed.jsonl](golden_eval_seed.jsonl); highlights:

| ID | Use case | Query basis | Sources found |
|---|---|---|---|
| hadith-001 | HADITH_TAKHRIJ | `key_id=762` | 9 books: Bukhari, Muslim, Abu Dawud, Nasa'i, Tirmidhi, Muwatta Malik, Musnad al-Tayalisi, Ibn al-Jarud (both editions) — 26 page references total |
| hadith-002 | HADITH_TAKHRIJ | `key_id=984` | Same 9-book spread, 33 page references |
| hadith-003 | HADITH_TAKHRIJ | `key_id=304` | Same 9-book spread, 60 page references |
| tafsir-001 | TAFSIR_BY_VERSE | Quran 16:67 (`verse_id=1968`) — the verse on dates/grapes, "sakar" and good provision | All 9 curated tafsir books |
| tafsir-002 | TAFSIR_BY_VERSE | `verse_id=32` — the Paradise-fruits verse | All 9 curated tafsir books |
| tafsir-003 | TAFSIR_BY_VERSE | `verse_id=918` — "we make some wrongdoers allies of others" | All 9 curated tafsir books |

Two candidate (unverified) fiqh examples are also included, referencing foundational madhhab texts
(al-Mudawwana, al-Umm) by title only — explicitly flagged `"confidence": "candidate"` and not to be
used for scoring until reviewed.

## 5. What this seed batch is (and isn't)

**Is:** proof that verifiable, zero-manual-authoring ground truth exists today for two of the four
use cases, and a concrete schema to keep extending. Six verified examples is far short of doc 06's
50–100 target — this is a seed, not the finished set.

**Isn't:** a substitute for fiqh/general-Q&A examples, which per §3.3/3.4 need actual expert input,
not more querying. Growing the hadith/tafsir side further is just running §3.1/3.2's method against
more `key_id`s — cheap, mechanical, worth doing before ADR-002's benchmark runs. Growing the fiqh/
general side is a different kind of work this document can't shortcut.

## Further work

- Extend §4 to 20–30 verified hadith/tafsir examples (mechanical, per §3.1/3.2) before running
  ADR-002's embedding benchmark.
- Recruit or schedule subject-matter review for the fiqh/general candidates before they count as
  scoring ground truth.
- Once a real ingestion pipeline exists, re-resolve every `shamela_page_id` in
  [golden_eval_seed.jsonl](golden_eval_seed.jsonl) to internal `page_id` values.
