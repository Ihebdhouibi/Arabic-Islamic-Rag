# 11 — Extending Extraction Coverage (and Contributing It Back)

> Follow-up to [10_ingestion_and_indexing_pipeline.md §1](10_ingestion_and_indexing_pipeline.md#1-finding-curated-graph-coverage-is-narrower-than-category-boundaries-suggest),
> which found that curated isnad/tafsir-verse coverage is 19 books, not the ~1,514 books in
> categories 06 and 03 combined. This document covers what closing that gap would actually take,
> and — since it's not yet confirmed whether the `AuthenticIlm` Hugging Face account is this
> project's own or a third party's — how to contribute the result back either way.

## 1. The gap, quantified

Direct counts against `book_metadata.parquet`:

| Category | Total books | Curated-coverage books | Remaining, uncovered |
|---|---|---|---|
| 06 — كتب السنة (hadith collections) | 1,241 | 10 | **1,232** |
| 03 — التفسير (tafsir) | 273 | 9 | **264** |

Closing even part of this gap directly expands the hadith-takhrij and tafsir-by-verse use cases
from doc 07 well beyond today's 19-book curated core — this is worth attempting for this project's
own sake, independent of whether it's ever contributed upstream.

## 2. Two different extraction problems, two different difficulties

**Tafsir–verse alignment is the tractable one.** The Quran is a closed, exactly-known set of
6,236 verses (`quran_verses.jsonl`), and per the original dataset card's data notes, Quranic
quotations in body text are already bracketed with `﴿...﴾` markers. That makes this largely a
**string-matching problem**, not one requiring an LLM: extract each `﴿...﴾`-delimited span per
page, normalize it (strip diacritics, unify alif/hamza variants — the same normalization
consideration from [03_embeddings_and_vector_stores.md](03_embeddings_and_vector_stores.md)), and
match it against `quran_verses.body`. The one real wrinkle is disambiguation: short phrases that
recur verbatim across multiple verses need either longer-match preference or surrounding-context
tie-breaking — a solvable, well-scoped problem, not an open-ended one.

**Isnad extraction is harder**, and is a genuine two-stage NLP task:

1. **Isnad-span detection** — classical isnad chains follow fairly formulaic openings (حدثنا،
   أخبرنا، حدثني، سمعت) followed by a sequence of `عن` (from so-and-so) links. Boundary detection
   is plausibly rule-based/regex to a first approximation, since the formulas are conventional.
2. **Narrator entity resolution** — each name mentioned in a detected chain has to be linked to a
   specific entry in the existing 18,989-row `narrators.jsonl`. This is the hard part: narrators
   are referenced inconsistently across texts (kunya, full name with nasab, short name), so exact
   string matching will under-match. Realistic options, in increasing cost and decreasing risk:
   - **Pure regex + fuzzy string matching** against `narrators.short_name`/`long_name` — cheap,
     moderate recall, will miss non-obvious name variants.
   - **LLM-assisted entity linking** on top of regex-detected candidate spans — the regex narrows
     ~1,232 books down to candidate isnad text; an LLM only resolves *which* narrator each name in
     that candidate refers to, rather than reading every page cold. This keeps LLM cost bounded to
     the candidate spans, not the whole corpus.
   - **Full LLM extraction per page** — most accurate, most expensive; only worth it if the hybrid
     approach's error rate proves unacceptable in validation (§3).

## 3. Validate before trusting anything — using data you already have for free

Before running any new extractor on the 1,232 + 264 uncovered books, run it on the **existing 19
curated books first** and compare its output against the existing `page_isnads`/`tafsir_xrefs`
rows as ground truth. This is a zero-cost evaluation set that already exists — treat it the way
doc 06 treats a golden dataset: measure precision/recall of detected isnad spans and narrator
links against Bukhari/Muslim/etc. before ever pointing the extractor at a book with no existing
answer key to check against.

## 4. Provenance matters once coverage is mixed-confidence

Once the graph contains both scholar-curated edges (the original 19 books) and automatically
extracted ones (anything new), the two should never be presented with equal authority. Tag every
graph edge with its provenance at load time — a `source: curated` vs. `source: extracted` field,
ideally with a confidence score for extracted edges — so retrieval and generation can distinguish
them. This connects directly to the citation-integrity requirement in
[07_recommended_architecture_for_shamela_rag.md §6](07_recommended_architecture_for_shamela_rag.md#6-generation-and-grounding--a-domain-specific-requirement-not-a-nice-to-have):
an isnad chain surfaced from automated extraction should be captioned as such, not presented with
the same weight as one that traces back to the original curation. Silently blurring "scholar
verified" and "automatically inferred" would undermine the entire trust argument this project's
architecture is built around.

## 5. Contributing back — two paths, since ownership isn't confirmed yet

Either way, the first step is identical: **build and validate the extension inside this project's
own pipeline** (doc 10 §3, Stage 7 — graph loading, with the `source` provenance tag from §4).
That's useful to this RAG system immediately, regardless of whether or when it goes anywhere else.
From there, the path forks:

- **If `AuthenticIlm` turns out to be this project's own account:** this is the simple case —
  regenerate the affected `_meta/*` files, version the update (Hugging Face datasets are git repos
  under the hood, so this is a normal commit/tag), and update the dataset card to describe the
  new coverage and, importantly, the extraction methodology and confidence caveats from §4 — so
  anyone downstream (including future you) knows which edges are curated and which are inferred.
- **If it's a third party's account:** don't submit a large unprompted parquet diff. Open a
  discussion on the dataset's Hugging Face "Community" tab describing the proposed extension,
  the methodology, and ideally a small validated sample (per §3) — and let the maintainer decide
  whether and how to fold it in. They may have scoped coverage to exactly these 19 books for
  reasons not visible from outside (a confidence bar they held the rest to, licensing constraints
  on specific critical editions, etc.), and the dataset's own license note about respecting the
  IP of muhaqqiqīn and publishers applies just as much to anything newly contributed as to the
  original data.

## 6. This is doc 05's "Path A," applied narrowly

[05_knowledge_graphs_and_graphrag.md §5](05_knowledge_graphs_and_graphrag.md#5-two-very-different-ways-a-graph-comes-into-existence)
drew a hard line between Path A (LLM/NLP-extracted graphs — expensive, approximate) and Path B
(curated structured data — cheap, exact), and this project's whole design leans on already being
mostly in Path B. Extending coverage is deliberately reintroducing Path A, but **narrowly** — only
for the two specific relationship types where a scoped, checkable extraction task exists (verse
alignment, isnad chains), not as a wholesale switch to LLM-extracting the rest of the corpus's
graph structure. §4's provenance tagging is what keeps that boundary honest once both kinds of
edges coexist in the same graph.
