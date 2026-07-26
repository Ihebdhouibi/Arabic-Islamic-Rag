# Evaluation Metrics — Worked Examples

> Part of the series indexed in [00_overview.md](00_overview.md). Extends
> [docs/technical_docs/06_evaluation_and_recent_advancements.md §2](../technical_docs/06_evaluation_and_recent_advancements.md)
> (which names precision@k, recall@k, MRR, and NDCG) with actual hand-computed arithmetic against
> a real example from this project's own
> [golden evaluation seed](../technical_docs/golden_eval_seed.jsonl), so the formulas aren't
> abstract by the time ADR-002's benchmark round actually runs.

## The example

Using `hadith-001` from the golden seed: hadith concordance key 762, with **9 distinct relevant
books** (Bukhari, Muslim, Abu Dawud, Nasa'i, Tirmidhi, Muwatta Malik, Musnad al-Tayalisi, and
Ibn al-Jarud in both its editions). Evaluated here at **book level** for a clean example (the
seed's actual entries are page-level; book-level grading is a reasonable simplification for
illustrating the metrics).

Suppose a retrieval run for this query returns the following top-10 books, in ranked order
(hypothetical result, constructed to have realistic gaps rather than a perfect run):

| Rank | Book | Relevant? |
|---|---|---|
| 1 | صحيح البخاري | ✓ |
| 2 | صحيح مسلم | ✓ |
| 3 | *(unrelated fiqh book)* | ✗ |
| 4 | سنن أبي داود | ✓ |
| 5 | سنن النسائي | ✓ |
| 6 | *(unrelated tafsir book)* | ✗ |
| 7 | سنن الترمذي | ✓ |
| 8 | موطأ مالك | ✓ |
| 9 | *(unrelated aqidah book)* | ✗ |
| 10 | مسند أبي داود الطيالسي | ✓ |

Both editions of Ibn al-Jarud's *al-Muntaqa* did **not** make the top 10 in this hypothetical run —
that gap is the point; a perfect run makes for a boring worked example.

## Precision@10 and Recall@10

**Precision@10** = relevant items retrieved ÷ items retrieved = **7/10 = 0.70**

**Recall@10** = relevant items retrieved ÷ total relevant items that exist = **7/9 ≈ 0.778**

These two numbers already tell a story precision alone can't: this run is fairly clean (70% of
what it returned was actually right) but is missing about 22% of the real concordance — exactly
the kind of gap this project's takhrij use case cares about, since a hadith-authenticity check
that silently drops two real citations is a meaningfully incomplete answer, not just a slightly
noisy one.

## MRR (Mean Reciprocal Rank)

MRR looks only at **where the first relevant result appears**: `RR = 1 / rank_of_first_hit`. Here,
the first relevant hit (Sahih al-Bukhari) is at rank 1, so `RR = 1/1 = 1.0` — a perfect score,
despite 2 of 9 relevant books being missing entirely from the top 10. This is exactly MRR's known
blind spot: it only rewards getting *something* right immediately, and says nothing about recall.
It's the right metric for "did we show the user a good answer fast" (e.g., a single-answer
lookup), and the wrong one on its own for judging takhrij completeness — pair it with recall@k,
never report it alone for this use case.

## NDCG@10 (Normalized Discounted Cumulative Gain)

NDCG rewards relevant results more when they appear **earlier**, via a logarithmic discount, and
normalizes against the best possible ordering so scores are comparable across queries with
different numbers of relevant items.

**Step 1 — DCG@10** of the actual ranking above (binary relevance: 1 or 0), using
`DCG = Σ relevance_i / log2(i + 1)` for rank `i` from 1 to 10:

| Rank i | Relevant | log2(i+1) | Contribution |
|---|---|---|---|
| 1 | 1 | 1.000 | 1.000 |
| 2 | 1 | 1.585 | 0.631 |
| 3 | 0 | — | 0 |
| 4 | 1 | 2.322 | 0.431 |
| 5 | 1 | 2.585 | 0.387 |
| 6 | 0 | — | 0 |
| 7 | 1 | 3.000 | 0.333 |
| 8 | 1 | 3.170 | 0.316 |
| 9 | 0 | — | 0 |
| 10 | 1 | 3.459 | 0.289 |

`DCG@10 = 1.000 + 0.631 + 0.431 + 0.387 + 0.333 + 0.316 + 0.289 = 3.386`

**Step 2 — IDCG@10**, the best possible DCG: since there are 9 relevant books total and we're only
looking at 10 slots, the ideal ranking places all 9 relevant books first, then 1 irrelevant one:

| Rank i | Relevant (ideal) | log2(i+1) | Contribution |
|---|---|---|---|
| 1–9 | 1 (all nine) | 1.000 … 3.322 | 1.000 + 0.631 + 0.500 + 0.431 + 0.387 + 0.356 + 0.333 + 0.316 + 0.301 |
| 10 | 0 | — | 0 |

`IDCG@10 = 4.254`

**Step 3 — NDCG@10** = `DCG@10 / IDCG@10 = 3.386 / 4.254 ≈ 0.796`

## What these four numbers say together, that any one alone wouldn't

| Metric | Value | What it captures |
|---|---|---|
| Precision@10 | 0.70 | Of what we showed, how much was right |
| Recall@10 | 0.778 | Of what exists, how much we found |
| MRR | 1.0 | Whether the very first result was trustworthy |
| NDCG@10 | 0.796 | Overall ranking quality, penalizing relevant items landing late |

A system tuned to maximize only MRR could look perfect (1.0, as above) while quietly missing
real citations — which is precisely why
[docs/technical_docs/14_golden_evaluation_dataset.md](../technical_docs/14_golden_evaluation_dataset.md)'s
examples should be scored on **recall and NDCG together**, not MRR alone, before ADR-002's
embedding benchmark is trusted as a basis for choosing a model.
