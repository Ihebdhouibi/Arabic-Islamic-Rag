# M2 Chunking — End-of-Milestone Review

Self-review of the `shamela_rag.chunking` package after M2 (issues #13–#25) merged. Purpose: assess
design coherence, surface gaps and real-corpus risks, and list what must be reconciled or tuned
before full-scale ingestion (M3+) and the M6 evaluation. This is an internal engineering review, not
a design change.

## 1. What was built

Eleven modules compose the structure-first, parent–child chunker:

| Module | Responsibility |
|---|---|
| `title_spans` | Parse inline `<span data-type=title id=toc-N>`; map `toc-N` → `shamela_title_id` |
| `tokens` | `TokenCounter` protocol + heuristic counter (swappable for a model tokenizer) |
| `boundaries` | Per-page fallback ladder (6 rungs) with source + confidence |
| `sections` | `build_sections`: trails, `path_source` (explicit vs derived), page ranges |
| `navigation` | Navigational-vs-content classification (heading-only → context, not a chunk) |
| `content_roles` | Split `body` vs `footnote`; never concatenate |
| `sizing` | `SizePolicy` + `split_section` (join-based, overlap, subheading→para→sentence→word) |
| `merge` | Conditional short-fragment merge with named-entity guard |
| `context_header` | Compact Arabic header; `99999` death-year → unknown; A/B toggle |
| `recovery` | Heading-recovery candidates (recorded only, never trusted yet) |
| `orchestrator` | `chunk_book` composing the above into verbatim `BookChunk` records |

105 unit tests, fully offline/deterministic, dependency-light.

## 2. What is solid

- **Verbatim source preservation.** Every emitted chunk's `source_text` is an exact slice of the
  page `body`/`footnotes`, offsets round-trip (`haystack[start:end] == source_text`), and the
  normalized `retrieval_text` is kept separate. The corpus text is never mutated by the chunker.
- **Auditable boundaries.** Each boundary records where it came from and a confidence, and ambiguous
  pages never fabricate offsets — exactly what we need to inspect/improve extraction later.
- **Correct, verified data handling.** Ordering by `page_id` (not `sequence_num`/`page_num`);
  `toc-N` → `shamela_title_id` (not the global `title_id`); inline markup treated as richer than
  `toc.jsonl`.
- **Config-driven, swappable.** Thresholds live in `SizePolicy`; the tokenizer is a protocol; the
  context header is produced separately so its dense-input weight can be A/B tested.
- **Honest derived-vs-explicit nesting** (`path_source`) and **footnote flagging** so editor notes
  are never attributed to the author.

## 3. Gaps and risks to address before scaling

Ordered by importance.

### 3.1 Two splitters diverged (highest priority)
`sizing.split_section` (join-based, applies **overlap** and the subheading→paragraph→sentence→word
**priority**) is **not** used by `orchestrator.chunk_book`. The orchestrator uses its own
offset-based `_split_offsets` (sentence-only, **no overlap**, no word fallback) so it can keep
`source_text` verbatim. Consequences:
- **Overlap is not applied in the real pipeline** — chunks are currently non-overlapping, despite
  the 64-token overlap being a design goal (cross-boundary context for retrieval).
- **The subheading/paragraph/word priority is unused** in practice.
- **Oversized segments with no sentence punctuation become a single over-max chunk** (classical
  prose frequently lacks punctuation) — a real quality risk.

**Fix direction:** apply the rich splitter to `retrieval_text`/embedding input while deriving
`source_text` from offset spans, or extend `_split_offsets` to honor the full priority + a word
fallback and to emit overlap on the *embedding* text only (never on `source_text`).

### 3.2 `merge` and `recovery` are not wired into the orchestrator
- `merge_short_fragments` and the `is_named_entity` flag are implemented and tested but **never
  called by `chunk_book`** — tiny fragments are not actually merged, and named-entity protection is
  inert in the real path.
- `recover_heading_candidates` runs nowhere in ingestion; candidates should at least be **counted
  and stored** for the M6 precision measurement.

### 3.3 "Round-trip verbatim" means "no mutation", not "lossless coverage"
Navigational headings and inline markup are intentionally excluded from chunks, so concatenating
chunks does **not** reproduce the whole book. The current gate only proves each chunk is an
un-mutated substring. We still need a **coverage/accounting check** (every source character is in a
chunk or explicitly classified as ignored heading/markup) — that is M6-06 (structural validation),
and it is not yet implemented.

### 3.4 Heuristic tokenizer skews every threshold
All sizes are measured with the word+punct heuristic, which **undercounts vs. a subword tokenizer**.
Real chunks will be larger than the nominal 128/768/448 targets. Must: wire the real Qwen3-8B/BGE-M3
tokenizer into `TokenCounter` once M3-02/03 land, then **re-tune thresholds in M6-04**.

### 3.5 Smaller but real
- `recovered_title` uses a **raw substring match** (no normalization) → misses when diacritics or
  markup differ; should try `normalize_for_index` matching.
- The **test fixture is 2 pages**; there is no golden test over a real large book with dense inline
  markup. Add a bigger real-book slice fixture before trusting the orchestrator at scale.
- Trail inheritance across a page with **no boundary** carries the last trail forward; fine within a
  section, but worth validating on multi-page sections.
- `min_content_tokens` (nav threshold) defaulting to 5 heuristic tokens can misflag genuinely terse
  sections; tune in M6.

## 4. Needs empirical tuning in M6 (not bugs)

`min_content_tokens`, all `SizePolicy` thresholds, overlap size, context-header length/inclusion,
heading-recovery precision (before enabling splits on candidates), and `recovered_title` match
strategy.

## 5. Recommended follow-ups (suggest as issues)

1. **Reconcile the two splitters**; wire `merge` + overlap + word-fallback into `chunk_book` (§3.1,
   §3.2). — high priority, do before/early in ingestion.
2. **Coverage accounting** in the structural-validation harness (§3.3) — this is M6-06; consider
   pulling part of it forward so ingestion reports lost text.
3. **Wire the real tokenizer** into `TokenCounter` after M3-02/03; re-run size tuning (M6-04).
4. **Bigger golden fixture** from a real book (a slice of `1021__أسد-الغابة`) for the orchestrator.
5. **Store heading-recovery candidates + boundary-confidence counts** during ingestion for M6.

## 6. Verdict

The M2 package is a clean, well-tested, auditable foundation and the hard invariant (no source-text
mutation) holds. The main debt is that the **orchestrator underuses the size policy** (no overlap,
simplified splitting) and that **merge/recovery are not yet integrated** — these should be closed
early in M3/ingestion rather than left to M6. None of the gaps block starting M3 (embedding
providers, Qdrant, ingestion), but §3.1–3.2 should be scheduled before the full-corpus embed run.
