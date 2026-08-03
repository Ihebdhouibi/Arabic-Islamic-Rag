#!/usr/bin/env bash
#
# create_issues.sh — populate GitHub labels, milestones, and the full General-Module
# issue backlog for the Arabic-Islamic-Rag repo.
#
# Source of truth: docs/implementation/general_module_issues.md
#
# Requirements: GitHub CLI (`gh`) authenticated with repo-admin rights.
#   gh auth login
#   gh auth status
#
# Usage:
#   REPO=Ihebdhouibi/Arabic-Islamic-Rag bash scripts/create_issues.sh
#
# Optional — also add every created issue to a GitHub Project (v2):
#   PROJECT_NUMBER=1 PROJECT_OWNER=Ihebdhouibi REPO=Ihebdhouibi/Arabic-Islamic-Rag \
#     bash scripts/create_issues.sh
#   (PROJECT_NUMBER is the number in the project URL; the `project` auth scope is required:
#    gh auth refresh -s project )
#
# The script is safe to re-run for labels and milestones (idempotent). NOTE: re-running
# WILL create duplicate issues — run the issue section only once. Use --labels-only or
# --milestones-only to run just those parts.

set -euo pipefail

REPO="${REPO:-Ihebdhouibi/Arabic-Islamic-Rag}"
PROJECT_NUMBER="${PROJECT_NUMBER:-}"
PROJECT_OWNER="${PROJECT_OWNER:-${REPO%%/*}}"

MODE="${1:-all}"   # all | --labels-only | --milestones-only | --issues-only

echo "Repo: $REPO"
[ -n "$PROJECT_NUMBER" ] && echo "Project: #$PROJECT_NUMBER (owner: $PROJECT_OWNER)"
echo

command -v gh >/dev/null || { echo "ERROR: gh CLI not found. Install: https://cli.github.com/"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "ERROR: run 'gh auth login' first."; exit 1; }

# ----------------------------------------------------------------------------- labels
ensure_label() { # name color description
  gh label create "$1" --repo "$REPO" --color "$2" --description "$3" --force >/dev/null 2>&1 || true
  echo "  label: $1"
}

create_labels() {
  echo "== Labels =="
  ensure_label infra            "5319e7" "Project foundation / infrastructure"
  ensure_label data             "0e8a16" "Data access / loaders / models"
  ensure_label chunking         "1d76db" "Chunking mechanism"
  ensure_label embeddings       "fbca04" "Embeddings / indexing"
  ensure_label retrieval        "d93f0b" "Retrieval pipeline"
  ensure_label generation       "b60205" "Answer generation"
  ensure_label eval             "0052cc" "Evaluation / benchmarks"
  ensure_label api              "006b75" "API / interface"
  ensure_label ci               "5319e7" "CI/CD"
  ensure_label docs             "c5def5" "Documentation"
  ensure_label test             "bfd4f2" "Tests"
  ensure_label good-first-issue "7057ff" "Good first issue"
  echo
}

# ------------------------------------------------------------------------- milestones
ensure_milestone() { # title description
  if gh api "repos/$REPO/milestones?state=all" --jq '.[].title' 2>/dev/null | grep -Fxq "$1"; then
    echo "  milestone exists: $1"
  else
    gh api "repos/$REPO/milestones" -f title="$1" -f description="$2" >/dev/null
    echo "  milestone created: $1"
  fi
}

M0="M0 — Foundation & infra"
M1="M1 — Data access"
M2="M2 — Chunking"
M3="M3 — Embeddings & indexing"
M4="M4 — Retrieval"
M5="M5 — Generation"
M6="M6 — Evaluation"
M7="M7 — API & integration"
M8="M8 — CI/CD & hardening"

create_milestones() {
  echo "== Milestones =="
  ensure_milestone "$M0" "Repo builds, Postgres+Qdrant run, CI green"
  ensure_milestone "$M1" "Read corpus files into typed models; reliable ordering"
  ensure_milestone "$M2" "Structure-first parent-child chunking; boundary ladder; size policy"
  ensure_milestone "$M3" "Dense (Qwen3-8B/BGE-M3) + sparse arms; ingestion into Qdrant+Postgres"
  ensure_milestone "$M4" "Hybrid retrieve + rerank + EN->AR translate + parent expansion"
  ensure_milestone "$M5" "Cited answer generation; preserve disagreement"
  ensure_milestone "$M6" "Golden set + retrieval metrics + 3-stage model comparison"
  ensure_milestone "$M7" "FastAPI /ask endpoint + citation schema"
  ensure_milestone "$M8" "Coverage gate, ingestion smoke test, contributing/branch docs"
  echo
}

# ----------------------------------------------------------------------------- issues
mk() { # title | milestone | comma,labels | body
  local title="$1" milestone="$2" labels="$3" body="$4"
  local label_args=() url
  IFS=',' read -ra L <<< "$labels"
  for l in "${L[@]}"; do label_args+=(--label "$l"); done
  url=$(gh issue create --repo "$REPO" --title "$title" --milestone "$milestone" \
        "${label_args[@]}" --body "$body")
  echo "  + $title  ->  $url"
  if [ -n "$PROJECT_NUMBER" ] && [ -n "$url" ]; then
    gh project item-add "$PROJECT_NUMBER" --owner "$PROJECT_OWNER" --url "$url" >/dev/null \
      && echo "      added to project #$PROJECT_NUMBER"
  fi
}

create_issues() {
  echo "== Issues =="
  local REF="Full detail: docs/implementation/general_module_issues.md"

  # ------------------------------------------------------------------ M0
  mk "M0-01 · Repo scaffolding & Python packaging" "$M0" "infra" \
"Create src/shamela_rag/ package layout, pyproject.toml (deps, entry points), code README, tests/ root; pin Python.
Depends on: —  |  Size: S  |  Branch: feature/m0-scaffolding
Done when: pip install -e . works in .venv; empty pytest run passes.
$REF"

  mk "M0-02 · Config & secrets management" "$M0" "infra" \
"pydantic-settings config: Postgres URL, Qdrant URL/collection, model names (Qwen/Qwen3-Embedding-8B, BAAI/bge-m3), corpus root, chunk-size params. .env.example; .env git-ignored.
Depends on: M0-01  |  Size: S  |  Branch: feature/m0-config
Done when: config loads + validates; one unit test.
$REF"

  mk "M0-03 · Linting, formatting, type-checking, pre-commit" "$M0" "infra,ci" \
"ruff (lint+format) + mypy + .pre-commit-config.yaml.
Depends on: M0-01  |  Size: S  |  Branch: feature/m0-lint
Done when: ruff check, ruff format --check, mypy src pass on skeleton.
$REF"

  mk "M0-04 · Structured logging" "$M0" "infra" \
"Central logging config (levels via env, JSON-or-plain toggle).
Depends on: M0-01  |  Size: S  |  Branch: feature/m0-logging
Done when: modules log through one configured logger; smoke test.
$REF"

  mk "M0-05 · Postgres + Qdrant via docker-compose" "$M0" "infra" \
"docker-compose.yml with Postgres (relational/provenance) and Qdrant (vectors). Init: enable pg_trgm; create Qdrant collection lazily from config.
Depends on: —  |  Size: M  |  Branch: feature/m0-compose
Done when: docker compose up gives reachable Postgres + Qdrant (/healthz).
$REF"

  mk "M0-06 · Migration framework + base schema" "$M0" "infra,data" \
"Alembic; first migration creates books, sections, chunks skeleton. chunks.source_text verbatim; separate retrieval_text normalized.
Depends on: M0-02, M0-05  |  Size: M  |  Branch: feature/m0-migrations
Done when: alembic upgrade head runs clean.
$REF"

  mk "M0-07 · CI pipeline (lint + test on PR)" "$M0" "ci" \
"GitHub Actions on PR to develop/stable-testing/main: ruff, mypy, pytest with Postgres + Qdrant service containers. Emits 'lint' and 'test' checks for branch protection.
Depends on: M0-03  |  Size: M  |  Branch: feature/m0-ci
Done when: CI green on a trivial PR; check names match plan §6.2.
$REF"

  # ------------------------------------------------------------------ M1
  mk "M1-01 · Corpus file loaders" "$M1" "data" \
"Streaming readers for pages.jsonl, toc.jsonl, book_metadata.json; tolerate null footnotes and truncated/oversized lines.
Depends on: M0-01  |  Size: M  |  Branch: feature/m1-loaders
Done when: yields typed records for fixture book 1021; unit tests on a tiny fixture.
$REF"

  mk "M1-02 · Domain models" "$M1" "data" \
"Book, Page, TocEntry models per verified schema incl. shamela_page_id, part, main_author_death_hijri, betaka_text, book_type_label, category_id, toc shamela_title_id (toc-N target) vs global title_id.
Depends on: M1-01  |  Size: S  |  Branch: feature/m1-models
Done when: models validate fixture records; tests.
$REF"

  mk "M1-03 · Page ordering & source-offset model" "$M1" "data,chunking" \
"Order by page_id (VERIFIED: sequence_num repeats in book 1021; page_id monotonic; never use printed page_num). Track per-page char offsets for source_offset spans.
Depends on: M1-02  |  Size: M  |  Branch: feature/m1-ordering
Done when: correctly-ordered stream + offsets; test asserts monotonic order and sequence_num not assumed unique.
$REF"

  mk "M1-04 · Book registry & genre routing hook" "$M1" "data" \
"category_id -> route. General module uses structural path for everything; hook must exist for later genres.
Depends on: M1-02  |  Size: S  |  Branch: feature/m1-registry
Done when: returns correct route from book_metadata.json; tests.
$REF"

  mk "M1-05 · Corpus discovery / manifest walk" "$M1" "data" \
"Enumerate book folders, locate files per book, skip malformed.
Depends on: M1-01  |  Size: S  |  Branch: feature/m1-discovery
Done when: lists all books under a root; handles a missing-file book; test.
$REF"

  # ------------------------------------------------------------------ M2
  mk "M2-01 · Arabic text normalization utilities" "$M2" "chunking,good-first-issue" \
"Diacritics, alif/hamza/ta-marbuta, tatweel, whitespace/\\r cleanup. Two variants: display-preserving (source_text untouched) vs index normalization (retrieval_text). Never mutate source_text.
Depends on: M0-01  |  Size: S  |  Branch: feature/m2-normalize
Done when: table-driven tests cover الصلاة/للصلاة/صلاته, hamza forms, ta-marbuta.
$REF"

  mk "M2-02 · Inline title-span parser (toc-N)" "$M2" "chunking" \
"Parse <span data-type='title' id=toc-N>…</span> from body (id may be unquoted). Map toc-N -> toc.shamela_title_id (NOT global title_id). Record char offset per occurrence.
Depends on: M1-01  |  Size: M  |  Branch: feature/m2-title-spans
Done when: extracts spans+offsets for fixture; test asserts toc-37 -> باب الهمزة; spans absent from toc.jsonl (toc-38/39) still captured.
$REF"

  mk "M2-03 · Token counter" "$M2" "chunking" \
"Token-length helper aligned to active embedding tokenizer (pluggable across Qwen3-8B / BGE-M3).
Depends on: M0-01  |  Size: S  |  Branch: feature/m2-tokens
Done when: stable counts; unit test on known strings.
$REF"

  mk "M2-04 · Boundary-detection fallback ladder" "$M2" "chunking" \
"Per occurrence, pick strongest boundary + record boundary_source+confidence: inline_toc > inline_title > recovered_title > toc_page_fallback > ambiguous_toc_page (no fabricated offsets) > paragraph_fallback.
Depends on: M2-02, M1-03  |  Size: L  |  Branch: feature/m2-boundary-ladder
Done when: each boundary gets source+confidence; ambiguous/low-confidence counted; tests per rung.
$REF"

  mk "M2-05 · Structural tree + derived context trail" "$M2" "chunking" \
"Build sections from parent_id where present; where null, derive trail from nearest active ordered heading (path_source = explicit_parent | derived_order). Compute page/offset ranges.
Depends on: M2-04, M1-03  |  Size: M  |  Branch: feature/m2-structural-tree
Done when: correct tree+trails; test covers a null-parent_id entry (the باقوم case).
$REF"

  mk "M2-06 · Navigational-vs-content classification" "$M2" "chunking" \
"Classify headings with no substantive body (volume labels, alphabet ranges, empty باب, dividers) as context nodes, not searchable chunks.
Depends on: M2-05  |  Size: S  |  Branch: feature/m2-nav-nodes
Done when: navigational nodes flagged, excluded from embedding, retained as parent context; tests.
$REF"

  mk "M2-07 · Content-role separation (body vs footnote)" "$M2" "chunking" \
"Emit content_role = body|footnote; never concatenate. Footnote chunks keep page linkage (body linkage only when marker reliable). Flag footnotes so downstream never auto-attributes to the author.
Depends on: M2-05  |  Size: M  |  Branch: feature/m2-content-roles
Done when: book with footnotes yields role-tagged linked chunks; books without unaffected; tests.
$REF"

  mk "M2-08 · Size & semantic policy (children)" "$M2" "chunking" \
"Config thresholds: nav->context only; entity <~128 keep atomic; coherent ~128-768 one child; >~768 split into ~384-512 paragraph-aligned children; ~64 overlap within-section only. Split priority: subheading>paragraph>Arabic sentence>token. Never cross into next section.
Depends on: M2-05, M2-03  |  Size: L  |  Branch: feature/m2-size-policy
Done when: synthetic long/short sections behave per policy; overlap never crosses sections; thresholds are config; tests.
$REF"

  mk "M2-09 · Short-fragment conditional merge" "$M2" "chunking" \
"Merge a sub-128 discursive fragment with adjacent sibling ONLY IF: neither a different named entity, same parent, same content_role, result under max, source order preserved. Keep original child offsets.
Depends on: M2-08  |  Size: S  |  Branch: feature/m2-merge-fragments
Done when: merges only under all conditions; named entries never merge; tests.
$REF"

  mk "M2-10 · Compact Arabic context header" "$M2" "chunking" \
"Prepend compact prefix (الكتاب/المؤلف/المسار/نوع المحتوى) to each child. Do NOT prepend full betaka_text. death year 99999 = unknown (omit from text, null in metadata). Store header separately for A/B (M6).
Depends on: M2-05  |  Size: S  |  Branch: feature/m2-context-header
Done when: stable header; 99999 handled; store-apart toggle; tests.
$REF"

  mk "M2-11 · Chunk & section models + metadata schema" "$M2" "chunking,data" \
"Finalize Section+Chunk models & DB columns: ids, book_id, title trail + path_source, author + death_hijri, category_id, book_type_label, part, page_id/offset range, content_role, boundary_source, confidence, source_text, retrieval_text, header, parent/child links.
Depends on: M2-08  |  Size: M  |  Branch: feature/m2-chunk-model
Done when: models + Alembic migration; tests.
$REF"

  mk "M2-12 · Optional heading-recovery candidates (measure, don't trust)" "$M2" "chunking" \
"Detect visible heading-like text absent from TOC and markup (Fatimah bint al-Khattab page). Record as candidates+confidence; do NOT split until precision measured (M6). Keep paragraph/max-size guards.
Depends on: M2-04  |  Size: M  |  Branch: feature/m2-heading-recovery
Done when: candidates recorded, never silently trusted; test on documented example.
$REF"

  mk "M2-13 · Per-book chunking orchestrator" "$M2" "chunking" \
"Compose: ordered stream -> boundary ladder -> structural tree/context -> role split -> size policy -> merge -> header -> models, into chunk_book(book).
Depends on: M2-04..M2-11  |  Size: M  |  Branch: feature/m2-orchestrator
Done when: end-to-end chunks for fixture pass golden snapshot; source text round-trips verbatim.
$REF"

  # ------------------------------------------------------------------ M3
  mk "M3-01 · Embedding provider interface" "$M3" "embeddings" \
"Abstract EmbeddingProvider (embed_documents, embed_query, dims, tokenizer, optional query_instruction recorded for eval). Fake in-memory provider for tests.
Depends on: M0-01  |  Size: S  |  Branch: feature/m3-embed-interface
Done when: interface + fake provider.
$REF"

  mk "M3-02 · Qwen3-Embedding-8B provider" "$M3" "embeddings" \
"Implement interface for Qwen/Qwen3-Embedding-8B (official query instruction/formatting; batching; device config).
Depends on: M3-01  |  Size: M  |  Branch: feature/m3-qwen3
Done when: embeds a batch; dims asserted; integration test skippable if weights absent in CI.
$REF"

  mk "M3-03 · BGE-M3 provider (dense + learned sparse)" "$M3" "embeddings" \
"Implement dense output; also expose BGE-M3 learned-sparse behind a flag for the M6 ablation. Batching + device config.
Depends on: M3-01  |  Size: M  |  Branch: feature/m3-bge-m3
Done when: dense embeds; learned-sparse retrievable; tests (real model skippable in CI).
$REF"

  mk "M3-04 · Qdrant collection schema (named vectors + payload)" "$M3" "embeddings,infra" \
"Collection with named dense vector(s) (dim per active model) + named sparse vector + payload for filter/citation fields and parent/child links. Cosine for dense.
Depends on: M0-05, M2-11  |  Size: M  |  Branch: feature/m3-qdrant-schema
Done when: collection created from config; upsert + dense NN + sparse query work in a test.
$REF"

  mk "M3-05 · Surface-form BM25 sparse arm" "$M3" "embeddings,retrieval" \
"Primary sparse index on lightly-normalized surface words + exact phrases (names/sects/titles precise) as Qdrant sparse vectors (IDF/BM25). Root expansion NOT in primary field.
Depends on: M3-04, M2-01  |  Size: M  |  Branch: feature/m3-bm25
Done when: exact-name query returns right chunk; test.
$REF"

  mk "M3-06 · Root-expansion field (separate, low-weight, gated)" "$M3" "embeddings,retrieval" \
"Root-normalized terms as a SEPARATE low-weight expansion field (via root dictionary), disabled by default; enabled only if it improves labeled retrieval (M6).
Depends on: M3-05, M3-11  |  Size: M  |  Branch: feature/m3-root-field
Done when: field builds and toggles; A/B hook ready; tests.
$REF"

  mk "M3-07 · Ingestion orchestrator (idempotent, resumable)" "$M3" "embeddings,data" \
"Per book: chunk -> dense-embed -> sparse-encode -> upsert to Qdrant; write source_text+provenance+metadata to Postgres. Idempotent, resumable, per-book progress, dry-run.
Depends on: M2-13, M3-02, M3-04, M3-05  |  Size: L  |  Branch: feature/m3-ingest
Done when: ingests fixture book fully; re-run doesn't duplicate; tests.
$REF"

  mk "M3-08 · Ingestion CLI" "$M3" "embeddings,api" \
"shamela-rag ingest --book <id> | --category <id> | --all, with --limit, --dry-run, --model.
Depends on: M3-07  |  Size: S  |  Branch: feature/m3-ingest-cli
Done when: ingests a single book; --help documented.
$REF"

  mk "M3-11 · Root-dictionary loader" "$M3" "data,embeddings" \
"Load _meta/root_dictionary.jsonl (1.95M entries) into a lookup for inflected-form -> root.
Depends on: M1-01  |  Size: M  |  Branch: feature/m3-root-dict
Done when: resolves known forms; perf/memory sane; tests on a sample.
$REF"

  # ------------------------------------------------------------------ M4
  mk "M4-00 · Query translation (EN→AR)" "$M4" "retrieval" \
"Detect language; translate English questions to Arabic on production retrieval path; preserve original for display; record translated form for eval parity.
Depends on: M0-02  |  Size: M  |  Branch: feature/m4-translate
Done when: English question translated before retrieval; Arabic passthrough; tests with fake translator.
$REF"

  mk "M4-01 · Dense retriever (Qdrant)" "$M4" "retrieval" \
"Embed query -> Qdrant dense NN with payload filters.
Depends on: M3-04, M3-02  |  Size: M  |  Branch: feature/m4-dense
Done when: ranked chunks for a query against the ingested book; test.
$REF"

  mk "M4-02 · Sparse retriever (Qdrant sparse / BM25)" "$M4" "retrieval" \
"Encode query to surface sparse representation -> Qdrant sparse search.
Depends on: M3-05  |  Size: M  |  Branch: feature/m4-sparse
Done when: exact-term queries (a scholar's name) return the right chunk; test.
$REF"

  mk "M4-03 · RRF fusion" "$M4" "retrieval,good-first-issue" \
"Reciprocal Rank Fusion over dense + sparse lists.
Depends on: M4-01, M4-02  |  Size: S  |  Branch: feature/m4-rrf
Done when: correct fused ranking on a hand-built example; unit test (no DB).
$REF"

  mk "M4-04 · Cross-encoder reranker" "$M4" "retrieval" \
"Reranker interface + multilingual cross-encoder (e.g. BGE-reranker-v2-m3); rerank top ~100 -> top ~10. Kept fixed across the M6 comparison.
Depends on: M4-03  |  Size: M  |  Branch: feature/m4-rerank
Done when: reranks a candidate list; fake reranker for tests; real one integration-tested (skippable in CI).
$REF"

  mk "M4-05 · Authority boost + ordering signals" "$M4" "retrieval" \
"Boost printed works over دروس مفرغة; optional ordering by death_hijri for debate-history questions.
Depends on: M4-04  |  Size: S  |  Branch: feature/m4-authority
Done when: boost changes ordering as expected on a fixture; test.
$REF"

  mk "M4-06 · Parent/neighbor context expansion" "$M4" "retrieval" \
"After a child matches, optionally expand to parent section or neighboring children (via Postgres provenance links) when the query needs it, without pulling unrelated siblings/entities.
Depends on: M4-04, M2-11  |  Size: M  |  Branch: feature/m4-expand
Done when: returns matched child + bounded context; never crosses into a different entity; test.
$REF"

  mk "M4-07 · Retrieval service (compose the pipeline)" "$M4" "retrieval" \
"One retrieve(question, k, filters) wiring translate -> dense+sparse -> RRF -> rerank -> boost -> expand, all configurable.
Depends on: M4-00, M4-05, M4-06  |  Size: M  |  Branch: feature/m4-service
Done when: final ranked, expanded, cite-ready passages for the three example questions; test.
$REF"

  # ------------------------------------------------------------------ M5
  mk "M5-01 · LLM generation provider interface" "$M5" "generation" \
"Abstract GenerationProvider (pluggable local/API per doc 09), streaming optional. Fake provider for tests.
Depends on: M0-01  |  Size: S  |  Branch: feature/m5-llm-interface
Done when: interface + fake provider.
$REF"

  mk "M5-02 · General Q&A prompt template" "$M5" "generation" \
"Enforce doc 07 §6 rules: cite every source, preserve disagreement, never present as an independent authority. Respect footnote flags (don't attribute editor notes to the author).
Depends on: M5-01  |  Size: M  |  Branch: feature/m5-prompt
Done when: renders with retrieved context; snapshot test; manual review on the examples.
$REF"

  mk "M5-03 · Answer assembly with citations" "$M5" "generation" \
"Post-process into answer + structured citations (book, author, page) mapped back to chunks; deflect when evidence is thin.
Depends on: M5-02, M4-07  |  Size: M  |  Branch: feature/m5-answer
Done when: answer + citations for a sample question; every citation resolves to a real chunk; test.
$REF"

  mk "M5-04 · General-module end-to-end service" "$M5" "generation,retrieval" \
"answer_general_question(q) composing retrieval + generation.
Depends on: M5-03  |  Size: M  |  Branch: feature/m5-e2e
Done when: cited answer end-to-end on the ingested book; smoke test.
$REF"

  # ------------------------------------------------------------------ M6
  mk "M6-01 · Golden evaluation dataset for the general module ⭐ ASSIGN TO COLLEAGUE" "$M6" "eval,docs" \
"Curate 50-150 questions with expected source passage(s) by book+page (+chunk once ingestion exists). Cover: scholar-on-scholar, author-on-movement, multi-party debate; exact scholar/narrator/sect/book names; semantic paraphrases; Arabic morphological variants; short bios + long discursive sections; multi-source disagreement; native-Arabic AND translated-English phrasings. JSONL per golden_eval_seed.jsonl; extend+document schema. NO CODING REQUIRED. Blocks M6-02/03/04.
Depends on: — (start immediately, in parallel)  |  Size: L  |  Branch: feature/m6-golden-set
$REF"

  mk "M6-02 · Retrieval metrics harness" "$M6" "eval" \
"Compute Recall@100 (pre-rerank), MRR@10 / nDCG@10, exact-entity retrieval accuracy, attribution/citation correctness, plus indexing throughput, query latency, vector/index storage. Per-question + aggregate.
Depends on: M6-01, M4-07  |  Size: M  |  Branch: feature/m6-metrics
Done when: report generated from golden set; unit tests on metric math.
$REF"

  mk "M6-03 · Three-stage model comparison (Qwen3-8B vs BGE-M3) ⭐ resolves ADR-002" "$M6" "eval,embeddings" \
"Same chunks/questions/limits/judgments; record each model's official query formatting. (1) Dense-only, reranker off. (2) Controlled hybrid: each dense + same surface BM25, same fusion. (3) BGE sparse ablation: BGE dense + BM25 vs + learned-sparse vs + both. Reranker fixed once introduced. Decide on retrieval quality first, then operational tradeoff. Only winner embeds full corpus.
Depends on: M6-02, M3-02, M3-03, M3-05  |  Size: L  |  Branch: feature/m6-model-ab
Done when: reproducible scripts + comparison table + written recommendation replacing ADR-002 shortlist.
$REF"

  mk "M6-04 · Chunk-size & context-prefix sweep" "$M6" "eval,chunking" \
"Sweep child target size and context-header length (repeated metadata can bias similarity), re-embed, re-measure; confirm/adjust M2-08/M2-10 defaults.
Depends on: M6-02  |  Size: M  |  Branch: feature/m6-sweeps
Done when: results table + recommendation committed.
$REF"

  mk "M6-05 · Single-book end-to-end smoke test (simple, early)" "$M6" "eval,test,good-first-issue" \
"One hand-written question against the ingested fixture book; assert known-correct passage is retrieved, expanded, cited. Runs in CI on a small fixture.
Depends on: M5-04  |  Size: S  |  Branch: feature/m6-smoke
Done when: deterministic smoke test passes and is wired into CI.
$REF"

  mk "M6-06 · Structural validation harness" "$M6" "eval,chunking,test" \
"Assert over a sample of books: every source char preserved or classified as ignored markup; ordering reproducible; every valid inline toc-N yields one boundary; toc-N -> shamela_title_id; chunks never cross sections (except approved merges); named entries never merged; overlap only within a section; footnotes distinguishable; ambiguous/missing boundaries counted.
Depends on: M2-13  |  Size: M  |  Branch: feature/m6-structural-validation
Done when: harness runs over >=20 books, reports zero hard violations (or lists them).
$REF"

  mk "M6-07 · (Optional) RAGAS faithfulness/context-relevance" "$M6" "eval" \
"Faithfulness + context-relevance scoring over a sample.
Depends on: M6-01, M5-04  |  Size: M  |  Branch: feature/m6-ragas
Done when: RAGAS report generated for the golden sample.
$REF"

  # ------------------------------------------------------------------ M7
  mk "M7-01 · FastAPI /ask endpoint" "$M7" "api" \
"POST /ask (accepts Arabic or English) -> cited answer; schemas; error handling; health check.
Depends on: M5-04  |  Size: M  |  Branch: feature/m7-api
Done when: returns a cited answer for a sample question; API test with fake generation provider.
$REF"

  mk "M7-02 · Citation response schema & formatting" "$M7" "api" \
"Stable JSON citation format (book, author, page, category, snippet, content_role); footnote citations clearly marked.
Depends on: M7-01  |  Size: S  |  Branch: feature/m7-citations
Done when: schema documented + validated; test.
$REF"

  mk "M7-03 · (Optional) minimal demo CLI" "$M7" "api,good-first-issue" \
"shamela-rag ask \"…\" prints answer + citations.
Depends on: M5-04  |  Size: S  |  Branch: feature/m7-cli-demo
Done when: works against the ingested book.
$REF"

  # ------------------------------------------------------------------ M8
  mk "M8-01 · Test-coverage gate in CI" "$M8" "ci,test" \
"Coverage reporting + minimum threshold on the 'test' check.
Depends on: M0-07  |  Size: S  |  Branch: feature/m8-coverage
Done when: CI fails below threshold; summary emitted.
$REF"

  mk "M8-02 · Ingestion smoke test in CI (tiny fixture)" "$M8" "ci,test" \
"Miniature fixture book (few pages + toc, incl. inline toc-N spans and a footnote) ingested end-to-end into Postgres + Qdrant service containers.
Depends on: M3-07  |  Size: M  |  Branch: feature/m8-ci-ingest
Done when: CI ingests the fixture and runs one retrieval assertion.
$REF"

  mk "M8-03 · Contributing & PR/branch docs" "$M8" "docs,ci" \
"CONTRIBUTING.md: branch model (main <- stable-testing <- develop <- feature/*), PR checklist, commit style, local Postgres+Qdrant setup. PR + issue templates.
Depends on: —  |  Size: S  |  Branch: feature/m8-contributing
Done when: docs + templates merged; referenced from README.
$REF"

  echo
  echo "Done. Created labels/milestones/issues on $REPO."
}

# ------------------------------------------------------------------------------- main
case "$MODE" in
  all)               create_labels; create_milestones; create_issues ;;
  --labels-only)     create_labels ;;
  --milestones-only) create_milestones ;;
  --issues-only)     create_issues ;;
  *) echo "Unknown mode: $MODE (use: all | --labels-only | --milestones-only | --issues-only)"; exit 1 ;;
esac
