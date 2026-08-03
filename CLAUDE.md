# CLAUDE.md — Working agreement for Arabic-Islamic-Rag

Guidance for AI coding assistants (Claude Code, Copilot, etc.) and human contributors in this repo.
Read this before making changes.

## Project

RAG / GraphRAG over the Shamela4 classical-Arabic corpus. We are currently implementing the
**general question-answering module**.

- Design docs: `docs/technical_docs/`
- Implementation plan + issue backlog: `docs/implementation/`
- Confirmed chunking mechanism: `review_general_module_chunking_embeddings_brief.md`

Stack: Python; PostgreSQL (relational / provenance) + Qdrant (vectors); embeddings
Qwen3-Embedding-8B / BGE-M3 (chosen by measurement); hybrid dense + surface-BM25 retrieval;
FastAPI.

## Rules (must follow)

1. **No AI attribution.** Never add "Generated with Claude", "Co-authored-by: Claude", Copilot, or
   any similar mention in commit messages, PR descriptions, or code. Describe the change only.
2. **No emojis.** Anywhere: code, comments, commit messages, PR descriptions, docs.
3. **Always end a task with a summary**, covering: What was done, Why, What to expect, and Edge
   cases to handle.
4. **Commit convention:** `<type>: <short imperative subject>`, using only these types:
   `feat`, `chore`, `docs`, `tests`, `bug`. Keep the subject short; add a body only when needed.
5. **Pre-commit must pass** before every commit (ruff lint + format, hygiene hooks, conventional
   commit-msg). Do not bypass with `--no-verify`.

## Branching & PRs

- Model: `main` (protected, release) <- `stable-testing` (release candidate) <- `develop`
  (integration) <- `feature/*` or `chore/*` (one branch per issue).
- One issue = one branch = one PR into `develop`. Squash-merge. No direct commits to protected
  branches.
- Use the terminal git CLI for all version-control operations.

## Definition of done

Code + tests, pre-commit clean, CI green, PR links its issue, reviewed. The corpus source text must
round-trip verbatim through the chunker (a hard validation gate).
