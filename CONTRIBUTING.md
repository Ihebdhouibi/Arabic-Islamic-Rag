# Contributing to Arabic-Islamic-Rag

Thanks for contributing. This guide covers the branch model, commit style, local setup, and the
checklist every change must pass.

## Branch model

```
main  <-  stable-testing  <-  develop  <-  feature/*  |  chore/*
```

- `main` — protected, release.
- `stable-testing` — release candidate.
- `develop` — integration branch; all feature work merges here.
- `feature/*` / `chore/*` — one branch per issue.

Rules:

- One issue = one branch = one PR into `develop`. Squash-merge.
- No direct commits to protected branches (`main`, `stable-testing`, `develop`).
- Use the terminal `git` CLI for version control.

## Commit style

Conventional commits, enforced by a `commit-msg` hook. Allowed types:

`feat`, `chore`, `docs`, `tests`, `bug`, `ci`

Format: `<type>: <short imperative subject>`. Add a body only when it adds information.

Two hard rules for all commits, code, comments, PRs, and docs:

- **No AI attribution** (no "Generated with ...", "Co-authored-by: ...", etc.). Describe the change only.
- **No emojis** anywhere.

## Local setup

Requires Python 3.11 and Docker.

```bash
# 1. Install the package with dev extras
pip install -e ".[dev]"

# 2. Install the git hooks (pre-commit + commit-msg)
pre-commit install && pre-commit install --hook-type commit-msg

# 3. Start Postgres + Qdrant (compose maps host ports 5433 and 6333)
docker compose up -d --wait

# 4. Apply migrations and run the tests
alembic upgrade head
pytest
```

Configuration is read from environment variables (prefix `SHAMELA_`) or a local `.env`; defaults
match the docker-compose profile, so no `.env` is required for local dev.

Integration tests that need Postgres or Qdrant skip automatically when those services are
unreachable; they run in CI where the service containers are up.

## Pull request checklist

Before opening a PR into `develop`:

- [ ] Branch is `feature/*` or `chore/*` off `develop`, scoped to a single issue.
- [ ] `pre-commit run --all-files` passes (ruff lint + format, hygiene hooks, conventional commit).
- [ ] `mypy src` is clean and `pytest` passes locally.
- [ ] New/changed behavior is covered by tests; coverage stays above the CI gate (85%).
- [ ] The PR description links its issue (e.g. `Closes #123`).
- [ ] Source corpus text still round-trips verbatim through the chunker (hard gate).

## Definition of done

Code + tests, pre-commit clean, CI green, PR links its issue, reviewed and squash-merged into
`develop`.
