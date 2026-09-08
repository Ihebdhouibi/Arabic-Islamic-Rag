# Sizing and latency benchmark

Measured on a **reduced ~10-book sample** (not the issue’s full 50), with OpenRouter
Qwen3-Embedding-8B dense vectors. Full-corpus figures are projections from the measured
per-page ratios below, against 8,589 books and 7,611,186 pages.

## Sample ingested

- Books: **10** (complete Postgres + Qdrant)
- Pages (distinct book/page pairs in chunks): **20,684**
- Chunk rows: **30,317** (includes some Postgres-only leftover chunks; see Notes)
- Qdrant points: **26,857** (dim 4096)

## Measured storage

| Store | Measured | Per page | Per book |
| --- | ---: | ---: | ---: |
| Postgres (database) | 0.072 GB | 3.6 KB | 5363.4 KB |
| Qdrant (storage dir) | 6.577 GB | 333.4 KB | 492596.6 KB |

### Postgres by table

| Table | Size |
| --- | ---: |
| chunks | 60.9 MB |
| sections | 4.4 MB |
| books | 64.0 KB |
| alembic_version | 24.0 KB |

## Full-corpus projection

Each row scales the measured value by its own per-page ratio, so the arithmetic is
visible and re-checkable: `projected = measured / sample_pages * 7,611,186`.

| Quantity | Measured | Per page | Projected (full corpus) |
| --- | ---: | ---: | ---: |
| Postgres database (GB) | 0.072 | 0.000003 | 26.4 |
| Chunk rows | 30,317.000 | 1.465722 | 11,155,885.0 |
| Qdrant points | 26,857.000 | 1.298443 | 9,882,693.0 |
| Qdrant on disk (GB) | 6.577 | 0.000318 | 2,420.1 |

## Retrieval latency

11 timed queries (warm-up excluded), budget 8,000 ms.

| Stage | p50 (ms) | p95 (ms) | mean (ms) |
| --- | ---: | ---: | ---: |
| translate | 0 | 0 | 0 |
| dense | 2,891 | 9,983 | 5,010 |
| sparse | 30 | 32 | 27 |
| fuse | 0 | 0 | 0 |
| hydrate | 31 | 31 | 28 |
| rerank | 175,405 | 186,343 | 162,838 |
| expand | 203 | 297 | 227 |
| **total** | **180,578** | **196,640** | **168,132** |

p95 total is **196,640 ms**, OVER the 8,000 ms budget.

## Notes

- Target sample: **10 fully ingested books** (dense via **OpenRouter + Qwen3-Embedding-8B**,
  4096-d), agreed with Iheb instead of a full 50-book overnight run.
- Measured on a **16 GB RAM laptop**; cross-encoder rerank ran on **CPU** (dominant latency).
- This measurement pass used `--skip-ingest` after ingest; latency timed hybrid retrieve
  (dense + BM25 sparse + local rerank). Arabic queries → translate stage ~0 ms.
- Latency p95 (~197s) is far over the 8s budget mainly because of CPU rerank, not OpenRouter dense.
- **Caveats (not fake, but biased):** one large book (~16k of 27k Qdrant points) dominates the
  sample, so full-corpus Qdrant GB is an upper-leaning extrapolation; Postgres page/chunk totals
  still include a few incomplete leftover rows without Qdrant points; per-book KB used the raw
  Postgres book-row count at measure time. Treat projections as order-of-magnitude, not final HW buy.
