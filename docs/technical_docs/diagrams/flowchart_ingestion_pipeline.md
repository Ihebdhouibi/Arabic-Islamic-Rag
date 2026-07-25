# Flowchart — Ingestion & Indexing Pipeline

> Part of the diagram set indexed in [../12_architectural_diagrams.md](../12_architectural_diagrams.md).
> Visualizes the 9 stages from
> [10_ingestion_and_indexing_pipeline.md §3](../10_ingestion_and_indexing_pipeline.md#3-pipeline-stages),
> with the two data findings from that document (§1–2) shown as explicit decision/branch points
> rather than buried in prose.

![Flowchart — Ingestion and Indexing Pipeline](flowchart_ingestion_pipeline.svg)

*Rendered from [`flowchart_ingestion_pipeline.mmd`](flowchart_ingestion_pipeline.mmd) — edit that
source and re-render if the pipeline design changes. Mermaid source reproduced below for inline
reference.*

```mermaid
flowchart TD
    A["Stage 0: Schema & ID reconciliation<br/>verify shamela_id/shamela_page_id resolution<br/>for every _meta table before any join logic"] --> B

    B["Stage 1: Metadata normalization<br/>load book_metadata.json → books table<br/>verify manifest.json SHA-256 checksums"] --> C

    C{"Stage 2: Genre + coverage routing<br/>category_id AND curated-coverage membership"}
    C -->|"category 06, in curated 10"| D1["HadithAtomicChunker"]
    C -->|"category 06, NOT in curated 10"| D2["TocAnchoredChunker (fallback)"]
    C -->|"category 03, in curated 9"| D3["VerseAnchoredChunker"]
    C -->|"category 03, NOT in curated 9"| D2
    C -->|"aqidah/prose w/ footnotes"| D4["DualStreamChunker"]
    C -->|"poetry"| D5["PoetryAtomicChunker"]
    C -->|"everything else"| D2

    D1 --> E
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E

    E["Stage 3: Chunking<br/>per-genre chunker applied,<br/>secondary split if a section is too large"] --> F

    F["Stage 4: Contextual prefixing<br/>cheap metadata breadcrumb by default;<br/>LLM-generated context for fiqh/ambiguous genres"] --> G

    G["Stage 5: Embedding<br/>batch-embed via chosen model (docs 03/09)<br/>content-hash each chunk for idempotent re-runs"] --> H

    G --> I

    H["Stage 6: Lexical / root-normalized index<br/>BM25 + root_dictionary-normalized field"] --> J
    I["Stage 7: Graph loading<br/>isnad edges, hadith concordance, verse links<br/>via resolved shamela_id/shamela_page_id"] --> J

    J["Stage 8: Validation / QA<br/>row-count reconciliation vs manifests<br/>golden-set spot check (doc 06)"]

    J -->|"pass"| K(["Ready for query-time retrieval"])
    J -->|"fail"| A
```

## Reading notes

- **Stage 2 is drawn as a decision diamond, not a straight arrow**, because it's the stage where
  [10 §1](../10_ingestion_and_indexing_pipeline.md#1-finding-curated-graph-coverage-is-narrower-than-category-boundaries-suggest)'s
  finding actually gets enforced: category membership alone is not sufficient to pick
  `HadithAtomicChunker`/`VerseAnchoredChunker` — a book also has to be in the resolved
  curated-coverage set, or it falls back to `TocAnchoredChunker` like any general book.
- **Stages 5 (embedding) and 7 (graph loading) run independently off Stage 4/3** — nothing about
  loading the curated graph tables depends on embeddings existing yet, so these can run in
  parallel rather than strictly sequentially, both feeding into the shared Stage 8 validation.
- **Stage 8 loops back to Stage 0 on failure**, not forward — per
  [10 §6](../10_ingestion_and_indexing_pipeline.md#6-what-was-actually-verified-vs-still-assumed),
  the kind of failure this pipeline is specifically designed to catch (a silent join mismatch like
  the `shamela_id` gotcha) traces back to a Stage 0 assumption that needs re-checking, not a
  downstream bug to patch locally.
