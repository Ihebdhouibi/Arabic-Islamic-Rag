# Architectural Diagrams

Formal structural diagrams for the Shamela RAG system. Each diagram exists as three files: a
rendered **`.svg` image** (open it directly for the picture), a **`.mmd` Mermaid source** file
(the editable source of truth), and a **`.md`** file with explanatory prose and cross-references
back to the docs the diagram is derived from — the `.md` embeds the `.svg` inline too, so you
don't need to open the image separately when reading the writeup. See
[../12_architectural_diagrams.md](../12_architectural_diagrams.md) for the narrative context
tying these together; this file is just the index.

| Diagram | Image | Source | Type | Covers |
|---|---|---|---|---|
| ERD — Data Model | [erd_data_model.svg](erd_data_model.svg) | [.mmd](erd_data_model.mmd) / [.md](erd_data_model.md) | Entity-relationship | The actual data schema — books, pages, TOC, narrators, isnad, cross-references, Quran verses — including the `shamela_id`/`shamela_page_id` join gotcha baked into the field comments |
| Class Diagram — System Architecture | [class_diagram_system_architecture.svg](class_diagram_system_architecture.svg) | [.mmd](class_diagram_system_architecture.mmd) / [.md](class_diagram_system_architecture.md) | UML class diagram | The proposed software structure — chunkers, stores, retrieval strategies, generation — translating docs 02/05/07/09's prose into concrete classes and interfaces |
| Flowchart — Ingestion Pipeline | [flowchart_ingestion_pipeline.svg](flowchart_ingestion_pipeline.svg) | [.mmd](flowchart_ingestion_pipeline.mmd) / [.md](flowchart_ingestion_pipeline.md) | Flowchart | The 9-stage ingestion pipeline from doc 10, with genre/coverage routing shown as an explicit decision point |
| Sequence Diagram — Query Lifecycle | [sequence_query_lifecycle.svg](sequence_query_lifecycle.svg) | [.mmd](sequence_query_lifecycle.mmd) / [.md](sequence_query_lifecycle.md) | Sequence diagram | One question's time-ordered path through the router, one of the four retrieval strategies, enrichment, and generation |

The `.svg` files were rendered from the `.mmd` sources via Mermaid's own engine (parse-validated
against `mermaid.parse()`, then rendered), not hand-drawn — if the design changes, edit the
`.mmd` file and re-render rather than editing the `.svg` directly.

These are design sketches, not committed code or a locked schema — they'll need revisiting once
the open decisions in [doc 07 §8](../07_recommended_architecture_for_shamela_rag.md#8-open-decisions-before-implementation-begins)
are actually resolved.
