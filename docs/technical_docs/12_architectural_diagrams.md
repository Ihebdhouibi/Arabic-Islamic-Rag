# 12 — Architectural Diagrams

> Companion to docs 05, 07, and 10. Those documents describe the data model, retrieval
> architecture, and ingestion pipeline in prose and tables; this document points to the formal
> diagram set that renders the same design visually — an ER diagram, a UML class diagram, an
> ingestion flowchart, and a query-time sequence diagram, all under
> [diagrams/](diagrams/README.md).

## Why these four, and why Mermaid

Doc 08 already covers *conceptual* illustrations (vector-store vs. knowledge-graph explanatory
figures, as standalone hand-drawn SVGs). This document's diagrams are a different kind:
**structural, implementation-facing diagrams** with standard notations — entity-relationship, UML
class, flowchart, sequence — that have purpose-built syntax in
[Mermaid](https://mermaid.js.org/). Hand-drawing these as SVG the way doc 08's figures were would
mean manually computing crow's-foot notation and UML class-box layout by hand for no benefit;
Mermaid's dedicated diagram types exist precisely so that doesn't need to happen.

Each diagram exists as three artifacts under [diagrams/](diagrams/README.md): a `.mmd` file (the
Mermaid source — plain text, diffable, the thing to edit when the design changes), a rendered
`.svg` image (generated from that source via Mermaid's own engine, not hand-drawn — an actual
picture to open or embed, per the request that prompted this doc), and a `.md` file that embeds
the image and carries the explanatory cross-references back to docs 02/05/07/09/10. Source and
rendered artifact are kept in sync manually for now — re-render the `.svg` from the `.mmd` after
any edit, the same discipline as keeping a diagram and its description from drifting apart in any
design doc.

## The four diagrams

- **[ERD — Shamela4 Data Model](diagrams/erd_data_model.md).** The actual schema — books, pages,
  TOC, narrators, isnad links, cross-reference tables, Quran verses — grounded in the fields
  verified across docs 05/10/11, with the `shamela_id`/`shamela_page_id` join gotcha from
  [10 §2](10_ingestion_and_indexing_pipeline.md#2-finding-hadith_xrefs-and-tafsir_xrefs-key-on-a-different-id-space-than-the-rest-of-the-corpus)
  written directly into the field comments so it can't be missed by whoever builds against it.
- **[Class Diagram — Proposed System Architecture](diagrams/class_diagram_system_architecture.md).**
  Translates the prose architecture from docs 02, 05, 07, and 09 into concrete classes and
  interfaces — the chunker hierarchy, the four retrieval-strategy implementations, the
  graph-augmented enrichment step, and the swappable `LLMClient` interface behind the
  router/generation split.
- **[Flowchart — Ingestion & Indexing Pipeline](diagrams/flowchart_ingestion_pipeline.md).** The
  9 stages from [doc 10 §3](10_ingestion_and_indexing_pipeline.md#3-pipeline-stages), with genre
  and curated-coverage routing drawn as an explicit decision point rather than assumed.
- **[Sequence Diagram — Query-Time Lifecycle](diagrams/sequence_query_lifecycle.md).** One
  question's time-ordered path through the query router, one of the four retrieval paths from
  [doc 07 §5](07_recommended_architecture_for_shamela_rag.md#5-query-routed-retrieval-one-path-per-use-case),
  and the shared reranking/enrichment/generation steps that follow regardless of path.

## What these diagrams are (and aren't)

These are **design sketches synthesized from the prior nine documents, not committed code or a
locked schema**. Class and method names are illustrative of responsibilities, not a frozen API;
the ER diagram documents the schema as inspected, not a schema this project has authored or can
change. They exist to make the architecture reviewable as a picture before implementation starts,
and to surface inconsistencies that prose alone tends to hide — drawing the class diagram, for
instance, is what made concrete exactly which four classes implement `RetrievalStrategy` and why
`GraphAugmentedEnricher` has to sit downstream of all of them, not just the graph-native two.

All four should be revisited once the open decisions from
[doc 07 §8](07_recommended_architecture_for_shamela_rag.md#8-open-decisions-before-implementation-begins)
(graph store choice, embedding model, router implementation, chunk sizing, rollout sequencing) are
actually resolved — at that point these diagrams stop being sketches and start being documentation
of real decisions.
