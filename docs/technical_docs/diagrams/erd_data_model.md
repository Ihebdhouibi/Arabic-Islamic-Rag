# ERD — Shamela4 Data Model

> Part of the diagram set indexed in [../12_architectural_diagrams.md](../12_architectural_diagrams.md).
> Grounded in the actual schemas inspected across docs 05, 10, and 11 — including the
> `shamela_id`/`shamela_page_id` join gotcha from
> [10_ingestion_and_indexing_pipeline.md §2](../10_ingestion_and_indexing_pipeline.md#2-finding-hadith_xrefs-and-tafsir_xrefs-key-on-a-different-id-space-than-the-rest-of-the-corpus),
> baked directly into the field comments below so it can't be missed by whoever builds against
> this diagram.

![ERD — Shamela4 Data Model](erd_data_model.svg)

*Rendered from [`erd_data_model.mmd`](erd_data_model.mmd) — edit that source and re-render if
the schema understanding changes. Mermaid source reproduced below for inline reference.*

```mermaid
erDiagram
    CATEGORY ||--o{ BOOK : "categorizes"
    AUTHOR ||--o{ BOOK : "writes (main_author_id)"
    BOOK ||--o{ PAGE : "contains"
    BOOK ||--o{ TOC_ENTRY : "contains"
    BOOK |o--o{ BOOK : "parent_id (multi-volume set)"
    TOC_ENTRY |o--o{ TOC_ENTRY : "parent_id (heading hierarchy)"
    TOC_ENTRY }o--|| PAGE : "anchors at page_id"
    PAGE ||--o{ PAGE_ISNAD : "isnad occurs on"
    PAGE_ISNAD }o--o{ NARRATOR : "narrator_ids (ordered chain)"
    BOOK ||--o{ HADITH_XREF : "resolved via shamela_id"
    HADITH_XREF }o--|| PAGE : "resolved via shamela_page_id"
    BOOK ||--o{ TAFSIR_XREF : "resolved via shamela_id"
    TAFSIR_XREF }o--|| PAGE : "resolved via shamela_page_id"
    TAFSIR_XREF }o--|| QURAN_VERSE : "key_id identifies verse"

    CATEGORY {
        int category_id PK
        string category_name_ar
    }
    AUTHOR {
        int author_id PK
        string name_ar
        int death_hijri "hijri year; negative = CE per dataset notes"
    }
    BOOK {
        int book_id PK "internal sequential id - used by pages, toc, page_isnads"
        int shamela_id "original Shamela numbering - THIS is what hadith_xrefs/tafsir_xrefs call book_id"
        string title_ar
        int category_id FK
        int main_author_id FK
        int book_type
        string book_type_label
        bool printed
        bool has_multi_part
        int parent_id FK "multi-volume parent, nullable"
    }
    PAGE {
        int page_id PK "internal sequential id"
        int book_id FK
        int shamela_page_id "original Shamela page numbering - THIS is what hadith_xrefs/tafsir_xrefs call page_id"
        string part "volume/part label, nullable"
        int page_num
        int sequence_num
        text body "may contain minimal HTML + Quranic quotes bracketed with the Arabic ornate parentheses"
        text footnotes "parallel commentary stream for matn/sharh books, nullable"
    }
    TOC_ENTRY {
        int title_id PK
        int book_id FK
        int page_id FK "page where heading appears"
        int parent_id FK "heading hierarchy, nullable = root"
        string title_text
    }
    NARRATOR {
        int id PK
        string short_name
        string long_name
        int narrator_type
        text biography_text "JSON-encoded structured fields (kunya, nasab, death date, tabaqa...)"
        text criticism_text "JSON-encoded jarh-wa-tadil quotes with source citations"
    }
    PAGE_ISNAD {
        int book_id FK "internal book_id - correct here, unlike hadith_xrefs"
        int shamela_page_id FK
        int_array narrator_ids "ordered chain, each element FK-many to NARRATOR.id"
    }
    HADITH_XREF {
        int id PK
        int key_id "groups rows = same underlying hadith across different books"
        int book_id "MISLEADING NAME: actually BOOK.shamela_id"
        int page_id "MISLEADING NAME: actually PAGE.shamela_page_id, scoped to the resolved book"
    }
    TAFSIR_XREF {
        int id PK
        int key_id "= QURAN_VERSE.id"
        int book_id "MISLEADING NAME: actually BOOK.shamela_id"
        int page_id "MISLEADING NAME: actually PAGE.shamela_page_id, scoped to the resolved book"
    }
    QURAN_VERSE {
        int id PK
        string body "primary diacritized text"
        string majma "alternate script variant"
        string amiri "alternate script variant"
    }
```

## Notes that don't fit cleanly into ER notation

- **`HADITH_XREF.key_id` is a grouping value, not a foreign key to a separate table.** Rows sharing
  the same `key_id` represent the same underlying hadith appearing in different books — the
  "SAME_HADITH_AS" edge from
  [08_fig2_knowledge_graph_structure.svg](../08_fig2_knowledge_graph_structure.svg) is built by
  grouping `HADITH_XREF` rows by `key_id` after resolving the book/page IDs, not by joining to
  another table.
- **`ROOT_DICTIONARY` (token → triliteral root) is intentionally omitted above.** It's a flat
  lookup table (`token`, `roots[]`) consulted at index/query time for root-normalized lexical
  search (per
  [03_embeddings_and_vector_stores.md §5](../03_embeddings_and_vector_stores.md#5-sparse-and-hybrid-representations)),
  not a relationship to any specific page or book — including it as an ER relationship would
  misrepresent a lexical lookup as a real join.
- **Curated coverage is a subset, not a property visible in this schema.** `PAGE_ISNAD` only has
  rows for 10 books and `TAFSIR_XREF` only for 9, out of 1,241 and 273 candidates respectively
  (per [10_ingestion_and_indexing_pipeline.md §1](../10_ingestion_and_indexing_pipeline.md#1-finding-curated-graph-coverage-is-narrower-than-category-boundaries-suggest)).
  The ER diagram shows what the relationship *looks like where it exists* — it doesn't show that
  it's sparse. Don't assume a `BOOK` row has isnad/tafsir coverage just because its category
  suggests it should.
