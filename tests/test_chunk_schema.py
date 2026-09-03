from __future__ import annotations

from shamela_rag.db.models import Base

_SECTION_COLUMNS = {
    "shamela_title_id",
    "title_trail",
    "depth",
    "path_source",
    "boundary_source",
    "confidence",
    "start_page_id",
    "end_page_id",
}

_CHUNK_COLUMNS = {
    "content_role",
    "source_text",
    "retrieval_text",
    "context_header",
    "part",
    "start_page_id",
    "end_page_id",
    "start_page_num",
    "end_page_num",
    "start_offset",
    "end_offset",
    "token_count",
}


_BOOK_COLUMNS = {
    "title_ar",
    "author_name_ar",
    "author_death_hijri",
    "category_id",
    "category_name_ar",
    "book_type_label",
}


def test_book_has_category_name_column() -> None:
    columns = set(Base.metadata.tables["books"].columns.keys())
    assert columns >= _BOOK_COLUMNS


def test_section_has_provenance_columns() -> None:
    columns = set(Base.metadata.tables["sections"].columns.keys())
    assert columns >= _SECTION_COLUMNS


def test_chunk_has_placement_and_text_columns() -> None:
    columns = set(Base.metadata.tables["chunks"].columns.keys())
    assert columns >= _CHUNK_COLUMNS


def test_source_and_retrieval_and_header_are_distinct_columns() -> None:
    columns = Base.metadata.tables["chunks"].columns
    assert "source_text" in columns
    assert "retrieval_text" in columns
    assert "context_header" in columns
