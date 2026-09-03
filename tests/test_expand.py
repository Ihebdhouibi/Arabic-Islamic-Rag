from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.chunking.orchestrator import BookChunk, chunk_book
from shamela_rag.chunking.sections import Section as ChunkSection
from shamela_rag.chunking.sections import build_sections
from shamela_rag.chunking.sizing import SizePolicy
from shamela_rag.data.models import load_book, load_toc
from shamela_rag.db.engine import get_engine, get_sessionmaker
from shamela_rag.db.models import Base, Book, Chunk, Section
from shamela_rag.retrieval.expand import (
    ChunkNotFoundError,
    ContextExpander,
    ExpandMode,
    ExpansionConfig,
)
from shamela_rag.retrieval.rerank import RerankedChunk

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "book_1021"
_FIXTURE_BOOK_ID = 1021
_SYNTH_BOOK_ID = 9_001
_FIXTURE_SPLIT_POLICY = SizePolicy(
    min_tokens=2, max_tokens=12, split_target_tokens=8, overlap_tokens=0
)


@dataclass(frozen=True)
class _SynthChunk:
    section_key: str
    content_role: str
    source_text: str
    page_id: int
    start_offset: int
    context_header: str = ""


def _hit(chunk_id: int, *, score: float = 1.0) -> RerankedChunk:
    return RerankedChunk(chunk_id=chunk_id, score=score, text="", payload={})


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = get_engine()
    try:
        with eng.connect():
            pass
    except Exception:  # noqa: BLE001
        pytest.skip("Postgres not reachable")
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        with Session(eng) as session, session.begin():
            session.execute(
                delete(Chunk).where(Chunk.book_id.in_([_SYNTH_BOOK_ID, _FIXTURE_BOOK_ID]))
            )
            session.execute(
                delete(Section).where(Section.book_id.in_([_SYNTH_BOOK_ID, _FIXTURE_BOOK_ID]))
            )
            session.execute(
                delete(Book).where(Book.book_id.in_([_SYNTH_BOOK_ID, _FIXTURE_BOOK_ID]))
            )


@pytest.fixture
def expander(engine: Engine) -> ContextExpander:
    return ContextExpander(get_sessionmaker(engine))


def _insert_synthetic_book(
    session: Session,
    *,
    chunks: list[_SynthChunk],
    sections: dict[str, str],
) -> dict[str, int]:
    session.add(Book(book_id=_SYNTH_BOOK_ID, title_ar="Synthetic"))
    session.flush()

    section_ids: dict[str, int] = {}
    for key, title in sections.items():
        row = Section(book_id=_SYNTH_BOOK_ID, title_text=title, title_trail=title, depth=0)
        session.add(row)
        session.flush()
        section_ids[key] = row.id

    chunk_ids: list[int] = []
    for spec in chunks:
        row = Chunk(
            book_id=_SYNTH_BOOK_ID,
            section_id=section_ids[spec.section_key],
            content_role=spec.content_role,
            source_text=spec.source_text,
            retrieval_text=spec.source_text,
            context_header=spec.context_header,
            start_page_id=spec.page_id,
            end_page_id=spec.page_id,
            start_offset=spec.start_offset,
            end_offset=spec.start_offset + len(spec.source_text),
            token_count=len(spec.source_text.split()),
        )
        session.add(row)
        session.flush()
        chunk_ids.append(row.id)
    return section_ids


def _seed_three_part_biography(session: Session) -> tuple[int, int, int, int, int]:
    _insert_synthetic_book(
        session,
        sections={"bio_a": "Biography A", "bio_b": "Biography B"},
        chunks=[
            _SynthChunk("bio_a", "body", "alpha one", 1, 0, "header A"),
            _SynthChunk("bio_a", "body", "alpha two", 1, 20, "header A"),
            _SynthChunk("bio_a", "body", "alpha three", 1, 40, "header A"),
            _SynthChunk("bio_b", "body", "beta one", 2, 0, "header B"),
            _SynthChunk("bio_b", "body", "beta two", 2, 20, "header B"),
        ],
    )
    rows = session.execute(
        select(Chunk.id, Chunk.source_text)
        .where(Chunk.book_id == _SYNTH_BOOK_ID)
        .order_by(Chunk.start_page_id, Chunk.start_offset, Chunk.id)
    ).all()
    by_text = {text: chunk_id for chunk_id, text in rows}
    return (
        by_text["alpha one"],
        by_text["alpha two"],
        by_text["alpha three"],
        by_text["beta one"],
        by_text["beta two"],
    )


def test_expand_empty_hits_returns_empty_list(expander: ContextExpander) -> None:
    assert expander.expand([]) == []


def test_expand_neighbor_window_includes_siblings_only(
    expander: ContextExpander, engine: Engine
) -> None:
    with Session(engine) as session, session.begin():
        first, middle, third, beta_one, _beta_two = _seed_three_part_biography(session)

    passages = expander.expand(
        [_hit(middle)],
        config=ExpansionConfig(mode=ExpandMode.NEIGHBORS, neighbor_window=1),
    )

    assert len(passages) == 1
    passage = passages[0]
    assert passage.chunk_ids == (first, middle, third)
    assert passage.parts[1].is_hit
    assert "alpha one" in passage.text
    assert "alpha three" in passage.text
    assert "beta one" not in passage.text
    assert "beta two" not in passage.text
    assert passage.text.startswith("header A")
    assert beta_one not in passage.chunk_ids


def test_expand_never_crosses_entity_sections(expander: ContextExpander, engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        first, middle, third, beta_one, beta_two = _seed_three_part_biography(session)

    passage = expander.expand(
        [_hit(beta_one)],
        config=ExpansionConfig(mode=ExpandMode.FULL_SECTION),
    )[0]

    assert passage.chunk_ids == (beta_one, beta_two)
    assert first not in passage.chunk_ids
    assert middle not in passage.chunk_ids
    assert third not in passage.chunk_ids


def test_expand_full_section_returns_all_siblings_in_order(
    expander: ContextExpander, engine: Engine
) -> None:
    with Session(engine) as session, session.begin():
        first, middle, third, _, _ = _seed_three_part_biography(session)

    passage = expander.expand(
        [_hit(middle)],
        config=ExpansionConfig(mode=ExpandMode.FULL_SECTION),
    )[0]

    assert passage.chunk_ids == (first, middle, third)
    assert (
        passage.text.index("alpha one")
        < passage.text.index("alpha two")
        < passage.text.index("alpha three")
    )


def test_expand_respects_token_cap_while_keeping_hit(
    expander: ContextExpander, engine: Engine
) -> None:
    with Session(engine) as session, session.begin():
        _insert_synthetic_book(
            session,
            sections={"long": "Long entry"},
            chunks=[
                _SynthChunk("long", "body", "one two three four five six", 1, 0),
                _SynthChunk("long", "body", "seven eight nine ten eleven twelve", 1, 100),
                _SynthChunk("long", "body", "thirteen fourteen fifteen sixteen", 1, 200),
            ],
        )
        rows = session.execute(
            select(Chunk.id).where(Chunk.book_id == _SYNTH_BOOK_ID).order_by(Chunk.start_offset)
        ).scalars()
        chunk_ids = list(rows)

    passage = expander.expand(
        [_hit(chunk_ids[1])],
        config=ExpansionConfig(mode=ExpandMode.FULL_SECTION, max_expanded_tokens=8),
    )[0]

    assert passage.hit_chunk_id == chunk_ids[1]
    assert chunk_ids[1] in passage.chunk_ids
    assert len(passage.chunk_ids) == 1


def test_expand_body_does_not_pull_footnote_sibling(
    expander: ContextExpander, engine: Engine
) -> None:
    with Session(engine) as session, session.begin():
        _insert_synthetic_book(
            session,
            sections={"entry": "Entry"},
            chunks=[
                _SynthChunk("entry", "body", "author prose", 1, 0),
                _SynthChunk("entry", ContentRole.FOOTNOTE.value, "editor note", 1, 50),
            ],
        )

    with Session(engine) as session:
        rows = session.execute(
            select(Chunk.id, Chunk.content_role).where(Chunk.book_id == _SYNTH_BOOK_ID)
        ).all()
        by_role = {role: chunk_id for chunk_id, role in rows}

    body_passage = expander.expand([_hit(by_role["body"])])[0]
    footnote_passage = expander.expand([_hit(by_role["footnote"])])[0]

    assert body_passage.chunk_ids == (by_role["body"],)
    assert footnote_passage.chunk_ids == (by_role["footnote"],)
    assert "editor note" not in body_passage.text
    assert "author prose" not in footnote_passage.text


def test_expand_none_mode_returns_hit_only(expander: ContextExpander, engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        first, middle, third, _, _ = _seed_three_part_biography(session)

    passage = expander.expand(
        [_hit(middle)],
        config=ExpansionConfig(mode=ExpandMode.NONE),
    )[0]

    assert passage.chunk_ids == (middle,)


def test_expand_null_section_id_is_atomic(expander: ContextExpander, engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        session.add(Book(book_id=_SYNTH_BOOK_ID, title_ar="Synthetic"))
        session.flush()
        row = Chunk(
            book_id=_SYNTH_BOOK_ID,
            section_id=None,
            content_role="body",
            source_text="standalone",
            retrieval_text="standalone",
        )
        session.add(row)
        session.flush()
        chunk_id = row.id

    passage = expander.expand([_hit(chunk_id)])[0]
    assert passage.chunk_ids == (chunk_id,)
    assert passage.text == "standalone"


def test_expand_missing_chunk_raises(expander: ContextExpander) -> None:
    with pytest.raises(ChunkNotFoundError, match="999999"):
        expander.expand([_hit(999_999)])


def test_expand_rejects_invalid_config(expander: ContextExpander) -> None:
    with pytest.raises(ValueError, match="neighbor_window"):
        expander.expand([_hit(1)], config=ExpansionConfig(neighbor_window=-1))
    with pytest.raises(ValueError, match="max_expanded_tokens"):
        expander.expand([_hit(1)], config=ExpansionConfig(max_expanded_tokens=0))


def test_expand_multiple_hits_preserves_order(expander: ContextExpander, engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        first, middle, third, beta_one, _ = _seed_three_part_biography(session)

    passages = expander.expand(
        [_hit(third, score=0.5), _hit(beta_one, score=0.9)],
        config=ExpansionConfig(neighbor_window=0),
    )

    assert [p.hit_chunk_id for p in passages] == [third, beta_one]
    assert passages[0].chunk_ids == (third,)
    assert passages[1].chunk_ids == (beta_one,)


@pytest.fixture
def seeded_fixture_book(engine: Engine) -> None:
    book_meta = load_book(_FIXTURE_DIR)
    sections = build_sections(list(load_toc(_FIXTURE_DIR)))
    chunks = chunk_book(_FIXTURE_DIR, policy=_FIXTURE_SPLIT_POLICY, min_content_tokens=1).chunks
    assert chunks

    with Session(engine) as session, session.begin():
        session.execute(delete(Chunk).where(Chunk.book_id == _FIXTURE_BOOK_ID))
        session.execute(delete(Section).where(Section.book_id == _FIXTURE_BOOK_ID))
        session.execute(delete(Book).where(Book.book_id == _FIXTURE_BOOK_ID))
        session.add(
            Book(
                book_id=book_meta.book_id,
                title_ar=book_meta.title_ar,
                author_name_ar=book_meta.main_author_name_ar,
                author_death_hijri=book_meta.main_author_death_hijri,
                category_id=book_meta.category_id,
                book_type_label=book_meta.book_type_label,
            )
        )
        session.flush()
        id_by_trail = _insert_sections(session, book_meta.book_id, sections)
        _insert_chunks(session, book_meta.book_id, chunks, id_by_trail, book_meta.title_ar)


def _insert_sections(
    session: Session, book_id: int, sections: list[ChunkSection]
) -> dict[tuple[str, ...], int]:
    id_by_trail: dict[tuple[str, ...], int] = {}
    for section in sorted(sections, key=lambda s: s.depth):
        parent_id = id_by_trail.get(section.trail[:-1]) if section.depth > 0 else None
        row = Section(
            book_id=book_id,
            parent_id=parent_id,
            shamela_title_id=section.shamela_title_id,
            title_text=section.title_text,
            title_trail=" > ".join(section.trail),
            depth=section.depth,
            path_source=section.path_source.value,
            start_page_id=section.start_page_id,
            end_page_id=section.end_page_id,
        )
        session.add(row)
        session.flush()
        id_by_trail[section.trail] = row.id
    return id_by_trail


def _resolve_section_id(
    chunk: BookChunk, id_by_trail: dict[tuple[str, ...], int], title_ar: str | None
) -> int | None:
    if chunk.trail in id_by_trail:
        return id_by_trail[chunk.trail]
    if chunk.trail and title_ar and chunk.trail[0] == title_ar and chunk.trail[1:] in id_by_trail:
        return id_by_trail[chunk.trail[1:]]
    return None


def _insert_chunks(
    session: Session,
    book_id: int,
    chunks: list[BookChunk],
    id_by_trail: dict[tuple[str, ...], int],
    title_ar: str | None,
) -> None:
    for chunk in chunks:
        session.add(
            Chunk(
                book_id=book_id,
                section_id=_resolve_section_id(chunk, id_by_trail, title_ar),
                content_role=chunk.content_role.value,
                source_text=chunk.source_text,
                retrieval_text=chunk.retrieval_text,
                context_header=chunk.context_header,
                start_page_id=chunk.page_id,
                end_page_id=chunk.page_id,
                start_page_num=chunk.page_num,
                end_page_num=chunk.page_num,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                token_count=chunk.token_count,
            )
        )
    session.flush()


def _body_sections(session: Session, book_id: int) -> list[tuple[int, list[int]]]:
    rows = session.execute(
        select(Chunk.section_id, Chunk.id)
        .where(Chunk.book_id == book_id, Chunk.content_role == ContentRole.BODY.value)
        .order_by(Chunk.section_id, Chunk.start_page_id, Chunk.start_offset, Chunk.id)
    ).all()
    grouped: dict[int, list[int]] = defaultdict(list)
    for section_id, chunk_id in rows:
        if section_id is not None:
            grouped[section_id].append(chunk_id)
    return list(grouped.items())


def _multi_chunk_body_sections(session: Session, book_id: int) -> list[tuple[int, list[int]]]:
    return [
        (section_id, ids) for section_id, ids in _body_sections(session, book_id) if len(ids) >= 2
    ]


def test_fixture_book_neighbor_expansion_stays_within_section(
    expander: ContextExpander, engine: Engine, seeded_fixture_book: None
) -> None:
    with Session(engine) as session:
        sections = _multi_chunk_body_sections(session, _FIXTURE_BOOK_ID)
        assert sections, "fixture book should contain at least one multi-chunk body section"
        section_id, chunk_ids = max(sections, key=lambda item: len(item[1]))
        middle = chunk_ids[len(chunk_ids) // 2]

    passage = expander.expand(
        [_hit(middle)],
        config=ExpansionConfig(mode=ExpandMode.NEIGHBORS, neighbor_window=1),
    )[0]

    assert passage.section_id == section_id
    assert set(passage.chunk_ids) <= set(chunk_ids)
    assert middle in passage.chunk_ids
    hit_parts = [part for part in passage.parts if part.is_hit]
    assert len(hit_parts) == 1
    assert hit_parts[0].chunk_id == middle


def test_fixture_book_full_section_never_pulls_other_sections(
    expander: ContextExpander, engine: Engine, seeded_fixture_book: None
) -> None:
    with Session(engine) as session, session.begin():
        sections = _multi_chunk_body_sections(session, _FIXTURE_BOOK_ID)
        assert sections, "need one multi-chunk section for full-section expansion"
        section_a, chunks_a = sections[0]
        hit = chunks_a[0]
        decoy = Section(
            book_id=_FIXTURE_BOOK_ID,
            title_text="decoy-entity",
            title_trail="decoy-entity",
            depth=0,
        )
        session.add(decoy)
        session.flush()
        decoy_chunk = Chunk(
            book_id=_FIXTURE_BOOK_ID,
            section_id=decoy.id,
            content_role=ContentRole.BODY.value,
            source_text="decoy neighboring biography text",
            retrieval_text="decoy neighboring biography text",
            start_page_id=999,
            end_page_id=999,
            start_offset=0,
            end_offset=32,
        )
        session.add(decoy_chunk)
        session.flush()
        decoy_id = decoy_chunk.id
        section_a_id = section_a
        expected = list(chunks_a)

    passage = expander.expand(
        [_hit(hit)],
        config=ExpansionConfig(mode=ExpandMode.FULL_SECTION),
    )[0]

    assert passage.section_id == section_a_id
    assert list(passage.chunk_ids) == expected
    assert decoy_id not in passage.chunk_ids
    assert "decoy neighboring biography text" not in passage.text


def test_fixture_book_expanded_text_is_verbatim_source(
    expander: ContextExpander, engine: Engine, seeded_fixture_book: None
) -> None:
    with Session(engine) as session:
        sections = _multi_chunk_body_sections(session, _FIXTURE_BOOK_ID)
        assert sections
        _section_id, chunk_ids = sections[0]
        hit = chunk_ids[0]
        sources = {
            chunk_id: text
            for chunk_id, text in session.execute(
                select(Chunk.id, Chunk.source_text).where(Chunk.id.in_(chunk_ids))
            ).all()
        }

    passage = expander.expand(
        [_hit(hit)],
        config=ExpansionConfig(mode=ExpandMode.FULL_SECTION, include_context_header=False),
    )[0]

    for part in passage.parts:
        assert part.source_text == sources[part.chunk_id]
    joined = "\n\n".join(sources[cid] for cid in passage.chunk_ids)
    assert passage.text == joined
