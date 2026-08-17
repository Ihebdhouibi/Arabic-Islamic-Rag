"""Streamlit demo for interactive general Q&A (M7-04).

Usage::

    pip install -e ".[demo,bge,rerank]"
    streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.demo.wiring import build_general_qa_service
from shamela_rag.factory import build_generation_provider
from shamela_rag.generation.answer import Answer
from shamela_rag.generation.service import GeneralQAService
from shamela_rag.retrieval.expand import ExpandedPassage
from shamela_rag.retrieval.filters import RetrievalFilter

_PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600&family=Source+Sans+3:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: "Source Sans 3", "IBM Plex Sans Arabic", sans-serif; }
[data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse 80% 50% at 50% -10%, #1a3a42 0%, transparent 55%),
              linear-gradient(180deg, #0f1419 0%, #151b22 40%, #0f1419 100%);
  color: #e8eef2;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #12181f; border-right: 1px solid #24303a; }
[data-testid="stSidebar"] * { color: #d7e0e6; }
.hero { text-align: center; padding: 2.5rem 1rem 1.25rem; }
.hero h1 { font-size: 2.1rem; font-weight: 700; margin: 0; color: #f4f7f9; }
.hero p { margin: 0.55rem 0 0; color: #9aafbb; font-size: 1.02rem; }
.chat-shell { max-width: 820px; margin: 0 auto; padding-bottom: 5.5rem; }
div[data-testid="stChatMessage"] {
  background: #1a222b; border: 1px solid #2a3742; border-radius: 16px;
  padding: 0.35rem 0.25rem; margin-bottom: 0.75rem;
}
div[data-testid="stChatInput"] { max-width: 820px; margin: 0 auto; }
div[data-testid="stChatInput"] textarea {
  border-radius: 18px !important; background: #1a222b !important;
  color: #f4f7f9 !important; border: 1px solid #33414c !important;
}
.diag-line, .cite-block { color: #c5d3db; font-size: 0.95rem; line-height: 1.55; }
.cite-title { color: #7ec8c4; font-weight: 600; }
.empty-hint { text-align: center; color: #7f93a0; padding: 4rem 1rem 2rem; font-size: 1.05rem; }
</style>
"""


@dataclass(frozen=True)
class _Turn:
    question: str
    answer: Answer | None
    passages: tuple[ExpandedPassage, ...]
    error: str | None


def _get_qa_service() -> GeneralQAService:
    if "qa_service" not in st.session_state:
        st.session_state.qa_service = build_general_qa_service(
            generation_provider=build_generation_provider()
        )
    return st.session_state.qa_service


def _parse_optional_int(raw: str, *, label: str) -> int | None:
    text = raw.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer when provided.") from exc


def _render_answer(turn: _Turn) -> None:
    if turn.error:
        st.error(turn.error)
        return
    assert turn.answer is not None
    answer = turn.answer
    st.markdown(answer.text)

    with st.expander("Retrieval diagnostics", expanded=False):
        st.write(f"Deflected: {answer.deflected}")
        if turn.passages:
            st.write("Passage scores (hit chunk id → score):")
            for passage in turn.passages:
                st.write(f"- chunk {passage.hit_chunk_id}: {passage.score}")
        else:
            st.write("No passages retrieved.")

    with st.expander("Citations", expanded=not answer.deflected):
        if not answer.citations:
            st.write("No citations.")
            return
        for citation in answer.citations:
            is_footnote = citation.content_role == ContentRole.FOOTNOTE.value
            st.markdown(
                f"**[{citation.marker}]** {citation.book_title} — {citation.author} — "
                f"page {citation.page}"
            )
            st.write(f"Footnote: {is_footnote}")
            st.write(citation.snippet)


def main() -> None:
    st.set_page_config(page_title="Shamela RAG", layout="centered")
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    if "turns" not in st.session_state:
        st.session_state.turns = []

    with st.sidebar:
        st.markdown("### Filters")
        st.caption("Optional book / category filters.")
        book_raw = st.text_input("Book id", value="")
        category_raw = st.text_input("Category id", value="")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.turns = []
            st.rerun()

    st.markdown(
        '<div class="hero"><h1>Shamela RAG</h1>'
        "<p>Ask in Arabic or English. Grounded answers from your ingested corpus.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    if not st.session_state.turns:
        st.markdown(
            '<div class="empty-hint">Your conversation appears here.<br/>'
            "Type a question in the box below.</div>",
            unsafe_allow_html=True,
        )

    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            _render_answer(turn)
    st.markdown("</div>", unsafe_allow_html=True)

    question = st.chat_input("Ask a question (Arabic or English)…")
    if question is None:
        return

    if not question.strip():
        st.session_state.turns.append(
            _Turn(question=question, answer=None, passages=(), error="Enter a non-empty question.")
        )
        st.rerun()
        return

    try:
        book_id = _parse_optional_int(book_raw, label="Book id")
        category_id = _parse_optional_int(category_raw, label="Category id")
    except ValueError as exc:
        st.session_state.turns.append(
            _Turn(question=question, answer=None, passages=(), error=str(exc))
        )
        st.rerun()
        return

    filters: RetrievalFilter | None = None
    if book_id is not None or category_id is not None:
        filters = RetrievalFilter(book_id=book_id, category_id=category_id)

    try:
        answer, passages = _get_qa_service().answer_with_retrieval(question, filters=filters)
        st.session_state.turns.append(
            _Turn(question=question, answer=answer, passages=tuple(passages), error=None)
        )
    except Exception as exc:  # noqa: BLE001 - surface wiring/runtime errors in the UI
        st.session_state.turns.append(
            _Turn(question=question, answer=None, passages=(), error=str(exc))
        )
    st.rerun()


if __name__ == "__main__":
    main()
