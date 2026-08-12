from __future__ import annotations

import pytest

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.generation import PromptPassage, render_general_qa_prompt

# Example used for snapshot / manual review (disagreeing body views + one footnote).
_EXAMPLE_QUESTION = "ما قول العلماء في خلق القرآن؟"
_EXAMPLE_PASSAGES = (
    PromptPassage(
        text="ذهبت المعتزلة إلى أن القرآن مخلوق.",
        book_title="الملل والنحل",
        author="الشهرستاني",
        page="42",
        content_role=ContentRole.BODY.value,
    ),
    PromptPassage(
        text="وقال أهل السنة إن القرآن كلام الله غير مخلوق.",
        book_title="العقيدة الطحاوية",
        author="الطحاوي",
        page="15",
        content_role=ContentRole.BODY.value,
    ),
    PromptPassage(
        text="تعليق المحقق: انظر الخلاف مفصلا في مظانه.",
        book_title="العقيدة الطحاوية",
        author="الطحاوي",
        page="15",
        content_role=ContentRole.FOOTNOTE.value,
    ),
)

_EXAMPLE_SNAPSHOT = """\
You answer using only the retrieved sources below. Follow every rule:

1. Cite every claim with book title, author, and page from the numbered sources.
2. When sources disagree, present each view with attribution; do not flatten them into one view.
3. Do not present yourself as an independent authority or issue novel rulings; report what the \
sources say.
4. Sources marked content_role=footnote (editor/muhaqqiq notes) must not be attributed to the \
book's author; treat them as editorial commentary only.

Question:
ما قول العلماء في خلق القرآن؟

Sources:
[1] book=الملل والنحل
    author=الشهرستاني
    page=42
    content_role=body
ذهبت المعتزلة إلى أن القرآن مخلوق.

[2] book=العقيدة الطحاوية
    author=الطحاوي
    page=15
    content_role=body
وقال أهل السنة إن القرآن كلام الله غير مخلوق.

[3] book=العقيدة الطحاوية
    author=الطحاوي
    page=15
    content_role=footnote
    note=editorial/footnote — do not attribute to the book's author
تعليق المحقق: انظر الخلاف مفصلا في مظانه.
"""


def test_render_snapshot_matches_example() -> None:
    assert render_general_qa_prompt(_EXAMPLE_QUESTION, _EXAMPLE_PASSAGES) == _EXAMPLE_SNAPSHOT


def test_instructions_encode_doc_07_rules() -> None:
    prompt = render_general_qa_prompt(_EXAMPLE_QUESTION, _EXAMPLE_PASSAGES)
    assert "Cite every claim" in prompt
    assert "disagreement" in prompt.lower() or "disagree" in prompt
    assert "independent authority" in prompt
    assert "content_role=footnote" in prompt
    assert "do not attribute to the book's author" in prompt


def test_footnote_passage_is_flagged() -> None:
    prompt = render_general_qa_prompt(
        "q",
        [
            PromptPassage(
                text="حاشية",
                book_title="كتاب",
                author="مؤلف",
                page="1",
                content_role=ContentRole.FOOTNOTE,
            )
        ],
    )
    assert "content_role=footnote" in prompt
    assert "editorial/footnote" in prompt


def test_empty_question_rejected() -> None:
    with pytest.raises(ValueError, match="question"):
        render_general_qa_prompt("  ", _EXAMPLE_PASSAGES)


def test_empty_passages_still_renders() -> None:
    prompt = render_general_qa_prompt("أين الدليل؟", [])
    assert "Question:\nأين الدليل؟" in prompt
    assert "no sources retrieved" in prompt
