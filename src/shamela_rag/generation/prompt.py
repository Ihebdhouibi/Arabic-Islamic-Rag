"""General Q&A prompt template (M5-02).

Renders a question plus retrieved passages into the LLM prompt, enforcing doc 07 §6:
cite every source, preserve disagreement, do not act as an independent authority, and never
attribute footnote/editor notes to the book's author.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shamela_rag.chunking.content_roles import ContentRole

_INSTRUCTIONS = """\
You answer using only the retrieved sources below. Follow every rule:

1. Cite every claim with book title, author, and page from the numbered sources.
2. When sources disagree, present each view with attribution; do not flatten them into one view.
3. Do not present yourself as an independent authority or issue novel rulings; report what the \
sources say.
4. Sources marked content_role=footnote (editor/muhaqqiq notes) must not be attributed to the \
book's author; treat them as editorial commentary only.
"""


@dataclass(frozen=True, slots=True)
class PromptPassage:
    """One retrieved passage prepared for the general Q&A prompt."""

    text: str
    book_title: str = ""
    author: str = ""
    page: str = ""
    content_role: str = ContentRole.BODY.value


def _role_value(role: str | ContentRole) -> str:
    if isinstance(role, ContentRole):
        return role.value
    return role.strip().lower() or ContentRole.BODY.value


def _format_passage(index: int, passage: PromptPassage) -> str:
    role = _role_value(passage.content_role)
    lines = [
        f"[{index}] book={passage.book_title or '(unknown)'}",
        f"    author={passage.author or '(unknown)'}",
        f"    page={passage.page or '(unknown)'}",
        f"    content_role={role}",
    ]
    if role == ContentRole.FOOTNOTE.value:
        lines.append("    note=editorial/footnote — do not attribute to the book's author")
    lines.append(passage.text.strip())
    return "\n".join(lines)


def render_general_qa_prompt(question: str, passages: Sequence[PromptPassage]) -> str:
    """Render the general-module Q&A prompt from a question and retrieved passages."""
    question_text = question.strip()
    if not question_text:
        raise ValueError("question must be non-empty")

    if passages:
        sources = "\n\n".join(
            _format_passage(index, passage) for index, passage in enumerate(passages, start=1)
        )
    else:
        sources = (
            "(no sources retrieved — say that the available evidence is insufficient; "
            "do not invent sources)"
        )

    return f"{_INSTRUCTIONS}\nQuestion:\n{question_text}\n\nSources:\n{sources}\n"
