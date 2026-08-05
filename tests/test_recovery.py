from __future__ import annotations

from shamela_rag.chunking.boundaries import Confidence
from shamela_rag.chunking.recovery import recover_heading_candidates


def test_numbered_entry_is_medium_confidence() -> None:
    candidates = recover_heading_candidates("١ - آبي اللحم الغفاري")
    assert len(candidates) == 1
    assert candidates[0].pattern == "numbered_entry"
    assert candidates[0].confidence is Confidence.MEDIUM


def test_short_bare_line_is_low_confidence() -> None:
    candidates = recover_heading_candidates("باب نادر")
    assert candidates[0].pattern == "short_line"
    assert candidates[0].confidence is Confidence.LOW


def test_full_sentence_is_not_a_candidate() -> None:
    assert recover_heading_candidates("هذا نص طويل ينتهي بنقطة كاملة وجملة تامة.") == []


def test_lines_inside_inline_spans_are_skipped() -> None:
    body = "<span data-type='title' id=toc-1>عنوان مُعلّم</span>"
    assert recover_heading_candidates(body) == []


def test_known_toc_titles_are_skipped() -> None:
    candidates = recover_heading_candidates("باب الطهارة", toc_titles=frozenset({"باب الطهارة"}))
    assert candidates == []


def test_unmarked_heading_recovered_next_to_marked_one() -> None:
    # The Fatimah case: a marked entry, then a later UNMARKED biography heading on the same page.
    body = (
        "<span data-type='title' id=toc-6342>فاطمة بنت الخطاب</span>\r"
        "نص ترجمتها الطويل هنا وكلام كثير ومفيد جدا.\r"
        "فاطمة بنت رسول الله صلى الله عليه وسلم\r"
        "ثم يبدأ نص الترجمة الاخرى"
    )
    texts = [c.text for c in recover_heading_candidates(body)]
    assert "فاطمة بنت الخطاب" not in texts  # marked -> skipped
    assert "فاطمة بنت رسول الله صلى الله عليه وسلم" in texts  # unmarked -> recovered
