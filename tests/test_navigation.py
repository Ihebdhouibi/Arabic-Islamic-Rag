from __future__ import annotations

from shamela_rag.chunking.navigation import (
    NodeKind,
    classify_body,
    is_navigational,
    substantive_text,
)


def test_heading_only_is_navigational() -> None:
    heading = "<span data-type='title' id=toc-1>حرف الألف</span>\r"
    assert classify_body(heading) is NodeKind.NAVIGATIONAL
    assert is_navigational("<span data-type='title'>باب</span>")


def test_heading_with_prose_is_content() -> None:
    body = (
        "<span data-type='title' id=toc-2>باب الطهارة</span>\r"
        "الطهارة في اللغة النظافة وفي الشرع رفع الحدث"
    )
    assert classify_body(body) is NodeKind.CONTENT
    assert not is_navigational(body)


def test_empty_body_is_navigational() -> None:
    assert classify_body("") is NodeKind.NAVIGATIONAL
    assert classify_body("   \r\r  ") is NodeKind.NAVIGATIONAL


def test_threshold_is_configurable() -> None:
    body = "كلمة واحدة فقط هنا"
    assert classify_body(body, min_content_tokens=100) is NodeKind.NAVIGATIONAL
    assert classify_body(body, min_content_tokens=1) is NodeKind.CONTENT


def test_substantive_text_removes_title_markup() -> None:
    assert substantive_text("<span data-type='title' id=toc-1>عنوان</span>\rمتن") == "متن"
