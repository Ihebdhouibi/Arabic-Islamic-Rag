from __future__ import annotations

from shamela_rag.data.domains import SuggestedDomain, suggested_domain_for_category


def test_maps_one_category_per_claimed_domain() -> None:
    assert suggested_domain_for_category(1) is SuggestedDomain.AQAID
    assert suggested_domain_for_category(6) is SuggestedDomain.HADITH
    assert suggested_domain_for_category(16) is SuggestedDomain.FIQH
    assert suggested_domain_for_category(22) is SuggestedDomain.FATWA


def test_tafsir_adjacent_categories_stay_null_until_core_confirms() -> None:
    assert suggested_domain_for_category(3) is None  # التفسير
    assert suggested_domain_for_category(4) is None  # علوم القرآن
    assert suggested_domain_for_category(5) is None  # التجويد والقراءات


def test_unmapped_and_none_return_null() -> None:
    assert suggested_domain_for_category(13) is None  # المنطق
    assert suggested_domain_for_category(26) is None  # تراجم
    assert suggested_domain_for_category(999) is None
    assert suggested_domain_for_category(None) is None
