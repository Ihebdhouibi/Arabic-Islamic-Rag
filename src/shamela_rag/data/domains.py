"""Map corpus ``category_id`` to a ``suggested_domain`` for downstream consumers (nullable).

Categories 3-5 stay ``None`` until a ``quran`` mapping is confirmed.
"""

from __future__ import annotations

from enum import StrEnum


class SuggestedDomain(StrEnum):
    QURAN = "quran"
    HADITH = "hadith"
    FIQH = "fiqh"
    AQAID = "aqaid"
    FATWA = "fatwa"


# First-pass map from issue #145 / corpus taxonomy. Unlisted ids -> None.
_CATEGORY_DOMAINS: dict[int, SuggestedDomain] = {
    1: SuggestedDomain.AQAID,  # العقيدة
    2: SuggestedDomain.AQAID,  # الفرق والردود
    6: SuggestedDomain.HADITH,  # كتب السنة
    7: SuggestedDomain.HADITH,  # شروح الحديث
    8: SuggestedDomain.HADITH,  # التخريج والأطراف
    9: SuggestedDomain.HADITH,  # العلل والسؤلات الحديثية
    10: SuggestedDomain.HADITH,  # علوم الحديث
    11: SuggestedDomain.FIQH,  # أصول الفقه
    12: SuggestedDomain.FIQH,  # علوم الفقه والقواعد الفقهية
    14: SuggestedDomain.FIQH,  # الفقه الحنفي
    15: SuggestedDomain.FIQH,  # الفقه المالكي
    16: SuggestedDomain.FIQH,  # الفقه الشافعي
    17: SuggestedDomain.FIQH,  # الفقه الحنبلي
    18: SuggestedDomain.FIQH,  # الفقه العام
    19: SuggestedDomain.FIQH,  # مسائل فقهية
    20: SuggestedDomain.FIQH,  # السياسة الشرعية والقضاء
    21: SuggestedDomain.FIQH,  # الفرائض والوصايا
    22: SuggestedDomain.FATWA,  # الفتاوى
}


def suggested_domain_for_category(category_id: int | None) -> SuggestedDomain | None:
    if category_id is None:
        return None
    return _CATEGORY_DOMAINS.get(category_id)
