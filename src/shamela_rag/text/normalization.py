"""Arabic text normalization.

Two variants, and a hard rule: neither is ever applied to the stored ``source_text``.
- ``normalize_for_display`` preserves letters and diacritics, only tidying line endings; use it
  when showing text back to a user.
- ``normalize_for_index`` is the aggressive form used to build ``retrieval_text``: it strips
  diacritics and tatweel, folds alef/hamza-on-alef/ta-marbuta/alef-maksura variants, and collapses
  whitespace, so diacritized and undiacritized spellings match. It intentionally does NOT do root
  normalization (that is a separate, gated lexical expansion field, M3-06).
"""

from __future__ import annotations

import re

# Tashkeel (harakat, tanwin, shadda, sukun, combining hamza/maddah, dagger alef, Quranic marks).
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_WHITESPACE = re.compile(r"\s+")
_TATWEEL = "\u0640"

# Letter foldings applied for indexing only.
_ALEF_VARIANTS = {"\u0622": "\u0627", "\u0623": "\u0627", "\u0625": "\u0627", "\u0671": "\u0627"}
_TA_MARBUTA = "\u0629"
_HEH = "\u0647"
_ALEF_MAKSURA = "\u0649"
_YEH = "\u064a"


def strip_diacritics(text: str) -> str:
    return _DIACRITICS.sub("", text)


def remove_tatweel(text: str) -> str:
    return text.replace(_TATWEEL, "")


def normalize_alef(text: str) -> str:
    for src, dst in _ALEF_VARIANTS.items():
        text = text.replace(src, dst)
    return text


def normalize_ta_marbuta(text: str) -> str:
    return text.replace(_TA_MARBUTA, _HEH)


def normalize_alef_maksura(text: str) -> str:
    return text.replace(_ALEF_MAKSURA, _YEH)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def normalize_for_display(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_for_index(text: str) -> str:
    text = strip_diacritics(text)
    text = remove_tatweel(text)
    text = normalize_alef(text)
    text = normalize_ta_marbuta(text)
    text = normalize_alef_maksura(text)
    return collapse_whitespace(text)
