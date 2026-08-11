from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from shamela_rag.data.root_dictionary import (
    RootDictionary,
    load_root_dictionary,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "root_dictionary_sample.jsonl"
_FULL_DICT = (
    Path(__file__).resolve().parents[1] / "Shamela4_Full_DB" / "_meta" / "root_dictionary.jsonl"
)


@pytest.fixture
def sample_dict() -> RootDictionary:
    return load_root_dictionary(_FIXTURE)


def test_resolves_known_salah_forms(sample_dict: RootDictionary) -> None:
    # Same forms cited in the chunking/normalization brief.
    assert "صلو" in sample_dict.lookup("الصلاة")
    assert "صلي" in sample_dict.lookup("الصلاة")
    assert sample_dict.lookup("للصلاة") == sample_dict.lookup("الصلاة")
    assert "صلو" in sample_dict.lookup("صلاته")


def test_resolves_known_kataba_forms(sample_dict: RootDictionary) -> None:
    assert "كتب" in sample_dict.lookup("كتاب")
    assert "كتب" in sample_dict.lookup("يكتب")
    assert "كتب" in sample_dict.lookup("كتب")


def test_unknown_token_returns_empty(sample_dict: RootDictionary) -> None:
    assert sample_dict.lookup("not-in-dictionary") == ()
    assert "not-in-dictionary" not in sample_dict


def test_len_and_contains(sample_dict: RootDictionary) -> None:
    assert len(sample_dict) == 6
    assert "الصلاة" in sample_dict


def test_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "roots.jsonl"
    path.write_text(
        "\n".join(
            [
                "not-json",
                json.dumps({"token": "حسن", "roots": ["حسن"]}, ensure_ascii=False),
                json.dumps({"token": 1, "roots": ["ء"]}, ensure_ascii=False),
                json.dumps({"token": "bad", "roots": "not-a-list"}, ensure_ascii=False),
                json.dumps(["not", "an", "object"], ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dictionary = load_root_dictionary(path)
    assert len(dictionary) == 1
    assert dictionary.lookup("حسن") == ("حسن",)


def test_duplicate_token_keeps_first(tmp_path: Path) -> None:
    path = tmp_path / "roots.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"token": "كتب", "roots": ["كتب"]}, ensure_ascii=False),
                json.dumps({"token": "كتب", "roots": ["توب"]}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_root_dictionary(path).lookup("كتب") == ("كتب",)


def test_sample_load_is_fast_and_small(sample_dict: RootDictionary) -> None:
    start = time.perf_counter()
    _ = sample_dict.lookup("الصلاة")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.01
    assert len(sample_dict) < 100


def test_large_synthetic_dict_loads_with_sane_memory(tmp_path: Path) -> None:
    path = tmp_path / "big.jsonl"
    n = 50_000
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(
                json.dumps({"token": f"توكن{i}", "roots": [f"جذر{i % 100}"]}, ensure_ascii=False)
            )
            fh.write("\n")

    start = time.perf_counter()
    dictionary = load_root_dictionary(path)
    elapsed = time.perf_counter() - start

    assert len(dictionary) == n
    assert dictionary.lookup("توكن42") == ("جذر42",)
    assert elapsed < 15.0


@pytest.mark.skipif(
    not _FULL_DICT.is_file(),
    reason="full Shamela root_dictionary.jsonl not present",
)
def test_full_dictionary_resolves_known_forms_when_available() -> None:
    dictionary = load_root_dictionary(_FULL_DICT)
    assert len(dictionary) > 1_000_000
    assert "صلو" in dictionary.lookup("الصلاة")
    assert "كتب" in dictionary.lookup("كتاب")
