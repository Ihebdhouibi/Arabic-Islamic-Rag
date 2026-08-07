"""Size & semantic policy: turn one structural section's text into embedding children.

Because ``split_section`` only ever sees a single section's text, a child can never cross into the
next structural section, and overlap is always within-section.

Policy (token counts are configurable starting values):
- a section at or under ``max_tokens`` becomes exactly one child (this covers both a short atomic
  entry and a normal coherent section);
- a section over ``max_tokens`` is split into ~``split_target_tokens`` children, preferring cut
  points in this order: internal subheading -> paragraph -> Arabic sentence -> word/token;
- consecutive children of an oversized section share ~``overlap_tokens`` of trailing content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from shamela_rag.chunking.title_spans import parse_title_spans
from shamela_rag.chunking.tokens import count_tokens

_PARAGRAPH = re.compile(r"(?:\r\n|\r|\n){2,}")
_SENTENCE = re.compile(r"(?<=[.!?\u061F\u061B])\s+")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class SizePolicy:
    min_tokens: int = 128
    max_tokens: int = 768
    split_target_tokens: int = 448
    overlap_tokens: int = 64


DEFAULT_POLICY = SizePolicy()


def _split_on_subheadings(text: str) -> list[str]:
    starts = [span.start for span in parse_title_spans(text)]
    if not starts:
        return [text]
    blocks: list[str] = []
    prev = 0
    for start in starts:
        if start > prev:
            blocks.append(text[prev:start])
        prev = start
    blocks.append(text[prev:])
    return [block for block in blocks if block.strip()]


def _split_by_words(text: str, target: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        current.append(word)
        if count_tokens(" ".join(current)) >= target:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def _atomize(text: str, policy: SizePolicy) -> list[str]:
    atoms: list[str] = []
    for block in _split_on_subheadings(text):
        for paragraph in _PARAGRAPH.split(block):
            for sentence in _SENTENCE.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if count_tokens(sentence) > policy.max_tokens:
                    atoms.extend(_split_by_words(sentence, policy.split_target_tokens))
                else:
                    atoms.append(sentence)
    return atoms


def _overlap_tail(atoms: list[str], overlap_tokens: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for atom in reversed(atoms):
        tokens = count_tokens(atom)
        if total + tokens > overlap_tokens:
            break
        tail.insert(0, atom)
        total += tokens
    return tail


def _pack(atoms: list[str], policy: SizePolicy) -> list[str]:
    children: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for atom in atoms:
        atom_tokens = count_tokens(atom)
        if current and current_tokens + atom_tokens > policy.split_target_tokens:
            children.append(" ".join(current))
            current = _overlap_tail(current, policy.overlap_tokens)
            current_tokens = sum(count_tokens(a) for a in current)
        current.append(atom)
        current_tokens += atom_tokens
    if current:
        children.append(" ".join(current))
    return children


def split_section(text: str, policy: SizePolicy = DEFAULT_POLICY) -> list[str]:
    if not text.strip():
        return []
    if count_tokens(text) <= policy.max_tokens:
        return [text]
    return _pack(_atomize(text, policy), policy)


def _hard_split_points(text: str) -> list[int]:
    points: set[int] = set()
    for span in parse_title_spans(text):
        if span.start > 0:
            points.add(span.start)
    for match in _PARAGRAPH.finditer(text):
        points.add(match.end())
    for match in _SENTENCE.finditer(text):
        points.add(match.end())
    return sorted(point for point in points if 0 < point < len(text))


def _word_ranges(text: str, start: int, end: int, target: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    seg_start = start
    for match in _WORD.finditer(text, start, end):
        if count_tokens(text[seg_start : match.end()]) >= target:
            ranges.append((seg_start, match.end()))
            seg_start = match.end()
    if seg_start < end:
        ranges.append((seg_start, end))
    return ranges


def _atom_ranges(text: str, policy: SizePolicy) -> list[tuple[int, int]]:
    bounds = [0, *_hard_split_points(text), len(text)]
    atoms: list[tuple[int, int]] = []
    for start, end in pairwise(bounds):
        if end <= start:
            continue
        if count_tokens(text[start:end]) > policy.max_tokens:
            atoms.extend(_word_ranges(text, start, end, policy.split_target_tokens))
        else:
            atoms.append((start, end))
    return atoms


def _overlap_tail_ranges(
    current: list[tuple[int, int]], text: str, overlap_tokens: int
) -> list[tuple[int, int]]:
    tail: list[tuple[int, int]] = []
    total = 0
    for start, end in reversed(current):
        tokens = count_tokens(text[start:end])
        if total + tokens > overlap_tokens:
            break
        tail.insert(0, (start, end))
        total += tokens
    return tail


def split_section_offsets(text: str, policy: SizePolicy = DEFAULT_POLICY) -> list[tuple[int, int]]:
    """Verbatim offset spans honoring the same priority + overlap as ``split_section``.

    Each ``(start, end)`` maps to an exact slice ``text[start:end]`` (so ``source_text`` stays
    verbatim); adjacent spans overlap by ~``overlap_tokens``.
    """
    if not text:
        return []
    if count_tokens(text) <= policy.max_tokens:
        return [(0, len(text))]
    children: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []
    current_tokens = 0
    for atom in _atom_ranges(text, policy):
        atom_tokens = count_tokens(text[atom[0] : atom[1]])
        if current and current_tokens + atom_tokens > policy.split_target_tokens:
            children.append((current[0][0], current[-1][1]))
            current = _overlap_tail_ranges(current, text, policy.overlap_tokens)
            current_tokens = sum(count_tokens(text[a:b]) for a, b in current)
        current.append(atom)
        current_tokens += atom_tokens
    if current:
        children.append((current[0][0], current[-1][1]))
    return children
