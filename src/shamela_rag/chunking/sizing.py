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

from shamela_rag.chunking.title_spans import parse_title_spans
from shamela_rag.chunking.tokens import count_tokens

_PARAGRAPH = re.compile(r"(?:\r\n|\r|\n){2,}")
_SENTENCE = re.compile(r"(?<=[.!?؟؛])\s+")


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
