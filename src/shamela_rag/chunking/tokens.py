"""Token counting for chunk sizing.

The size policy (M2-08) needs a length measure. This exposes a ``TokenCounter`` protocol so a real
model tokenizer (Qwen3-8B / BGE-M3, wired when embeddings land in M3) can be swapped in, with a
dependency-free heuristic as the default. The heuristic counts word runs plus individual
punctuation; it undercounts relative to a subword tokenizer, so it is a stable proxy, not an exact
match.
"""

from __future__ import annotations

import re
from typing import Protocol

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    def count(self, text: str) -> int:
        return len(_TOKEN.findall(text))


_default: TokenCounter = HeuristicTokenCounter()


def count_tokens(text: str) -> int:
    return _default.count(text)
