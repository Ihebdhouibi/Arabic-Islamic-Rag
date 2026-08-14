"""General-module end-to-end Q&A service (M5-04).

Composes retrieval and generation: ``answer`` runs the retrieval pipeline for a question, then
assembles a cited answer from the resulting passages. This is the top-level entry point the API
(M7) will call.
"""

from __future__ import annotations

from shamela_rag.generation.answer import Answer, AnswerAssembler
from shamela_rag.retrieval.expand import ExpandedPassage
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.service import RetrievalService


class GeneralQAService:
    def __init__(self, *, retrieval_service: RetrievalService, assembler: AnswerAssembler) -> None:
        self._retrieval = retrieval_service
        self._assembler = assembler

    def answer(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: RetrievalFilter | None = None,
    ) -> Answer:
        answer, _passages = self.answer_with_retrieval(question, k=k, filters=filters)
        return answer

    def answer_with_retrieval(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: RetrievalFilter | None = None,
    ) -> tuple[Answer, list[ExpandedPassage]]:
        """Like ``answer``, but also returns the retrieved passages (scores for diagnostics)."""
        passages = self._retrieval.retrieve(question, k=k, filters=filters)
        return self._assembler.assemble(question, passages), passages
