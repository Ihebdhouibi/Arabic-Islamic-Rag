# 06 — RAG Evaluation and Recent Advancements

> Part of a 7-document technical series on building a production RAG system over the Shamela digital library (~8,589 books, ~7.6M pages of classical Arabic Islamic scholarship). This document covers **evaluation methodology** and **the current state of RAG research/tooling**. Chunking, embeddings, retrieval strategies, and knowledge graphs/GraphRAG are each covered in depth in their own documents and are only referenced here as pointers.

## Part 1 — Evaluation

### 1. Why RAG Evaluation Isn't Generic LLM Evaluation

Evaluating a plain LLM is already hard — you're scoring open-ended text against fuzzy notions of quality. RAG adds a second, coupled system in front of the LLM: a retriever. A wrong final answer therefore has at least three distinct causes, each requiring a different fix: (1) **bad retrieval** — the model never had the right evidence, and no amount of prompting fixes that; (2) **good retrieval, bad use of it** — the right passage was in context but the model ignored, misread, or blended it with its own (possibly wrong) parametric knowledge; or (3) **both**.

If you only measure end-to-end answer quality, you can't tell these apart, and you won't know whether to invest in the retriever, the prompt, or the model. This is why RAG evaluation must decompose into **retrieval metrics**, **generation metrics**, and **end-to-end metrics**, measured both separately and together. For Shamela, a wrong answer about a fiqh ruling could stem from retrieving the wrong madhhab's text entirely, or from correctly retrieving it but paraphrasing away a qualifying condition — the difference between debugging the index and debugging the prompt.

### 2. Core Retrieval Metrics

These metrics ask: *given a query, did the retriever surface the right passages, in a useful order?* They require a labeled set of (query, relevant-document) pairs — a "golden" or "qrels" set.

- **Precision@k** — of the top-*k* retrieved chunks, what fraction are actually relevant? Matters when context budget is tight or irrelevant chunks actively confuse the generator (a real risk with dense classical commentary, where an off-topic passage can derail synthesis).
- **Recall@k** — of all relevant chunks in the corpus, what fraction did the top-*k* capture? Matters when a question genuinely needs multiple sources (a comparative fiqh question spanning several madhhabs) — missing even one produces an incomplete answer that no downstream generation step can repair.
- **Mean Reciprocal Rank (MRR)** — the average, across queries, of 1/(rank of the first relevant result). Right metric when there's typically one canonical correct passage and you care how quickly it surfaces — e.g., "what is the ruling on X" lookups with one authoritative source.
- **Normalized Discounted Cumulative Gain (NDCG)** — accounts for *graded* relevance (highly relevant vs. marginally relevant, not just relevant/irrelevant) and discounts lower-ranked items against an ideal ordering. Right choice when relevance is a spectrum, as in a scholarly corpus where a passage might be tangential, on-topic, or the definitive locus classicus for a question.

None of these metrics say anything about what the LLM does with the retrieved text — that's Part 2's job.

### 3. Core Generation / Answer-Quality Metrics

Once the LLM has produced an answer from retrieved context, RAG-specific quality dimensions include:

- **Faithfulness / groundedness** — does every claim in the answer actually follow from the retrieved context, or did the model add unsupported claims (hallucinate beyond its evidence)? Arguably the single most important RAG-specific metric: it catches "sounds right but isn't sourced" answers, the failure mode most dangerous for a religious-text application.
- **Answer relevance** — does the answer address the question asked, independent of whether it's grounded? A perfectly faithful answer that dodges the question still fails the user.
- **Context relevance / context precision** — of the context handed to the LLM, how much was actually useful? A generation-side view of a retrieval-side concept — it tells you whether the *augmentation* step is feeding the model noise.
- **Context recall** — did the retrieved context, as a whole, contain everything needed for a complete, correct answer? Checked against a reference answer, and the generation-facing analogue of retrieval recall@k.

These four metrics are rarely computed with exact-match or string-overlap scoring, because "did the answer follow from the context" requires something closer to entailment judgment than string matching. The dominant approach as of 2023–2026 is **LLM-as-a-judge**: prompting a strong (often different-from-the-generator) LLM to score faithfulness and relevance, frequently by decomposing the answer into atomic claims and checking each against the retrieved context. This has real tradeoffs: it's dramatically more **scalable than human annotation** (thousands of triples scored for the cost of API calls, which is what makes continuous regression testing feasible at all); it isn't free (**cost** compounds with claim-decomposition approaches that make multiple calls per example); and it carries **judge bias** — LLM judges documentedly favor longer or more confident-sounding answers, are sensitive to prompt phrasing, can share blind spots with a closely related generator model, and typically lack the specialized fiqh/hadith training needed to judge faithfulness correctly in this corpus. That last point is exactly why automated scores should be periodically spot-checked against human judgment rather than trusted blindly (Section 6).

### 4. Evaluation Frameworks and Tools

Several open-source and commercial tools implement these ideas so you don't have to build LLM-as-a-judge scoring from scratch:

- **RAGAS** — the open-source framework and paper that popularized reference-free RAG metrics (faithfulness, answer relevance, context precision/recall) computed via LLM judges. Es et al., *"Ragas: Automated Evaluation of Retrieval Augmented Generation"* ([arXiv:2309.15217](https://arxiv.org/abs/2309.15217); [EACL 2024](https://aclanthology.org/2024.eacl-demo.16/)). Often the first tool teams reach for, since its metric names map directly onto Section 3.
- **TruLens** — best known for formalizing the **"RAG triad"** (context relevance, groundedness, answer relevance), each LLM-judged, with tracing tooling to pinpoint which stage failed on a given example ([RAG Triad docs](https://www.trulens.org/getting_started/core_concepts/rag_triad/)).
- **DeepEval** — an open-source, Pytest-style framework for unit-testing LLM/RAG outputs, with 50+ metrics (faithfulness, answer relevancy, contextual precision/recall, hallucination); "write evaluations like you write tests" ([docs](https://deepeval.com/), [GitHub](https://github.com/confident-ai/deepeval)).
- **Arize Phoenix** — open-source LLM observability (tracing + evaluation + datasets) on OpenTelemetry; best known for tracing a live RAG request end-to-end and attaching evaluators to individual spans, useful for finding *where* a failure occurred ([docs](https://arize.com/docs/phoenix), [GitHub](https://github.com/Arize-ai/phoenix)).
- **LangSmith** — LangChain's platform, best known for dataset-driven evaluation tightly integrated with LangChain/LangGraph pipelines, including intermediate-step visibility for debugging ([tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial), [Ragas+LangSmith](https://www.langchain.com/blog/evaluating-rag-pipelines-with-ragas-langsmith)).

These overlap heavily in the metrics they compute and differ more in integration surface and workflow style than in evaluation theory. Picking one matters less than building the golden dataset described next — the metric plumbing is a commodity; domain-specific ground truth is the hard part.

### 5. Building a Domain-Specific Golden Dataset

General-purpose RAG benchmarks (academic QA datasets built on Wikipedia or English news) are close to useless as a primary signal for a classical Arabic religious-text system. They are **English-centric**, telling you little about retrieval over classical Arabic (Quranic Arabic, hadith terminology, madhhab-specific fiqh vocabulary) — a register current embedding models are comparatively undertrained on. And they are **general-domain**: "good retrieval" on them means matching Wikipedia-style facts, not this corpus's actual failure modes — conflating similarly-named scholars, missing isnad chains, collapsing distinctions between madhhabs, or matching a topical keyword ("zakat," "talaq") without matching the actual ruling. A leaderboard score tells you a model is broadly competent; it does not tell you whether *your* system retrieves *this specific* Hanafi ruling on a specific narration when a user asks about it.

The practical alternative, worth prioritizing before any framework in Section 4, is a **hand-curated golden set**: a modest collection of representative (question, expected answer, expected source book/page) triples, built by someone who understands the domain. Even 50–100 well-chosen examples — spanning direct factual lookups, comparative madhhab questions, multi-hop synthesis across books, and edge cases with genuine scholarly disagreement — is enough to catch regressions when chunking, embedding models, or prompts change (run the set before/after and diff the scores), give the metrics in Sections 2–3 concrete ground truth instead of relying purely on reference-free judging, and surface corpus-specific failure modes before they erode production trust. Treat it as a living regression-test suite, checked into version control and extended as new failure modes surface — not a one-time benchmark exercise.

### 6. Human-in-the-Loop Evaluation

Automated metrics — even well-designed, LLM-judged ones — cannot fully substitute for expert review here. Two things need validation a generic judge model can't reliably provide: **factual correctness** at the level of classical scholarship (correct attribution, accurate transmission of a ruling's actual condition and scope), and **scholarly nuance** — e.g., correctly representing that a ruling is disputed between madhhabs rather than flattening a live disagreement into one confident answer, or preserving a majority/minority distinction within a single madhhab. A judge without training in fiqh and hadith sciences can rate a fluent, well-cited, but subtly oversimplified answer as "faithful," because those metrics check consistency with the retrieved text, not correctness against the actual body of scholarship.

The practical implication: subject-matter-expert review should be a standing part of the evaluation loop, not a one-time audit — periodically reviewing production answers and helping refine the golden dataset (Section 5), with particular attention to cases involving disagreement between schools of thought. How to structure that review, and how to encode "this is disputed" as a first-class answer category rather than a failure, is picked up in more depth in the domain-synthesis document later in this series.

## Part 2 — Recent Advancements

RAG has moved fast since 2020. This section places the major developments on a timeline and explains, for each, what changed and why it matters — without re-deriving techniques covered in depth elsewhere in this series.

### 7. Contextual Retrieval (Anthropic, Sept 2024)

Anthropic's Contextual Retrieval prepends a short, LLM-generated explanatory context to each chunk before embedding and indexing (both for embeddings and for the BM25 side of hybrid search), addressing the classic problem of chunks losing meaning when split out of their surrounding document. Anthropic reported reducing failed retrievals by 49%, and by 67% when combined with reranking ([Anthropic, "Introducing Contextual Retrieval," Sept 2024](https://www.anthropic.com/engineering/contextual-retrieval)). This is covered in full depth, including implementation guidance, in the chunking-strategy document — it's listed here only for timeline placement.

### 8. RAPTOR, Self-RAG, CRAG, GraphRAG — Timeline Pointers

Four techniques mark the shift from naive to "advanced/modular" RAG through 2023–2024, each covered in depth in the retrieval-strategies or knowledge-graph documents:

- **RAPTOR** (Jan 2024) builds a recursive tree of clustered-and-summarized chunks so retrieval can operate at multiple levels of abstraction, not just flat passages ([Sarthi et al., arXiv:2401.18059](https://arxiv.org/abs/2401.18059)).
- **Self-RAG** (Oct 2023) trains a model to decide on-demand whether to retrieve at all, and to self-critique its own generations against retrieved passages using special reflection tokens ([Asai et al., arXiv:2310.11511](https://arxiv.org/abs/2310.11511)).
- **CRAG (Corrective RAG)** (Jan 2024) adds a lightweight evaluator that grades retrieved documents' quality and triggers corrective actions (e.g., falling back to web search) when retrieval confidence is low ([Yan et al., arXiv:2401.15884](https://arxiv.org/abs/2401.15884)).
- **GraphRAG** (Apr 2024, Microsoft) builds an LLM-extracted knowledge graph plus community summaries over a corpus to answer global, sensemaking-style questions that flat retrieval struggles with ([Edge et al., arXiv:2404.16130](https://arxiv.org/abs/2404.16130)).

### 9. Long-Context LLMs vs. RAG: Do We Still Need Retrieval?

As commercial models pushed context windows to 1M+ tokens, a natural question emerged: if you can stuff an entire corpus (or a large chunk of it) into the prompt, why bother with retrieval? The evidence as of 2025–2026 is genuinely mixed. Comparative studies find results diverge largely by model capability — strong long-context models reason well over large stuffed contexts, weaker ones degrade — and that neither approach dominates universally; the right choice depends on task type, corpus size, and how dynamic the data is ([Li et al., "Long Context vs. RAG for LLMs," arXiv:2501.01880](https://arxiv.org/pdf/2501.01880)).

Three counterarguments keep RAG relevant for a system like this one: **cost** — attention scales poorly with context length, and re-processing a huge context on every query is dramatically more expensive at scale than retrieving a small slice, compounding fast across millions of queries against an 8,589-book corpus; **lost-in-the-middle persists** — even long-context-native models still underweight information in the middle of a long input relative to the ends ([Liu et al., arXiv:2307.03172](https://arxiv.org/abs/2307.03172)), so more tokens in context isn't the same as more *effective* use of them; and **freshness, citability, and access control** — a corpus that gets corrected or filtered by permissions can't be baked into a fixed prompt or fine-tuned weights without a rebuild, whereas RAG's index updates incrementally, filters per-user, and — critically here — supports precise page-level citation far more reliably than an answer drawn diffusely from a million-token context.

The pragmatic 2025–2026 consensus is convergence, not a winner: use retrieval to cut a huge, dynamic, access-controlled corpus to a small, citable context, and let longer context windows make what *is* retrieved more forgiving of imperfect chunking — not a substitute for retrieval ([RAGFlow, "From RAG to Context," 2025](https://ragflow.io/blog/rag-review-2025-from-rag-to-context); [Elastic, "RAG vs. Long-Context LLM"](https://www.elastic.co/search-labs/blog/rag-vs-long-context-model-llm)).

### 10. Agentic RAG: Retrieval as a Tool Call

The most significant architectural shift of 2024–2026 is treating retrieval not as a fixed pipeline stage but as **one tool among several that an agentic LLM decides to invoke, repeatedly, interleaved with reasoning**. Instead of "always retrieve top-*k*, then generate once," an agentic system lets the model decide *whether* to search, *what* to search for, *how many times* (issuing follow-up queries when the first pass is insufficient), and whether to reach for a different tool entirely (web search, code execution, a structured-metadata query) within a single reasoning loop — the pattern popularized by ReAct-style prompting and now built directly into agent/function-calling frameworks ([MarkTechPost, "What is Agentic RAG?," Aug 2025](https://www.marktechpost.com/2025/08/27/what-is-agentic-rag-use-cases-and-top-agentic-rag-tools-2025/)). This generalizes the "Modular RAG" iterative/adaptive-retrieval concept from the fundamentals document by giving the model explicit agency over when and how to call the retriever, rather than encoding that logic in a fixed graph. For Shamela, it's attractive for multi-hop questions — an agent retrieves a ruling, notices it depends on a hadith's authenticity grading, and issues a follow-up retrieval for that grading before answering — but it also multiplies the number of places retrieval quality can go wrong, which is exactly why the per-stage metrics in Part 1 matter more, not less, as systems become agentic.

### 11. Small, Efficient, and Local Retrieval Models

A quieter but well-supported trend is the push toward smaller, efficient embedding models that run on-device or on modest hardware instead of only via API. Google's EmbeddingGemma (Sept 2025) runs in under 200MB of RAM with quantization, explicitly positioned for on-device semantic search and RAG ([Google Developers Blog, Sept 2025](https://developers.googleblog.com/en/introducing-embeddinggemma/)); compact open models such as Qwen3-Embedding-0.6B and nomic-embed-text have become common defaults for self-hosted RAG stacks where data can't leave a controlled environment. This matters for Shamela to the extent data sovereignty, cost control, or offline access are requirements — a genuine trend, but an infrastructure choice layered on top of the same evaluation and retrieval fundamentals covered elsewhere in this series, not a replacement for them.

### 12. Milestone Timeline

| Period | Milestone | Why it mattered |
|---|---|---|
| 2020 | Original RAG paper ([Lewis et al., arXiv:2005.11401](https://arxiv.org/abs/2005.11401)) | Introduced RAG as a formal architecture: frozen parametric model + non-parametric retrieval index, trained end-to-end. Foundational citation for the entire field. |
| 2020–2022 | Dense retrieval era (DPR and successors) | Dense embedding-based retrieval matured as the default alternative/complement to sparse (BM25) search. |
| 2022–2023 | Naive RAG production adoption | LLM APIs + vector databases made "embed, index, retrieve top-*k*, stuff into prompt" a widely deployed pattern, exposing its failure modes (Section 4 of the fundamentals doc) at scale. |
| 2023 | HyDE, early advanced-RAG techniques | Query-side transformations (e.g., embedding a hypothetical answer rather than the raw query) started addressing retrieval-precision gaps in naive pipelines. |
| Oct 2023 | Self-RAG ([arXiv:2310.11511](https://arxiv.org/abs/2310.11511)) | Adaptive, self-critiquing retrieval — retrieval becomes a decision, not a constant. |
| Sept 2023 | RAGAS ([arXiv:2309.15217](https://arxiv.org/abs/2309.15217)) | Popularized reference-free, LLM-judged RAG evaluation metrics still in wide use today. |
| Jan 2024 | RAPTOR ([arXiv:2401.18059](https://arxiv.org/abs/2401.18059)), CRAG ([arXiv:2401.15884](https://arxiv.org/abs/2401.15884)) | Hierarchical/multi-level retrieval and self-correcting retrieval pipelines. |
| Apr 2024 | GraphRAG ([arXiv:2404.16130](https://arxiv.org/abs/2404.16130)) | Knowledge-graph-augmented retrieval mainstreamed for global/sensemaking queries. |
| Sept 2024 | Contextual Retrieval ([Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)) | Simple, high-leverage fix for context loss at chunk boundaries; widely adopted. |
| 2024–2026 | Long-context debate, agentic RAG, small/local retrieval models | Field shifts from "which single technique" to "how do retrieval, agency, and context length compose" — the current, unsettled frontier. |

### 13. What This Means for Production Systems Today

None of the advancements above substitute for getting the fundamentals right. Most documented production RAG failures trace back to unglamorous basics, not a missing exotic technique: chunks that split arguments or isnad chains, no hybrid (dense + keyword) retrieval, no reranking step, and — most relevant here — **no evaluation harness at all**, so regressions ship silently. Barnett et al.'s empirical survey of real deployments across research, education, and biomedical domains found exactly these basics to be the dominant failure modes, not exotic edge cases ([Barnett et al., 2024, arXiv:2401.05856](https://arxiv.org/abs/2401.05856)).

The practical recommendation for this project, in order: (1) get chunking, hybrid retrieval, and reranking solid first (covered in the chunking and retrieval-strategies documents) — cheap, well-understood, and address the majority of real-world failures; (2) build the evaluation harness in parallel, not after — a 50–100 example golden dataset (Section 5) plus one framework from Section 4, wired into the development workflow so every pipeline change is measured, not eyeballed; (3) only then reach for RAPTOR, GraphRAG, agentic retrieval, or contextual embeddings — each genuinely valuable for specific query patterns, but each adds engineering and inference cost that's only justified once you can *measure* whether it improved your metrics on your own golden set. Adopting the newest published technique without a harness to confirm it helped is trend-chasing, not engineering — treat the golden dataset as the thing you build first, not last.

## Further Reading / Citations

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Es, S., James, J., Espinosa Anke, L., & Schockaert, S. (2023). *Ragas: Automated Evaluation of Retrieval Augmented Generation.* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) · [EACL 2024](https://aclanthology.org/2024.eacl-demo.16/) · [GitHub](https://github.com/explodinggradients/ragas)
- TruLens — RAG Triad documentation: [trulens.org](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
- DeepEval: [deepeval.com](https://deepeval.com/) · [GitHub](https://github.com/confident-ai/deepeval)
- Arize Phoenix: [arize.com/docs/phoenix](https://arize.com/docs/phoenix) · [GitHub](https://github.com/Arize-ai/phoenix)
- LangSmith RAG evaluation tutorial: [docs.langchain.com](https://docs.langchain.com/langsmith/evaluate-rag-tutorial); *Evaluating RAG pipelines with Ragas + LangSmith*: [langchain.com/blog](https://www.langchain.com/blog/evaluating-rag-pipelines-with-ragas-langsmith)
- Sarthi, P., et al. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* [arXiv:2401.18059](https://arxiv.org/abs/2401.18059)
- Asai, A., et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- Yan, S.-Q., et al. (2024). *Corrective Retrieval Augmented Generation.* [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)
- Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
- Anthropic (2024). *Introducing Contextual Retrieval.* [anthropic.com/engineering/contextual-retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- Liu, N. F., et al. (2023/2024). *Lost in the Middle: How Language Models Use Long Contexts.* [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) · [TACL 2024](https://aclanthology.org/2024.tacl-1.9/)
- Li, Z., et al. (2025). *Long Context vs. RAG for LLMs: An Evaluation and Revisits.* [arXiv:2501.01880](https://arxiv.org/pdf/2501.01880)
- RAGFlow (2025). *From RAG to Context — A 2025 Year-End Review of RAG.* [ragflow.io/blog](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
- Elastic. *RAG vs. Long-Context LLM.* [elastic.co/search-labs](https://www.elastic.co/search-labs/blog/rag-vs-long-context-model-llm)
- MarkTechPost (2025). *What is Agentic RAG? Use Cases and Top Agentic RAG Tools.* [marktechpost.com](https://www.marktechpost.com/2025/08/27/what-is-agentic-rag-use-cases-and-top-agentic-rag-tools-2025/)
- Google Developers Blog (2025). *Introducing EmbeddingGemma.* [developers.googleblog.com](https://developers.googleblog.com/en/introducing-embeddinggemma/)
- Barnett, S., et al. (2024). *Seven Failure Points When Engineering a Retrieval Augmented Generation System.* [arXiv:2401.05856](https://arxiv.org/abs/2401.05856)
- Gao, Y., et al. (2023/2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
