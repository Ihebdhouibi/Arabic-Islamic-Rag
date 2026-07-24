# 01 — RAG Fundamentals

> Part of a 7-document technical series on building a production RAG system over the Shamela digital library (~8,589 books, ~7.6M pages of classical Arabic Islamic scholarship). This document covers **RAG fundamentals only**. Chunking, embeddings, retrieval strategies, knowledge graphs/GraphRAG, and evaluation are each covered in depth in their own documents — they are only mentioned here as pointers.

## 1. What RAG Is, and Why It Exists

A large language model (LLM) is a frozen artifact. Everything it "knows" was baked into its weights during pretraining, on a snapshot of text that ends at some cutoff date. This creates three structural problems for any application that wants to answer questions grounded in real, current, or private knowledge:

- **Knowledge cutoff.** The model has no idea about anything that happened, was published, or was written after its training data was collected. It also has no idea about anything that was never public in the first place — your organization's internal documents, a licensed manuscript corpus, or (in our case) the specific critical editions and page layouts of a specific digital library.
- **Hallucination.** When an LLM doesn't know an answer, it does not reliably say "I don't know." It produces fluent, plausible-sounding text that may be entirely fabricated — a fake citation, a misattributed hadith, a plausible-sounding but nonexistent verse reference. The model is a next-token predictor optimized for fluency, not a database with an integrity constraint.
- **Domain and private data.** Most valuable use cases involve data the model never saw during pretraining: your company's contracts, a hospital's patient records, or a 7.6-million-page corpus of classical Arabic religious texts. The model cannot answer questions about content it was never exposed to, no matter how well it reasons.

There are three broad ways to get an LLM to work with knowledge it doesn't already have: **retrieval-augmented generation (RAG)**, **fine-tuning**, and **long-context prompting**. RAG's core idea, introduced by Lewis et al. in 2020, is to keep the LLM's parameters frozen and instead retrieve relevant text from an external knowledge store at query time, then feed that text into the model's context window alongside the user's question ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401)). The model's job shifts from "recall this from memory" to "synthesize an answer from the evidence I just handed you."

Why retrieval instead of fine-tuning as the default? Fine-tuning bakes knowledge into weights, which means: every time the underlying corpus changes you must retrain; you cannot easily point to *which* document justified an answer (the knowledge is diffused across billions of parameters, not attributable to a specific source); and training runs are expensive and slow compared to updating a search index. RAG decouples "what the model knows how to do" (reason, write, summarize — learned once during pretraining) from "what facts it has to work with" (swapped in per-query from an index that can be updated in minutes). This decoupling is what makes RAG the default architecture for knowledge-intensive applications, and it is exactly why it matters for a corpus like Shamela: the corpus is huge, mostly static but occasionally corrected/re-edited, and — critically — every claim needs to be traceable to a specific book, volume, and page.

## 2. The Canonical RAG Pipeline

A RAG system is a pipeline, not a single model call. Each stage has its own failure modes and its own document later in this series; here is the map.

1. **Ingestion (parsing/cleaning).** Raw source material — in our case Shamela's book database with its markup conventions, footnotes, page-break markers, and OCR artifacts from older scans — is parsed into clean, structured text. This stage resolves format-specific quirks (e.g., Shamela's internal tagging for page numbers, poetry, Quranic ayat, hadith isnad chains) into a normalized representation the rest of the pipeline can consume. Garbage in at this stage propagates through every later stage, so ingestion quality is disproportionately important and easy to underestimate.

2. **Chunking.** Documents are split into smaller passages sized to fit inside an embedding model's input window and to keep retrieved context focused. Chunk boundaries matter enormously: a chunk cut mid-argument, or one that separates a hadith's text from its chain of narrators, can destroy the very context a retriever needs to match on. (Full treatment in the chunking-strategy document — this is one of the highest-leverage decisions in the whole system, especially for classical Arabic texts with nested commentary structures.)

3. **Embedding.** Each chunk is converted into a dense numerical vector by an embedding model, such that semantically similar text ends up close together in vector space. This is what allows retrieval by *meaning* rather than exact keyword match — a query about "ruling on combining prayers while traveling" can match a passage that never uses those exact words. (Model selection, especially for Arabic and classical/religious registers, is covered in the embeddings document.)

4. **Indexing/storage.** Vectors (and usually the original text plus metadata — book title, author, page number, category) are stored in a structure optimized for fast similarity search: a vector database or a vector index (e.g., HNSW-based ANN structures), often alongside a traditional keyword/full-text index for hybrid search. This is also where metadata filters live (e.g., restrict retrieval to a given madhhab or genre).

5. **Query-time retrieval.** When a user asks a question, the query itself is embedded (and/or transformed — see "advanced RAG" below) and used to search the index for the top-*k* most relevant chunks. This is the step that determines what evidence the model will even have a chance to reason over; if the right passage isn't retrieved here, no amount of generation quality can recover it.

6. **Reranking (optional but recommended).** Initial retrieval (especially pure vector similarity) is optimized for recall and speed, not precision — it returns a candidate set that is *probably* relevant. A reranker, typically a cross-encoder model that jointly scores the query and each candidate passage, reorders that candidate set by finer-grained relevance before anything is passed to the LLM. This is one of the cheapest, highest-ROI additions to a naive pipeline.

7. **Augmentation/prompt construction.** The retrieved (and reranked) chunks are assembled into a prompt alongside the user's question, usually with instructions about how to use the sources, how to cite them, and what to do if the retrieved evidence is insufficient. Prompt construction also has to manage ordering and context-length budgets — as discussed below, *where* in the prompt a passage sits affects how well the model uses it.

8. **Generation.** The LLM produces an answer conditioned on the augmented prompt. Its job is now closer to reading comprehension and synthesis than open-ended recall, which is a large part of why RAG reduces (but does not eliminate) hallucination.

9. **Citation/grounding (optional but, for this project, essential).** The system attributes claims in the generated answer back to specific retrieved sources — ideally at the level of a specific book/page rather than a vague "based on the provided context." For a religious text corpus this is not a nice-to-have polish feature; see Section 6.

## 3. Naive RAG, Advanced RAG, and Modular RAG

The most widely cited framework for classifying RAG architectures comes from Gao et al.'s 2023/2024 survey, *"Retrieval-Augmented Generation for Large Language Models: A Survey"* ([arXiv:2312.10997](https://arxiv.org/abs/2312.10997)). It proposes three tiers of maturity:

- **Naive RAG** is the pipeline described in Section 2 executed in its simplest form: chunk the corpus, embed the chunks, embed the query, retrieve top-*k* by vector similarity, stuff the results into a prompt, generate. There is no query preprocessing, no reranking, no feedback loop, and no routing logic. It is easy to build and is where almost every RAG project starts — and it is also where most of the well-documented failure modes in Section 4 live.

- **Advanced RAG** adds optimizations before and after the core retrieval step without changing the overall linear shape of the pipeline. **Pre-retrieval** techniques include query rewriting/expansion (reformulating a vague user question into one or more better search queries), query routing (deciding which index or retrieval strategy a query should use), and hypothetical document embeddings (generating a hypothetical answer and embedding *that* to search, since it may resemble the target passage more than the raw question does). **Post-retrieval** techniques include reranking, context compression/filtering (dropping irrelevant retrieved chunks before they reach the LLM), and reordering retrieved passages to counteract position bias.

- **Modular RAG** treats the pipeline as a set of interchangeable, composable modules rather than a fixed sequence — search, memory, routing, and fusion modules that can be added, removed, or rearranged, including patterns like iterative retrieval (retrieve, generate a partial answer, retrieve again based on what's still missing), and adaptive retrieval (deciding at runtime whether retrieval is even necessary for a given query). This is the tier where techniques like agentic RAG and multi-hop retrieval live.

| Dimension | Naive RAG | Advanced RAG | Modular RAG |
|---|---|---|---|
| Pipeline shape | Fixed, linear | Fixed, linear, with pre/post steps added | Flexible graph of interchangeable modules |
| Query handling | Used as-is | Rewritten/expanded/routed | Routed, decomposed, iteratively refined |
| Retrieval | Single pass, top-*k* similarity | Single pass + reranking/filtering | Multi-pass, iterative, adaptive (can skip retrieval) |
| Typical failure addressed | None (baseline) | Irrelevant chunks, position bias | Multi-hop questions, insufficient single-pass evidence |
| Engineering complexity | Low | Moderate | High |
| Good starting point for | Prototypes, proof of concept | Most production systems | Systems with complex, multi-part queries |

For a corpus like Shamela — where a single fiqh question might legitimately require synthesizing rulings from multiple madhhabs across multiple books, or a historical question might need cross-referencing seerah and hadith sources — naive RAG is a reasonable Phase 1 milestone but Advanced RAG (at minimum: query rewriting for classical Arabic normalization + reranking) should be treated as the real production baseline. Modular/agentic patterns (iterative retrieval, routing between hadith/tafsir/fiqh indices) are a natural Phase 3 once the simpler tiers are validated.

## 4. Key Failure Modes of Naive RAG

Naive RAG fails in specific, well-documented ways, not just "sometimes it's wrong":

- **Lost in the middle.** Liu et al. (2023) showed that LLMs use long contexts unevenly: performance on tasks requiring identifying relevant information is highest when that information is near the beginning or end of the context, and degrades substantially when it sits in the middle — even for models explicitly designed for long contexts ([Liu et al., 2023, arXiv:2307.03172](https://arxiv.org/abs/2307.03172); published as [Liu et al., 2024, TACL](https://aclanthology.org/2024.tacl-1.9/)). Practical implication: if you retrieve 10 chunks and dump them into the prompt in arbitrary order, the model may effectively ignore the most relevant one if it lands mid-context. This motivates result reordering and keeping retrieved-context lists as short and precise as possible rather than relying on "just retrieve more and let the model sort it out."

- **Irrelevant or low-precision retrieval.** Vector similarity is a proxy for relevance, not relevance itself. A chunk can be semantically close to a query's *topic* while being useless for actually answering it (e.g., retrieving a passage that merely mentions the word "zakat" in an unrelated aside rather than one stating the actual ruling). This is what reranking is designed to fix, but naive RAG has no such correction step.

- **Chunking destroying context.** If a chunk boundary falls in the middle of an argument, splits a hadith from its isnad, or separates a ruling from the condition that qualifies it, retrieval can surface text that is misleading in isolation even though it was accurate in its original place. This is arguably the single most underestimated failure mode in naive pipelines, because it's invisible until you inspect actual retrieved passages against their source.

- **Stale indexes.** An index is a snapshot. If the underlying corpus is corrected (e.g., a critical edition supersedes an older scan, or metadata is fixed), and the index isn't rebuilt or incrementally updated, the system will confidently serve outdated or superseded content — a subtler version of the same staleness problem RAG was supposed to solve for the LLM's parametric knowledge.

- **Lack of citation/grounding.** Naive RAG concatenates retrieved text into a prompt and asks the model to "answer using the above," but nothing forces the model to actually restrict itself to that evidence, or to say which part of the answer came from which source. Without an explicit grounding/citation mechanism, the model can still blend retrieved content with its own parametric (and possibly wrong) prior knowledge, producing an answer that looks well-sourced but isn't.

Barnett et al.'s empirical study of RAG systems in production, *"Seven Failure Points When Engineering a Retrieval Augmented Generation System"*, documents these and related failures (missing content, wrong chunk size, incorrect specificity, incomplete answers) from real deployments across research, education, and biomedical domains, and makes the broader point that RAG robustness is discovered through operation, not designed perfectly up front ([Barnett et al., 2024, arXiv:2401.05856](https://arxiv.org/abs/2401.05856)).

## 5. When RAG Is (and Isn't) the Right Tool

RAG, long-context prompting, and fine-tuning are not mutually exclusive, and mature systems often combine them — but each has a distinct sweet spot.

| Factor | Favors RAG | Favors long-context | Favors fine-tuning |
|---|---|---|---|
| Corpus size | Large / doesn't fit in any context window (Shamela's 7.6M pages certainly doesn't) | Small enough to fit entirely in the model's context window | N/A — not about fitting data in, but about behavior |
| Data freshness | Changes frequently; index can be updated without retraining | Static or slow-changing | Static; retraining needed for updates |
| Need for citations/auditability | High — retrieval naturally tracks source documents | Possible but must be engineered explicitly | Low — knowledge is diffused in weights, not attributable |
| Cost at scale/query volume | Lower — only relevant chunks are sent per query | Higher — full context resent (or cached) every query; can be 20x+ costlier at volume | Amortized training cost, cheap inference, but retraining is expensive |
| Latency | Extra retrieval step adds latency, but per-query payload is small | No retrieval step, but processing a huge context is itself slow | Fastest at inference (no retrieval, no huge context) |
| Goal: change model behavior/style/format | Not applicable — RAG changes *what* the model knows, not *how* it behaves | Not applicable | This is exactly fine-tuning's strength |
| Goal: answer with traceable evidence from a huge, evolving corpus | Best fit | Poor fit past small corpora | Poor fit |

Fine-tuning teaches a model *how* to reason, write, or format; RAG (and long context) supply *what* it should reason about. For a project like Shamela — a multi-million-page corpus that must remain auditable down to book/page and where correctness must be sourced rather than "remembered" by the model — RAG is not just a good option, it is close to a hard requirement. Long-context prompting alone doesn't scale to a corpus this size, and fine-tuning cannot make a model "know" 7.6 million pages of specific text, nor would it give you a citation trail even if it could.

## 6. Why This Matters More for a Religious Text Corpus

In most RAG applications, a wrong or uncited answer is a quality bug: annoying, sometimes costly, but recoverable. In a corpus of classical Islamic scholarship, the calculus is different. A fabricated hadith attribution, a fiqh ruling presented without its source, or a tafsir claim that blends the actual text of a classical scholar with the model's own paraphrase, is not merely "low quality" output — it is a claim about religious authority and textual authenticity. Users of such a system are often implicitly trusting it the way they would trust a library catalog or a scholar's citation, not the way they'd shrug off a wrong answer from a general chatbot.

This reframes several RAG properties from "nice engineering hygiene" to "non-negotiable requirement": citation grounding (Section 2, stage 9) must be treated as a first-class output, not a formatting afterthought; retrieval precision matters more than usual because presenting the *wrong* hadith or ruling with confidence is worse than saying nothing; and staleness handling matters because critical editions and scholarly corrections are a normal part of how this literature is maintained — the index has to reflect the text that scholars actually recognize as authoritative, not a frozen first pass.

Concretely, this means the system should be designed so that every generated claim is traceable to a specific book, volume, and page in the underlying corpus, and the generation stage should be constrained (through prompting and evaluation) to refuse or hedge rather than fabricate when retrieval doesn't surface adequate evidence. This theme — hallucination and citation as trust/authority failures specific to religious and scholarly domains — is expanded substantially in the evaluation document later in this series, which covers how to actually measure faithfulness, attribution accuracy, and refusal behavior for this kind of corpus.

## 7. Further Reading

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) — the original RAG paper.
- Gao, Y., et al. (2023/2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* [arXiv:2312.10997](https://arxiv.org/abs/2312.10997) — source of the Naive/Advanced/Modular RAG taxonomy used in Section 3.
- Liu, N. F., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* [arXiv:2307.03172](https://arxiv.org/abs/2307.03172); published version in *Transactions of the Association for Computational Linguistics*, 12:157–173 (2024): [ACL Anthology](https://aclanthology.org/2024.tacl-1.9/).
- Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., & Abdelrazek, M. (2024). *Seven Failure Points When Engineering a Retrieval Augmented Generation System.* CAIN '24. [arXiv:2401.05856](https://arxiv.org/abs/2401.05856).
- Pinecone. *Retrieval Augmented Generation* learning series — practical, vendor-neutral-ish walkthrough of RAG pipeline components. [pinecone.io/learn/series/rag](https://www.pinecone.io/learn/series/rag/).
- Meilisearch. *RAG vs. long-context LLMs: A side-by-side comparison* — cost and use-case comparison referenced in Section 5. [meilisearch.com/blog/rag-vs-long-context-llms](https://www.meilisearch.com/blog/rag-vs-long-context-llms).

**Next in this series:** chunking strategies, embedding model selection for Arabic/classical text, retrieval strategies (hybrid search, reranking in depth), knowledge graphs and GraphRAG, and evaluation methodology for RAG over religious/scholarly corpora.
