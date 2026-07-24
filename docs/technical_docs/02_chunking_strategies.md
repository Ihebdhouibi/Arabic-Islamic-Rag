# Chunking Strategies for Retrieval-Augmented Generation

> Document 2 of 7 in this technical series. This doc assumes you already understand the basic RAG loop (index → retrieve → generate) from Document 1. It focuses entirely on **chunking**: how you cut source documents into the units that get embedded, indexed, and retrieved. The deep application of these ideas to the Shamela corpus (8,589 books, ~7.6M pages of classical Arabic Islamic texts) lives in a separate synthesis document; here we build the conceptual toolkit.

## Why chunking is a first-class design problem

Every RAG system embeds and indexes *chunks*, not whole documents. The chunk is the atomic unit of retrieval: it is what gets scored against a query, what gets stuffed into the LLM's context window, and what — implicitly — the LLM is asked to reason about as a coherent piece of information. Get chunking wrong and every downstream stage inherits the damage: embeddings represent incoherent or truncated ideas, retrieval returns fragments that don't answer the question, and generation hallucinates to fill in gaps the retriever couldn't supply.

Chunking sits at the intersection of three constraints that are often in tension:

- **Semantic coherence** — a chunk should represent one complete idea, argument, narration, or fact, not half of one.
- **Retrieval granularity** — smaller chunks let the retriever pinpoint precisely relevant material; larger chunks preserve context but dilute the similarity signal with irrelevant surrounding text.
- **Budget constraints** — embedding model context limits, vector-store payload sizes, and the LLM's generation context window all impose hard ceilings on how much and how you can chunk.

There is no universal "right" chunk size. The right strategy depends on document structure, query patterns, embedding model, and downstream task. What follows is a survey of the major strategies in use today, roughly ordered from naive to increasingly structure- and semantics-aware.

---

## 1. Fixed-size / naive token-window chunking

**What it is.** Split the raw text into windows of *N* tokens (or characters), typically with some overlap (e.g., 512 tokens with 50-token overlap) so that ideas spanning a boundary aren't completely severed.

**How it works.** A tokenizer counts tokens (or characters) linearly through the document and cuts every *N* units, sliding the window forward by `N - overlap` each time. No attention is paid to sentence, paragraph, or semantic boundaries.

**Pros**
- Trivial to implement, extremely fast, deterministic.
- Produces uniformly sized chunks, which simplifies batching and cost estimation for embedding APIs.
- Works acceptably on unstructured, prose-heavy text with weak internal structure.

**Cons**
- Routinely slices sentences, footnotes, tables, and — critically for our domain — a hadith's isnad from its matn, or a poetic verse (bayt) in half.
- Overlap mitigates but does not solve boundary damage; it also duplicates content across chunks, inflating index size and retrieval noise.
- Ignores document semantics entirely, so two unrelated ideas can end up concatenated in one chunk while a single coherent idea gets split across two.

**When to use it.** As a fast baseline, for homogeneous unstructured text with no reliable structure to exploit, or as a fallback for content types no smarter splitter handles well. It is a reasonable starting point to benchmark smarter strategies against — but for a structured, classical-text corpus like Shamela, it should not be the terminal design, because it will routinely fracture isnad chains, verses, and cross-referenced tafsir comments.

---

## 2. Recursive character/structure-based splitting

**What it is.** Instead of cutting blindly at a fixed offset, try to split on the "most natural" separator that keeps chunks under the size limit, falling back to coarser splitting only when necessary. This is the strategy popularized by LangChain's `RecursiveCharacterTextSplitter`.

**How it works.** The splitter is given an ordered list of separators, by default `["\n\n", "\n", " ", ""]` — paragraph breaks first, then line breaks, then spaces, then raw characters as a last resort. It recursively tries the first separator; if the resulting piece is still larger than `chunk_size`, it re-splits that piece using the next separator in the list, and so on, down to individual characters if nothing else works. A `chunk_overlap` parameter still stitches adjacent chunks together for context continuity. LangChain also ships language-aware variants (Markdown, HTML, code) with separators tuned to headers, `##` levels, or function/class boundaries.

**Pros**
- Respects natural document structure (paragraphs, then sentences, then words) far better than blind character counting, at almost the same implementation cost.
- Configurable separator hierarchy makes it easy to adapt to a document format (e.g., add `"۔"` or `"۔ "`-style Arabic sentence terminators, or split preferentially on section markers).
- Still fast, deterministic, and cheap — no embedding calls or LLM calls required at chunk time.

**Cons**
- Still fundamentally size-driven: it stops trying to find semantic breakpoints once the size limit is roughly satisfied, so it can still bisect a coherent argument if no natural separator appears near the target size.
- Separator lists are language- and genre-specific; classical Arabic typography (limited modern punctuation, run-on paragraphing in older print editions) can starve the splitter of good breakpoints, causing it to fall back to word- or character-level splits more often than for modern English prose.

**When to use it.** The default, pragmatic choice for most general-purpose RAG pipelines and a strong improvement over naive fixed-size splitting whenever documents have *some* typographic structure (paragraphs, line breaks) even without a formal outline. It's a good "second baseline" to compare more expensive strategies against.

---

## 3. Semantic chunking (embedding-similarity breakpoint detection)

**What it is.** Instead of relying on typographic separators, detect topic shifts by comparing the embeddings of adjacent sentences (or small sentence groups) and cutting where similarity drops sharply. This approach was popularized by Greg Kamradt's "5 Levels of Text Splitting" work and implemented in LlamaIndex as `SemanticSplitterNodeParser`.

**How it works.** The document is first split into sentences. A sliding window groups each sentence with a small buffer of neighbors (e.g., one sentence before and after) to give the embedding model enough context. Each windowed group is embedded, and the cosine distance between embeddings of consecutive windows is computed. Distances are converted to a distribution, and a breakpoint is placed wherever the distance exceeds a percentile threshold (e.g., the 95th percentile) — i.e., wherever the topic shifts more sharply than "normal" for that document. Sentences between breakpoints become one chunk.

**Pros**
- Adapts chunk boundaries to actual topic shifts rather than arbitrary size or punctuation, which can meaningfully improve retrieval precision on discursive, multi-topic prose.
- No fixed chunk size is imposed a priori; chunks naturally vary to match content density.

**Cons**
- Computationally expensive relative to rule-based splitting: every sentence window requires an embedding call before you even build your retrieval index.
- Breakpoint thresholds are dataset-sensitive and require tuning; a threshold tuned for modern English news prose will not transfer cleanly to classical Arabic religious argumentation with different rhetorical rhythms.
- Evidence on real-world benefit is mixed: a 2024 NAACL Findings paper, *"Is Semantic Chunking Worth the Computational Cost?"* (Qu, Tu & Bao, arXiv:2410.13070), found that semantic chunking's retrieval gains over simple fixed-size baselines were inconsistent across tasks and often did not justify the extra embedding compute.
- Purely lexical/embedding-based breakpoints know nothing about domain structure — they cannot tell that an isnad chain "belongs" attached to its matn even if the embedding drift between them looks like a topic shift.

**When to use it.** Long, unstructured discursive prose where topic boundaries aren't marked typographically and you can afford the extra embedding calls at ingestion time — e.g., long-form essays, transcripts, or narrative fiction. Given the mixed empirical evidence and its cost, treat it as one candidate to A/B test, not a default.

---

## 4. Document-structure-aware / hierarchical chunking

**What it is.** Use the document's own logical structure — headings, chapter/section markers, a table of contents, Markdown header levels, HTML `<hN>` tags — as the chunking boundaries, rather than inferring boundaries from size or embeddings.

**How it works.** A structure-aware splitter first parses the document into its hierarchy (book → chapter → section → paragraph, or `h1` → `h2` → `h3` → body text). Each leaf section becomes a candidate chunk, further subdivided by a recursive/size-based splitter only if it still exceeds the size limit. Critically, each chunk can retain metadata about its position in the hierarchy (chapter title, section title, page range), which can be prepended to the chunk text or stored as retrievable metadata for filtering and citation.

**Pros**
- Chunks correspond to units the author/editor intended as coherent (a chapter, a numbered ruling, a named section) — this is usually the strongest available proxy for "one idea per chunk."
- Preserves rich metadata (title trail, page numbers) essentially for free, which is invaluable for citation and for hybrid filtered search.
- Cheap: no embedding or LLM calls needed if the structure (headings, TOC) is already known or easily parsed.

**Cons**
- Entirely dependent on structure existing and being reliably extractable; scanned or poorly-OCR'd documents may have no machine-readable headings.
- Sections vary wildly in length — a "section" can be one sentence or twenty pages — so you still need a secondary size-based splitter as a fallback for oversized sections, and a merging step for undersized ones.
- Structure boundaries don't always align with retrieval-optimal boundaries (a heading might introduce a topic that isn't fully "about" that heading for its first few sentences).

**When to use it.** Whenever reliable structure exists — and this is the single most relevant strategy for a corpus like Shamela's, where books already carry a hierarchical table of contents mapped to page IDs. TOC-anchored chunking lets you use author-intended boundaries (chapters, bab/fasl divisions) as the primary chunk grid, falling back to recursive splitting only inside oversized sections. This is also the standard approach for technical documentation, legal codes, and any Markdown/HTML corpus with real heading hierarchies.

---

## 5. Proposition-based chunking / atomic fact extraction

**What it is.** Decompose text into "propositions": short, self-contained, atomic statements, each expressing a single fact and rewritten (typically by an LLM) so it makes sense without surrounding context — for example, resolving pronouns and implicit references. This is the approach formalized in the "Dense X Retrieval" paper.

**How it works.** *Dense X Retrieval: What Retrieval Granularity Should We Use?* (Chen et al., arXiv:2312.06648) introduces propositions as an alternative retrieval unit to passages or sentences, and constructs FactoidWiki — Wikipedia decomposed into ~250 million propositions — to test the idea. An LLM is prompted to read a passage and emit a list of atomic, decontextualized factual statements (e.g., turning "He was born there in 1980" into "Ibn X was born in [city] in 1980"). Each proposition is then embedded and indexed as its own retrievable unit, sometimes alongside the passage it came from for provenance.

**Pros**
- The paper's experiments show retrieval by proposition outperforms passage- or sentence-level retrieval on open-domain QA accuracy and downstream QA performance at a fixed retrieved-word budget, because propositions are maximally information-dense and self-contained.
- Excellent for precise fact retrieval and for building knowledge-graph-adjacent representations (each proposition is close to a subject-predicate-object triple in spirit).

**Cons**
- Requires an LLM call per passage at ingestion time, making it one of the most compute- and cost-intensive chunking strategies at scale — expensive to run over millions of pages.
- Decontextualization by an LLM can introduce factual drift or hallucinated resolution of ambiguous references, which is a serious concern for a corpus with theological and legal authority.
- Loses some rhetorical and argumentative structure (chains of reasoning, qualifications, scholarly caveats) that classical texts rely on — a proposition-atomized fatwa may lose the conditions attached to its ruling.

**When to use it.** High-value, fact-dense corpora where atomic-fact retrieval materially improves QA accuracy and where ingestion cost is acceptable — e.g., building a fact-checking index or a knowledge base layer on top of a primary passage-level index, rather than as the sole chunking strategy for an entire multi-million-page corpus.

---

## 6. Late chunking

**What it is.** A 2024 technique from Jina AI that inverts the traditional order of operations: instead of chunking text and then embedding each chunk independently (losing cross-chunk context), you embed the *entire* document first with a long-context embedding model, and only afterward pool the resulting token-level embeddings into chunk-level vectors.

**How it works.** Described in *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models* (Günther et al., arXiv:2409.04701) and Jina AI's engineering blog, the method runs the transformer's token embedding layer over the full document (up to the model's context limit, e.g., 8192 tokens for `jina-embeddings-v2`), producing a token embedding for every token *in the context of the whole document*. Chunk boundaries are then applied *after* this pass, and each chunk's final embedding is a mean-pooling of the token embeddings that fall within its span. Because every token embedding was computed with full-document attention, even a chunk near the end of a document "knows about" content at the start (e.g., what a pronoun or "the aforementioned ruling" refers to).

**Pros**
- Each chunk embedding carries global document context without needing to duplicate that context as literal text in the chunk (unlike Contextual Retrieval, below) — no extra tokens are added to the chunk.
- Directly fixes the classic "pronoun/reference resolution" failure mode of chunking, where a chunk like "This ruling applies to it as well" is meaningless in isolation.
- Requires no LLM calls — only a single forward pass of a long-context embedding model per document, which is cheaper than proposition extraction or LLM-based contextualization.

**Cons**
- Requires a long-context embedding model that exposes token-level embeddings before pooling (not all embedding APIs support this); document length is still capped by the model's max context (e.g., 8k tokens), so very long documents must be processed in overlapping macro-segments.
- Implementation is more involved than any rule-based splitter — requires custom pooling logic, not just an off-the-shelf splitter call.
- Newer technique (published September 2024) with a smaller ecosystem of battle-tested tooling compared to recursive or structure-based splitting, though adoption is growing (Weaviate, Elasticsearch, Milvus, and Qdrant have all published integrations).

**When to use it.** Long documents where cross-references, anaphora, and document-wide context matter but you want to avoid the token overhead and LLM cost of explicit contextualization — a strong candidate for long classical commentaries where later passages implicitly reference earlier definitions or rulings, provided your embedding model supports the required token-level access.

---

## 7. Contextual Retrieval (Anthropic, September 2024)

**What it is.** A technique from Anthropic that addresses context loss by prepending a short, LLM-generated explanation of *where a chunk fits in the document* to the chunk itself, before embedding it and indexing it for BM25 — rather than changing how the chunk boundaries are drawn.

**How it works.** Described in Anthropic's engineering post *"Introducing Contextual Retrieval"* (anthropic.com/engineering/contextual-retrieval, September 19, 2024), the method takes each standard chunk (produced by whatever splitter you like) plus the full original document, and asks an LLM (with prompt caching to keep costs low) to generate 50–100 tokens of situating context — e.g., "This chunk is from an SEC filing on ACME Corp's Q2 2023 revenue; it discusses the 3% year-over-year decline in the widgets division." This generated context is prepended to the chunk text before it is embedded (**Contextual Embeddings**) and before it is indexed into a lexical index (**Contextual BM25**). Anthropic reports this combination reduces failed retrievals by 49%, and by 67% when combined with reranking, relative to naive embedding-only retrieval, in their evaluation.

**Pros**
- Directly attacks the "chunk lacks context" problem without requiring a specialized embedding model — works with any embedding provider and any chunking strategy underneath it.
- Improves both dense (embedding) and sparse (BM25) retrieval simultaneously, since the context is added to both representations.
- Prompt caching (holding the full document in cache while generating context for each of its chunks) keeps the added LLM cost comparatively low for a corpus that is chunked once and re-queried many times.

**Cons**
- Still requires one LLM call per chunk at ingestion time (mitigated, not eliminated, by caching), so it's meaningfully more expensive than any non-LLM strategy — a real consideration at the scale of millions of chunks.
- Prepended context adds tokens to every embedding and index entry, slightly increasing storage and embedding cost per chunk.
- Quality depends on the contextualizing LLM correctly summarizing document-level context; errors or omissions there propagate into every affected chunk.

**When to use it.** High-stakes corpora where retrieval failures are costly and where chunks are inherently ambiguous out of context — a very good fit conceptually for classical commentary literature, where a chunk discussing "this narration" or "the aforementioned scholar's opinion" is uninterpretable without knowing which book, chapter, and prior discussion it belongs to. Anthropic explicitly positions it as complementary to good chunking, not a replacement for it — you still choose a base chunking strategy (e.g., structure-aware) and then apply contextualization on top.

---

## 8. Agentic / LLM-driven chunking

**What it is.** Let an LLM read the document and directly decide where the semantically coherent breakpoints are, rather than inferring them from embeddings or fixed rules. This is Kamradt's "Level 5: reasoned chunking" and is implemented, for example, in Chroma's research code as an `LLMChunker`.

**How it works.** The document (or a sliding window of it) is passed to an LLM with a prompt asking it to identify natural breakpoints — e.g., "insert a marker between chunks such that each chunk is a complete, self-contained unit of meaning" — or to directly output the segmented chunks. Because LLMs have a limited context window, long documents must be processed in overlapping segments, with the model asked to only commit to boundaries that aren't near the segment's own edges (to avoid missing context on either side).

**Pros**
- Chroma's technical report on chunking evaluation found LLM-based semantic chunking achieved the highest recall (91.9%) among all strategies tested, reflecting the LLM's ability to use genuine language understanding — not just embedding distance — to find true topic boundaries.
- Can incorporate domain instructions directly in the prompt (e.g., "never split a hadith's isnad from its matn," "keep a Quranic verse and its immediate tafsir comment together"), making it uniquely flexible for domain-specific rules without custom code.

**Cons**
- The most expensive strategy computationally and financially: an LLM call (or several, for long documents processed in windows) per document, at million-page scale, is a significant infrastructure and cost commitment.
- Slowest to run at ingestion time; not practical for real-time or frequently re-indexed corpora.
- Non-deterministic: different runs (or model versions) can produce different boundaries, complicating reproducibility and versioning of the index.

**When to use it.** Best reserved for smaller, high-value corpora, or for a one-time high-quality pass over content where retrieval quality matters more than ingestion cost — or as a strategy applied selectively to a subset of especially difficult documents (irregular structure, mixed genres) rather than uniformly across an entire multi-million-page corpus.

---

## 9. Domain-specific / entity-atomic chunking

**What it is.** The general principle that the "correct" chunk size is often not a token count at all, but the domain's own natural atomic unit of meaning — one FAQ pair, one legal clause, one code function, one hadith, one verse-plus-commentary.

**How it works.** Rather than imposing an external splitter, you write a parser specific to the corpus's known schema. Examples:
- **Q&A datasets**: one chunk = one question-answer pair, never split.
- **Legal text**: one chunk = one numbered clause or section of a statute, since clauses are the unit lawyers cite and reason about.
- **Source code**: one chunk = one function or class (as LangChain's and Kamradt's "document-specific" splitters do via AST-aware parsing), preserving the unit a developer would actually want returned.
- **Hadith collections**: one chunk = one hadith (isnad + matn together), since that is the unit scholars narrate, grade, and cite — never a page or fixed window that might bisect a chain of narrators.
- **Tafsir**: one chunk = one verse plus its associated exegetical comment(s), since tafsir is structured around specific ayat, not arbitrary page breaks.
- **Poetry**: one chunk boundary never falls mid-verse (bayt); a verse (and often a matching couplet/misra pair) is the smallest indivisible unit.

**Pros**
- When a genuine atomic unit exists, this is close to a strict upper bound on achievable precision: the chunk cannot be topically incoherent because it *is* the domain's unit of coherence by definition.
- Typically implementable with a deterministic parser (no embeddings, no LLM calls) once the schema is known, making it cheap and fast at scale.
- Naturally carries rich, exact metadata (hadith grading, verse number, clause number) useful for filtering and citation.

**Cons**
- Requires upfront investment in a domain-aware parser; only works where the corpus has (or can be given) reliable machine-readable structure marking these units.
- Atomic units vary enormously in length (a one-line hadith vs. a multi-page one; a two-line verse vs. a long qasida with running commentary), so size-based secondary handling is still needed for outliers.
- Not always a single clean unit — some genres deliberately interleave units (matn/sharh dual streams, interlinear commentary) that need a chunking *model*, not just a splitter, to decide whether to keep the base text and its commentary together or separate them into parallel indices.

**When to use it.** Whenever the domain has a well-defined atomic unit, this should be the *default* starting point, ahead of any generic splitter — generic strategies should only be layered on top of an atomic unit that is itself too large to embed as one chunk (e.g., a very long hadith commentary), never used in place of respecting the atomic boundary.

---

## 10. Chunk size and overlap tradeoffs

Three budgets interact whenever you set chunk size:

**Embedding model context limits.** Every embedding model has a maximum input length (512 tokens for many older BERT-based models, up to 8192 tokens for long-context models like `jina-embeddings-v2`/`v3`). Chunks must fit within this limit or be truncated silently — a common, hard-to-detect source of degraded retrieval quality. Long-context embedding models raise the ceiling but don't eliminate the tradeoff, since a single embedding vector representing an 8k-token chunk is a coarser semantic summary than one representing a 200-token chunk (information gets averaged away).

**Retrieval precision vs. recall.** Smaller chunks yield sharper, more targeted similarity matches — precision — because the embedding represents a narrow, specific idea, so a query about that idea produces a high, unambiguous similarity score. Larger chunks improve recall for questions that need broader context (a chunk covering an entire argument is more likely to "contain" the answer even if the query's phrasing doesn't closely match any single sentence), but they dilute the similarity signal, since the embedding is an average over more, and more varied, content. Pinecone's own guidance and multiple production write-ups converge on 400–512 tokens with roughly 10–20% overlap as a reasonable general-purpose default, but this should be validated empirically against your own corpus and queries (e.g., sweeping chunk sizes of 200/400/600 tokens and measuring recall@k), not assumed.

**Generation context window budget.** Every chunk retrieved consumes tokens in the LLM's context window at generation time, competing with the system prompt, conversation history, and the model's own output budget. Smaller chunks let you fit more distinct pieces of evidence (higher k) into a fixed context budget, which helps when an answer needs synthesis across several sources; larger chunks reduce k but give each piece of evidence more surrounding support, reducing the risk that a fact is retrieved without the qualifications or conditions attached to it in the source. Overlap adds a further tax here: any duplicated text between adjacent retrieved chunks is wasted context-window budget.

The overarching lesson: chunk size is not a single global knob to tune once. It's a design decision made per content type, in light of a specific embedding model's context limit, a specific retrieval precision/recall target, and a specific generation budget — which is exactly why structure-aware and domain-atomic chunking (Sections 4 and 9) tend to outperform any single fixed-size choice on heterogeneous corpora.

---

## 11. Comparison table

| Strategy | Implementation complexity | Context preservation | Computational cost (ingestion) | Best-fit content | Notable adopters / tools |
|---|---|---|---|---|---|
| Fixed-size token window | Very low | Poor — arbitrary cuts | Very low | Homogeneous unstructured prose; quick baselines | LangChain `CharacterTextSplitter`, most vector-DB quickstarts |
| Recursive character/structure splitting | Low | Moderate — respects paragraphs/sentences | Very low | General-purpose text with light typographic structure | LangChain `RecursiveCharacterTextSplitter`, LlamaIndex `SentenceSplitter` |
| Semantic chunking (embedding breakpoints) | Medium | Good for topic shifts, blind to domain structure | Medium (1 embedding call per sentence window) | Long discursive prose without typographic structure | LlamaIndex `SemanticSplitterNodeParser`, Kamradt's "Level 4" |
| Document-structure-aware / hierarchical | Medium | Very good — matches author-intended units | Low (parsing only, no ML calls) | Docs with real headings/TOC: technical docs, legal codes, books with tables of contents | Markdown/HTML splitters, custom TOC-driven parsers |
| Proposition-based / atomic facts | High | Excellent for isolated facts, poor for argument chains | High (1+ LLM call per passage) | Fact-dense QA corpora, knowledge-graph construction | Dense X Retrieval / FactoidWiki (arXiv:2312.06648) |
| Late chunking | High (engineering) | Excellent — full-document context in every chunk embedding | Medium (1 long-context embedding pass per document) | Long documents with cross-references/anaphora | Jina AI (`jina-embeddings-v2/v3`), Weaviate, Elasticsearch, Qdrant, Milvus integrations |
| Contextual Retrieval | Medium–High | Excellent — explicit situating context per chunk | High (1 LLM call per chunk, cached) | High-stakes corpora where chunks are ambiguous out of context | Anthropic (Claude + prompt caching), AWS Bedrock Knowledge Bases |
| Agentic / LLM-driven chunking | High | Excellent — genuine semantic judgment | Very high (1+ LLM call per document/window) | Small, high-value corpora; one-time high-quality passes | Chroma `LLMChunker` (Chroma Technical Report) |
| Domain-specific / entity-atomic | Medium (custom parser) | Excellent when a true atomic unit exists | Low (deterministic parsing) | Q&A pairs, legal clauses, code functions, hadith, verses | Custom pipelines per domain; the standard in legal & code RAG |

---

## 12. Applying these concepts to structured multi-genre corpora

Classical religious and legal text collections — of which Shamela is a canonical large-scale example — are unusual among RAG corpora in that they are simultaneously large-scale *and* richly structured, which is a combination generic chunking research (mostly built and evaluated on Wikipedia, news, or SEC filings) doesn't directly address. The practical implication is that the strategies above should be combined hierarchically rather than chosen exclusively. A book's existing table of contents, already mapped to page IDs, is a gift: it gives you author-intended, hierarchical boundaries (Section 4) essentially for free, without any embedding or LLM cost, and should be treated as the primary chunking grid wherever it exists — with recursive or semantic splitting invoked only as a fallback inside oversized leaf sections.

Genre matters more than book-level structure in several important cases, which is why entity-atomic chunking (Section 9) needs to sit *above* structure-based chunking in the decision hierarchy, not below it. Hadith collections should never be windowed by page or token count; the atomic unit is one hadith — isnad and matn together — because splitting a chain of narrators from its narration destroys the very thing a hadith scholar or student would want retrieved intact and gradable. Tafsir works are organized around specific verses, so verse-anchoring, not page-anchoring, should define chunk boundaries, with each chunk ideally carrying the verse's reference as metadata for precise citation. Poetry embedded within prose commentary imposes a hard constraint that generic splitters routinely violate: a verse (bayt) — and often its two hemistichs together — must never be bisected, regardless of what a token-count splitter would otherwise do at that offset.

The matn/sharh (base text/commentary) pattern found in many classical works is the trickiest structural case, because it is not simply hierarchical — it's two parallel streams referring to each other. A commentary passage may be incomprehensible without its corresponding base-text lemma, but concatenating them naively can also conflate two authors' distinct voices and centuries-apart contexts in one chunk. This is a case where techniques like Contextual Retrieval (Section 7) or late chunking (Section 6) are conceptually attractive: rather than merging matn and sharh into one chunk, keep them as separate, atomically-chunked units, but annotate the sharh chunk with situating context (which matn lemma it comments on, from which book) so it remains interpretable in isolation. The deep design decisions for these cases — exact chunk schemas, how to store isnad/matn separately for hadith while keeping them jointly retrievable, and how to handle books that lack a usable TOC — are the subject of the corpus-specific synthesis document; the point to take from this section is that no single strategy from the table above is sufficient alone, and a production design for this kind of corpus is necessarily a layered combination of structure-aware, domain-atomic, and context-preserving techniques.

---

## Further reading

- Anthropic — [Introducing Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) (engineering blog, Sept 19, 2024) and the accompanying [Claude Cookbook guide](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide).
- Jina AI — [Late Chunking in Long-Context Embedding Models](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) and the paper Günther et al., [*Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*](https://arxiv.org/abs/2409.04701) (arXiv:2409.04701); reference implementation at [github.com/jina-ai/late-chunking](https://github.com/jina-ai/late-chunking).
- Chen et al., [*Dense X Retrieval: What Retrieval Granularity Should We Use?*](https://arxiv.org/abs/2312.06648) (arXiv:2312.06648) — the propositions paper and FactoidWiki dataset.
- Chroma — [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking) (Smith & Troynikov, Chroma Technical Report, July 2024); code and `LLMChunker`/`ClusterSemanticChunker` implementations at [github.com/brandonstarxel/chunking_evaluation](https://github.com/brandonstarxel/chunking_evaluation).
- Qu, Tu & Bao, [*Is Semantic Chunking Worth the Computational Cost?*](https://arxiv.org/abs/2410.13070) (arXiv:2410.13070, NAACL 2025 Findings) — a critical counterpoint on semantic chunking's real-world benefit.
- LangChain — [RecursiveCharacterTextSplitter reference](https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter) and [text splitter integrations](https://docs.langchain.com/oss/python/integrations/splitters).
- LlamaIndex — [Semantic Chunker documentation](https://developers.llamaindex.ai/python/examples/node_parsers/semantic_chunking/) (`SemanticSplitterNodeParser`).
- Pinecone — [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/).
- Greg Kamradt — [5 Levels of Text Splitting](https://x.com/GregKamradt/status/1699465826485862543) (original thread) and the [Full Stack Retrieval tutorial notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb).
- Weaviate — [Late Chunking: Balancing Precision and Cost in Long Context Retrieval](https://weaviate.io/blog/late-chunking) (independent engineering write-up and integration).
