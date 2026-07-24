# 05 — Knowledge Graphs and GraphRAG

> Part of a 7-document series on building a production RAG system over the Shamela digital
> library (~8,589 classical Arabic Islamic texts). This document covers knowledge graphs and
> GraphRAG in depth. See the companion documents for fundamentals, chunking, embeddings,
> retrieval strategies, and evaluation.

## 1. What "graph" actually means here

If you've only worked with vector search, the word "graph" in GraphRAG probably sounds like
marketing. It isn't. It refers to a specific, much older data structure, and understanding it is
the whole point of this document.

A **knowledge graph** is made of two things:

- **Nodes** (also called entities or vertices) — a person, a place, a book, a concept, a Quran
  verse, a hadith. In the Shamela corpus: a narrator like *Abu Hurayrah*, a book like *Sahih
  al-Bukhari*, a verse like *2:255 (Ayat al-Kursi)*, a triliteral root like *ك-ت-ب*.
- **Edges** (relationships) — a *typed, directional* link between two nodes. Not "these two
  things are similar," but "this specific relationship holds." In the Shamela corpus:
  *narrator A → NARRATED-FROM → narrator B*, *tafsir book X → COMMENTS-ON → verse 2:255*,
  *hadith in book Y → SAME-HADITH-AS → hadith in book Z*, *word form → DERIVED-FROM → root*.

That's it. A knowledge graph is a set of (subject, predicate, object) facts — the same shape as
a relational database's foreign keys, just thought of as a network you can traverse rather than
a set of tables you join. If you've ever drawn an entity-relationship diagram, you've drawn a
knowledge graph.

### The contrast with a vector store

A vector store also connects pieces of text to each other, but only in one specific, unlabeled
way. Every chunk gets an embedding, and "closeness" in that embedding space means "these two
chunks talk about semantically similar things." That's the *only* relationship a vector index
knows. It has no concept of:

- **Direction** — "A cites B" is not the same as "B cites A," but cosine similarity is symmetric.
- **Type** — a vector index can't distinguish "narrated from," "refuted by," and "commented on."
  Everything collapses into one undifferentiated notion of similarity.
- **Multi-hop structure** — a vector index returns a flat top-*k* list. It has no native
  operation for "follow this relationship, then follow the next one, three times."

Put concretely: if narrator A narrated hundreds of hadiths from narrator B, their biography
embeddings might not even be particularly close — two people's biographies rarely read as
"similar text," even though they are tightly *related*. A knowledge graph captures exactly the
relationship a vector store misses, by storing it explicitly instead of hoping similarity
approximates it.

## 2. Why graphs matter for retrieval

Three classes of questions expose where pure vector search struggles and graphs help.

**Multi-hop reasoning.** "Which narrators did Ahmad ibn Hanbal narrate from, and is any link in
that chain classified as *da'if* (weak) by the critics?" Answering this means: (1) find every
NARRATED-FROM edge from Ahmad ibn Hanbal, (2) follow each resulting narrator's own NARRATED-FROM
edges up the chain, (3) check every narrator touched against their jarh wa ta'dil record for a
weak grading. That's a graph traversal — essentially free once isnad links are edges, and nearly
impossible to do reliably by asking a vector index for "chunks about his teachers," since nothing
guarantees you get the full chain rather than just the most *semantically prominent* mentions.

**Global / holistic questions.** "What are the major theological disputes running through this
collection of Hanafi fiqh texts?" No single chunk answers this — the answer is a synthesis across
potentially thousands of documents. Top-*k* vector retrieval returns a handful of locally relevant
passages and misses the forest for the trees — precisely the failure mode Microsoft's GraphRAG
paper set out to fix (see §4).

**Exact structured lookups.** "Show every book that comments on verse 2:255" or "list every
hadith appearing in both Sahih al-Bukhari and Sahih Muslim." These aren't fuzzy semantic
questions — they're exact joins over known relationships. Vector similarity is the wrong tool for
an exact join: embeddings retrieve things *about* verse 2:255 without guaranteeing completeness,
and have no way to express "all of them, no more, no less."

## 3. What vector similarity structurally cannot do

To make the contrast concrete, here is specifically what breaks:

- **Transitive / multi-hop chains.** Similarity is not transitive. If chunk A is similar to chunk
  B, and chunk B is similar to chunk C, nothing guarantees A is similar to C — so you cannot chain
  vector search hops to reconstruct a narrator chain, a citation path, or a derivation chain of
  Arabic word forms back to a root.
- **Guaranteed exact-match completeness.** Similarity search returns "the *k* most similar
  things," ranked by a continuous score with no natural cutoff for "all of them." A structured
  join (`WHERE verse_id = '2:255'`) returns exactly the right set, every time. Retrieval by
  embedding will nearly always leave something out, or let something irrelevant in, once you need
  a *complete, correct* set rather than "probably the most relevant handful."
- **Flattening relational structure.** Embedding a passage compresses it into a single dense
  vector that captures "aboutness," but throws away the explicit structure of *who said what about
  whom*. A biography chunk about narrator A and a chunk of A's isnad both embed as "text about
  narrator A" — the vector doesn't preserve that one is a biographical judgment and the other a
  chain membership, or how either connects to a third narrator two hops away.

None of this makes vector search bad — it is excellent at open-ended, fuzzy, semantic questions
("find passages discussing the permissibility of X"). It simply is not the right structure for
questions that are inherently about relationships and joins rather than aboutness.

## 4. Microsoft's GraphRAG: extraction, communities, and global summarization

In April 2024, Microsoft Research published *"From Local to Global: A Graph RAG Approach to
Query-Focused Summarization"* (arXiv:2404.16130), releasing the implementation as
[microsoft/graphrag](https://github.com/microsoft/graphrag), docs at
[microsoft.github.io/graphrag](https://microsoft.github.io/graphrag/). The paper's starting point
is exactly §2's "global questions" problem: naive RAG fails on sensemaking questions like "what
are the main themes in this dataset?" because no single chunk — or top-*k* set of chunks —
contains that answer.

GraphRAG's pipeline has three stages:

1. **LLM-driven extraction.** An LLM reads through the source text and extracts entities (people,
   organizations, concepts) and relationships between them, building a knowledge graph
   automatically from *unstructured* text — no pre-existing structured data required. This is the
   expensive part: every chunk of the corpus gets one or more LLM calls just to build the graph,
   before a single user question is ever asked.
2. **Community detection.** The resulting graph is clustered into hierarchical **communities**
   using the Leiden algorithm — a modern graph-clustering method that groups densely-interconnected
   nodes and, applied recursively, produces communities at multiple levels of granularity (broad
   top-level themes down to tight, specific clusters).
3. **Community summarization.** An LLM generates a natural-language summary of each community.
   For a "global" question, GraphRAG retrieves and synthesizes across these pre-computed community
   summaries (a "map-reduce" over summaries) instead of searching for individual chunks — which is
   how it answers holistic questions that plain top-*k* retrieval cannot.

Microsoft later published a lower-cost variant, **LazyGraphRAG**, which defers most of the
expensive extraction/summarization work until query time and reported much lower indexing costs
for comparable quality (see the [Microsoft Research
blog](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)).
That evolution is telling: the original GraphRAG's biggest practical drawback was the cost of
step 1 — LLM extraction over an entire corpus.

## 5. Two very different ways a graph comes into existence

This is the distinction that matters most for this project, so it's worth stating plainly.

**Path A — extract the graph from raw text with an LLM.** This is what the Microsoft GraphRAG
paper does, and what most GraphRAG tutorials assume you need. It's necessary when the source is
unstructured prose and no one has already labeled the entities and relationships in it. It is
also expensive (one or more LLM calls per chunk, repeated for summarization), *approximate* (LLMs
miscategorize entities, merge distinct people who share a name, miss relationships, or invent ones
that aren't there — errors that then compound through community detection and summarization), and
must be partially re-run whenever source documents change.

**Path B — the graph already exists as curated, structured data.** If someone has already done
the work of identifying entities and relationships — a foreign-key table, an expert-curated
dataset — you don't need an LLM to *discover* the graph; you need to *load* it. This is cheap,
exact (no hallucinated relationships), and stable (it only changes when the underlying data does).

**This Shamela corpus is unusually, and importantly, in Path B already.** The pre-extracted tables
described at the top of this document are not raw text waiting to be mined — they are already a
graph, just sitting in relational form:

| Existing table | Becomes... |
|---|---|
| ~19,000 narrators + biography/jarh-wa-ta'dil text | Nodes, with the criticism text as node attributes |
| ~35,000 isnad links (narrator ↔ page/hadith) | NARRATED-FROM / TRANSMITTED-IN edges |
| Hadith concordance across books | SAME-HADITH-AS edges linking hadith nodes across book boundaries |
| Verse → tafsir commentary links | COMMENTS-ON edges from tafsir-book/passage nodes to verse nodes |
| Word-form → root mapping | DERIVED-FROM edges from inflected forms to triliteral roots |

That is a five-relationship-type knowledge graph, already curated by scholars over centuries and
digitized, with zero LLM extraction cost and zero hallucination risk. A typical GraphRAG
deployment spends most of its setup cost and error budget on step 1 above (extraction). This
project can skip step 1 almost entirely for these five relationship types and go straight to
querying and hybrid retrieval (§7). LLM extraction still has a role — e.g., for relationships
genuinely locked in unstructured prose, such as "this fiqh passage disputes that one's ruling" —
but it is the exception here, not the starting point.

## 6. Graph databases, and when you don't need one

Once you have a graph — extracted or curated — you need somewhere to query it.

- **Neo4j**, the most widely used dedicated graph database, stores nodes and edges natively and is
  queried with **Cypher**, a declarative language purpose-built for graph patterns (e.g.,
  `MATCH (a:Narrator)-[:NARRATED_FROM*1..5]->(b:Narrator) WHERE b.grade = 'weak' RETURN a,b` reads
  almost like the English question). It fits naturally when the graph is large, traversal patterns
  are varied, or you want built-in graph algorithms ([Neo4j: What is
  GraphRAG?](https://neo4j.com/blog/genai/what-is-graphrag/)).
- **Amazon Neptune**, AWS's managed graph database, supports both the property-graph model (via
  Gremlin/openCypher) and RDF/SPARQL — relevant if the rest of the stack is already on AWS.
- **Relational tables with recursive queries** are frequently enough. Postgres's recursive common
  table expressions (`WITH RECURSIVE ...`) can walk a self-referencing
  `narrator_id -> narrated_from_id` edge table to arbitrary depth with no separate graph engine.

You do **not** always need a dedicated graph database. The deciding factors are graph *size* and
*query complexity*, not the presence of relationships per se. Tens of thousands of narrator nodes
and tens of thousands of isnad edges — the actual scale here — sit comfortably within what an
indexed Postgres table with a recursive CTE can traverse with good performance. Dedicated graph
databases earn their operational cost at a different scale (millions to billions of nodes/edges)
or when query patterns are too varied for hand-written recursive SQL to stay maintainable. For
this corpus, starting with well-indexed relational tables and adding a graph engine later if
traversal needs outgrow it is a reasonable, low-risk sequencing.

## 7. Hybrid Graph + Vector RAG

In practice, production systems rarely pick one or the other — they combine both, because they
answer different question shapes.

**Query routing.** A router (rule-based or a small LLM classifier) inspects the incoming question
and decides: is this a semantic/open-ended question ("find passages about the permissibility of
listening to music") that should go to vector search, or a structured/relational question
("who did narrator X narrate from") that should go to a graph query? LangChain's graph-integration
tooling (`langchain-neo4j`, including
[`GraphCypherQAChain`](https://reference.langchain.com/v0.3/python/neo4j/chains/langchain_neo4j.chains.graph_qa.cypher.GraphCypherQAChain.html),
which translates a natural-language question into Cypher, executes it, and feeds results back
through an LLM) is built for exactly this pattern, as is
[LlamaIndex's Knowledge Graph Index and Knowledge Graph RAG Query Engine](https://developers.llamaindex.ai/python/examples/query_engine/knowledge_graph_rag_query_engine/).

**Graph-augmented retrieval (neighborhood expansion).** Rather than routing to one or the other,
retrieve with vector search first, then *expand* each hit through its graph neighborhood before
handing context to the LLM. Concretely: a vector search returns a hadith passage; the system then
automatically pulls in the biography and jarh-wa-ta'dil verdict for every narrator in that
hadith's isnad, plus any SAME-HADITH-AS matches in other books — context vector search alone would
never surface, because narrator-biography text and hadith-passage text don't necessarily embed as
"similar." This pattern — vector search finds the entry point, graph traversal enriches it — is
the approach Neo4j documents in its GraphRAG guidance ([Neo4j: What is
GraphRAG?](https://neo4j.com/blog/genai/what-is-graphrag/)), and is functionally what LangChain's
and LlamaIndex's graph-retriever integrations support.

Both patterns are complementary: a mature system typically routes clearly-structured questions to
the graph and uses neighborhood expansion to enrich answers to clearly-semantic ones.

## 8. Comparison: Vector RAG vs. GraphRAG vs. Hybrid

| Dimension | Vector RAG | GraphRAG (LLM-extracted) | Hybrid Graph + Vector |
|---|---|---|---|
| Best for | Open-ended, fuzzy, semantic questions | Global/holistic sensemaking across a corpus | Both, routed or combined per question |
| Multi-hop precision | Poor — no native traversal | Good if the extracted graph is accurate | Good — exact if curated, approximate if LLM-extracted |
| Exact structured-lookup precision | Poor — no exact-match guarantee | Good | Good |
| Open-ended semantic precision | Good | Weaker for narrow lookups | Good — defers to vector search |
| Setup cost | Low (embed and index) | High (extraction + community detection + summarization) | Low if structured data exists; moderate otherwise |
| Infrastructure complexity | Low (one vector index) | Moderate–high (graph store, extraction pipeline) | Moderate (vector index + graph store + router) |
| Reuse of pre-existing structured data | No — re-embedded as flat text | Partial — still extracts unless graph is supplied | Yes — the main advantage when structured data exists |
| Maintenance as source changes | Re-embed changed chunks | Re-run extraction on affected portions | Re-embed chunks; re-sync graph edges (cheap if structured) |

## 9. Honest limitations and costs

GraphRAG is not a free upgrade, and it's worth being blunt about where it costs the most and helps
the least:

- **LLM-extraction-based graph construction is genuinely expensive**, because it requires running
  an LLM over every chunk of the corpus (and again for community summarization) before any user
  ever asks a question. Reported real-world indexing costs have varied wildly — from a few dollars
  for small corpora to five figures for large ones in early implementations — part of why
  Microsoft's follow-up work, LazyGraphRAG, focused specifically on cutting that cost (see the
  [LazyGraphRAG blog post](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
  and Microsoft's own [cost breakdown](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/graphrag-costs-explained-what-you-need-to-know/4207978)).
- **Extraction errors compound.** An LLM that misidentifies an entity, merges two distinct people,
  or fabricates a relationship poisons every downstream step: the community it's clustered into,
  that community's summary, and any answer built on it — with no default reconciliation step to
  catch it. This is much of the substance behind community pushback weighing GraphRAG's overhead
  against simpler alternatives (see ["You probably don't need
  GraphRAG"](https://medium.com/@amrwrites/you-probably-dont-need-graphrag-0bc9cf671db1)).
- **Graphs need maintenance.** A graph is a snapshot of relationships at build time; as source
  documents change, it drifts out of sync unless something re-extracts or re-syncs it. That's true
  whether the graph came from an LLM or a structured table, though keeping a curated table in sync
  is far cheaper — a data pipeline problem, not an LLM re-run.
- **Graph traversal does not replace semantic search.** A user asking "find me discussions of
  patience in adversity" is not asking a graph question, and forcing that query through traversal
  or community summaries will underperform plain vector search. The two techniques complement each
  other; neither subsumes the other.

## 10. Where this corpus's structured data changes the calculus

Most GraphRAG write-ups — including the original Microsoft paper — implicitly assume you're
starting from raw, unstructured text and must pay the full extraction cost to get a graph at all.
That is the expensive, error-prone, hardest part of the whole approach, and it's also the part
this project can mostly skip.

The narrator table, the isnad links, the hadith concordance, and the verse-to-tafsir links are
already a hand-curated, scholarly-vetted graph — arguably higher quality than most GraphRAG
pipelines ever produce from LLM extraction, since it was built by generations of hadith scholars
practicing jarh wa ta'dil, not inferred by a language model reading text once. Loading these
tables as nodes and edges (into Postgres recursive-query tables, or a graph database if scale
later warrants it) gets this project directly to GraphRAG's traversal and query capabilities —
multi-hop isnad analysis, exact verse-to-commentary lookups, cross-book hadith matching — without
spending the extraction budget or inheriting the error rate a from-scratch deployment would.

The remaining, genuinely unstructured relationships here — a fiqh passage disputing another
scholar's ruling, a commentary implicitly building on an earlier one without formal citation — are
where LLM-based extraction, used sparingly and validated, still earns its cost, as a targeted
supplement rather than the whole strategy. The architecture that falls out of this is a hybrid:
structured tables as the graph's backbone, vector search as the default for open-ended questions,
and LLM extraction applied narrowly where no structured signal exists. Wiring that together —
router logic, neighborhood-expansion queries, schema — is the subject of the synthesis document
later in this series.

## Further reading

- Edge, D. et al., ["From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization"](https://arxiv.org/abs/2404.16130) — arXiv:2404.16130, the original paper.
- [microsoft/graphrag](https://github.com/microsoft/graphrag) — official implementation, docs at
  [microsoft.github.io/graphrag](https://microsoft.github.io/graphrag/).
- [GraphRAG: New tool for complex data discovery, now on GitHub](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/)
  — Microsoft Research's announcement.
- [LazyGraphRAG: Setting a new standard for quality and cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
  and [GraphRAG costs explained](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/graphrag-costs-explained-what-you-need-to-know/4207978)
  — cost-reduction follow-up and Microsoft's own cost accounting.
- [Neo4j: What is GraphRAG?](https://neo4j.com/blog/genai/what-is-graphrag/) — graph traversal,
  Cypher, and vector+graph retrieval patterns.
- [Neo4j LangChain integration docs](https://neo4j.com/developer/genai-ecosystem/langchain/) and
  [`GraphCypherQAChain` reference](https://reference.langchain.com/v0.3/python/neo4j/chains/langchain_neo4j.chains.graph_qa.cypher.GraphCypherQAChain.html)
  — text-to-Cypher question answering.
- [LlamaIndex Knowledge Graph Index](https://developers.llamaindex.ai/python/examples/index_structs/knowledge_graph/knowledgegraphdemo/)
  and [Knowledge Graph RAG Query Engine](https://developers.llamaindex.ai/python/examples/query_engine/knowledge_graph_rag_query_engine/).
- ["You probably don't need GraphRAG"](https://medium.com/@amrwrites/you-probably-dont-need-graphrag-0bc9cf671db1)
  and ["The GraphRAG Cost Cliff"](https://medium.com/graph-praxis/the-graphrag-cost-cliff-how-33-000-became-33-in-eighteen-months-be1b0fbe37e4)
  — pragmatic, cost-focused community perspectives.
