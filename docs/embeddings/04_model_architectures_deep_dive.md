# Model Architectures — Deep Dive

> Part of the series indexed in [00_overview.md](00_overview.md). Doc 01 covered the three
> retrieval-time architectures (bi-encoder, cross-encoder, late-interaction). This doc goes one
> level deeper on two things
> [docs/technical_docs/03_embeddings_and_vector_stores.md](../technical_docs/03_embeddings_and_vector_stores.md)
> covers only at survey depth: the encoder **backbone family** a model is built from, and the
> actual mechanics of Matryoshka embeddings.

## 1. Two backbone families: bidirectional encoders vs. repurposed decoders

**Bidirectional encoder-only models** (the BERT lineage — what BGE-M3, multilingual-e5, and most
of the "classic" embedding models in doc 03's landscape are built on) process the entire input at
once, with every token able to attend to every other token in both directions. This is the
natural fit for producing one good summary vector per passage, and it's why encoder-only models
have dominated the embedding leaderboards for years — the architecture is purpose-built for
exactly this job.

**Decoder-based / LLM-repurposed embedders** are a newer, growing category: take a causal
(left-to-right) decoder LLM — the same architecture family used for text generation — and adapt it
to produce embeddings, typically via last-token pooling (per
[01_how_embeddings_work.md §1](01_how_embeddings_work.md#1-producing-a-single-embedding-the-bi-encoder-pipeline))
plus contrastive fine-tuning. Examples include e5-mistral and the GTE-Qwen family. The appeal:
these models inherit a much larger pretraining base (the same LLM pretraining that makes modern
chat models fluent) and tend to generalize better to instructions and longer context, at the cost
of being far larger and more expensive to run than a purpose-built encoder-only model of
comparable embedding quality. **For this project specifically**, per
[docs/technical_docs/13_architecture_decisions.md ADR-002](../technical_docs/13_architecture_decisions.md#adr-002-embedding-model--no-model-chosen-yet-the-evaluation-protocol-is-the-decision),
the shortlist (BGE-M3, multilingual-e5-large) is entirely encoder-only — a reasonable starting
point given cost, but a decoder-based embedder is worth adding to that benchmark round if
classical-register Arabic turns out to need more nuanced contextual understanding than the
encoder-only shortlist provides. Not assumed here — a candidate to test, not a recommendation.

## 2. Matryoshka embeddings — the actual mechanism, not just the name

Doc 03 mentions Matryoshka embeddings as letting you truncate dimensions for a storage/speed
tradeoff. Here's *why* truncation works instead of just destroying the embedding:

**The problem it solves:** a normal embedding model's 1024 dimensions are not independently
useful — the training objective (per
[01_how_embeddings_work.md §2](01_how_embeddings_work.md#2-training-an-embedding-model-contrastive-learning))
never encourages the *first* 256 dimensions to be independently meaningful. Truncating a
normally-trained embedding to its first 256 dimensions would throw away information essentially
at random, since nothing during training organized the dimensions by importance.

**The fix Matryoshka training makes:** during contrastive training, the loss is computed and
summed not just on the full-size embedding, but *also* on several truncated prefixes of it (e.g.,
the first 768, first 512, first 256, first 128 dimensions), all from the **same** underlying
vector. This forces the model to pack the most important, most broadly useful signal into the
*earliest* dimensions — nested the way a matryoshka doll nests smaller dolls inside bigger ones,
which is exactly the name's origin (from the "Matryoshka Representation Learning" paper cited in
doc 03).

**The practical payoff:** a Matryoshka-trained embedding can be truncated to a shorter prefix
*after training, at query time*, with a graceful, measured quality degradation instead of a random
one — useful for trading storage/speed against quality without retraining or re-embedding
anything, since the same base vectors serve every truncation length. Not every model supports
this — it has to be trained in specifically, per §2 above; truncating a non-Matryoshka model's
embedding is just destructive.

## 3. Where this matters for ADR-004's chunk-size decision revisited

[docs/technical_docs/13_architecture_decisions.md ADR-004](../technical_docs/13_architecture_decisions.md#adr-004-chunk-sizing--512–1024-token-target-for-prose-genres-natural-unit-sizing-elsewhere)
is explicitly marked provisional pending real measurement. One more variable worth folding into
that eventual benchmark round, now that backbone family is on the table: a decoder-based embedder
with a much larger effective context window could plausibly tolerate *larger* chunks than an
encoder-only model without the same pooling-dilution problem described in
[01_how_embeddings_work.md §4](01_how_embeddings_work.md#4-why-this-matters-for-chunk-level-design-decisions-already-made) —
worth testing chunk size and backbone family together, not as two independent decisions, once the
golden evaluation set from
[docs/technical_docs/14_golden_evaluation_dataset.md](../technical_docs/14_golden_evaluation_dataset.md)
is large enough to support that kind of joint comparison.
