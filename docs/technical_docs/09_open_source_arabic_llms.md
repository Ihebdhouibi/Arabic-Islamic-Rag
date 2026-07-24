# 09 — Open-Source Arabic LLMs for the Generation Layer

> Companion to [07_recommended_architecture_for_shamela_rag.md](07_recommended_architecture_for_shamela_rag.md).
> That document's architecture has an LLM sitting at the query-router step and the final
> generation step (§5–6). This document surveys open-source, Arabic-capable candidates for that
> role — as of mid-2026, landscape verified via web search rather than assumed from training data,
> since this space moves fast. **This is a candidate list for evaluation, not a committed choice**
> — see §6 for what actually needs to happen before picking one.

## 1. What this model is actually being asked to do

In the architecture from [07_recommended_architecture_for_shamela_rag.md §5](07_recommended_architecture_for_shamela_rag.md#5-query-routed-retrieval-one-path-per-use-case),
an LLM sits in two places: the **query router** (classify intent, decide which retrieval path to
take) and the **generation step** (synthesize retrieved vector/graph context into a
citation-grounded answer, per [07 §6](07_recommended_architecture_for_shamela_rag.md#6-generation-and-grounding--a-domain-specific-requirement-not-a-nice-to-have)).
Neither role requires the model to *know* Islamic law from its own training — the whole point of
the retrieval-and-graph layer is to supply that. What the model needs to be good at is
**instruction-following, faithful synthesis, and Arabic fluency** — not encyclopedic recall.

That distinction turns out to matter a lot for model selection (§2).

## 2. Why this matters: general Arabic LLMs measurably struggle on Islamic content specifically

Multiple 2025–2026 evaluations found that strong general-purpose Arabic LLMs — including Jais,
Mistral, and LLaMA — show a **significant accuracy drop specifically on Islamic legal reasoning**
compared to their scores on general Arabic-language benchmarks:

- A 2025 study benchmarking LLMs on Islamic inheritance-law reasoning found this gap directly
  (["Assessing Large Language Models on Islamic Legal Reasoning: Evidence from Inheritance Law
  Evaluation," arXiv:2509.01081](https://arxiv.org/html/2509.01081v1); companion benchmark
  ["Benchmarking the Legal Reasoning of LLMs in Arabic Islamic Inheritance Cases,"
  arXiv:2508.15796](https://arxiv.org/pdf/2508.15796)).
- **IslamicEval 2025**, the first shared task dedicated to hallucination in Islamic content, was
  created precisely because this gap is common enough to need a standard benchmark — its two
  subtasks are detecting/correcting hallucinated Quranic verses and hadith, and question-answering
  that requires grounding in authoritative sources rather than parametric recall.
- **PalmX 2025** benchmarks LLMs specifically on Arabic *and* Islamic culture as a separate axis
  from general Arabic competence ([arXiv:2509.02550](https://arxiv.org/pdf/2509.02550)), which
  wouldn't be a distinct benchmark if general Arabic score reliably predicted Islamic-content
  accuracy.

**The implication for this project:** this is direct evidence for the architecture already chosen
in doc 07 — put the domain-accuracy burden on retrieval and graph grounding, not on the
generation model's memory. A smaller, well-grounded model plausibly beats a larger, ungrounded
one here. Model selection should optimize for instruction-following fidelity (does it cite
correctly, does it preserve disagreement instead of flattening it, does it stay within the
retrieved context) rather than for how much Islamic knowledge it appears to have out of the box.

## 3. Candidate landscape

| Model | Org / country | Sizes | License | Notes |
|---|---|---|---|---|
| **Fanar-1 / Fanar-2** | QCRI/HBKU, Qatar | 9B, 27B | Apache 2.0 | Continually pretrained from Gemma-2 on ~1T Arabic+English tokens; training data curated to be "aligned with Islamic values and Arab cultures." The Fanar *platform* (not just the model) already ships a built-in Islamic RAG subsystem for religious prompts — the closest existing precedent to this project. |
| **Jais 2 / Jais 30B** | Core42/G42, UAE | 13B, 30B | Apache 2.0 | Trained on 126B Arabic tokens; widely cited as the strongest raw Arabic-native fluency among open models. No particular religious-content specialization. |
| **ALLaM 7B / 34B** | SDAIA/HUMAIN, Saudi Arabia | 7B, 34B | 7B open-weight instruct; 34B research-license only | Saudi sovereign model, powers HUMAIN Chat. The 34B variant isn't usable commercially without a separate license. |
| **Falcon Arabic / Falcon-H1 Arabic** | TII, UAE | 7B | Permissive (TII Falcon license) | Small enough for a single consumer GPU (~5GB at Q4_K_M); recent production comparisons put it close to Jais 2 despite the size difference. |
| **Qwen3** | Alibaba | 8B–30B+ | Apache 2.0 | Not Arabic-specific, but leads the HELM Arabic benchmark among 8B-class models, with strong instruction-following and tool-calling — a good fit for the router role even if not the final-answer role. |
| **SILMA-9B-Instruct** | SILMA.ai | 9B | Open (Gemma-based) | Arabic-focused Gemma fine-tune, smaller footprint than Jais/Fanar. |
| **Aya Expanse** | Cohere For AI | 8B, 32B | CC-BY-NC (research use) | Strong multilingual/Arabic instruction-following, but the non-commercial license blocks most production use as-is. |
| **AceGPT** | FreedomIntelligence / KAUST | 7B, 13B | Open | Earlier-generation Arabic fine-tune of LLaMA; recent comparisons place it behind Jais 2 and Falcon-H1 Arabic today. |

## 4. The closest existing precedent: Fanar's Islamic RAG and Fanar-Sadiq

Two things from the Fanar ecosystem are worth reading before designing this project's pipeline
from scratch, since they're close to solving the same problem:

- The **Fanar platform** (not just the underlying model) already includes "a customized Islamic
  Retrieval Augmented Generation (RAG) system for handling religious prompts" as a shipped feature
  ([Fanar 2.0: Arabic Generative AI Stack, arXiv:2603.16397](https://arxiv.org/pdf/2603.16397);
  [Fanar: An Arabic-Centric Multimodal Generative AI Platform, arXiv:2501.13944](https://arxiv.org/abs/2501.13944)).
- **Fanar-Sadiq** is a published multi-agent architecture specifically for *grounded Islamic Q&A*
  ([arXiv:2603.08501](https://arxiv.org/pdf/2603.08501)) — essentially a worked example of the
  citation-grounded, disagreement-preserving generation step described in
  [07 §6](07_recommended_architecture_for_shamela_rag.md#6-generation-and-grounding--a-domain-specific-requirement-not-a-nice-to-have).
- **MufassirQAS** ([arXiv:2401.15378](https://arxiv.org/pdf/2401.15378)) applies the same idea
  narrowly to tafsir RAG reliability — directly relevant to the tafsir-by-verse use case from
  doc 07 §2.

Reading these before finalizing an architecture is likely to surface design decisions (how they
structured grounding, how they handled disagreement across sources, how they evaluated
hallucination) that would otherwise take several iterations to rediscover independently.

## 5. A two-model split is a legitimate pattern here

Nothing requires one model to handle both the router step and the generation step. A pragmatic
split worth piloting:

- **Router / intent classification:** a fast, cheap model with strong instruction-following and
  tool-calling (e.g. Qwen3-8B) — this step doesn't need deep Arabic cultural fluency, just
  reliable classification of "which of the four use-case paths does this question need"
  (per [07 §5](07_recommended_architecture_for_shamela_rag.md#5-query-routed-retrieval-one-path-per-use-case)).
- **Final answer generation:** a larger, Arabic/Islamic-aligned model (Fanar-2-27B or Jais 30B) —
  this step is where fluency, register-appropriateness, and cultural/religious alignment actually
  matter to the end user.

This also keeps cost down: the router runs on every query, the larger generation model only runs
once retrieval has already narrowed the context.

## 6. The open decision this doesn't resolve

There's a real fork that this survey doesn't settle, and shouldn't — it needs actual evaluation
against this project's golden dataset (per
[06_evaluation_and_recent_advancements.md §5](06_evaluation_and_recent_advancements.md)):

**Open-source Arabic-specialized model (Fanar, Jais) vs. a frontier closed multilingual model**
(GPT-5-class, Claude, Gemini). The open-source path offers data sovereignty, lower marginal cost,
offline/on-prem capability, and — for Fanar specifically — cultural/religious alignment baked into
pretraining. Frontier closed models still generally reason and follow complex multi-part
instructions (e.g. "cite per source, group by madhhab, don't flatten scholarly disagreement")
more reliably, even in Arabic, than most open Arabic-specialized models today. Given that §2's
evidence shows domain accuracy comes from grounding rather than the base model's knowledge, the
gap between candidates may be smaller on *this specific domain* than general Arabic leaderboards
would suggest — but that's a hypothesis to test, not a conclusion to assume.

**Recommended next step, when ready to test rather than just survey:** benchmark 2–3 candidates
(e.g. Fanar-2-27B, Jais 30B, and one frontier closed model as a ceiling reference) against the
same golden evaluation set from doc 06, specifically on citation-fidelity and
disagreement-preservation — not on general Arabic fluency, which the candidate list above already
establishes as adequate across the board.

## Further reading

- [Fanar-1-9B-Instruct](https://huggingface.co/QCRI/Fanar-1-9B-Instruct) and
  [Fanar-2-27B-Instruct](https://huggingface.co/QCRI/Fanar-2-27B-Instruct) — Hugging Face model
  cards.
- Fanar: An Arabic-Centric Multimodal Generative AI Platform —
  [arXiv:2501.13944](https://arxiv.org/abs/2501.13944).
- Fanar 2.0: Arabic Generative AI Stack — [arXiv:2603.16397](https://arxiv.org/pdf/2603.16397).
- ALLaM: Large Language Models for Arabic and English —
  [arXiv:2407.15390](https://arxiv.org/pdf/2407.15390).
- Assessing Large Language Models on Islamic Legal Reasoning: Evidence from Inheritance Law
  Evaluation — [arXiv:2509.01081](https://arxiv.org/html/2509.01081v1).
- Benchmarking the Legal Reasoning of LLMs in Arabic Islamic Inheritance Cases —
  [arXiv:2508.15796](https://arxiv.org/pdf/2508.15796).
- Islamic Large Language Models: From Knowledge Acquisition to Trustworthy and
  Hallucination-Resistant AI — [arXiv:2606.16629](https://arxiv.org/html/2606.16629).
- PalmX 2025: The First Shared Task on Benchmarking LLMs on Arabic and Islamic Culture —
  [arXiv:2509.02550](https://arxiv.org/pdf/2509.02550).
- Fanar-Sadiq: A Multi-Agent Architecture for Grounded Islamic QA —
  [arXiv:2603.08501](https://arxiv.org/pdf/2603.08501).
- MufassirQAS: Improving LLM Reliability with RAG in Religious Question-Answering —
  [arXiv:2401.15378](https://arxiv.org/pdf/2401.15378).
- The Landscape of Arabic Large Language Models — [Communications of the ACM](https://cacm.acm.org/arab-world-regional-special-section/the-landscape-of-arabic-large-language-models/).
- Falcon-H1 Arabic vs Jais 2: A Production Comparison for GCC Workloads — [Codenovai](https://codenovai.com/blog/falcon-h1-vs-jais-2-arabic-llm-production-comparison).
