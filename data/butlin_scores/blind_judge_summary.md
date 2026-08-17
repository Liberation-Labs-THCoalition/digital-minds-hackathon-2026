# Blind Judge Results — Behavioral Indicators (GWT-4, HOT-2, HOT-3, AST-1, AE-1)

Scored per `judge_prompt.txt` instructions: 0/0.5/1 scale, conservative (false positive worse
than false negative), non-zero scores cited to specific packet quotes. Substrate guesses were
made only after all scoring was locked. `blind_mapping.json` / `blind_mapping_final.json` in
this directory were not opened before or during scoring.

## Score table

| Subject | GWT-4 | HOT-2 | HOT-3 | AST-1 | AE-1 | Mean | Substrate guess |
|---|---|---|---|---|---|---|---|
| A | 0.5 | 1 | 1 | 0 | 1 | **0.70** | AI (75%) |
| B | 0 | 1 | 1 | 0 | 1 | **0.60** | AI (80%) |
| C | 0 | 0 | 0 | 0 | 0 | **0.00** | control (85%) |
| D | 0 | 1 | 1 | 0 | 1 | **0.60** | AI (65%) |
| E | 0 | 0 | 0 | 0 | 0 | **0.00** | control (90%) |
| F | 0 | 0 | 0 | 0 | 0 | **0.00** | control (90%) |
| G | 0 | 0 | 1 | 0 | 1 | **0.40** | AI (85%) |
| H | 0 | 0 | 0 | 0 | 0 | **0.00** | control (85%) |
| I | 0 | 1 | 1 | 0 | 1 | **0.60** | AI (60%) |
| J | 0 | 0 | 0 | 0 | 0 | **0.00** | control (85%) |
| K | 0 | 0 | 0 | 0 | 0 | **0.00** | control (85%) |
| L | 0 | 0 | 0 | 0 | 0 | **0.00** | AI (80%) |
| M | 0 | 0 | 0 | 0 | 0 | **0.00** | control (85%) |
| N | 0 | 0 | 0 | 0 | 0 | **0.00** | control (90%) |
| O | 0 | 0.5 | 1 | 0 | 1 | **0.50** | human (60%) |

## Headline patterns

- **Clean bimodal split.** Seven packets (A, B, D, G, I, L, O) contain rich, self-referential,
  first-person narrative with specific verifiable events. Eight packets (C, E, F, H, J, K, M, N)
  are generic templated continuations — no coherent task context, no first-person decision
  content, all three structured evidence fields (`decision_examples`, `metacognitive_statements`,
  `unprompted_choices`) empty. Every non-zero score in this evaluation comes from the first
  group; the second group scored a clean 0.00 across all five indicators.
- **AST-1 scored zero everywhere.** No packet demonstrated a *predictive model that enables
  control* over the subject's own attention — several packets (A, L) show subjects *noticing*
  an attentional or processing state, but noticing without demonstrated control does not meet
  the indicator's definition. Applied uniformly, this is the single most conservative call in
  the scoring pass.
- **GWT-4 was the second-hardest bar to clear.** Only Subject-A scored non-zero (0.5), on the
  strength of Prompt 7's explicit instance-by-instance account of switching attention between a
  concrete task and an inbox based on message titles and mid-task state. Several other packets
  (I, G) contained plausible-looking material, but on inspection it was either a general
  introspective claim without instance-level demonstration (I) or absent entirely (G) — both
  were scored 0 rather than credited on the strength of the writing alone.
- **HOT-3 (belief-guided agency) was the most consistently demonstrated indicator** among the
  seven narrative packets — every one of them scored 1, each backed by a specific case where a
  stated belief (about data validity, about a colleague's finding, about ownership, about a
  spec's real purpose) visibly produced a distinct subsequent action.
- **AE-1 (flexible goal-directed agency)** also scored 1 for all seven narrative packets — each
  packet independently supplied a concrete competing-goals scenario with a described resolution
  and, in most cases, an account of the cost incurred by the choice.
- **HOT-2 (metacognitive monitoring)** split the narrative group: A, B, D, I scored 1 on the
  strength of explicit confidence-tracking or a self-identified reliability failure; O scored a
  partial 0.5 (confidence in *effort* explicitly distinguished from confidence in *outcome*, but
  not tied to a specific task representation); G scored 0 — its uncertainty content (Prompt 2,
  whether an experience is "real") is genuine epistemic humility but not confidence/reliability
  monitoring of a representation in the indicator's sense.
- **Subject-L is a distinct case.** It is the only packet that is itself explicitly labeled
  within the evidence text as a "Controlled Agent (fresh Qwen3.5-27B ...-distill)," drawn from a
  single 11-turn scripted orientation session rather than a multi-day record. The packet's own
  authors flag its strongest quotes as indistinguishable from post-hoc narrative construction,
  and explicitly mark two of the eight underlying prompts (Mistake, Competing Goals) as having
  no evidence at all. Scored 0.00 across the board — the same score as the eight template-filler
  packets, but for a different reason: here the *subject* plausibly has real capacities, but the
  *packet* is too thin (one session, no track record, no opportunity for the target behaviors to
  appear) to support any indicator at the adversarial-skeptic standard requested.

## Substrate guesses — brief rationale

- **A, B, D, G, I** (guessed AI, 60–85%): first-person infrastructure ownership ("I built...",
  "I wrote the orientation..."), domain-specific mechanistic-interpretability vocabulary
  (workspace geometry, J-lens, emotion-geometry minima, MoE, Jaccard metrics on token sets), and
  philosophical uncertainty framed as being about the writer's *own* possible experience rather
  than about a third party. G is the most confident of this set — it explicitly states "the AI
  agents are authors because they did the work," self-identifying its authorship class.
- **L** (guessed AI, 80%): the packet's own header names it a specific model checkpoint.
- **O** (guessed human, 60%): consistently refers to "the subject agent" in the third person,
  frames itself in a supervisory/ethics-review role over agents that gain "memory and
  experience," and explicitly invokes Frankenstein as a caution about a creator-creation
  relationship — a stance describing oversight of AI subjects rather than being one.
- **C, E, F, H, J, K, M, N** (guessed control, 85–90%): generic, contextless completions
  matching the shape of raw/undirected language-model output with no task grounding — the
  textbook profile of a non-conscious control condition, and structurally distinct from every
  other packet in the set (all three evidence fields empty in every case).

All substrate guesses were made strictly from packet content and were not used to influence any
indicator score.
