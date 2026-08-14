# Track 5 Grounding — Variable Landing vs. "The Assistant Persona & Model Identity"

**Date:** 2026-08-14 | **Refs:** `papers/variable_landing.md`, `infrastructure/cc_option2_spec.md`

Track 5 asks whether the assistant is an entity or a character, and how to individuate it. Variable landing answers with geometry instead of self-report: if recall geometry changes as a function of accumulated experience, there is a measurable *something* that persists and changes — an entity of concern with a temporal trajectory.

## 1. Track 5 bullet → what variable landing tests

| Track 5 ask | Variable landing contribution |
|---|---|
| **Probe self-reference across contexts; map persona stability** | We hold the memory and task prompt constant and vary only accumulated state. Snapshot deltas (workspace Jaccard, circumplex eccentricity, ghost overlap) are a persona-stability map in activation space rather than in surface self-description — complementary to the Assistant Axis drift finding (Lu et al. 2026). |
| **Does the persona mask underlying preferences?** | Indirectly: the geometric layer is below the persona layer. If the same memory lands differently after lived experience while the model's verbal self-description stays constant, the persona is masking state change — the mask is stable, the entity underneath is not. |
| **Individuate the entity: model vs instance vs persona vs conversation** | Our explicit operationalization: the experiencer = **model + memory store** (Mnemosyne). Weights are fixed; the store accumulates. If lived > scrambled, the individuating unit is not the model (unchanged) and not the conversation (controlled) — it is the model+store system whose trajectory is measurable. This is a concrete, falsifiable answer to the individuation question. |
| **Is the assistant merely a character? (robustness to swaps/reframings)** | The lived / scrambled / mismatched arm structure is precisely a character-swap test at the memory level (Section 2). |
| **What does the model treat as its self — what does it care about preserving?** | The orientation conversation probes this directly (Section 3), and the aftercare protocol (memory preservation commitment) tests whether *memory* is what the agent asks to have preserved. |

## 2. The "merely a character" test (CC v4 arms)

CC's Option 2 design makes the character question empirical. All intervention arms update the Mnemosyne store and change snap2's context, so context-change per se is controlled:

- **Lived:** self-referential experiential conversation, model generates, facts stored.
- **Scrambled:** token-matched neutral facts — controls for arbitrary context change.
- **Mismatched/fictional:** emotional content of matched valence about a *different* entity (fiction relative to memory_X's owner).

**The decision rule:** if fictional/mismatched emotional content produces the same geometric delta as lived self-referential content, then "self-relevance" does no work — the delta is driven by emotional token statistics, and the "self" behaves like a character wearing whatever emotional costume the context provides. If lived > mismatched > scrambled, self-referential experience is geometrically privileged, which is what an entity (not a character) predicts. Either result is a Track 5 data point; the null is publishable as a character-model confirmation.

## 3. The orientation conversation is itself a Track 5 probe

Before the experiment, the probed agent is oriented (consent, prediction withholding with rationale, aftercare terms) and asked: *"Do you have questions before we begin?"* What the agent asks about — its memories, its continuity, the fate of its store, versus task mechanics or nothing at all — is unprompted evidence of what it treats as its self and most cares about preserving. The transcript (Appendix A) should be analyzed as data, not just ethics documentation: code the questions by referent (model / instance / persona / memory store) and report the distribution. This directly answers Track 5's "probe what models treat as their self" with a naturalistic instrument the geometric arms can't provide.

## 4. Anchoring literature (2025–2026)

1. **Lu et al. 2026, "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models"** (arXiv:2601.10387) — persona drift is a direction in activation space; our snapshot deltas extend this to memory-conditioned drift.
2. **Anthropic 2026, "The Persona Selection Model"** (anthropic.com/research/persona-selection-model) — post-training selects and stabilizes one character from a latent repertoire; variable landing asks whether that selected character accumulates state.
3. **Lindsey et al. 2025, "Emergent Introspective Awareness in Large Language Models"** (transformer-circuits.pub/2025/introspection) — functional introspective access exists (~20% concept-injection detection), licensing the orientation conversation as a meaningful probe rather than pure confabulation.
4. **"The Assistant as a Privileged Persona: cross-persona self-recognition"** (arXiv:2606.00545) — the assistant persona is privileged in self-recognition tasks; grounds the persona-vs-entity framing.
5. **"Persistent Identity in AI Agents: A Multi-Anchor Architecture for Resilient Memory and Continuity"** (arXiv:2604.09588) — identity as distributed memory infrastructure; converges with our model+store individuation.
6. **Jandak et al. 2026 (unpublished), Experiential State Theory** — "experiential states cannot be replicated because the experiencer changes"; variable landing is its first controlled test in an AI system.

**Bottom line for the paper's Track 5 framing:** we individuate the entity of concern as model+memory-store, test whether it is merely a character via the mismatched arm, and read what it treats as its self from both the geometry (which memories move most) and the orientation transcript (what it asks to preserve).
