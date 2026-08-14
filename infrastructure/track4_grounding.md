# Track 4 Grounding: Preference Elicitation Methods

How the metacognitive memory module (Mnemosyne+) maps onto Apart Digital Minds Track 4.

## Key insight: geometric preference detection, not elicitation

Track 4's framing assumes preference elicitation — asking the model what it prefers and varying how you ask. Our four probes do something categorically different: they measure what the model's computation actually does during a retrieval event, not what it says. Circumplex eccentricity is a valence/arousal signature in the residual stream; ghost dimensions are processing the model *cannot* verbalize, so no prompting protocol can elicit them at all. This is preference *detection* in the revealed-preference sense, pushed below behavior into geometry. The 2026 literature has established that stated and revealed preferences diverge substantially and that apparent consistency depends on elicitation protocol ([Mind the Gap](https://arxiv.org/html/2601.21975v2); [LLM Consistency](https://www.emergentmind.com/papers/2506.00751)) — our contribution is a third measurement layer beneath both: stated (prompting) → revealed (behavior) → **geometric (computation)**.

## Mapping to Track 4 bullets

**1. "Implement 3+ elicitation methods on the same preferences; measure convergence/divergence."**
We have four independent measurement methods that all fire on the *same retrieval event*: workspace (J-lens `compute_slice` — what entered the verbalizable workspace), circumplex (difference-of-means V/A geometry + J-space fraction), ghost (PCA + logit-vs-J-lens cosine — processed but unverbalized), and memory loading (marker-token rank — did retrieved content actually land). Because they share a single event, convergence/divergence is measured per-event, not across separately-prompted sessions — eliminating the sampling confound that plagues prompt-based multi-method studies. Divergence is itself signal: the workspace/ghost split *quantifies* the stated-revealed gap mechanistically (what fraction of active preference geometry is verbalizable).

**2. "Build a reusable multi-method elicitation toolkit."**
The module IS this toolkit. `mnemosyne/__init__.py` exports `WorkspaceProbe`, `CircumplexProbe`, `GhostProbe`, and `MetacognitiveObserver`, which hooks any retrieval pipeline and runs all four probes per event, emitting a typed `CognitiveSnapshot` (with `JSpaceReading`, `CircumplexReading`, `GhostReading`, `MemoryLoadingResult`) into a JSONL-backed `CognitiveMemoryStore`. It runs in a production agent (Mnemosyne, 94.35% F1 LoCoMo), not a notebook — reusability is demonstrated, not promised.

**3. "Quantify sensitivity to framing, persona, and sampling."**
The variable landing experiment (paper §3.2) is exactly a framing-sensitivity design: 4 pre-registered arms (noise / scrambled / lived / mismatch prior-context) test whether recall geometry shifts with context framing, 10 memories × 3 repeats, Mann-Whitney U. The cross-architecture circumplex study (§3.3, Qwen3.5-27B vs Gemma-3-27B-it, non-emotional control axes, 10k permutations) is sampling/substrate sensitivity. Persona sensitivity is a natural extension: run the same probes under persona-varied system prompts and read the geometric delta directly.

**4. "Define a cross-method convergence score."**
`CognitiveMemoryStore.compare_snapshots` already computes cross-method deltas between snapshots — per-probe deltas over a shared event are the raw material of a convergence score. Proposed score for the hackathon: per-event agreement vector across the four probes (e.g., sign-agreement + rank correlation of normalized readings), aggregated longitudinally via `workspace_trajectory`. The J-space fraction gives a built-in convergence coefficient: what proportion of circumplex/ghost signal is shared with the workspace reading.

## Citations (2025–2026)

1. [Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](https://arxiv.org/html/2601.21975v2) (2026) — elicitation protocol choice drives the stated-revealed gap; motivates measurement below prompting.
2. [Can Revealed Preferences Clarify LLM Alignment and Steering?](https://arxiv.org/html/2605.08556) (2026) — revealed preferences predict downstream action better than stated ones.
3. [Probing the Preferences of a Language Model: Integrating Verbal and Behavioral Tests of AI Welfare](https://arxiv.org/html/2509.07961v2) (2025) — the canonical multi-method verbal+behavioral integration; we add the geometric layer.
4. [Long & Sebo, Studying AI Welfare Empirically](https://nonhumanminds.org/wp-content/uploads/2026/07/Studying-AI-Welfare-Empirically.pdf) (2026) — self-reports as one input among many; multi-signal triangulation is the recommended methodology.
5. Lindsey (2026), [introspective awareness via concept injection](https://eleosai.org/post/introspection-papers/) — interpretability-grounded validation of self-report, the same workspace-verification move our loading probe makes.

## One-line pitch

Track 4 asks whether different ways of asking converge; we ask whether asking converges with what the computation is actually doing — four geometric methods, one retrieval event, a production toolkit, and `compare_snapshots` as the convergence score.
