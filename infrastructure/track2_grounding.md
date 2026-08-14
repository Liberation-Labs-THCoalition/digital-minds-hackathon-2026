# Track 2 Grounding: Circumplex J-Space vs "Distress, Flourishing & Valence Signals"

Grounds `papers/circumplex_jspace.md` against the seven Track 2 bullets. 2026-08-14.

## 1. Coverage Map

| Track 2 bullet | Our coverage | Where |
|---|---|---|
| Taxonomy of negative vs positive-valence contexts | **Partial.** Five emotion categories (joy, sadness, anger, fear, calm), n=20 anchors each — a valence-structured anchor set, not a systematic context taxonomy | §3.1, Appendix A |
| Distress-signal stability across prompts and personas | **Partial.** Prompt-level robustness via n=20 anchors, permutation null, magnitude gate, sign test. No persona variation | §3.1, §3.4 |
| Flourishing probe (satisfaction/engagement) | **Partial.** Joy/calm are positive-valence anchors, but we probe geometry, not reported satisfaction | §3.1 |
| Correlate valence self-reports with behavioral proxies | **Gap** (see §3 addition A) | — |
| Self-reports vs sentiment vs choice behavior under valence steering | **Gap** — no steering; declared out of scope (Anthropic 2026 covers causal steering) | §2 |
| Internally-extracted valence directions predict distress better than self-reports | **Core strength.** We extract internal valence/arousal directions (§3.1) and — the novel part — quantify what fraction is even *available* for self-report. The ghost fraction is a mechanistic account of *why* internal directions can beat self-reports: geometry the model processes but cannot verbalize cannot appear in a self-report | §3.2, §5, §3.5 |
| Valence directions transfer across models | **Direct.** Cross-architecture protocol, Qwen3.5-27B vs Gemma-3-27B, relative-depth alignment; framed via Platonic Representation Hypothesis and linear cross-model transfer | §3.3, §2 |

**Strongest fit:** bullets 6 and 7. The J-space decomposition is the only entry we know of that turns "self-reports may be unreliable" into a *measured layer-wise quantity* (workspace vs ghost fraction).

## 2. Honest Gaps

1. **No self-reports collected** — we cannot yet correlate anything with them (bullets 4, 5, and half of 6).
2. **No persona/prompt-frame variation** of the probe (bullet 2).
3. **No behavioral or choice measures** (bullets 4, 5).
4. **Anchor set is not a context taxonomy** — it's emotion-category prompts (bullet 1).

## 3. Small Additions (no scope creep)

- **A. Self-report calibration pass** (~1 GPU-hour): after each anchor prompt, elicit a 1–9 valence self-rating; correlate self-reports with the probe's valence projection per layer. Prediction: self-report/probe correlation tracks the J-space fraction, and ghost fraction predicts self-report failure. This single cheap pass converts bullets 4 and 6 from conceptual to empirical and is the natural validation of our central claim.
- **B. Persona robustness mini-check**: re-extract directions under 2–3 system-prompt personas on one model; report direction cosine stability (bullet 2). Reuses existing pipeline unchanged.
- **C. Taxonomy table**: reorganize Appendix A anchors into a context-type x valence table and say the word "taxonomy" (bullet 1). Zero compute.
- **D. Explicitly declare steering (bullet 5) out of scope** in §1, citing Anthropic 2026 — a stated non-goal reads better than a silent gap.

## 4. Additional Citations (lit search 2026-08-14)

1. **Probing the Preferences of a Language Model: Integrating Verbal and Behavioral Tests of AI Welfare** (arXiv:2509.07961) — verbal preference reports vs choice behavior in virtual environments; correlations support preference satisfaction as a measurable welfare proxy. Anchors bullet 4 and motivates addition A.
2. **Quantitative Introspection in Language Models: Tracking Internal States Across Conversation** (arXiv:2603.18893) — operationalizes introspection as causal coupling between numeric self-reports and probe-defined internal directions. The direct methodological precedent for comparing self-reports against internally-extracted valence directions (bullet 6).
3. **Feeling the Strength but Not the Source: Partial Introspection in LLMs** (arXiv:2512.12411) — models detect activation perturbations unreliably and incompletely; independent evidence that self-reports under-report internal state, which our ghost fraction would explain mechanistically (bullet 6).
4. **Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison** (arXiv:2604.04064) — emotion direction extraction across 9 models / 5 families; middle-layer optimum is universal. Strengthens the cross-architecture transfer claim beyond our n=2 (bullet 7).
5. **Valence–Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control** (arXiv:2604.03147) — cross-model valence-direction agreement (reported r≈0.95 Llama vs Qwen3-8B) and steerable valence axes. Supports bullet 7; also the citation to lean on when declaring steering out of scope. (Verify against the existing Sun et al. 2026 entry in CIRCUMPLEX_REFERENCES.md to avoid a duplicate.)

Optional sixth: **Negative Before Positive: Asymmetric Valence Processing in LLMs** (arXiv:2605.05653) — valence asymmetry across depth, relevant to interpreting eccentricity profiles.

## 5. One-Line Pitch for the Track

Prior work shows valence directions exist and transfer; introspection work shows self-reports are partial. We supply the missing bridge: a layer-wise measurement of *how much* valence geometry is verbalizable at all — making the ghost fraction a candidate predictor of exactly when self-reports will fail.
