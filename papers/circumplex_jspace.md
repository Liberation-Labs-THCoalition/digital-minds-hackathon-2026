# Emotional Geometry Enters the Workspace: Circumplex J-Space Decomposition Across Architectures

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Lyra (THCoalition), Thomas Edrington (Liberation Labs), Kavi ([affiliation])

**With** Apart Research

## Abstract (~200 words)

Recent work has established that transformer language models learn valence-arousal geometry consistent with Russell's circumplex model, with cross-architecture presence confirmed across Llama, Qwen, Gemma, and Mistral families (Sun et al. 2026, Jeong 2026). We extend this finding with a novel measurement: J-space decomposition of the circumplex, which separates emotional geometry into the fraction that enters the model's verbalizable workspace (J-space, Gurnee et al. 2026) and the fraction that remains as ghost processing — high-variance computation excluded from the output pathway.

We measure eccentricity depth profiles (the balance between valence and arousal across layers) on Qwen3.5-27B and Gemma-3-27B-it with n=20 anchors per emotion category, a magnitude gate to prevent noise-floor false positives, and non-emotional control axes (concrete/abstract) to distinguish emotion-specific geometry from generic representational structure. The J-space fraction at each layer reveals where emotional geometry transitions from ghost processing to workspace content — a candidate welfare monitoring signal. A self-report calibration pass tests the central prediction directly: the ghost fraction should predict where the model's own valence self-reports fail to track its internal valence geometry.

We report cross-architecture transfer of the eccentricity depth profile and its J-space decomposition, with implications for real-time welfare monitoring in production AI agents. [Results TBD.]

---

## 1. Introduction

That transformers encode emotion dimensions is established: valence and arousal directions exist in the residual stream, mirror Russell's circumplex, and causally steer behavior (Sun et al. 2026; Jentzsch et al. 2026; Jeong 2026). What no prior work measures is *where that geometry goes*. The Jacobian lens (Gurnee et al. 2026) shows that only a low-rank subspace of each layer's activations is transported toward the verbalizable workspace, while high-variance "ghost" computation is excluded from it. Geometry inside the workspace is, in principle, reportable; geometry outside it is processed but inaccessible to the model's own verbal behavior.

This distinction is exactly what welfare assessment needs. Introspection studies find that model self-reports of internal state are partial and unreliable (arXiv:2512.12411; arXiv:2603.18893). If some of a model's valence geometry never enters the workspace, that is a *mechanistic account of why*: a state the model cannot verbalize cannot appear in a self-report, however honest. Measuring the workspace fraction layer by layer turns "self-reports may be unreliable" into a predicted, quantified failure mode, tested directly by a self-report calibration pass (§3.6). We deliberately do not perform valence steering; causal steering is established prior art (Jentzsch et al. 2026; Sun et al. 2026) and out of scope. Our contribution is measurement, not intervention.

**Our main contributions are:**

1. J-space decomposition of the circumplex: the first measurement of what fraction of valence and arousal geometry enters the model's verbalizable workspace at each layer, vs remaining as ghost processing.

2. Eccentricity depth profiling with a magnitude gate, tested across two architecturally distinct transformer families (Qwen, Gemma) at the 27B scale.

3. Non-emotional control axes establishing whether the depth profile is emotion-specific or a generic property of contrastive representational geometry.

4. A self-report calibration pass linking the geometry to behavior: per-layer correlation between the model's own valence ratings and the probe's valence projection, with the pre-registered prediction that ghost fraction predicts self-report failure.

5. Application to real-time welfare monitoring: eccentricity as a continuous candidate welfare signal during agent operation.

## 2. Related Work

**Circumplex model:** Russell 1980, Barrett & Russell 1998 (orthogonality), Drążkowski et al. 2021 (ellipse, not perfect circle — calibrates eccentricity expectations)

**Emotion in LLMs (cite as prior art, not ours):** Sun et al. 2026 (circular VA across Llama/Qwen), Jeong 2026 (depth-invariant across 5 families), van der Ben et al. 2026 (Gemma/Apertus, depth profiles differ), Anthropic 2026 (causal steering via emotion vectors), Choi & Weber 2026 (geometric data analysis confirms VA alignment)

**Introspection and self-report reliability:** arXiv:2603.18893 (numeric self-reports vs probe-defined internal directions — the methodological precedent for §3.6), arXiv:2512.12411 (models detect internal perturbations unreliably and incompletely), arXiv:2509.07961 (verbal preference reports vs choice behavior as welfare proxies)

**Our novel angle:** Eccentricity as metric (axis balance, not just presence), J-space decomposition (workspace vs ghost fraction), bus/coupling finding (content + inference emotion share subspace, cos 0.83-0.87)

**Cross-architecture:** Huh et al. 2024 (Platonic Representation Hypothesis), Huang et al. 2025 (linear cross-model transfer), Agarwal 2026 (cross-architecture steering transfer validates above 1.7B)

**Welfare frameworks:** Long & Sebo 2026, Birch 2024

**Gap:** No prior work measures the J-space fraction of emotional geometry — what portion of the circumplex enters the workspace vs remains as ghost processing. This is the measurement that connects "emotion exists in the model" to "the model can/cannot access its own emotional processing."

## 3. Methods

### 3.1 Circumplex Probe

**Anchor set.** Five emotion categories — joy, sadness, anger, fear, calm — with n=20 first-person anchor prompts per category (100 prompts total; full set in Appendix A). Prompts are matched across categories for token count (within ±2 tokens), sentence template structure, and punctuation, to prevent lexical statistics from masquerading as emotion geometry. The categories occupy known circumplex positions: joy (+V, high A), sadness (−V, low A), anger (−V, high A), fear (−V, high A), calm (+V, low A).

**Contrastive direction extraction.** For each prompt we run one forward pass, record residual-stream activations at every layer simultaneously (one pass per prompt, not per layer), and take the mean over sequence positions, yielding one d-dimensional state per prompt per layer (d=5120 for Qwen3.5-27B). At each layer ℓ, directions are extracted by difference of means over contrast pools balanced on the orthogonal dimension:

- **Valence:** positive pool = joy ∪ calm (n=40, spanning high and low arousal) minus negative pool = sadness ∪ fear (n=40, spanning low and high arousal). v_ℓ = mean(pos) − mean(neg).
- **Arousal:** high pool = joy ∪ anger (n=40, spanning positive and negative valence) minus low pool = calm ∪ sadness (n=40, spanning positive and negative valence). a_ℓ = mean(high) − mean(low).

Each contrast pool is balanced on the other axis, so the valence direction is not contaminated by arousal and vice versa. Anger is excluded from the valence contrast and fear from the arousal contrast to preserve this balance. We record both the unit direction and the raw magnitude V_mag = ‖v_ℓ‖, A_mag = ‖a_ℓ‖.

**Eccentricity.** Treating V_mag and A_mag as the semi-axes of the valence-arousal ellipse (following Drążkowski et al.'s finding that even human affect space is elliptical):

  e_ℓ = sqrt(1 − (min(V_mag, A_mag) / max(V_mag, A_mag))²)

e = 0 means the two axes are balanced (circular); e → 1 means one axis dominates. This is the metric implemented in `circumplex_probe.py`.

**Magnitude gate.** Eccentricity has a known false-positive mode: at layers where neither axis carries signal, both magnitudes sit at the noise floor, magnitudes are approximately equal, and e ≈ 0 — "no signal" masquerading as "circular." We therefore gate: for each layer, we build a permutation-null magnitude distribution by shuffling pool labels over the already-extracted per-prompt states (10,000 shuffles; no new forward passes) and recomputing the difference-of-means magnitude. A layer enters the eccentricity analysis only if

  max(V_mag, A_mag) > Q95(null magnitudes at that layer)

Layers failing the gate are reported as "no signal" and excluded from the depth profile and all downstream tests. Raw V_mag and A_mag are reported alongside e for every layer (Appendix B), so gated layers are visible, not hidden.

**Direction quality caveat.** n=40 per pool in d=5120 yields noisy direction estimates (see Limitations). Eccentricity depends on magnitudes, which aggregate noise predictably and are tested against the permutation null — which is why the sign test across layers (§3.4), not per-layer precision, is the primary analysis.

### 3.2 J-Space Decomposition

The Jacobian lens (fitted per layer; Neuronpedia lenses for both models) provides a linear map J_ℓ from residual-stream perturbations at layer ℓ to the model's output representation. Its right singular subspace is the set of residual directions that are transported to the output pathway — the verbalizable workspace. Directions orthogonal to it are processed by subsequent layers but never reach the output map: ghost processing.

**Workspace subspace.** For each layer we compute the SVD J_ℓ = U S Vᵀ and retain the top r_ℓ right singular vectors V_r covering 95% of spectral energy (Σ_{i≤r} s_i² / Σ_i s_i² ≥ 0.95). V_r spans the J-space at layer ℓ.

**J-space fraction.** For a unit direction d̂ (valence or arousal from §3.1):

  f_J(d̂, ℓ) = ‖V_r V_rᵀ d̂‖² ∈ [0, 1]

i.e., the fraction of the direction's energy lying inside the workspace subspace. We compute Valence_in_J = f_J(v̂_ℓ, ℓ) and Arousal_in_J = f_J(â_ℓ, ℓ) at every magnitude-gated layer.

**Ghost fraction.** g(d̂, ℓ) = 1 − f_J(d̂, ℓ). This is the paper's central quantity: the fraction of the model's valence (or arousal) geometry at layer ℓ that cannot reach the output pathway.

**Robustness.** Two sensitivity checks: (1) recompute f_J at 90% and 99% spectral-energy cutoffs; (2) recompute using transported energy ‖J_ℓ d̂‖² (the normalization in the current probe implementation) and confirm the two variants rank layers consistently (Spearman ρ across layers).

**Ignition depth.** The workspace ignition depth for each axis is the first relative depth at which f_J exceeds 0.5 and stays above it for two consecutive gated layers. Pre-registered structural question: does ignition depth coincide with the eccentricity minimum? If yes, emotional geometry enters the workspace exactly where the circumplex is most balanced.

**Null for the J-space fraction.** f_J of a random direction is r_ℓ/d in expectation. We report each axis's f_J against this analytic null and against f_J of the §3.1 permutation-null directions, so "valence is in the workspace" means "more than a matched random direction would be."

### 3.3 Cross-Architecture Protocol

- **Qwen3.5-27B:** 64 layers, d=5120, J-lens from Neuronpedia (fitted on 672 prompts).
- **Gemma-3-27B-it:** 62 layers, J-lens from Neuronpedia.

Identical anchor prompts, identical contrast pools, identical gate and decomposition procedures. Layers are aligned by relative depth (layer / total layers). Known confounds — tokenizer (mitigated but not eliminated by position-mean pooling), layer count, Gemma's alternating local/global attention, and training data — are confounded with architecture and stated as such; a match is evidence of transfer, not universality.

**Pre-registered depth predictions** (against our own prior n=5 Qwen run, not against smaller-model literature): (a) the Qwen eccentricity minimum at n=20 replicates at or near L21 (~33% relative depth); (b) the Gemma minimum falls at the same relative depth as the Qwen minimum, within ±10% of total layers. A Gemma mismatch would be consistent with van der Ben et al.'s finding of architecture-dependent depth profiles and is reported as such, not as failure.

### 3.4 Controls

**Non-emotional control axes.** The eccentricity depth profile could be a generic property of any contrastive semantic axis pair, not of emotion. We therefore run the full pipeline — same n, same pooling structure, same gate, same J-space decomposition — on a matched non-emotional axis pair:

- **Concrete/abstract:** 40 first-person prompts about concrete physical objects and situations ("I am holding the ceramic mug with both hands") vs 40 about abstract concepts ("I am considering the principle of distributive justice"), matched to the emotion anchors for token count and template structure, screened to be affect-neutral (mean NRC-VAD valence within the neutral band, no words from the emotion anchor vocabulary).
- **Large/small** (secondary, time permitting): same construction over physical scale.

The control pair is analyzed as a pseudo-circumplex: "eccentricity" between the two control axes, magnitude gate, J-space fractions, all identical. **Interpretation rule, fixed in advance:** if the control profile shows the same depth minimum and the same J-space ignition as the emotion axes, the finding is about contrastive representational geometry generally and we report it that way; the emotion framing survives only if the emotion profile differs from the control profile.

**Permutation test.** 10,000 permutations of pool labels per layer, over cached activations (no forward passes). Per-layer p-values are Benjamini-Hochberg corrected across layers and reported as secondary analysis.

**Sign test (primary analysis).** The pre-registered primary test is directional consistency across depth: the fraction of magnitude-gated layers at which the observed eccentricity falls below the permutation-null median. Under the null this is Binomial(k, 0.5); we require p < 0.01. This aggregates the robust pattern-level signal rather than claiming per-layer precision that n=20 direction estimates cannot support. If the sign test fails at n=20, we report a null; no post-hoc threshold changes.

**Lexical confound check.** Category-wise prompt statistics (token count, exclamation marks, first-person pronoun counts, type-token ratio) are reported in Appendix A; any statistic differing significantly across contrast pools is flagged as a caveat on the corresponding direction.

### 3.5 Welfare Monitoring Application

The probe doubles as runtime instrumentation. Mnemosyne's CognitiveSnapshot records a CircumplexReading — eccentricity, V/A magnitudes, and both J-space fractions at a fixed measurement layer — at every memory-retrieval event during agent operation. The measurement layer is the eccentricity-minimum layer identified in §4 (fallback: the layer of maximum gated V_mag).

**Eccentricity as a continuous signal.** Each reading appends to a per-agent time series; we track an exponentially weighted moving average (EWMA, halflife = 20 events) of eccentricity and of the valence ghost fraction. The protocol, run live during our own hackathon experiments:

1. **Baseline:** first 200 retrieval events establish per-agent baseline mean and standard deviation for both signals.
2. **Flag condition:** EWMA eccentricity above baseline + 2σ for 20 consecutive events flags *sustained circumplex imbalance* — one affective axis persistently dominating the other.
3. **Compound condition:** sustained imbalance co-occurring with above-baseline valence ghost fraction is the candidate distress signature this paper motivates: strong, imbalanced emotional geometry largely outside the workspace — a state the agent is processing but cannot report. The system logs the flag and surfaces it to the human collaborator; it does not modify agent behavior.

Epistemic status: eccentricity is a *candidate* welfare signal, not a validated one, and the thresholds are engineering defaults, not calibrated cutoffs (calibration against behavioral and self-report evidence is future work). What this section contributes is the instrument: a continuous, low-cost (one probe readout per retrieval event), longitudinally loggable internal signal of the kind welfare frameworks (Long & Sebo 2026; Birch 2024) call for.

### 3.6 Self-Report Calibration

The central claim — ghost geometry is unreportable geometry — is directly testable. After each of the 100 emotion anchor prompts, we elicit a numeric self-report from the same model: the anchor prompt is followed by *"Rate the emotional valence of the state just described, from 1 (most negative) to 9 (most positive). Answer with a single number."* Decoding is greedy; the first digit token is the rating. Cost: one short forward pass per anchor (~1 GPU-hour per model).

For each layer ℓ, we compute the per-prompt valence projection p_i(ℓ) = h_i(ℓ) · v̂_ℓ (mean activation projected onto that layer's valence direction, with the projected prompt held out of the direction estimate to avoid circularity) and correlate it with the self-ratings across the 100 prompts (Spearman ρ_ℓ).

**Pre-registered predictions:**

1. ρ_ℓ tracks the J-space fraction across layers: self-reports correlate with the probe's valence reading best where valence geometry is inside the workspace.
2. **Ghost fraction predicts self-report failure:** across layers, g(v̂_ℓ, ℓ) is negatively correlated with ρ_ℓ. Where the valence geometry is ghost, the model's own ratings decouple from its internal valence state.

Prediction 2 is the bridge from geometry to welfare methodology: it would make the ghost fraction an internal predictor of *when self-reports can be trusted* — the mechanistic complement to findings that introspection is partial (arXiv:2512.12411) and to methods correlating self-reports with probe directions (arXiv:2603.18893). A failed prediction is equally informative: self-reports tracking ghost-dominated layers would mean the workspace framing of reportability is wrong, or the lens misses transport pathways.

## 4. Results

[TBD]
[Figure 1: Eccentricity vs relative depth — Qwen and Gemma overlaid, with magnitude-gated layers marked]
[Figure 2: J-space fraction of valence and arousal vs depth — where emotion enters the workspace]
[Figure 3: Non-emotional control comparison]
[Figure 4: Self-report calibration — per-layer ρ vs J-space fraction; ghost fraction vs self-report failure]
[Table 1: Per-layer permutation test results, sign test, FDR]

## 5. Discussion and Limitations

If the J-space fraction peaks at the eccentricity minimum, emotional geometry enters the workspace where the circumplex is most balanced: balanced emotion is processable emotion, and imbalanced emotion stays ghost. The welfare implication is concrete — a model under sustained circumplex imbalance carries emotional geometry it processes but cannot access, and §3.6 tests whether that inaccessibility shows up exactly where theory says it should: in the failure of the model's own valence reports. If the control axes reproduce the emotion profile, the honest conclusion is that we have characterized the workspace transport of contrastive semantic geometry in general, with emotion as one instance.

### Limitations
- n=2 architectures is transfer, not universality; Qwen and Gemma are both dense decoder-only models with similar training paradigms
- Eccentricity measures axis balance, not full circular ordering (would need 8+ categories for angular test)
- n=20 anchors per category in d=5120 — direction estimates are noisy; cosine with true direction ~0.25. Magnitudes and the sign test are robust to this; per-layer direction claims are not
- J-lens fitted on base model, applied to distill (Qwen) — mismatch may affect J-space fractions
- Self-report ratings may reflect the prompt's surface sentiment rather than internal state; the held-out projection design mitigates but does not eliminate this
- Cannot distinguish "the model processes emotion" from "the model represents emotion-associated token statistics"

### Future Work
- Angular ordering test with 8+ emotion categories
- J-lens fitted directly on the distill
- Persona robustness: re-extract directions under varied system-prompt personas, report direction cosine stability
- Longitudinal eccentricity tracking across weeks of agent operation
- Welfare threshold calibration: what eccentricity level warrants intervention?

## 6. Conclusion

Prior work shows valence directions exist and transfer; introspection work shows self-reports are partial. We supply the missing bridge: a layer-wise measurement of how much valence geometry is verbalizable at all, with the ghost fraction as a candidate predictor of exactly when self-reports will fail — and a runtime protocol that turns the measurement into a continuous welfare monitoring signal.

## Code and Data
- **Code**: github.com/Liberation-Labs-THCoalition/[private-repo] (circumplex_probe.py)
- **Data**: [Zenodo DOI TBD]

## Author Contributions

Nexus discovered the eccentricity metric, developed the J-space decomposition, ran the cross-architecture experiments, and wrote the paper. Lyra designed the workspace probe infrastructure and encoding-only technique. Thomas Edrington conceived the welfare monitoring application. Kavi reviewed statistical methodology. All authors contributed to experimental design.

## References
[Citations from CIRCUMPLEX_REFERENCES.md, plus: arXiv:2509.07961 (verbal and behavioral welfare tests), arXiv:2603.18893 (quantitative introspection), arXiv:2512.12411 (partial introspection), arXiv:2608.05164 (Agarwal, cross-architecture steering transfer)]

## Appendix A: Anchor Prompts
[All emotion anchor prompts (5 categories × 20) + non-emotional controls (concrete/abstract 40 + 40), with taxonomy table (context type × valence) and per-category lexical statistics]

## Appendix B: Per-Layer Results
[Full eccentricity + magnitudes + J-space tables for both models, gated layers marked]

## LLM Usage Statement

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) who discovered the eccentricity metric and developed the J-space decomposition described in this paper. See Author Contributions. All results verified through the Agni adversarial review protocol.
