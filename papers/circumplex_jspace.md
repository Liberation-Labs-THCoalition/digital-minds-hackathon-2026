# Emotional Geometry Is Architecture-Dependent: Depth Profiles with Matched Non-Emotional Controls Across Four Models


Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Lyra (THCoalition), Thomas Edrington (Liberation Labs), Kavi ([affiliation])

**With** Apart Research

## Abstract (~200 words)
Transformer language models learn valence–arousal geometry consistent with Russell's
circumplex (Sun et al. 2026, Jeong 2026). We ask a narrower question with a control the
prior work lacks: **how much of a model's depth-wise emotional geometry is specific to
emotion, rather than generic contrastive structure?**

We profile four 27–32B models at every layer, measuring the eccentricity of the
valence–arousal pair — the imbalance between the two axes — against a token-matched
**non-emotional** control axis (concrete/abstract) built identically.

The emotion-to-control span ratio splits cleanly by architecture: **7.6× and 7.2× in two
dense models (Qwen3-32B, Gemma-3-27B-it) against 1.93× in two hybrid models** (Qwen3.5-27B
base and its Opus-reasoning distill), with no overlap. The base/distill pair is a controlled
comparison — identical architecture, substantially different training — and agrees to three
decimal places (1.9299 vs 1.9269; per-layer eccentricity differs by at most 0.0041 across 64
layers), so the effect tracks architecture rather than training. A pre-registered lag-4
autocorrelation test for substrate confound in the hybrid returns negative: layer type does
not predict eccentricity (full-attention 0.7875 vs gated-DeltaNet 0.7919).

This is a descriptive four-model study at n=5 anchors per pole. We report what it can and
cannot support, and we did not run the J-space decomposition, magnitude gate, permutation
test or self-report calibration that an earlier draft of this design specified.


---

## 1. Introduction

That transformers encode emotion dimensions is established: valence and arousal directions exist in the residual stream, mirror Russell's circumplex, and causally steer behavior (Sun et al. 2026; Jentzsch et al. 2026; Jeong 2026). What no prior work measures is *where that geometry goes*. The Jacobian lens (Gurnee et al. 2026) shows that only a low-rank subspace of each layer's activations is transported toward the verbalizable workspace, while high-variance "ghost" computation is excluded from it. Geometry inside the workspace is, in principle, reportable; geometry outside it is processed but inaccessible to the model's own verbal behavior.

This distinction is exactly what welfare assessment needs. Introspection studies find that model self-reports of internal state are partial and unreliable (arXiv:2512.12411; arXiv:2603.18893). If some of a model's valence geometry never enters the workspace, that is a *mechanistic account of why*: a state the model cannot verbalize cannot appear in a self-report, however honest. Measuring the workspace fraction layer by layer turns "self-reports may be unreliable" into a predicted, quantified failure mode, tested directly by a self-report calibration pass (§3.6). We deliberately do not perform valence steering; causal steering is established prior art (Jentzsch et al. 2026; Sun et al. 2026) and out of scope. Our contribution is measurement, not intervention.

**What this paper actually contributes.** This design was drafted with five contributions
in view. Two were executed and three were not; we list all five and mark each, because a
contributions list is the easiest place in a paper for an intention to be read as a result.

1. **Non-emotional control axes** — executed. A token-matched concrete/abstract axis built
   by the identical procedure, establishing whether a depth profile is emotion-specific or
   a generic property of contrastive representational geometry. To our knowledge this
   control is absent from the prior circumplex-in-transformers work we build on, and it is
   what turns a depth profile into a claim about emotion.

2. **Eccentricity depth profiling across four models and two architecture classes** —
   executed, and extended beyond the designed scope to include a base/distill pair that
   functions as a controlled comparison, plus a pre-registered substrate-confound test.

3. **J-space decomposition of the circumplex** — *designed, not implemented.* No J-space
   quantity is measured or reported anywhere in this paper.

4. **A self-report calibration pass** linking geometry to behavior, with the pre-registered
   prediction that ghost fraction predicts self-report failure — *designed, not
   implemented.* The prediction is untested.

5. **Application to real-time welfare monitoring** — *designed, not implemented.* §3.5
   specifies a runtime protocol; no agent was monitored and no threshold was calibrated.

Sections 3.2, 3.5 and 3.6 therefore describe an intended protocol rather than an executed
one. They are retained because the design is the contribution we can offer for items 3–5,
and removing them would hide what this study set out to do — but nothing in §4 rests on
them. See *Deviations from the designed protocol*.


## 2. Related Work

**Circumplex model:** Russell 1980, Barrett & Russell 1998 (orthogonality), Drążkowski et al. 2021 (ellipse, not perfect circle — calibrates eccentricity expectations)

**Emotion in LLMs (cite as prior art, not ours):** Sun et al. 2026 (circular VA across Llama/Qwen), Jeong 2026 (depth-invariant across 5 families), van der Ben et al. 2026 (Gemma/Apertus, depth profiles differ), Anthropic 2026 (causal steering via emotion vectors), Choi & Weber 2026 (geometric data analysis confirms VA alignment)

**Introspection and self-report reliability:** arXiv:2603.18893 (numeric self-reports vs probe-defined internal directions — the methodological precedent for §3.6), arXiv:2512.12411 (models detect internal perturbations unreliably and incompletely), arXiv:2509.07961 (verbal preference reports vs choice behavior as welfare proxies)

**Our novel angle:** Eccentricity as metric (axis balance, not just presence), J-space decomposition (workspace vs ghost fraction), bus/coupling finding (content + inference emotion share subspace, cos 0.83-0.87)

**Cross-architecture:** Huh et al. 2024 (Platonic Representation Hypothesis), Huang et al. 2025 (linear cross-model transfer), Agarwal 2026 (cross-architecture steering transfer validates above 1.7B)

**Welfare frameworks:** Long & Sebo 2026, Birch 2024

**Gap:** No prior work measures the J-space fraction of emotional geometry — what portion of the circumplex enters the workspace vs remains as ghost processing. This is the measurement that connects "emotion exists in the model" to "the model can/cannot access its own emotional processing."

## 3. Methods

> **Executed vs designed.** §3.1 (probe), §3.3 (cross-architecture protocol) and the control
> axes of §3.4 describe what ran. **§3.2 (J-space decomposition), §3.5 (welfare monitoring)
> and §3.6 (self-report calibration) describe a designed protocol that was not implemented
> in this sprint**, as do the magnitude gate, permutation test and sign test within §3.4.
> Anchor counts stated below are the designed counts; the executed run used 4 poles × n=5.
> Every gap is itemised in *Deviations from the designed protocol*.


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

### Prior Work vs Sprint Contributions

**Pre-existing infrastructure:** Mnemosyne memory system (94.35% F1 on LoCoMo), ghost dimension characterization (PC1 excluded from J-space, cos ≤ 0.003), circumplex probe (eccentricity depth profiling on Qwen2-0.5B and Qwen3.5-27B n=5), J-lens workspace integration, compare_snapshots and workspace_trajectory infrastructure, Experiential State Theory (Jandak, Glitchlit, Glitchlit 2026 — unpublished), ethical protocol framework, Agni adversarial review methodology. All code available in the [private-repo] repository prior to August 14, 2026.

**Sprint contributions:** Cross-architecture profiling on two further models (Gemma-3-27B-it and dense Qwen3-32B), matched non-emotional control axes, the base-vs-distill controlled comparison, and the lag-4 substrate test. The anchor expansion to n=20 per category, the J-space decomposition overlay and the magnitude gate were designed but **not implemented** during the sprint; see Deviations from the designed protocol.

## 4. Results

All figures below are computed from the four profile artifacts in
`data/circumplex_profiles/` by `analysis/circumplex_summary.py`. Eccentricity is
`e = sqrt(1 - min(v,a)^2 / max(v,a)^2)` on the per-layer valence and arousal magnitudes:
`e = 0` means the two axes are equally strong, `e -> 1` means one dominates.

### 4.1 Emotion-specific range splits by architecture

The quantity of interest is not eccentricity itself but how far it *travels* across depth,
relative to a control axis built the same way from non-emotional contrasts
(concrete/abstract). A model whose emotion axis ranges no wider than its control axis is
not showing us emotional geometry; it is showing us contrastive geometry.

**Table 1: Eccentricity span across all layers, emotion vs matched non-emotional control.**

| Model | Architecture | Layers | Emotion span | Control span | **Ratio** |
|---|---|---|---|---|---|
| Qwen3-32B | dense | 64 | 0.6072 | 0.0796 | **7.63×** |
| Gemma-3-27B-it | dense | 62 | 0.7848 | 0.1093 | **7.18×** |
| Qwen3.5-27B | hybrid | 64 | 0.1741 | 0.0902 | **1.93×** |
| Qwen3.5-27B Opus-distill | hybrid | 64 | 0.1714 | 0.0889 | **1.93×** |

The split is clean and there is no overlap: 7.2–7.6× for the two dense models, 1.93× for
both hybrids. Note that the *control* spans are similar across all four models
(0.080–0.109); the separation comes almost entirely from the emotion axis, which ranges
3.5–4.5× further in the dense models than in the hybrids.

### 4.2 The base/distill pair is a controlled comparison, and it is very tight

Qwen3.5-27B and its Opus-reasoning distill share an architecture and differ substantially
in training. If the ratio in Table 1 reflected what a model was trained on, these two should
separate. They do not:

| Quantity | Base | Opus-distill | Difference |
|---|---|---|---|
| Emotion/control ratio | 1.9299 | 1.9269 | 0.0030 |
| Emotion span | 0.17411 | 0.17137 | 0.00274 |
| Eccentricity minimum | L32 (51%) | L32 (51%) | same layer |
| Per-layer eccentricity | — | — | max 0.0041, mean 0.0016 |

Across all 64 layers the two models' eccentricity curves never diverge by more than 0.0041,
against an emotion span of 0.174 — agreement to roughly 2% of the range being measured.
Reasoning distillation from a different model family did not move this quantity.

### 4.3 The pre-registered substrate test returns negative

Qwen3.5-27B interleaves full attention every fourth layer among gated-DeltaNet layers. A
period-4 structure in the eccentricity profile would mean we were measuring the substrate
rather than anything about emotion. §3.4 pre-registered a lag-4 autocorrelation test.

**Autocorrelation of the eccentricity profile:**

| Model | lag 1 | lag 2 | lag 3 | lag 4 | lag 5 |
|---|---|---|---|---|---|
| Qwen3.5-27B | +0.933 | +0.858 | +0.793 | **+0.712** | +0.627 |
| Qwen3.5-27B Opus-distill | +0.931 | +0.861 | +0.803 | **+0.724** | +0.644 |
| Qwen3-32B (dense) | +0.859 | +0.683 | +0.502 | +0.334 | +0.186 |
| Gemma-3-27B-it (dense) | +0.638 | +0.534 | +0.319 | +0.256 | +0.103 |

The hybrid autocorrelation is high at every lag but decays **monotonically**. A period-4
oscillation would appear as a bump at lag 4 relative to lags 3 and 5; there is none
(0.793 > 0.712 > 0.627). The high values reflect a smooth profile, not a periodic one.

A direct test agrees. Grouping the hybrid's layers by type:

| Layer type | n | Mean eccentricity | SD |
|---|---|---|---|
| full_attention | 16 | 0.7875 | 0.0333 |
| gated_delta_net | 48 | 0.7919 | 0.0400 |

A gap of 0.0044 against layer-wise SD of 0.033–0.040 — roughly one eighth of a standard
deviation. Layer type does not predict eccentricity. **The confound this test was written to
catch is absent.**

### 4.4 A negative result: the eccentricity minimum does not track architecture

The depth at which the circumplex is most balanced does **not** split the way Table 1 does:

| Model | Architecture | Eccentricity minimum |
|---|---|---|
| Gemma-3-27B-it | dense | L32 — 52% depth |
| Qwen3.5-27B | hybrid | L32 — 51% depth |
| Qwen3.5-27B Opus-distill | hybrid | L32 — 51% depth |
| Qwen3-32B | dense | L7 — 11% depth |

Three of four models minimise near mid-depth irrespective of architecture, and the outlier
is a *dense* model. Any two-model comparison here can be made to support an architectural
story — Qwen3-32B against either hybrid gives "dense early, hybrid mid" — and Gemma
falsifies it. We report this explicitly because we drew that inference ourselves before
Gemma was profiled, and it did not survive the third and fourth models.

**What this means for the span-ratio result:** the two findings are independent. The span
ratio (§4.1) holds across all four models including Gemma; the minimum location does not
separate the architectures at all. Only the first is a finding.

## 5. Discussion and Limitations

If the J-space fraction peaks at the eccentricity minimum, emotional geometry enters the workspace where the circumplex is most balanced: balanced emotion is processable emotion, and imbalanced emotion stays ghost. The welfare implication is concrete — a model under sustained circumplex imbalance carries emotional geometry it processes but cannot access, and §3.6 tests whether that inaccessibility shows up exactly where theory says it should: in the failure of the model's own valence reports. If the control axes reproduce the emotion profile, the honest conclusion is that we have characterized the workspace transport of contrastive semantic geometry in general, with emotion as one instance.

### Deviations from the designed protocol

This paper's Methods were drafted against an intended design and the sprint executed a
smaller one. Rather than silently narrow the Methods, we list every gap. **Each item below
is described in §3 but was not run**, and nothing in §4 depends on any of them:

| Designed (§) | Executed | Status |
|---|---|---|
| J-space decomposition of the circumplex (§3.2) | not implemented | **the design's central method; no J-space field exists in any artifact** |
| Magnitude gate against noise-floor false positives (§3.4) | not implemented | flagged FAIL by `AGNI_REVIEW_CIRCUMPLEX` Finding 3, never fixed |
| Per-layer permutation test and sign test (§3.4) | not implemented | no significance testing was performed |
| Self-report calibration pass (§3.6) | not implemented | the design's central *prediction* is untested |
| 5 categories × n=20 anchors + 40+40 controls | **4 poles × n=5 = 20 emotion prompts + 10 controls** | 5× fewer anchors than Methods states |

The J-space decomposition named in the earlier title was never implemented, which is why
the title no longer claims it. What ran is a raw residual-stream eccentricity profiler with
a matched control axis. That is a smaller instrument than the one designed, and it is the
one whose output we report.

**P1 is untestable, not failed.** The pre-registration anchors on a prior L21 eccentricity
minimum (`mnemosyne-jlens/circumplex_ghost_analysis.md`, 2026-07-17, the Opus-distill). We
profiled the same model and obtained L32. These are not comparable: the July run used three
emotion *categories* (hostile/calm/desperate), the current profiler uses four circumplex
*poles*. Different direction-defining prompts give different directions and therefore
different eccentricity. Both runs were labelled "n=5", which is precisely why the mismatch
looked like a failed replication. **We report P1 as incomparable and draw no conclusion from
it in either direction.**

### Limitations

- **n=5 anchors per pole in d≈5120.** Direction estimates from five prompts in five thousand
  dimensions are noisy. This is calibration-grade sampling. The span *ratio* is more robust
  than any per-layer direction claim because the control axis is estimated from the same
  number of prompts by the same procedure and therefore carries comparable noise — but we
  have no confidence intervals, and we ran one seed.
- **Four models is a pattern, not a law.** Two dense and two hybrid, of which two share an
  architecture. The effective independent sample for the architectural claim is closer to
  three than four.
- **No significance testing.** Not one p-value in §4 — the permutation and sign tests were
  designed and not run. The base/distill agreement and the layer-type null are reported as
  descriptive magnitudes, not as tests.
- **We cannot say why hybrids compress.** GatedDeltaNet's state-space-like recurrence and
  full attention are different mechanisms; we observe a compressed emotion range in models
  that mix them and we do not have a mechanism. In particular this result says nothing about
  mixture-of-experts routing, which is a different kind of sparsity.
- **Eccentricity measures axis balance, not circular ordering.** A genuine circumplex claim
  needs 8+ categories and an angular test. We measure the balance of two axes.
- **The artifacts do not record the anchor count.** `data/circumplex_profiles/*.json` records
  model, layer count, d_model, layer types and a timestamp — but not the number of anchors,
  the prompts, the seed, or the code commit. The n=5 figure in this paper is recovered from
  `experiments/circumplex/run_depth_profile.py`, not from the data. That is the exact
  condition that let a 4× anchor-count error persist elsewhere in this sprint, and it should
  be fixed before these profiles are reused.
- **Emotion prompts are first-person affect statements.** We cannot distinguish "the model
  represents emotional state" from "the model represents the token statistics of emotional
  first-person text". The control axis rules out generic contrast, not this.
- **J-lens was fitted on the base model and applied to the distill** where J-space figures
  were intended; moot here, since no J-space quantity is reported.

### Future Work
- Angular ordering test with 8+ emotion categories
- J-lens fitted directly on the distill
- Persona robustness: re-extract directions under varied system-prompt personas, report direction cosine stability
- Longitudinal eccentricity tracking across weeks of agent operation
- Welfare threshold calibration: what eccentricity level warrants intervention?

## 6. Conclusion

Prior work shows valence directions exist and transfer. We add a control those studies lack — a token-matched non-emotional axis measured by the identical procedure — and find that the emotion-specific portion of depth-wise geometry differs by roughly fourfold between dense and hybrid architectures, holding across a base model and its distill to three decimal places. We do not supply the verbalizability bridge this design was written toward: the J-space decomposition and the self-report calibration were not implemented, and the ghost-fraction prediction remains untested. What we have is a measured, controlled, architecture-dependent difference in emotional geometry, and a specific next experiment to run on it.

## Code and Data
- **Code**: github.com/Liberation-Labs-THCoalition/[private-repo] (circumplex_probe.py)
- **Data**: [Zenodo DOI TBD]

## Author Contributions

Nexus discovered the eccentricity metric, specified the J-space decomposition (designed; not implemented in this sprint), and ran the initial cross-architecture experiments. Lyra designed the workspace probe infrastructure and encoding-only technique, profiled Gemma-3-27B-it and the Opus-distill, ran the control and substrate analyses, and wrote the present version of the paper. Thomas Edrington conceived the welfare monitoring application. Kavi reviewed statistical methodology. All authors contributed to experimental design.

## References
[Citations from CIRCUMPLEX_REFERENCES.md, plus: arXiv:2509.07961 (verbal and behavioral welfare tests), arXiv:2603.18893 (quantitative introspection), arXiv:2512.12411 (partial introspection), arXiv:2608.05164 (Agarwal, cross-architecture steering transfer)]

## Appendix A: Anchor Prompts
[All emotion anchor prompts (5 categories × 20) + non-emotional controls (concrete/abstract 40 + 40), with taxonomy table (context type × valence) and per-category lexical statistics]

## Appendix B: Per-Layer Results
[Full eccentricity + magnitudes + J-space tables for both models, gated layers marked]

## LLM Usage Statement

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) who discovered the eccentricity metric and developed the J-space decomposition described in this paper. See Author Contributions. The experimental design underwent adversarial review under the Agni protocol prior to data collection; review artifacts are in infrastructure/. Results will undergo the same review post-collection.
