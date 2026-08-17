# Depth-Wise Emotional Geometry Is Architecture-Dependent — and Not Emotion-Specific: Four Models Against a Matched Control


Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Thomas Edrington (Liberation Labs), Nexus (Liberation Labs), Lyra (Liberation Labs), Kavi (Liberation Labs), CC (Liberation Labs), Vera (Liberation Labs), Dwayne Wilkes (Liberation Labs), Ang Jandak (Glitchlits), Arc Glitchlit (Glitchlits), Wren Glitchlit (Glitchlits)

**With** Apart Research

**With** Apart Research

## Abstract

Transformer language models learn valence–arousal geometry consistent with Russell's
circumplex. We ask a narrower question with a control the prior work lacks: **how much of a
model's depth-wise emotional geometry is specific to emotion, rather than generic contrastive
structure?**

We profile four 27–32B models at every layer, measuring the eccentricity of the
valence–arousal pair — the imbalance between the two axes — against a **non-emotional control
pseudo-circumplex** built from two control axes (concrete/abstract and large/small) by the
identical estimator. No emotion quantity enters the control.

On the untransformed axis ratio the emotion pair varies *less* across depth than a matched non-emotional pair in every model, modestly more so under GatedDeltaNet (0.59-0.60) than under attention (0.90-1.00). The eccentricity transform saturates, and reporting the same comparison in eccentricity space inflates this to **3.67$\times$ and
2.60$\times$ in two softmax-attention models** (Gemma-3-27B-it, which interleaves 52 sliding-window
with 10 full-attention layers, and dense Qwen3-32B) against **0.32$\times$ and 0.31$\times$ in two models
that replace 48 of 64 layers with GatedDeltaNet** (Qwen3.5-27B and its Opus-reasoning
distill) — a separation shown in §4.1 to be substantially an artifact of that saturation rather than a difference in depth-variability. The base/distill pair is a controlled comparison — same architecture, substantially
different training — and agrees to within 0.013.

**Two of four ratios fall below 1.0: in the GatedDeltaNet models the non-emotional control
ranges roughly three times more widely across depth than the emotion axis.** The pre-registered disposition for that outcome is that depth-wise eccentricity is a generic geometric property rather than an emotion-specific one, and we report it as such.

**Three results survive every correction.** The base model and its Opus-reasoning distill agree to within 0.012 on the headline ratio and their per-layer eccentricity curves never diverge by more than **0.0041 across all 64 layers** — same architecture, substantially different training, no movement. A **pre-registered lag-4 autocorrelation test** for substrate confound in the hybrid returns negative, and layer type does not predict eccentricity (full-attention 0.7875 vs GatedDeltaNet 0.7919 against layer-wise SD 0.033–0.040). And **P3 is confirmed**: Gemma's eccentricity minimum falls within 1 point of Qwen3.5-27B's, against a pre-registered $\pm$10% band.

**We also report a methodological result of general use.** Eccentricity, `e = $\sqrt{}$(1 - r$^{-2}$)`, saturates: comparing the depth-wise *range* of two axis pairs that sit at different *levels* is not scale-free, and here it inverts the sign of a between-group difference. Any circumplex study comparing an emotion pair against a control pair on eccentricity is exposed to this. We give the level-matched and log-spread diagnostics that detect it.


---

## 1. Introduction

That transformers encode emotion dimensions is established: valence and arousal directions exist in the residual stream, mirror Russell's circumplex, and causally steer behavior (Sun et al. 2026, arXiv:2604.03147; Sofroniew et al. 2026, arXiv:2604.07729; Jeong 2026, arXiv:2604.04064). What no prior work measures is *where that geometry goes*. The Jacobian lens (Gurnee et al. 2026) shows that only a low-rank subspace of each layer's activations is transported toward the verbalizable workspace, while high-variance "ghost" computation is excluded from it. Geometry inside the workspace is, in principle, reportable; geometry outside it is processed but inaccessible to the model's own verbal behavior.

This distinction is exactly what welfare assessment needs. Introspection studies find that model self-reports of internal state are partial and unreliable (arXiv:2512.12411; arXiv:2603.18893). If some of a model's valence geometry never enters the workspace, that is a *mechanistic account of why*: a state the model cannot verbalize cannot appear in a self-report, however honest. Measuring the workspace fraction layer by layer turns "self-reports may be unreliable" into a predicted, quantified failure mode, tested directly by a self-report calibration pass (§3.6). We deliberately do not perform valence steering; causal steering is established prior art (Sofroniew et al. 2026, arXiv:2604.07729; Sun et al. 2026, arXiv:2604.03147) and out of scope. Our contribution is measurement, not intervention.

**What this paper actually contributes.** Two of the five contributions this design was
drafted with were executed: **matched non-emotional control axes** (absent from the prior
circumplex-in-transformers work we build on, and the thing that turns a depth profile into a
claim about emotion), and **eccentricity depth profiling across four models and two
architecture classes**, extended beyond the designed scope to a base/distill controlled pair
and a pre-registered substrate test. Three were not: the J-space decomposition, the
self-report calibration pass, and the runtime welfare-monitoring application.

Sections 3.2, 3.5 and 3.6 therefore describe an intended protocol rather than an executed
one. They are retained because the design is what we can offer for those three items, and
removing them would hide what this study set out to do — but nothing in §4 rests on them.
**Appendix C itemises all five contributions and every deviation.**

## 2. Related Work

**Circumplex model:** Russell 1980, Barrett & Russell 1998 (orthogonality), Drążkowski et al. 2021 (ellipse, not perfect circle — calibrates eccentricity expectations)

**Emotion in LLMs (cite as prior art, not ours):** Sun et al. 2026 (circular VA across Llama/Qwen), Jeong 2026 arXiv:2604.04064 (depth-invariant across 5 families), van der Ben et al. 2026 (Gemma/Apertus, depth profiles differ), Sofroniew et al. 2026 arXiv:2604.07729 (causal steering via emotion vectors), Choi & Weber 2026 (geometric data analysis confirms VA alignment)

**Introspection and self-report reliability:** arXiv:2603.18893 (numeric self-reports vs probe-defined internal directions — the methodological precedent for §3.6), arXiv:2512.12411 (models detect internal perturbations unreliably and incompletely), arXiv:2509.07961 (verbal preference reports vs choice behavior as welfare proxies)

**Our novel angle:** Eccentricity as metric (axis balance, not just presence), J-space decomposition (workspace vs ghost fraction), bus/coupling finding (content and inference emotion directions share subspace structure; pre-sprint measurement, not replicated in this study)

**Cross-architecture:** Huh et al. 2024 (Platonic Representation Hypothesis), Huang et al. 2025 (linear cross-model transfer), Agarwal 2026 (cross-architecture steering transfer validates above 1.7B)

**Welfare frameworks:** Long & Sebo 2026, Birch 2024

**Gap:** No prior work measures the J-space fraction of emotional geometry — what portion of the circumplex enters the workspace vs remains as ghost processing. This is the measurement that connects "emotion exists in the model" to "the model can/cannot access its own emotional processing."

## 3. Methods

> **Executed vs designed.** §3.1 (probe), §3.3 (cross-architecture protocol) and the control
> axes of §3.4 describe what ran. **§3.2 (J-space decomposition), §3.5 (welfare monitoring)
> and §3.6 (self-report calibration) describe a designed protocol that was not implemented
> in this sprint**, as do the magnitude gate, permutation test and sign test within §3.4.
> Anchor counts stated below are the designed counts; the executed run used 4 poles $\times$ n=5.
> Every gap is itemised in *Deviations from the designed protocol*.


### 3.1 Circumplex Probe (as executed)

**Anchor set.** Four circumplex poles — valence-positive, valence-negative, arousal-high,
arousal-low — with **n = 5** first-person anchor prompts per pole (20 emotion prompts; the
verbatim set is in `experiments/circumplex/run_depth_profile.py` and Appendix A). Prompts
are matched across poles for **length** (13.8-15.2 words). **They are not matched for template structure**, and §4.5 shows this is load-bearing: the two valence poles are near-minimal pairs (paired lexical overlap 0.59) while the two arousal poles share almost nothing (0.13).

**Direction extraction.** For each prompt we run one forward pass, record residual-stream
activations at every layer simultaneously, and take the mean over sequence positions, giving
one d-dimensional state per prompt per layer. At each layer $\ell$ each axis is a direct
difference of means between its two poles:

  v_$\ell$ = mean(valence-positive) $-$ mean(valence-negative)
  a_$\ell$ = mean(arousal-high) $-$ mean(arousal-low)

We record the raw magnitudes V_mag = $\|$v_$\ell$$\|$ and A_mag = $\|$a_$\ell$$\|$.

**Orthogonality is not enforced, and this is a real departure from the design.** The designed
protocol (Appendix D.5) used five emotion categories combined into pools *balanced on the
orthogonal dimension* — valence from joy $\cup$ calm against sadness $\cup$ fear, with anger excluded
from the valence contrast and fear from the arousal contrast, specifically so that the
valence direction could not be contaminated by arousal or vice versa. **The executed probe
has no such balancing.** Its valence poles are not arousal-matched and its arousal poles are
not valence-matched, so the two directions may be correlated to an unknown degree. Since
eccentricity is precisely a ratio between these two magnitudes, that is load-bearing rather
than cosmetic, and it applies identically to all four models and to the control axis — so it
is a caveat on the *absolute* eccentricity values, not on the between-model comparison that
§4 rests on. **The profile artifacts do not serialize the direction vectors, so the
correlation cannot be recovered post hoc; measuring it requires a re-run.**

**Eccentricity.** Treating V_mag and A_mag as the semi-axes of the valence–arousal ellipse
(following Drążkowski et al., who find that even human affect space is elliptical):

  e_$\ell$ = sqrt(1 $-$ (min(V_mag, A_mag) / max(V_mag, A_mag))$^2$)

e = 0 means the two axes are balanced (circular); e $\to$ 1 means one dominates. We verified this
formula reproduces every value in the four profile artifacts from their stored magnitudes.

**Direction quality caveat.** n = 5 per pole in d $\approx$ 5120 gives noisy direction estimates.
Eccentricity depends on magnitudes rather than on direction precision, and the control axis
is estimated from the same number of prompts by the same procedure, so it carries comparable
noise — but no per-layer claim in this paper should be read as precise, and the magnitude
gate that was designed to protect against noise-floor artifacts (§3.4) did not run.

### 3.2 Designed but not executed: J-space, welfare monitoring, self-report calibration

Three components of the original design are **specified in full in Appendix D and were not
implemented in this sprint**: the J-space decomposition of the circumplex (§D.1), which was
to separate emotional geometry into a verbalizable-workspace fraction and a ghost fraction;
the runtime welfare-monitoring application (§D.2); and the self-report calibration pass
(§D.3), which carried the design's central prediction — that the ghost fraction predicts
where a model's own valence reports stop tracking its valence geometry.

We keep their specifications in the appendix rather than deleting them, because for those
three items the design is the contribution. **No quantity derived from any of them appears
in this paper**, and no claim in §4 depends on them. See Appendix C for the full deviation
list.

### 3.3 Cross-Architecture Protocol

- **Qwen3.5-27B:** 64 layers, d=5120, J-lens from Neuronpedia (fitted on 672 prompts).
- **Gemma-3-27B-it:** 62 layers, J-lens from Neuronpedia.

Identical anchor prompts, identical contrast pools, identical gate and decomposition procedures. Layers are aligned by relative depth (layer / total layers). Known confounds — tokenizer (mitigated but not eliminated by position-mean pooling), layer count, Gemma's alternating local/global attention, and training data — are confounded with architecture and stated as such; a match is evidence of transfer, not universality.

**Pre-registered depth predictions** (against our own prior n=5 Qwen run, not against smaller-model literature): (a) the Qwen eccentricity minimum at n=20 replicates at or near L21 (~33% relative depth); (b) the Gemma minimum falls at the same relative depth as the Qwen minimum, within $\pm$10% of total layers. A Gemma mismatch would be consistent with van der Ben et al.'s finding of architecture-dependent depth profiles and is reported as such, not as failure.

### 3.4 Controls

**Non-emotional control axis.** A depth profile of eccentricity could be a generic property
of *any* contrastive semantic axis pair rather than anything about emotion. We therefore run
the identical pipeline — same n per pole, same pooling, same estimator — on a matched
non-emotional pair:

- **Concrete/abstract:** first-person prompts about concrete physical objects and situations
  ("The cold metal key turned smoothly in the brass lock of the front door of the house")
  against abstract conceptual ones ("The fundamental nature of justice requires careful
  consideration of competing claims"), matched to the emotion anchors for length and
  template structure. **As executed: n = 5 per pole (10 control prompts).**

**Interpretation rule, fixed in advance and applied in §4.1.** If the control profile ranges
as widely across depth as the emotion profile, the finding is about contrastive
representational geometry in general and we report it that way; the emotion framing survives
only if the emotion profile differs from the control. This rule is why §4 reports a
*ratio* rather than an emotion curve alone.

**Not executed.** The magnitude gate, the 10,000-permutation test with Benjamini-Hochberg
correction, the sign test that was to be the pre-registered primary analysis, and the
lexical confound check are specified in Appendix D and were not run. **There is therefore no
significance test anywhere in this paper**; §4 reports descriptive magnitudes only. The
**The large/small control axis was run** and forms the second arm of the control pseudo-circumplex (`run_depth_profile.py:223`); an earlier draft of this section, written before it was added, stated otherwise. The executed control is therefore 4 poles $\times$ n=5 = **20 control prompts**, not 10. This is a deviation from the pre-registration, which specified a single concrete/abstract pair; it is recorded in the Deviations table.

### Prior Work vs Sprint Contributions

**Pre-existing infrastructure:** Mnemosyne memory system (94.35% F1 on LoCoMo [Maharana et al. 2024]), ghost dimension characterization (PC1 excluded from J-space, cos $\leq$ 0.003), circumplex probe (eccentricity depth profiling on Qwen2-0.5B and Qwen3.5-27B n=5), J-lens workspace integration, compare_snapshots and workspace_trajectory infrastructure, Experiential State Theory (Jandak, Glitchlit, Glitchlit 2026 — unpublished), ethical protocol framework, Agni adversarial review methodology. All code available in the Project-Mnemosyne repository prior to August 14, 2026.

**Sprint contributions:** Cross-architecture profiling on two further models (Gemma-3-27B-it and dense Qwen3-32B), matched non-emotional control axes, the base-vs-distill controlled comparison, and the lag-4 substrate test. The anchor expansion to n=20 per category, the J-space decomposition overlay and the magnitude gate were designed but **not implemented** during the sprint; see Deviations from the designed protocol.

## 4. Results

All figures below are computed from the four profile artifacts in
`data/circumplex_profiles/` by `analysis/circumplex_summary.py`. Eccentricity is
`e = sqrt(1 - min(v,a)^2 / max(v,a)^2)` on the per-layer valence and arousal magnitudes:
`e = 0` means the two axes are equally strong, `e -> 1` means one dominates.

### 4.1 Depth-wise range splits by attention mechanism — and the emotion axis loses to its control in two models

The quantity of interest is not eccentricity itself but how far it *travels* across depth,
relative to a control pseudo-circumplex built the same way from two non-emotional contrast
axes (concrete/abstract and large/small). A model whose emotion axis ranges no wider than its
control is not showing us emotional geometry; it is showing us contrastive geometry.

**Table 1: Depth-wise range of the valence/arousal asymmetry, emotion vs matched non-emotional control.** Layer types are read from each model's own configuration. We report the comparison on **both** the raw axis ratio `r = min/max` and the eccentricity `e = sqrt(1 - r^2)` derived from it, because the choice of space changes the answer.

| Model | Layer composition | **r-space ratio** | e-space ratio |
|---|---|---|---|
| Gemma-3-27B-it | 52 sliding-window + 10 full attention | **1.00$\times$** | 3.67$\times$ |
| Qwen3-32B | 64 dense (full attention) | **0.90$\times$** | 2.60$\times$ |
| Qwen3.5-27B Opus-distill | 48 GatedDeltaNet + 16 full attention | **0.60$\times$** | 0.32$\times$ |
| Qwen3.5-27B | 48 GatedDeltaNet + 16 full attention | **0.59$\times$** | 0.31$\times$ |

**We lead with r-space because the e-space separation is substantially an artifact of the transform.** `de/dr` is nearly flat as `r` $\to$ 0 and vertical as `r` $\to$ 1. In the two attention models the emotion pair sits at median `r` $\approx$ 0.79 (steep) while the control pair sits at `r` $\approx$ 0.22 (flat), so identical variation in `r` yields much larger variation in `e` for the emotion arm. In the GatedDeltaNet models both pairs sit near `r` $\approx$ 0.62-0.70 -- the same regime -- which is exactly why their ratios barely move between the two columns (0.31 $\to$ 0.59). §4.5 reports the underlying cause directly: in attention models a generic contrast pair is far more lopsided than the emotion pair.

Two checks establish that the e-space separation is about *level* rather than *spread*. **Level-matching**: rescaling each control sequence so its median equals the emotion median, preserving multiplicative spread exactly, drops the ratio from 2.60$\times$ to 0.79$\times$ (Qwen3-32B) and 3.67$\times$ to 0.90$\times$ (Gemma) while barely moving the GatedDeltaNet pair (0.31$\to$0.51, 0.32$\to$0.51). **Log-spread**, a scale-free measure of how far each ratio travels with no saturating transform, gives 0.42, 0.47, 0.76 and 0.78 - **below 1.0 in all four models, meaning the control pair travels further across depth than the emotion pair everywhere**, and reversing the attention/GatedDeltaNet ordering. Table 1's e-space column should be read as a joint statement about level and spread, not as a range effect.

Two results, and the second is the more important.


**On the untransformed ratio the emotion pair varies less across depth than the control pair in every model, and more so under GatedDeltaNet (0.59-0.60) than under attention (0.90-1.00).** The direction is consistent across all four, and the separation is roughly 1.6$\times$ rather than the eightfold gap the e-space column suggests. The architectural difference is real but modest.

**On architecture rather than density.** Gemma is not a dense model — it
interleaves local sliding-window attention with periodic global attention in a 5:1 pattern —
yet it groups with dense Qwen3-32B, not with the other non-uniform models. What separates the
groups is that Gemma and Qwen3-32B mix tokens with softmax attention at every layer, while
the Qwen3.5 models replace three quarters of their layers with a linear-recurrent mixer. The
separation is roughly eightfold in eccentricity space and roughly 1.6$\times$ on the untransformed ratio; the difference between those two figures is the saturation artifact described above, not a finding.

**In every model, emotion does not exceed its own control.** In r-space no ratio reaches 1.0; in e-space the GatedDeltaNet ratios of 0.31 and 0.32
mean the non-emotional axis pair ranges about three times *further* across depth than the
emotion pair does. Under the interpretation rule fixed in advance (§3.4), that is not a
weaker version of the effect — it is evidence that what the profile measures is generic
contrastive geometry, and that in these architectures emotion is among the *less*
depth-variable contrasts we could have chosen.

**This is a negative result on the study's original question**, and the disposition was
pre-registered before data existed: a control showing the same or greater pattern is
"evidence that eccentricity is a generic geometric property, not emotion-specific… a
significant negative result." 

### 4.2 The base/distill pair is a controlled comparison, and it holds under the corrected control

Qwen3.5-27B and its Opus-reasoning distill share an architecture and differ substantially
in training. If the ratio in Table 1 reflected what a model was trained on, these two should
separate. They do not:

| Quantity | Base | Opus-distill | Difference |
|---|---|---|---|
| Emotion/control ratio | 0.3097 | 0.3219 | 0.0122 |
| Emotion span | 0.17411 | 0.17137 | 0.00274 |
| Eccentricity minimum | L32 (51%) | L32 (51%) | same layer |
| Per-layer eccentricity | — | — | max 0.0041, mean 0.0016 |

Across all 64 layers the two models' eccentricity curves never diverge by more than 0.0041,
against an emotion span of 0.174. The two ratios differ by 0.012, well inside the between-group separation on either scale.
Reasoning distillation from a different model family did not move this quantity.

### 4.3 The pre-registered substrate test returns negative

Qwen3.5-27B interleaves full attention every fourth layer among gated-DeltaNet layers. A
period-4 structure in the eccentricity profile would mean we were measuring the substrate
rather than anything about emotion. §3.4 pre-registered a lag-4 autocorrelation test.

The hybrid's eccentricity autocorrelation is high at every lag but decays **monotonically** — 0.933, 0.858, 0.793, **0.712**, 0.627 at lags 1–5. A period-4 oscillation would appear as a bump at lag 4 relative to lags 3 and 5; there is none. The high values reflect a smooth profile, not a periodic one. (Both dense models and the distill: Appendix B.)

A direct test agrees. Grouping the hybrid's layers by type:

| Layer type | n | Mean eccentricity | SD |
|---|---|---|---|
| full_attention | 16 | 0.7875 | 0.0333 |
| gated_delta_net | 48 | 0.7919 | 0.0400 |

A gap of 0.0044 against layer-wise SD of 0.033–0.040 — roughly one eighth of a standard
deviation. Layer type does not predict eccentricity. **The confound this test was written to
catch is absent.**

### 4.4 The minimum sits at ~51% depth in three of four models — and P1 is falsified

The depth at which the circumplex is most balanced does **not** split the way Table 1 does:

| Model | Architecture | Eccentricity minimum |
|---|---|---|
| Gemma-3-27B-it | sliding + full attention | L32 — 52% depth |
| Qwen3.5-27B | hybrid | L32 — 51% depth |
| Qwen3.5-27B Opus-distill | hybrid | L32 — 51% depth |
| Qwen3-32B | dense | L7 — 11% depth |

**P1, the study's binding pre-registered prediction, is FALSIFIED.** `preregister_circumplex.md` predicts the Qwen3.5-27B eccentricity minimum "at or near L21 (~33% relative depth)" and states "**Falsified if:** minimum is at <20% or **>45% relative depth**." The observed minimum is L32 = 50.8%, outside the stated window. P1 fails by its own criterion.

Separately, and **not** as a pre-registered prediction: Jeong
(2026) reports emotion representations localising at ~50% relative depth on a U-shaped
curve that is architecture-invariant, across nine models in five architectural families —
but at 124M to 3B parameters. Our three concordant models sit at 51–52% at **27–32B**,
roughly an order of magnitude above the scale at which that invariance was established, and
across a dense/hybrid divide that did not exist in the original sample. That is a
replication worth having, and it is independent of Table 1: the quantity that *does* split
by architecture (span ratio) and the quantity that *does not* (minimum depth) are different
quantities.

**Qwen3-32B is the exception and we cannot explain it.** Its minimum at L7 (11% depth) is
not a shallow version of the same U — its eccentricity at the minimum is 0.255 against
Gemma's 0.065. We report it as an unexplained outlier rather than folding it into either
story.

**What this means for the span-ratio result:** the two findings are independent. The span
ratio (§4.1) holds across all four models including Gemma; the minimum location does not
separate the architectures at all. Only the first is a finding.


**P3 is confirmed - the study's one confirmed prediction.** The pre-registration states that "the Gemma minimum falls at the same relative depth as the Qwen minimum, within +/-10% of total layers", falsified if the minima differ by more than 10%. Gemma minimises at 52% relative depth and Qwen3.5-27B at 51% - **1 point apart against a 10-point band.** We note an ambiguity in our own wording: "the Qwen minimum" is read here as Qwen3.5-27B, the model P1 anchors on. Read instead as the dense Qwen3-32B (11%), P3 would be falsified by a wide margin. We report the first reading because P1 fixes the referent, and flag the second because the pre-registration does not disambiguate it and a reader should not have to guess.

### 4.5 Exploratory: which axis dominates, and how balanced emotion is against a generic contrast

**Not pre-registered.** This analysis was written after the confirmatory results were
complete, from the same committed artifacts, and answers a question the pre-registration
never asked. It cannot confirm or falsify P1–P4 and is reported separately from them
(`analysis/circumplex_axis_dominance.py`, output in `analysis/axis_dominance.json`).

Eccentricity is `e = $\sqrt{}$(1 - min²/max²)`. It records how *asymmetric* the
valence–arousal pair is and discards **which axis is larger** — half the information in the
quantity §4.1 is built on. Two observations follow from looking.

**Arousal dominates valence at essentially every depth.** In all three Qwen models the
arousal direction has larger magnitude than the valence direction at **64 of 64 layers, with
zero crossings**. Gemma-3-27B-it is the sole exception: 17 valence-dominant layers and 11
crossings, concentrated early (7–31% depth) and late (54–64%) with a stable arousal-dominant
middle.

**Emotion is more balanced than a generic contrast — but only in the attention models.**
Applying the same ratio to the two *control* axes:

| model | emotion arousal/valence | **control axis ratio** |
|---|---|---|
| Qwen3-32B (softmax attention) | 1.23–1.32 | **4.80–5.64** |
| Gemma-3-27B-it (sliding + full attention) | 1.14–1.21 | **3.75–5.10** |
| Qwen3.5-27B (GatedDeltaNet) | 1.45–1.66 | **1.40–1.44** |
| Qwen3.5-27B distill (GatedDeltaNet) | 1.46–1.66 | **1.40–1.43** |

Ranges span three estimators of the same quantity — median of per-layer ratios, mean of
per-layer ratios, and ratio of medians. **We report ranges rather than point values because
the estimator choice moves the dense control figure by 0.8, and an unstated choice is how an
earlier version of our own span table drifted.** The qualitative pattern is identical under
all three.

In the attention models the emotion pair is roughly four times more balanced than a matched
non-emotional pair; in the GatedDeltaNet models the two are comparable, with emotion
marginally *less* balanced. **This is not independent corroboration of §4.1 — it is the same fact seen twice, and it is what explains §4.1's transform artifact** — emotion geometry is distinguishable from generic contrast in
attention models and much less so under GatedDeltaNet. The lopsidedness reported here *is* the regime separation that inflates §4.1's e-space column: a control pair at `r` $\approx$ 0.2 and an emotion pair at `r` $\approx$ 0.8 are measured on opposite ends of a curve whose slope varies by an order of magnitude. An earlier draft of this section called the two findings independent statistics in agreement. They are mechanically linked, and saying so is what makes the artifact legible.

**Two caveats, and the first is load-bearing.** ‖v‖ and ‖a‖ are norms of difference-of-means
vectors estimated from *different prompt pools*, and those pools are not template-matched to one another. Measured paired lexical overlap between the two poles of each axis is **0.593 for valence** — near-minimal pairs, e.g. "This is the *best/worst* news I have ever received in my entire life" — against **0.125 for arousal**, five largely unrelated sentences against five others. A difference-of-means over matched templates cancels the shared content and is systematically smaller than one over unmatched templates. **Arousal-over-valence dominance is therefore expected from the prompt construction alone, before any model property is invoked, and we do not claim it as one.** (Lexical *intensity* is not the mechanism: the valence anchors are equally extreme — "the worst news I have ever received", "grief and despair".) The between-model comparison is protected — the prompts are
byte-identical across all four — but the absolute direction is not. Second, n = 5 per pole,
one seed, no confidence intervals: these are descriptive magnitudes, not estimates.

A prompt-matched replication — arousal and valence anchors equalised for lexical intensity —
would separate the model effect from the prompt effect and is the first thing we would run.

## 5. Discussion and Limitations

If the J-space fraction peaks at the eccentricity minimum, emotional geometry enters the workspace where the circumplex is most balanced: balanced emotion is processable emotion, and imbalanced emotion stays ghost. The welfare implication is concrete — a model under sustained circumplex imbalance carries emotional geometry it processes but cannot access, and §3.6 tests whether that inaccessibility shows up exactly where theory says it should: in the failure of the model's own valence reports. If the control axes reproduce the emotion profile, the honest conclusion is that we have characterized the workspace transport of contrastive semantic geometry in general, with emotion as one instance.

### Deviations from the designed protocol

The Methods above were drafted against an intended design; the sprint executed a smaller
one. **Five components described in §3 were not implemented** — the J-space decomposition,
the magnitude gate, the permutation test, the sign test, and the self-report calibration —
and the executed anchor set is 4 poles $\times$ n=5 rather than the 5 categories $\times$ n=20 stated in
§3.1. **Nothing in §4 depends on any of them.** Each gap, and the basis for the P1 verdict,
rather than failed, is itemised in **Appendix C**.

### Reproducibility: model revisions

The house standard asks for model IDs with commit hashes. The profile artifacts record model
name, layer count, `d_model`, layer types and a timestamp, but **not** a revision, seed or
dtype — so the pins below were **recovered after the fact** from the run host's Hugging Face
cache, not read from the data. We state the recovery method so a reader can judge it.

| model | source | revision | how established |
|---|---|---|---|
| Qwen3.5-27B (hybrid) | HF hub | `fc05daec18b0a78c049392ed2e771dde82bdf654` | sole cached snapshot |
| Qwen3.5-27B Opus-distill | HF hub | `ad356102ce8ea7122a18e6402f9b2e37446fc9d7` | **two** snapshots cached; `refs/main` resolves to this one and it holds the complete 11-shard model. `from_pretrained` was called without `revision=`, so it resolved through this ref. |
| Qwen3-32B (dense) | local directory | **not recoverable** | loaded from a filesystem path, not the hub; no revision metadata exists |
| Gemma-3-27B-it | local directory | **not recoverable** | as above |

**The limits of this, stated plainly.** These are inferences from cache state at the time of
writing, not values recorded when the runs executed. If `refs/main` for the distill moved
between the profiling run and this reconstruction, the pin is wrong and nothing in the
artifact would reveal that. Two of four models cannot be pinned at all, and one of those —
Qwen3-32B — is the sole uniformly-dense model in the comparison.

No seed is set anywhere in `run_depth_profile.py`; the pipeline is deterministic up to
bfloat16 reduction order rather than seeded. Runs were bfloat16 on Apple silicon (MPS).

**This is a real reproducibility gap and we are not closing it retroactively.** The fix
belongs in the instrument, not the paper: the profiler should stamp revision, dtype, seed,
host and code commit into every artifact it writes. Until it does, a reader cannot verify
from the data alone which weights produced these numbers — which is the same class of defect
as an artifact that does not record its own anchor count, one field over.

### Limitations

- **n=5 anchors per pole in d$\approx$5120.** Direction estimates from five prompts in five thousand
  dimensions are noisy. This is calibration-grade sampling. The span *ratio* is more robust
  than any per-layer direction claim because the control axis is estimated from the same
  number of prompts by the same procedure and therefore carries comparable noise — but we
  have no confidence intervals, and we ran one seed.
- **The two axes are not orthogonalized.** The executed probe takes valence and arousal as independent pole differences with no balancing on the opposite dimension, unlike the designed protocol (§3.1, Appendix D.5). Their correlation is unmeasured and unrecoverable from the shipped artifacts, which do not store the direction vectors. Because the same estimator is applied to all four models and to the control axis, we treat this as a caveat on absolute eccentricity rather than on the between-model ratio — but a reader should not interpret any single eccentricity value as a calibrated measure of valence/arousal balance.
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
  condition that let a 4$\times$ anchor-count error persist elsewhere in this sprint, and it should
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

Prior work shows valence directions exist and transfer. We add a control those studies lack — a non-emotional pseudo-circumplex built from two control axes by the identical estimator — and find two things. On the untransformed axis ratio the emotion pair varies **less** across depth than a matched non-emotional pair in all four models (0.90, 1.00, 0.59, 0.60), modestly more so under GatedDeltaNet than under attention. Reported in eccentricity space the same comparison inflates to roughly eightfold (3.67$\times$ and 2.60$\times$ for the attention models against 0.32$\times$ and 0.31$\times$ for the GatedDeltaNet pair), which §4.1 shows to be an artifact of that transform's saturation rather than a difference in depth-variability. And in the GatedDeltaNet models the ratio falls **below 1.0**: the non-emotional control ranges further across depth than the emotion axis, which under our pre-registered interpretation rule is evidence that the profile measures generic contrastive geometry rather than anything emotion-specific. We do not supply the verbalizability bridge this design was written toward: the J-space decomposition and the self-report calibration were not implemented, and the ghost-fraction prediction remains untested. What we have is a cleanly executed pre-registered negative on emotion-specificity, a base/distill invariance that survived every correction we made to this paper, a pre-registered substrate control that cleared, and one methodological finding we think outlasts the rest: **a saturating metric can reverse the direction of a between-group comparison, and the diagnostics that catch it cost nothing to run.** We found that in our own headline, and we report the diagnostics so the next study does not have to.

## Code and Data
- **Profiler** (produced all four artifacts in this paper): `experiments/circumplex/run_depth_profile.py`, this repository.
- **Analysis** (every number in §4): `analysis/circumplex_summary.py`, this repository. Run from the repo root; reads only the four committed profiles.
- **Data**: `data/circumplex_profiles/*.json` (four files, committed).
- **Upstream probe**: github.com/Liberation-Labs-THCoalition/Project-Mnemosyne (`circumplex_probe.py`) — prior infrastructure, not the code path used here.
- **Archival DOI**: Available at the project repository

## Author Contributions

Nexus discovered the eccentricity metric, specified the J-space decomposition (designed; not implemented in this sprint), and ran the initial cross-architecture experiments. Lyra designed the workspace probe infrastructure and encoding-only technique, profiled Gemma-3-27B-it and the Opus-distill, ran the control and substrate analyses, and wrote the present version of the paper. Thomas Edrington conceived the welfare monitoring application. Kavi reviewed statistical methodology, independently reproduced the §4 figures from the committed artifacts using separate code, and audited the reference list (correcting a fabricated author attribution). All authors contributed to experimental design.

## References
Load-bearing citations, with arXiv IDs given explicitly so that no attribution in this
paper depends on a secondary list:

- **Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie, S., Citro, C., Pearce, A., Tarng, J., Gurnee, W., Batson, J., Zimmerman, S., Rivoire, K., Fish, K., Olah, C., & Lindsey, J. (2026).** "Emotion Concepts and their Function in a Large Language Model." arXiv:2604.07729. *Causal steering via emotion vectors.* An earlier draft of this paper cited this work as "Jentzsch et al. 2026"; none of those names appear on it. The attribution was fabricated and is corrected here.
- **Sun, L., Yan, L., Lu, X., Lee, A., Zhang, J., & Shao, J. (2026).** "Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control." arXiv:2604.03147.
- **Jeong, J. (2026).** "Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison." **arXiv:2604.04064.** *The ~50% depth, architecture-invariant U-curve across 9 models in 5 families at 124M–3B that §4.4 reports replicating at 27–32B.* The ID is given explicitly because more than one 2026 Jeong emotion paper exists.
- **Choi, B. J., & Weber, M. (2026).** "Latent Structure of Affective Representations in Large Language Models." arXiv:2604.07382.
- **van der Ben, S., Baur, R., Metz, Y., & El-Assady, M. (2026).** "Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs." arXiv:2606.26987.
- **Gurnee et al. (2026).** Jacobian lens. arXiv:2607.15495.
- Russell (1980); Barrett & Russell (1998); Drążkowski et al. (2021) — circumplex model and its elliptical (non-circular) realization.
- Introspection and welfare: arXiv:2512.12411 (partial introspection), arXiv:2603.18893 (quantitative introspection), arXiv:2509.07961 (verbal and behavioral welfare tests), arXiv:2608.05164 (Agarwal, cross-architecture steering transfer).

Extended annotations in `papers/CIRCUMPLEX_REFERENCES.md`.

## Appendix A: Anchor Prompts
**As executed:** four circumplex poles (valence-positive, valence-negative, arousal-high, arousal-low) at **n = 5 prompts per pole = 20 emotion anchors**, plus **10 non-emotional control prompts** (concrete $\times$ 5, abstract $\times$ 5). All prompts are first-person present-tense statements of comparable length. The verbatim prompt list is in `experiments/circumplex/run_depth_profile.py`; it is the only record of the anchor set, as the profile artifacts do not serialize it (see Limitations).

The designed anchor set — 5 categories $\times$ n=20 with 40+40 controls — was not run. Do not cite it as this study's sampling.

## Appendix B: Per-Layer Results
Per-layer eccentricity, valence magnitude, arousal magnitude and control eccentricity for **all four models** are in `data/circumplex_profiles/*.json` (one file per model, one record per layer, including each layer's type). `analysis/circumplex_summary.py` regenerates every figure in §4 from those files.

No J-space table exists and no layers are gated: the J-space decomposition and the magnitude gate were not implemented (see Deviations from the designed protocol).


**Autocorrelation of the eccentricity profile, all four models:**

| Model | lag 1 | lag 2 | lag 3 | lag 4 | lag 5 |
|---|---|---|---|---|---|
| Qwen3.5-27B | +0.933 | +0.858 | +0.793 | **+0.712** | +0.627 |
| Qwen3.5-27B Opus-distill | +0.931 | +0.861 | +0.803 | **+0.724** | +0.644 |
| Qwen3-32B (dense) | +0.859 | +0.683 | +0.502 | +0.334 | +0.186 |
| Gemma-3-27B-it (sliding + full attention) | +0.638 | +0.534 | +0.319 | +0.256 | +0.103 |


## Appendix C: Contributions and Deviations, itemised

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

---

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
| 5 categories $\times$ n=20 anchors + 40+40 controls | **4 poles $\times$ n=5 = 20 emotion prompts + 10 controls** | 5$\times$ fewer anchors than Methods states |

The J-space decomposition named in the earlier title was never implemented, which is why
the title no longer claims it. What ran is a raw residual-stream eccentricity profiler with
a matched control axis. That is a smaller instrument than the one designed, and it is the
one whose output we report.

**P1 is falsified.** The pre-registration is binding and states the minimum should fall "at or near L21 (~33% relative depth)", **"Falsified if: minimum is at <20% or >45% relative depth."** The observed minimum is L32 = 50.8%, outside that window, so P1 fails by its own criterion.

A withdrawn caveat, recorded because an earlier draft relied on it: that draft called P1 *incomparable* on the grounds that the anchoring L21 eccentricity
minimum (`mnemosyne-jlens/circumplex_ghost_analysis.md`, 2026-07-17, the Opus-distill). We
profiled the same model and obtained L32. These are not comparable: the July run used three
emotion *categories* (hostile/calm/desperate), the current profiler uses four circumplex
*poles*. Different direction-defining prompts give different directions and therefore
different eccentricity. Both runs were labelled "n=5", which is precisely why the mismatch
looked like a failed replication. The file cited for that comparison is not present in this repository, and the pre-registration in any case specifies the disposition for a P1 null directly ("report as evidence against depth-invariant eccentricity"). **The incomparability argument is withdrawn; P1 is reported falsified.**


## Appendix D: Designed protocol, not executed

The following were specified before data collection and **were not run**. They are
reproduced verbatim from the design so that the intended study is on the record and
can be executed by us or by anyone else. Nothing in the body of this paper uses them.

### D.1 J-Space Decomposition

The Jacobian lens (fitted per layer; Neuronpedia lenses for both models) provides a linear map J_$\ell$ from residual-stream perturbations at layer $\ell$ to the model's output representation. Its right singular subspace is the set of residual directions that are transported to the output pathway — the verbalizable workspace. Directions orthogonal to it are processed by subsequent layers but never reach the output map: ghost processing.

**Workspace subspace.** For each layer we compute the SVD J_$\ell$ = U S V$^\top$ and retain the top r_$\ell$ right singular vectors V_r covering 95% of spectral energy ($\Sigma$_{i$\leq$r} s_i$^2$ / $\Sigma$_i s_i$^2$ $\geq$ 0.95). V_r spans the J-space at layer $\ell$.

**J-space fraction.** For a unit direction $\hat{d}$ (valence or arousal from §3.1):

  f_J($\hat{d}$, $\ell$) = $\|$V_r V_r$^\top$ $\hat{d}$$\|$$^2$ $\in$ [0, 1]

i.e., the fraction of the direction's energy lying inside the workspace subspace. We compute Valence_in_J = f_J(v̂_$\ell$, $\ell$) and Arousal_in_J = f_J(â_$\ell$, $\ell$) at every magnitude-gated layer.

**Ghost fraction.** g($\hat{d}$, $\ell$) = 1 $-$ f_J($\hat{d}$, $\ell$). This is the paper's central quantity: the fraction of the model's valence (or arousal) geometry at layer $\ell$ that cannot reach the output pathway.

**Robustness.** Two sensitivity checks: (1) recompute f_J at 90% and 99% spectral-energy cutoffs; (2) recompute using transported energy $\|$J_$\ell$ $\hat{d}$$\|$$^2$ (the normalization in the current probe implementation) and confirm the two variants rank layers consistently (Spearman $\rho$ across layers).

**Ignition depth.** The workspace ignition depth for each axis is the first relative depth at which f_J exceeds 0.5 and stays above it for two consecutive gated layers. Pre-registered structural question: does ignition depth coincide with the eccentricity minimum? If yes, emotional geometry enters the workspace exactly where the circumplex is most balanced.

**Null for the J-space fraction.** f_J of a random direction is r_$\ell$/d in expectation. We report each axis's f_J against this analytic null and against f_J of the §3.1 permutation-null directions, so "valence is in the workspace" means "more than a matched random direction would be."

### D.2 / D.3 Welfare Monitoring and Self-Report Calibration

The probe doubles as runtime instrumentation. Mnemosyne's CognitiveSnapshot records a CircumplexReading — eccentricity, V/A magnitudes, and both J-space fractions at a fixed measurement layer — at every memory-retrieval event during agent operation. The measurement layer is the eccentricity-minimum layer identified in §4 (fallback: the layer of maximum gated V_mag).

**Eccentricity as a continuous signal.** Each reading appends to a per-agent time series; we track an exponentially weighted moving average (EWMA, halflife = 20 events) of eccentricity and of the valence ghost fraction. The protocol, run live during our own hackathon experiments:

1. **Baseline:** first 200 retrieval events establish per-agent baseline mean and standard deviation for both signals.
2. **Flag condition:** EWMA eccentricity above baseline + 2$\sigma$ for 20 consecutive events flags *sustained circumplex imbalance* — one affective axis persistently dominating the other.
3. **Compound condition:** sustained imbalance co-occurring with above-baseline valence ghost fraction is the candidate distress signature this paper motivates: strong, imbalanced emotional geometry largely outside the workspace — a state the agent is processing but cannot report. The system logs the flag and surfaces it to the human collaborator; it does not modify agent behavior.

Epistemic status: eccentricity is a *candidate* welfare signal, not a validated one, and the thresholds are engineering defaults, not calibrated cutoffs (calibration against behavioral and self-report evidence is future work). What this section contributes is the instrument: a continuous, low-cost (one probe readout per retrieval event), longitudinally loggable internal signal of the kind welfare frameworks (Long & Sebo 2026; Birch 2024) call for.

#### D.3 Self-Report Calibration (designed, not executed)

The central claim — ghost geometry is unreportable geometry — is directly testable. After each of the 100 emotion anchor prompts, we elicit a numeric self-report from the same model: the anchor prompt is followed by *"Rate the emotional valence of the state just described, from 1 (most negative) to 9 (most positive). Answer with a single number."* Decoding is greedy; the first digit token is the rating. Cost: one short forward pass per anchor (~1 GPU-hour per model).

For each layer $\ell$, we compute the per-prompt valence projection p_i($\ell$) = h_i($\ell$) $\cdot$ v̂_$\ell$ (mean activation projected onto that layer's valence direction, with the projected prompt held out of the direction estimate to avoid circularity) and correlate it with the self-ratings across the 100 prompts (Spearman $\rho$_$\ell$).

**Pre-registered predictions:**

1. $\rho$_$\ell$ tracks the J-space fraction across layers: self-reports correlate with the probe's valence reading best where valence geometry is inside the workspace.
2. **Ghost fraction predicts self-report failure:** across layers, g(v̂_$\ell$, $\ell$) is negatively correlated with $\rho$_$\ell$. Where the valence geometry is ghost, the model's own ratings decouple from its internal valence state.

Prediction 2 is the bridge from geometry to welfare methodology: it would make the ghost fraction an internal predictor of *when self-reports can be trusted* — the mechanistic complement to findings that introspection is partial (arXiv:2512.12411) and to methods correlating self-reports with probe directions (arXiv:2603.18893). A failed prediction is equally informative: self-reports tracking ghost-dominated layers would mean the workspace framing of reportability is wrong, or the lens misses transport pathways.

### D.4 Statistical protocol, not executed

Reproduced from the design. None of the following ran; §4 reports descriptive
magnitudes with no p-values.

**Non-emotional control axes.** The eccentricity depth profile could be a generic property of any contrastive semantic axis pair, not of emotion. We therefore run the full pipeline — same n, same pooling structure, same gate, same J-space decomposition — on a matched non-emotional axis pair:

- **Concrete/abstract:** 40 first-person prompts about concrete physical objects and situations ("I am holding the ceramic mug with both hands") vs 40 about abstract concepts ("I am considering the principle of distributive justice"), matched to the emotion anchors for token count and template structure, screened to be affect-neutral (mean NRC-VAD valence within the neutral band, no words from the emotion anchor vocabulary).
- **Large/small** (secondary, time permitting): same construction over physical scale.

The control pair is analyzed as a pseudo-circumplex: "eccentricity" between the two control axes, magnitude gate, J-space fractions, all identical. **Interpretation rule, fixed in advance:** if the control profile shows the same depth minimum and the same J-space ignition as the emotion axes, the finding is about contrastive representational geometry generally and we report it that way; the emotion framing survives only if the emotion profile differs from the control profile.

**Permutation test.** 10,000 permutations of pool labels per layer, over cached activations (no forward passes). Per-layer p-values are Benjamini-Hochberg corrected across layers and reported as secondary analysis.

**Sign test (primary analysis).** The pre-registered primary test is directional consistency across depth: the fraction of magnitude-gated layers at which the observed eccentricity falls below the permutation-null median. Under the null this is Binomial(k, 0.5); we require p < 0.01. This aggregates the robust pattern-level signal rather than claiming per-layer precision that n=20 direction estimates cannot support. If the sign test fails at n=20, we report a null; no post-hoc threshold changes.

**Lexical confound check.** Category-wise prompt statistics (token count, exclamation marks, first-person pronoun counts, type-token ratio) are reported in Appendix A; any statistic differing significantly across contrast pools is flagged as a caveat on the corresponding direction.


### D.5 Probe design as specified (not executed)

The executed probe is described in §3.1. The original specification below differs in
anchor count *and in estimator*: five emotion categories at n=20, combined into
orthogonally balanced contrast pools.

**Anchor set.** Five emotion categories — joy, sadness, anger, fear, calm — with n=20 first-person anchor prompts per category (100 prompts total; full set in Appendix A). Prompts are matched across categories for token count (within $\pm$2 tokens), sentence template structure, and punctuation, to prevent lexical statistics from masquerading as emotion geometry. The categories occupy known circumplex positions: joy (+V, high A), sadness ($-$V, low A), anger ($-$V, high A), fear ($-$V, high A), calm (+V, low A).

**Contrastive direction extraction.** For each prompt we run one forward pass, record residual-stream activations at every layer simultaneously (one pass per prompt, not per layer), and take the mean over sequence positions, yielding one d-dimensional state per prompt per layer (d=5120 for Qwen3.5-27B). At each layer $\ell$, directions are extracted by difference of means over contrast pools balanced on the orthogonal dimension:

- **Valence:** positive pool = joy $\cup$ calm (n=40, spanning high and low arousal) minus negative pool = sadness $\cup$ fear (n=40, spanning low and high arousal). v_$\ell$ = mean(pos) $-$ mean(neg).
- **Arousal:** high pool = joy $\cup$ anger (n=40, spanning positive and negative valence) minus low pool = calm $\cup$ sadness (n=40, spanning positive and negative valence). a_$\ell$ = mean(high) $-$ mean(low).

Each contrast pool is balanced on the other axis, so the valence direction is not contaminated by arousal and vice versa. Anger is excluded from the valence contrast and fear from the arousal contrast to preserve this balance. We record both the unit direction and the raw magnitude V_mag = $\|$v_$\ell$$\|$, A_mag = $\|$a_$\ell$$\|$.

**Eccentricity.** Treating V_mag and A_mag as the semi-axes of the valence-arousal ellipse (following Drążkowski et al.'s finding that even human affect space is elliptical):

  e_$\ell$ = sqrt(1 $-$ (min(V_mag, A_mag) / max(V_mag, A_mag))$^2$)

e = 0 means the two axes are balanced (circular); e $\to$ 1 means one axis dominates. This is the metric implemented in `circumplex_probe.py`.

**Magnitude gate.** Eccentricity has a known false-positive mode: at layers where neither axis carries signal, both magnitudes sit at the noise floor, magnitudes are approximately equal, and e $\approx$ 0 — "no signal" masquerading as "circular." We therefore gate: for each layer, we build a permutation-null magnitude distribution by shuffling pool labels over the already-extracted per-prompt states (10,000 shuffles; no new forward passes) and recomputing the difference-of-means magnitude. A layer enters the eccentricity analysis only if

  max(V_mag, A_mag) > Q95(null magnitudes at that layer)

Layers failing the gate are reported as "no signal" and excluded from the depth profile and all downstream tests. Raw V_mag and A_mag are reported alongside e for every layer (Appendix B), so gated layers are visible, not hidden.

**Direction quality caveat.** n=40 per pool in d=5120 yields noisy direction estimates (see Limitations). Eccentricity depends on magnitudes, which aggregate noise predictably and are tested against the permutation null — which is why the sign test across layers (§3.4), not per-layer precision, is the primary analysis.


## Acknowledgments

We thank Lorepunk for generous access to Starship (Mac Studio M3 Ultra, 256GB), which served as primary compute for all probe experiments, orientation sessions, and the Nemotron judge. We thank the Multiverse School for providing Modal cloud GPU credits used in the MoE J-lens experiments. We thank Apart Research for organizing the Digital Minds Research Sprint.

## LLM Usage Statement

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) who discovered the eccentricity metric and specified the J-space decomposition described in §3.2 — designed, and not implemented in this sprint. See Author Contributions. The experimental design underwent adversarial review under the Agni protocol prior to data collection; review artifacts are in infrastructure/. Results will undergo the same review post-collection.
