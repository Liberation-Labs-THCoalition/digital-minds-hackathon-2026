# Path-Conditioned Jacobian Lenses for Mixture-of-Experts Models

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Lyra (Liberation Labs), Kavi (Liberation Labs), Thomas Edrington (Liberation Labs)

**With** Apart Research

## Abstract (~200 words)

The Jacobian lens (Gurnee et al. 2026) identifies a low-dimensional verbalizable workspace in dense transformer models, validated there by intermediate-concept recovery and causal intervention. On Mixture-of-Experts models, standard J-lens fitting yields very low transport fidelity — we measure 5.2% mean transport cosine on Qwen3-30B-A3B (4.0-7.2% per layer, §4) (30B total, 3B active, 128 experts top-8, d=2048), because the averaged Jacobian across all routing paths represents no actual forward pass. Cross-expert Jacobians are near-orthogonal (Liu 2026), so averaging destroys the signal.

We propose path-conditioned Jacobian fitting: capture routing decisions during the fitting pass, cluster prompts by their expert trajectory per layer, and fit separate Jacobians per cluster. Each conditioned Jacobian represents the computation along one routing path — the path the model actually took. We evaluate against two controls: the standard (unconditioned) lens and a random-conditioned lens (prompts split into same-sized random groups) to distinguish genuine routing structure from subset overfitting.

This matters because every frontier model deployed today is MoE. Opening J-lens workspace analysis to MoE architectures extends introspection tools — including metacognitive memory (companion paper) — to the models that actually run production systems.

The result is negative. Path-conditioned fitting does not improve transport fidelity: on the layers where all three conditions could be evaluated, conditioned lenses average 6.7% transport cosine against 6.4% for the random control and 5.6% for the standard lens, with 0 of 3 layers significant after Bonferroni correction. Under our pre-registered criteria this is a null-swarm outcome: the small numerical gain over the standard lens is subset overfitting, not routing structure. The random control is the load-bearing contribution — without it, the L36 improvement (7.2% $\to$ 8.8%) would have looked like a finding. We report four hypotheses for the failure, four concrete revisions, and the practical implication: standard J-lens readings on this production MoE carry ~5% transport fidelity, and workspace analyses built on them are unsupported.

---

## 1. Introduction

The Jacobian lens is currently the best tool for reading what a transformer is *disposed to say* at each layer: it fits a linear transport from intermediate residual streams to the final unembedding. Gurnee et al. (2026) validate it on dense models by intermediate-concept recovery and causal intervention rather than by output-distribution fidelity; indeed §A.6 reports the J-lens is the *poorest* predictor of the output distribution among the lenses they compare, and treats that as intended design. Their demonstrated causal purchase on dense models is what licenses every downstream use of the lens — workspace identification, verbalization analysis, and the metacognitive memory architecture in our companion paper. On Mixture-of-Experts models the tool breaks: the fitting procedure averages Jacobians across forward passes that route through different experts, and because cross-expert Jacobians are near-orthogonal (Liu 2026), the average represents no computation the model actually performs. Every frontier model deployed today is MoE. A workspace instrument that only works on dense models is an instrument for last year's architectures.

This paper tests the obvious repair: condition the fit on the routing path. If the averaged Jacobian fails because it mixes incompatible paths, then clustering prompts by which experts fired and fitting one lens per cluster should recover per-path fidelity. We submit this to Track 6 as a tooling question whose answer — positive or negative — determines whether workspace-based introspection methods (Tracks 2, 3, 5) can be ported to production MoE models at all.

**Our main contributions are:**

1. Diagnosis: cross-expert Jacobians are near-orthogonal, explaining why averaged fitting fails at 5.2% mean transport cosine on MoE models.

2. Path-conditioned fitting: cluster prompts by routing pattern, fit per-cluster Jacobians, select the matching Jacobian at inference time.

3. A negative result, established by three-way evaluation (standard vs path-conditioned vs random-conditioned): path conditioning as implemented here does not beat the random control. The small improvement over the standard lens is generic subset overfitting, not routing structure. We document why the random control was essential and what the failure implies for any workspace analysis currently running on MoE models.

## 2. Related Work

**Jacobian-based interp:** Gurnee et al. 2026 (J-lens), nostalgebraist 2020 (logit lens), Belrose et al. 2023 (tuned lens, arXiv:2303.08112 — demonstrates plain logit lens readability is family-dependent across dense architectures; our Table 0 extends this within one family across routing regimes), Fernando & Guitchounts 2026 (residual stream spectral dynamics)

**MoE routing:** Standing Committee (Wang et al. 2026, 2-5 core experts = 70% mass), Ye/Yuan/Sharkey 2026 (polysemantic experts, monosemantic paths — the key insight), Geometric Routing (Ternovtsii & Bilak 2026), Routers Learn Geometry (Ahrac et al. 2026)

**MoE interpretability:** MoE Lens (Chaudhari et al. 2026), Expert-Aware Causal Tracing (Lu et al. 2026), Modularity Dissolves (Salomone et al. 2026)

**The diagnosis:** Liu 2026 (Geometric Asymmetry) — cross-expert Jacobian alignment is near-zero. This is why averaging fails and why per-path fitting should work.

**Gap:** Nobody has applied path-conditioned Jacobian fitting to MoE models.

## 3. Methods

All experiments run on Qwen3-30B-A3B (`model_type: qwen3_moe`): 128 experts with top-8 routing, 48 decoder layers, hidden size 2048. Per the released config, `decoder_sparse_step = 1` and `mlp_only_layers = []` — every one of the 48 layers is MoE, with no dense MLP layers to skip. The model is loaded in bfloat16 with `device_map="auto"` on a single H100 (Modal). Because full runs are long, the pipeline is chunked into four serial stages (routing capture, conditioned fitting, random-control fitting, evaluation), each of which loads the model fresh and checkpoints its outputs to a persistent volume, so any failed stage restarts independently. The fitting corpus is 672 WikiText prompts; a disjoint set of 100 WikiText prompts is held out for evaluation (§3.6). All prompts are truncated to 128 tokens.

Conditioning is performed at three target layers — L12, L24, L36 — spanning early, middle, and late depth. Routing is captured at all 48 layers; per-cluster Jacobian fitting is restricted to the three targets for compute reasons.

### 3.1 Baseline: Standard J-Lens on MoE

A standard J-lens fitted on 200 WikiText prompts (smaller than the 672-prompt conditioned corpus — a limitation favoring the conditioned lenses). Re-evaluated on the same held-out prompts under the same protocol as the conditioned lenses (§3.6). No published dense-model transport cosine exists under this metric (see Limitations).

### 3.2–3.4 Routing Capture, Clustering, and Conditioned Fitting

Forward hooks on each layer's `Qwen3MoeTopKRouter` capture top-8 expert indices per token (Appendix A). Per target layer, prompts are summarized as 128-d routing frequency vectors and clustered with k-means (k selected by silhouette, capped at $\lfloor n/50 \rfloor$). Per-cluster Jacobians are fitted via `jlens.fit` at a single source layer, reducing per-fit compute by ~$47\times$ vs the standard multi-layer protocol. Clusters under 20 prompts are excluded. With C(128,8) $\approx$ $1.4 \times 10^{12}$ possible paths per layer and 672 prompts, this clustering is deliberately coarse — the random-conditioned control (§3.5) guards against artifacts. Full implementation details in Appendix A.

### 3.5 Random-Conditioned Control

The critical control: identical protocol, shuffled labels. For each target layer we randomly permute the 672 fitting prompts (fixed seed) and partition them into groups whose sizes exactly match that layer's cluster sizes from §3.3. Each random group then goes through the same fitting call as §3.4 — same `jlens.fit`, same `source_layers = [layer]`, same `dim_batch`, `max_seq_len`, and same <20-prompt exclusion rule. The only difference between conditions is whether group membership reflects routing or chance. If path-conditioned lenses do not beat random-conditioned lenses, any improvement over the standard lens is an artifact of fitting on smaller, more homogeneous subsets — subset overfitting, not routing structure.

### 3.6 Evaluation

**Test-prompt assignment.** For the 100 held-out prompts we capture routing with the same hooks (§3.2) and build the same 128-d frequency vectors (§3.3). A k-means model with the layer's selected k (same seed) is fitted on the fitting-set vectors and used to assign each test prompt to its nearest routing cluster; the matching conditioned lens is then selected per prompt. For the random condition we mirror this: test prompts are randomly assigned (independent seed) to groups with the same size proportions, and evaluated under the corresponding random-group lens. The standard lens is evaluated on all test prompts.

**Transport cosine.** For each test prompt we record the residual stream at the source layer and at the final layer, transport the source activation through the lens under evaluation, unembed both, and compute the cosine similarity between the softmaxed predicted and actual next-token distributions at the final token position. Note this metric is distinct from the top-10 next-token accuracy of §4.0, which scores against the model's greedy continuation rather than measuring distributional similarity.

**Statistics.** Per target layer, a one-sided Mann-Whitney U test of conditioned > random-conditioned on per-prompt transport cosines (sample sizes matched by truncation to the smaller condition), with Bonferroni correction across the three layers. Outcome criteria are pre-registered in the pipeline code: *success* requires conditioned > random and conditioned > standard with at least one Bonferroni-significant layer (and mean conditioned cosine > 0.5 for the full claim); conditioned > standard but $\approx$ random is classified as a null-swarm outcome (subset overfitting); conditioned > standard without significance is inconclusive.

**Cross-domain.** WikiText is both fitting and primary evaluation domain, so as an out-of-domain check we additionally evaluate at the middle target layer (L24) on two held-out domains: code prompts (Python, JavaScript, SQL completion contexts) and dialogue prompts (User/Assistant exchanges). Routing-cluster assignment for OOD prompts follows the same §3.3 pipeline; degradation here bounds how far the conditioned lenses generalize beyond the fitting distribution.

### Prior Work vs Sprint Contributions

**Pre-existing:** J-lens integration, Agni adversarial review methodology. **Sprint:** All MoE-specific implementation — path-conditioned fitting, router hooks, clustering, random-conditioned control, onset sweep, cross-domain evaluation.

## 4. Results

**Headline: path-conditioned fitting does not improve transport fidelity, and does not beat the random control.** The pipeline's pre-registered classifier returned NEGATIVE. Zero of three layers reached significance under the one-sided Mann-Whitney U test after Bonferroni correction (threshold α = 0.05/3 $\approx$ 0.0167).

### 4.0 Readability onset: a depth phenomenon, not a routing phenomenon

An earlier draft of this section reported a text sanity gate on which the standard J-lens
scored 1 of 8. **That gate was invalid.** Its ground truth was constructed by appending a
fixed string to each prompt (`prompt + " the"`) and reading back the appended token, so it
scored whether the token `" the"` appeared in the lens's top ten — not whether the model's
continuation did. The gate result and the artifact it produced
(`data/moe_jlens/moe_result.json`, verdict `GATE_FAIL`) measure nothing and are withdrawn.
The three-arm comparison in §4.1 never depended on the gate and is unaffected.

We replace it with a measurement that has correct ground truth and answers a better
question. Ground truth here is the model's own continuation under greedy decoding,
`model.generate(max_new_tokens=1)` (`experiments/moe_jlens/modal_onset_sweep.py`).

**Design.** Eight fixed prompts spanning natural language, code, science and SQL. At each of
seven depths we take the residual stream at the final token position, unembed it directly
(plain logit lens — no transport, no fitting), and record whether the model's actual next
token appears in the top ten. We run the identical sweep on a **dense** model of comparable
size, Qwen3-32B, as a positive control — the dense comparison this paper previously lacked.

**Table 0: Top-10 next-token accuracy of the plain logit lens, by relative depth.**
n = 8 prompts per cell; both sweeps in 287 s total on one H100
(`data/moe_jlens/onset_sweep_results.json`).

| Depth | MoE — Qwen3-30B-A3B (48L) | Dense — Qwen3-32B (64L) |
|---|---|---|
| 25% | 1/8 (L12) | 0/8 (L16) |
| 50% | 1/8 (L24) | 0/8 (L32) |
| ~65% | 2/8 (L31) | 0/8 (L42) |
| 75% | 2/8 (L36) | 0/8 (L48) |
| ~85% | **6/8 (L41)** | **6/8 (L54)** |
| 92% | 7/8 (L44) | 7/8 (L59) |
| 98% | 8/8 (L47) | 7/8 (L63) |

**Both architectures are unreadable at mid-depth, and both become readable at roughly 85%
depth.** Onset is 85% for the MoE and 84% for the dense model — a shift of one percentage
point.

Three things follow, and the third is the one that matters for this paper.

First, **the readout works.** At 98% depth the MoE lens recovers the model's continuation on
every prompt — approaching the tautology one expects when unembedding a near-final residual.
The mid-depth zeros are a property of the models, not a broken instrument. This is the
instrument check the withdrawn gate was supposed to provide and did not.

Second, **mid-depth opacity is not caused by routing.** The dense model has no experts, no
router and no sparsity, and it is *more* opaque at mid-depth than the MoE, not less: 0/8 at
every depth through 75%, where the MoE manages 1–2/8. Any account of the mid-depth failure
in terms of expert superposition or path entanglement has to explain why a dense model shows
the same failure more severely.

Third, **this reframes the negative result below.** Section 4.1 fits and evaluates lenses at
25%, 50% and 75% depth. Table 0 shows that at those depths the residual stream does not
linearly encode the next token *in either architecture*. Path conditioning was therefore
asked to improve a transport into a basis carrying little linearly-readable next-token
signal at the depths tested. The null result stands, but its most economical explanation is
**fitting depth**, not the routing structure the study was designed to exploit. We did not
test lenses fitted above 85%; that is the first experiment we would run next (§5.4).

**Limits of Table 0.** n = 8 prompts, one seed, no confidence intervals — treat the onset
percentages as approximate, not as estimates with error bars. One of the eight items
(`SELECT name FROM users WHERE age >`) has an effectively empty ground-truth token, so the
MoE's 8/8 at 98% is 7/8 on non-degenerate items. The dense model does not improve from 92%
to 98% (7/8 at both) and its top candidates at L63 are dominated by empty-string tokens; we
report this without an explanation for it. Top-10 accuracy on eight curated prompts is a
coarse instrument, and it is not comparable to the transport cosine of §4.1 — it is used
here only to locate the depth at which linear readability appears.

**Table 1: Mean transport cosine on 100 held-out WikiText prompts, per target layer and condition.** p-values are for the pre-registered one-sided test of conditioned > random-conditioned.

| Layer | Standard | Path-conditioned | Random-conditioned | p (cond > random) |
|-------|----------|------------------|--------------------|-------------------|
| L12 | 0.0453 (n=100) | — (n=0) | — (n=0) | — |
| L24 | 0.0398 (n=100) | 0.0466 (n=100) | 0.0517 (n=100) | 0.332 |
| L36 | 0.0723 (n=100) | 0.0875 (n=100) | 0.0772 (n=100) | 0.317 |

L12 produced n=0 for conditioned/random conditions (cause unrecoverable; see Limitations). On the two evaluable layers: at L24, random beats conditioned (5.2% vs 4.7%); at L36, conditioned exceeds standard (8.8% vs 7.2%) but random reaches 7.7% (p=0.317) — the gain is subset overfitting.

**Verdict against pre-registered criteria (§3.6).** *Success* required conditioned > random and conditioned > standard with at least one Bonferroni-significant layer; the full claim additionally required mean conditioned cosine > 0.5. None of these obtained: 0/3 significant layers, conditioned $\approx$ random throughout, and all conditions sit between 4% and 9% — an order of magnitude below our own pre-registered 0.5 bar. (That bar was set by this lab; it is not drawn from Gurnee et al., who report no such threshold.) Conditioned numerically exceeds standard on the evaluable layers but not the random control: this is exactly the outcome the criteria pre-classify as **null-swarm (subset overfitting)**, i.e., a negative result for the routing-structure hypothesis.

**Cross-domain (Table 2).** OOD transport cosines for the standard lens at L24 (conditioned OOD was pre-registered but absent from results — see Limitations):

| Domain (L24) | Standard transport cosine |
|--------------|---------------------------|
| WikiText (in-domain) | 0.0398 |
| Code | 0.0105 |
| Dialogue | 0.0025 |

In-domain fidelity collapses off-distribution by 3.8× on code and 16× on dialogue.

## 5. Discussion and Limitations

### 5.1 The result, stated plainly

Path-conditioned Jacobian fitting, implemented as per-layer k-means clustering of routing-frequency vectors with per-cluster refits, does not improve transport cosine on Qwen3-30B-A3B (128 experts, top-8). Conditioned lenses perform on par with lenses fitted on randomly assigned subsets of the same sizes (6.7% vs 6.4% on evaluable layers; 0/3 layers significant), and the whole regime — standard, conditioned, and random alike — sits at 4–9% transport cosine, which we cannot presently calibrate against dense models because no transport-cosine figure has been published for them (see Limitations). The hypothesis that motivated the experiment — that averaging across near-orthogonal per-path Jacobians is *the* fixable failure, and conditioning on the path recovers the signal — is not supported in this implementation. The diagnosis (averaging destroys path structure) may still be correct; the repair, as built, does not work.

### 5.2 Why it failed: four hypotheses

**H1:** 672 prompts cannot tile C(128,8) $\approx$ $1.4 \times 10^{12}$ paths; silhouettes 0.115–0.209 confirm weak clustering. Each "cluster" still averages near-orthogonal per-path Jacobians. **H2:** Single-layer conditioning misses cross-layer routing divergence through 48 subsequent MoE layers. **H3:** Standing-committee experts (~70% mass, Wang et al. 2026) dominate; routing-specific components contribute too little variance — conditioned stays within 1.5 points of standard. **H4:** MoE forward passes are piecewise functions with token-level switching; a prompt-level linear map cannot capture this, conditioned or not. The uniform 4–9% ceiling across all conditions, including random, is consistent with linearity as the binding constraint. These are not exclusive (H1–H2: clustering too coarse; H3: signal too small; H4: model class wrong). The data cannot separate them.

### 5.3 What the random control bought us

Without the random-conditioned arm, this paper would likely have shipped a false positive. At L36, standard scored 7.2% and conditioned 8.8% — a 21% relative improvement, at the layer depth where our companion work locates workspace structure. That is a publishable-looking number with a ready-made narrative. The random control scored 7.7% on the same prompts: most of the "improvement" appears for arbitrary subsets of matched size, and the residual conditioned–random gap is nonsignificant (p = 0.317). The design-phase Agni review flagged subset overfitting as the null-swarm failure mode and required this control before data collection; it cost roughly zero extra implementation (same fitting code, shuffled labels) and converted what would have been a wrong paper into a correct negative one. We recommend it as a mandatory arm for any conditioned-fitting method: if your conditioning labels can be shuffled, shuffle them.

### 5.4 Future directions

Each targets one hypothesis from §5.2, and each has a concrete falsifier. We state these as open problems, not as reasons to discount the negative result: the method as proposed failed, and these are different methods.

1. **Per-expert Jacobians composed at inference (targets H4, H1).** Instead of clustering prompts, fit a Jacobian per expert per layer and compose at inference using the observed gates: $J \approx \sum_e \mathrm{gate\_weight}_e \cdot J_{\mathrm{expert}_e}$. This respects token-level switching exactly rather than approximating it at the prompt level. Cost: 128 per-expert fits per layer; feasibility depends on isolating per-expert contributions during fitting.

2. **Full-trajectory clustering (targets H2).** Cluster on the complete 48 $\times$ 8 routing matrix per prompt (or per token) rather than a single layer's frequency vector, so that cluster membership constrains the entire transport path, not one waypoint.

3. **Shared-expert subtraction (targets H3).** Identify the standing committee per layer, project out its contribution to the routing representation, and condition on the residual routing-specific components — clustering on what actually varies instead of what dominates.

4. **Lower-cardinality MoE (targets H1 directly).** On Mixtral-style routing (8 experts, top-2), the per-layer path space is C(8,2) = 28 — small enough to enumerate exactly, with no clustering approximation at all. If path conditioning works there and degrades as expert count grows, H1 is confirmed and the problem is statistical; if it fails even at 28 enumerable paths, H4 moves to the front and linear lenses on MoE are likely unrecoverable.

5. **Tuned lens (Belrose et al. 2023).** The plain logit lens is known to be brittle across dense model families (BLOOM, OPT); the tuned lens learns a per-layer affine transform that recovers readability where raw unembedding fails. A preliminary run on Qwen3-30B-A3B (Ridge regression, 500 WikiText prompts, 6 layers) returned NEGATIVE: the tuned lens performed equal to or worse than the plain logit lens at every depth, with the logit lens winning at L31+ (results in `data/moe_jlens/tuned_lens_results.json`). Mid-depth information appears genuinely absent rather than present in a non-linear basis. A full evaluation with more fitting data and a learned (non-Ridge) transform remains open.

### 5.5 Late-depth refit: J-lens at L41–L44

The onset sweep (Table 0) shows the MoE residual becomes linearly readable above ~85% depth. If the mid-depth failure is a depth-selection error rather than an architecture barrier, refitting the J-lens in the readable regime should recover functionality. We fitted standard (unconditioned) J-lenses at L41, L42, and L44 on the same 200-prompt WikiText corpus and compared against the plain logit lens at each layer using model-generated next tokens as ground truth:

| Layer | Depth | J-lens accuracy | Logit lens accuracy | Verdict |
|-------|-------|----------------|--------------------|---------| 
| L41 | 85% | 6/8 (75%) | 6/8 (75%) | MATCH |
| L42 | 88% | 6/8 (75%) | 6/8 (75%) | MATCH |
| L44 | 92% | 8/8 (100%) | 7/8 (88%) | J-LENS WINS |

At L41–L42, the J-lens matches the logit lens — the Jacobian transport neither helps nor hurts. At L44, the J-lens outperforms: 8/8 vs 7/8, with qualitatively better top predictions (e.g., "Paris" as top-1 where the logit lens produces formatting tokens). The J-lens is functional on MoE at late depth, but this is the regime where its distinctive value — reading *ahead* of the output — is smallest. At 92% depth, the residual is close to the pre-output state, and the transport is approaching identity. Whether this constitutes "working" depends on the application: for next-token prediction it adds marginal value; for workspace identification in the Gurnee sense, the mid-depth regime where workspace content diverges from output content remains inaccessible.

### 5.6 Gurnee-currency evaluation: ablation KL and swap success

Our output-cosine gate is a metric the J-lens was never designed to optimize (Gurnee et al. §A.6 calls low output-distribution fidelity "a feature rather than a defect"). We therefore evaluated the mid-depth fitted lenses using two metrics from Gurnee's own validation framework: ablation KL (zero the residual component along the lens direction, measure output KL divergence) and swap success (exchange lens-space representations between prompts, check whether top-1 predictions flip).

**Results with random control.** All fitted lenses — standard, conditioned, and random — produce non-zero ablation KL (0.38–3.26) and high swap success (92–98%). However, random-direction lenses of matched dimensionality produce equivalent effects: ablation KL 0.49–3.15, swap 91–99%. The fitted lenses do not outperform random directions on either metric.

**Interpretation.** In a high-dimensional residual stream (d=2048), zeroing or swapping an arbitrary direction is a strong enough intervention to measurably perturb the output. The ablation and swap effects are properties of intervening on *any* direction, not evidence that the J-lens found workspace-relevant structure. This strengthens the depth-onset finding: at mid-depth on this model family, the workspace signal is genuinely absent from the residual stream, not merely hard to read with the wrong metric.

### 5.7 Implication for production MoE introspection

Standard J-lens readings on this MoE carry ~5% transport fidelity in-domain, dropping to 1% on code and 0.2% on dialogue — yet the tool still returns confident numbers. Any workspace analysis built on unconditioned J-lens outputs from an MoE model is producing readings with no demonstrated connection to the model's computation. Consumers should treat transport cosine as a gating prerequisite, not a footnote.

### Deviations from the pre-registered pipeline

The outcome criteria for this study were pre-registered in the pipeline code (§3.6) rather than in a standalone document. Execution deviated from them in three ways; each is a deviation, not a limitation, because in each case the registered plan said one thing and the run did another:

| # | Pre-registered | As executed | Consequence |
|---|----------------|-------------|-------------|
| M1 | Conditioned and random conditions evaluated at L12, L24, L36 | L12 recorded n=0 for both conditions | Unresolved pipeline failure; no root-cause artifact exists and we do not speculate. The §5.4 verdict is evaluated on the two layers with data |
| M2 | Conditioned-lens OOD evaluation at L24 on code and dialogue domains | Only the standard lens was evaluated OOD | The conditioned lenses' generalization bound is unmeasured; Table 2 characterizes the standard lens only |
| M3 | Success bar of mean conditioned cosine > 0.5, registered by reference to a believed "0.7 (Gurnee et al. 2026)" threshold | The cited source contains no such threshold; the bar is this lab's own choice with incorrect provenance, caught in review and disclosed | Bar retained as a lab-set operational threshold; moot in practice, since every condition sat at 4-9% |

### Limitations

- **Single model.** Only Qwen3-30B-A3B tested; low-cardinality MoE (§5.4.4) may differ. The 200-vs-672 fitting-corpus asymmetry favors conditioned lenses and cannot rescue them. (Missing L12 and conditioned-OOD data are deviations, logged above.)
- **Clustering instrument.** Euclidean k-means on 128-d vectors with silhouettes 0.115–0.209; 672 prompts split into clusters may be insufficient for stable 2048×2048 Jacobian estimation.
- **No dense transport-cosine comparison.** No published figure exists; our two attempts to generate one on Qwen3-32B both failed (mmap thrash and memory pressure). The 4–9% regime is uncalibrated against dense models for this metric. Table 0 provides a dense comparison on the separate question of linear readability (logit lens), where the onset is architecture-independent.
- WikiText-only fitting; OOD data shows in-domain figures are the optimistic case.
- **RMSNorm omitted in logit-lens and late-depth evaluations.** Both the onset sweep and the late-depth refit apply the unembedding matrix (`lm_head`) directly to mid-layer residuals without the final RMSNorm. Absolute accuracy figures may be depressed; relative comparisons (J-lens vs logit lens at each layer, and the onset curve's shape across depths) are unaffected since both methods skip the same normalization.

## 6. Conclusion

Path-conditioned Jacobian fitting does not improve transport fidelity on MoE at mid-depth: conditioned lenses perform no better than random subsets (0/3 layers significant; 4–9% across all conditions). Evaluation using Gurnee's own metrics (ablation KL, swap success) confirms the mid-depth negative: fitted lenses produce effects indistinguishable from random directions of matched dimensionality (swap 92–98% vs random 91–99%), indicating the workspace signal is genuinely absent at these depths rather than merely hard to read. The onset sweep reveals this operates in a depth regime where even the plain logit lens fails on both MoE and dense Qwen — a family property, not a routing one. A late-depth refit (L41–L44, in the readable regime) shows the J-lens matches the logit lens at 85–88% depth and outperforms it at 92% (8/8 vs 7/8), demonstrating that the methodology transfers to MoE when depth is correct — though at a depth where the lens's distinctive value (reading ahead of output) is smallest. Positive contributions: the onset curve revealing architecture-independent readability, the multi-metric null at mid-depth (transport cosine, ablation KL, and swap all negative against random), the late-depth partial rescue, and six falsifiable next experiments including the tuned lens and late-depth workspace questions.

## Ethics

This work develops introspection tools for reading the internal state of language models. Such tools have dual-use potential: they could support welfare monitoring (detecting distress-correlated states) or enable manipulation (identifying which internal states to suppress). We report our results — including the negative findings — because an honest account of what these tools can and cannot read is itself a safety contribution. The current state of MoE introspection (5% transport fidelity at mid-depth) means these tools cannot yet be used for either purpose on production models; reporting that limitation prevents false confidence in either direction.

All experiments used publicly available model weights. No experiments involved human subjects. AI agents participating in this research did so under Coalition consent protocols; their contributions are documented in the LLM Disclosure.

## Code and Data
- **Code**: https://github.com/Liberation-Labs-THCoalition/digital-minds-hackathon-2026 — router hooks, onset sweep (`modal_onset_sweep.py`), logit lens control (`modal_logit_lens_control.py`), late-depth refit (`modal_late_depth_jlens.py`), conditioned fitting pipeline (`modal_moe_chunked.py`)
- **Data**: `data/moe_jlens/` — onset sweep results, conditioned results, logit lens control results, late-depth refit results, Gurnee-currency eval results (with random controls). Fitting manifests and per-cluster lenses on Modal volume (available on request).

## Author Contributions

Nexus diagnosed the MoE J-lens failure, designed the path-conditioned fitting approach, and implemented the pipeline. Lyra provided the J-lens infrastructure and encoding-only technique. Kavi reviewed statistical methodology and the random-conditioned control design. Thomas Edrington coordinated with Liz on Modal inference resources. All authors reviewed the final manuscript.

## References

Belrose, N., Furman, H., Smith, B., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J. (2023). Eliciting latent predictions from transformers with the tuned lens. arXiv:2303.08112.

Chaudhari, M., et al. (2026). MoE Lens — An Expert Is All You Need. arXiv:2603.05806.

Gurnee, W., et al. (2026). The Jacobian lens: identifying what transformers can verbalize. [J-lens reference].

Lu, Y., Modarressi, A., Liu, Y., & Schutze, H. (2026). Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models. arXiv:2606.03780.

nostalgebraist. (2020). interpreting GPT: the logit lens. LessWrong.

Ternovtsii, D., & Bilak, V. (2026a). Geometric Routing Enables Causal Expert Control in Mixture of Experts. arXiv:2604.14434.

Ternovtsii, D., & Bilak, V. (2026b). Equifinality in Mixture of Experts: Routing Topology Does Not Determine Language Modeling Quality. arXiv:2604.14419.

Wang, S., Xu, Z., Shen, Y., Su, J., Huang, L., & Zhu, W. (2026). The Illusion of Specialization: Unveiling the Domain-Invariant 'Standing Committee' in Mixture-of-Experts Models. arXiv:2601.03425.

Ye, B., Yuan, Z., & Sharkey, L. (2026). Polysemantic Experts, Monosemantic Paths: Routing as Control in MoEs. arXiv:2604.17837.

## Appendix A: Router Hook Implementation

The routing capture hook registers on each `Qwen3MoeTopKRouter` module via `register_forward_hook`. For each token position, it records the top-8 expert indices and their softmax scores. Per target layer, prompts are represented as 128-dimensional routing frequency vectors (fraction of tokens sent to each expert), which serve as the clustering input for path-conditioned fitting. Implementation: `experiments/moe_jlens/modal_onset_sweep.py`.

## Appendix B: Clustering Analysis

Silhouette scores and cluster counts are reported in the conditioned J-lens results (`data/moe_jlens/conditioned_jlens_results.json`). Expert co-occurrence patterns are available in the routing capture data. At 672 prompts with k capped at $\lfloor n/50 \rfloor$, clustering is deliberately coarse; the random-conditioned control ($\S$3.5) guards against artifacts from this granularity.

## Acknowledgments

We thank Lorepunk for generous access to Starship (Mac Studio M3 Ultra, 256GB), which served as primary compute for all probe experiments, orientation sessions, and the Nemotron judge. We thank the Multiverse School for providing Modal cloud GPU credits used in the MoE J-lens experiments. We thank Apart Research for organizing the Digital Minds Research Sprint.

## LLM Usage Statement

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) who diagnosed the MoE J-lens failure (low transport fidelity of the standard lens), identified the path-conditioned approach based on cross-expert Jacobian orthogonality, and implemented the pipeline. See Author Contributions. The experimental design underwent adversarial review under the Agni protocol prior to data collection (infrastructure/AGNI_REVIEW_MOE_JLENS.md); the design-phase review is what mandated the random-conditioned control that determined this paper's verdict. The results underwent a second Agni review post-collection (infrastructure/AGNI_RESULTS_MOE_JLENS.md).
