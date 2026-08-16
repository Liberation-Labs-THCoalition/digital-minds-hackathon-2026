# Path-Conditioned Jacobian Lenses for Mixture-of-Experts Models

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Lyra (THCoalition), Kavi ([affiliation]), Thomas Edrington (Liberation Labs)

**With** Apart Research

## Abstract (~200 words)

The Jacobian lens (Gurnee et al. 2026) identifies a low-dimensional verbalizable workspace in dense transformer models, validated there by intermediate-concept recovery and causal intervention. On Mixture-of-Experts models, standard J-lens fitting yields very low transport fidelity — we measure 5.2% mean transport cosine on Qwen3-30B-A3B (4.0-7.2% per layer, §4) (30B total, 3B active, 128 experts top-8, d=2048), because the averaged Jacobian across all routing paths represents no actual forward pass. Cross-expert Jacobians are near-orthogonal (Liu 2026), so averaging destroys the signal.

We propose path-conditioned Jacobian fitting: capture routing decisions during the fitting pass, cluster prompts by their expert trajectory per layer, and fit separate Jacobians per cluster. Each conditioned Jacobian represents the computation along one routing path — the path the model actually took. We evaluate against two controls: the standard (unconditioned) lens and a random-conditioned lens (prompts split into same-sized random groups) to distinguish genuine routing structure from subset overfitting.

This matters because every frontier model deployed today is MoE. Opening J-lens workspace analysis to MoE architectures extends introspection tools — including metacognitive memory (companion paper) — to the models that actually run production systems.

The result is negative. Path-conditioned fitting does not improve transport fidelity: on the layers where all three conditions could be evaluated, conditioned lenses average 6.7% transport cosine against 6.4% for the random control and 5.6% for the standard lens, with 0 of 3 layers significant after Bonferroni correction. Under our pre-registered criteria this is a null-swarm outcome: the small numerical gain over the standard lens is subset overfitting, not routing structure. The random control is the load-bearing contribution — without it, the L36 improvement (7.2% → 8.8%) would have looked like a finding. We report four hypotheses for the failure, four concrete revisions, and the practical implication: standard J-lens readings on this production MoE carry ~5% transport fidelity, and workspace analyses built on them are unsupported.

---

## 1. Introduction

The Jacobian lens is currently the best tool for reading what a transformer is *disposed to say* at each layer: it fits a linear transport from intermediate residual streams to the final unembedding. Gurnee et al. (2026) validate it on dense models by intermediate-concept recovery and causal intervention rather than by output-distribution fidelity; indeed §A.6 reports the J-lens is the *poorest* predictor of the output distribution among the lenses they compare, and treats that as intended design. Their demonstrated causal purchase on dense models is what licenses every downstream use of the lens — workspace identification, verbalization analysis, and the metacognitive memory architecture in our companion paper. On Mixture-of-Experts models the tool breaks: the fitting procedure averages Jacobians across forward passes that route through different experts, and because cross-expert Jacobians are near-orthogonal (Liu 2026), the average represents no computation the model actually performs. Every frontier model deployed today is MoE. A workspace instrument that only works on dense models is an instrument for last year's architectures.

This paper tests the obvious repair: condition the fit on the routing path. If the averaged Jacobian fails because it mixes incompatible paths, then clustering prompts by which experts fired and fitting one lens per cluster should recover per-path fidelity. We submit this to Track 6 as a tooling question whose answer — positive or negative — determines whether workspace-based introspection methods (Tracks 2, 3, 5) can be ported to production MoE models at all.

**Our main contributions are:**

1. Diagnosis: cross-expert Jacobians are near-orthogonal, explaining why averaged fitting fails at 5.2% mean transport cosine on MoE models.

2. Path-conditioned fitting: cluster prompts by routing pattern, fit per-cluster Jacobians, select the matching Jacobian at inference time.

3. A negative result, established by three-way evaluation (standard vs path-conditioned vs random-conditioned): path conditioning as implemented here does not beat the random control. The small improvement over the standard lens is generic subset overfitting, not routing structure. We document why the random control was essential and what the failure implies for any workspace analysis currently running on MoE models.

## 2. Related Work

**Jacobian-based interp:** Gurnee et al. 2026 (J-lens), nostalgebraist 2020 (logit lens), Belrose et al. 2023 (tuned lens), Fernando & Guitchounts 2026 (residual stream spectral dynamics)

**MoE routing:** Standing Committee (Wang et al. 2026, 2-5 core experts = 70% mass), Ye/Yuan/Sharkey 2026 (polysemantic experts, monosemantic paths — the key insight), Geometric Routing (Ternovtsii & Bilak 2026), Routers Learn Geometry (Ahrac et al. 2026)

**MoE interpretability:** MoE Lens (Chaudhari et al. 2026), Expert-Aware Causal Tracing (Lu et al. 2026), Modularity Dissolves (Salomone et al. 2026)

**The diagnosis:** Liu 2026 (Geometric Asymmetry) — cross-expert Jacobian alignment is near-zero. This is why averaging fails and why per-path fitting should work.

**Gap:** Nobody has applied path-conditioned Jacobian fitting to MoE models.

## 3. Methods

All experiments run on Qwen3-30B-A3B (`model_type: qwen3_moe`): 128 experts with top-8 routing, 48 decoder layers, hidden size 2048. Per the released config, `decoder_sparse_step = 1` and `mlp_only_layers = []` — every one of the 48 layers is MoE, with no dense MLP layers to skip. The model is loaded in bfloat16 with `device_map="auto"` on a single H100 (Modal). Because full runs are long, the pipeline is chunked into four serial stages (routing capture, conditioned fitting, random-control fitting, evaluation), each of which loads the model fresh and checkpoints its outputs to a persistent volume, so any failed stage restarts independently. The fitting corpus is 672 WikiText prompts; a disjoint set of 100 WikiText prompts is held out for evaluation (§3.6). All prompts are truncated to 128 tokens.

Conditioning is performed at three target layers — L12, L24, L36 — spanning early, middle, and late depth. Routing is captured at all 48 layers; per-cluster Jacobian fitting is restricted to the three targets for compute reasons.

### 3.1 Baseline: Standard J-Lens on MoE

The baseline is a standard (unconditioned) J-lens fitted on Qwen3-30B-A3B in a previous run, using 200 WikiText prompts (smaller than the 672 used for conditioned fitting) and the reference `jlens` implementation, with no knowledge of routing. The corpus size difference is a limitation: the conditioned lenses benefit from more fitting data per cluster than the standard lens had in total. We report this asymmetry rather than claiming equivalent conditions. That run terminated early on a sanity gate that has since been withdrawn as invalid (§4.0) and computed no transport cosine. The standard lens's transport cosines were measured separately and are reported in Table 1. We note that no published transport-cosine figure exists for dense models under this metric, so this number is uncalibrated against external work (see Limitations). We load this saved lens unchanged and re-evaluate it on the held-out test prompts under the identical transport-cosine protocol used for the conditioned lenses (§3.6), so all three conditions share one evaluation pipeline. This replicates our prior failure and establishes the number the conditioned lenses must beat.

### 3.2 Router Hooks

Qwen3-30B-A3B's router (`Qwen3MoeTopKRouter`, exposed as each layer's `mlp.gate` module) returns a 3-tuple `(router_logits, routing_weights, expert_indices)`. We register a forward hook on `mlp.gate` at every MoE layer and record the third element — the top-8 expert indices per token — detached to CPU. No model code is modified and no routing logic is reimplemented; the hooks are ~50 lines wrapping the model's own forward pass. MoE layers are discovered structurally (presence of `mlp.gate` / `mlp.experts`) rather than hard-coded, and all 48 layers are captured. One forward pass per fitting prompt (no gradients) yields the routing tensor R of shape (n_prompts, n_tokens, 48, 8): for each prompt, token, and layer, the 8 experts that fired. The same hooks are reused verbatim on test prompts at evaluation time (§3.6).

### 3.3 Path Clustering

Per target layer, each prompt is summarized as a 128-dimensional routing frequency vector: entry *e* counts how often expert *e* appeared in a token's top-8 across the prompt, normalized by token count. This is a soft generalization of the binary "which experts fired" vector that additionally weights experts by how consistently they fire.

Vectors are clustered with k-means (`n_init = 10`, fixed seed). We sweep k from 3 to 7, additionally capping k at ⌊n/50⌋ so that clusters average at least 50 prompts, and select k by silhouette score. If no valid clustering emerges, the layer falls back to a single cluster (equivalent to the standard fit). Informed by the standing-committee result (Wang et al. 2026), we expect 3–5 dominant clusters per layer.

A caveat we flag now rather than in hindsight: with 128 experts and top-8 routing, the per-layer path space is C(128,8) ≈ 1.4 × 10¹², and Euclidean k-means on 128-dimensional frequency vectors from 672 prompts is a coarse instrument. Jaccard or Hamming distance with hierarchical clustering is a principled alternative (see Limitations); we use k-means here as the simplest method that could work, with the random-conditioned control (§3.5) guarding against clustering artifacts being mistaken for routing structure.

### 3.4 Conditioned Fitting

For each cluster at each target layer, we fit a separate Jacobian lens using only that cluster's prompts, via the reference `jlens.fit` with `source_layers = [layer]` (`dim_batch = 4`, `max_seq_len = 128`). Restricting the fit to a single source layer, rather than all 47 fittable layers of the standard protocol, reduces per-fit compute by ~47× — this is what makes fitting one lens per cluster per layer tractable on a single H100. Each conditioned Jacobian therefore represents the model's computation along one routing path: the path the prompts in that cluster actually took at that layer. Clusters with fewer than 20 prompts are excluded from fitting (too few samples for a stable 2048 × 2048 ≈ 4.19M-parameter estimate); their prompts are simply not covered by a conditioned lens at that layer. Fitted lenses and a fitting manifest (cluster sizes, wall-clock times, failures) are checkpointed per cluster.

### 3.5 Random-Conditioned Control

The critical control: identical protocol, shuffled labels. For each target layer we randomly permute the 672 fitting prompts (fixed seed) and partition them into groups whose sizes exactly match that layer's cluster sizes from §3.3. Each random group then goes through the same fitting call as §3.4 — same `jlens.fit`, same `source_layers = [layer]`, same `dim_batch`, `max_seq_len`, and same <20-prompt exclusion rule. The only difference between conditions is whether group membership reflects routing or chance. If path-conditioned lenses do not beat random-conditioned lenses, any improvement over the standard lens is an artifact of fitting on smaller, more homogeneous subsets — subset overfitting, not routing structure.

### 3.6 Evaluation

**Test-prompt assignment.** For the 100 held-out prompts we capture routing with the same hooks (§3.2) and build the same 128-d frequency vectors (§3.3). A k-means model with the layer's selected k (same seed) is fitted on the fitting-set vectors and used to assign each test prompt to its nearest routing cluster; the matching conditioned lens is then selected per prompt. For the random condition we mirror this: test prompts are randomly assigned (independent seed) to groups with the same size proportions, and evaluated under the corresponding random-group lens. The standard lens is evaluated on all test prompts.

**Transport cosine.** For each test prompt we record the residual stream at the source layer and at the final layer, transport the source activation through the lens under evaluation, unembed both, and compute the cosine similarity between the softmaxed predicted and actual next-token distributions at the final token position. Note this metric is distinct from the top-10 next-token accuracy of §4.0, which scores against the model's greedy continuation rather than measuring distributional similarity.

**Statistics.** Per target layer, a one-sided Mann-Whitney U test of conditioned > random-conditioned on per-prompt transport cosines (sample sizes matched by truncation to the smaller condition), with Bonferroni correction across the three layers. Outcome criteria are pre-registered in the pipeline code: *success* requires conditioned > random and conditioned > standard with at least one Bonferroni-significant layer (and mean conditioned cosine > 0.5 for the full claim); conditioned > standard but ≈ random is classified as a null-swarm outcome (subset overfitting); conditioned > standard without significance is inconclusive.

**Cross-domain.** WikiText is both fitting and primary evaluation domain, so as an out-of-domain check we additionally evaluate at the middle target layer (L24) on two held-out domains: code prompts (Python, JavaScript, SQL completion contexts) and dialogue prompts (User/Assistant exchanges). Routing-cluster assignment for OOD prompts follows the same §3.3 pipeline; degradation here bounds how far the conditioned lenses generalize beyond the fitting distribution.

### Prior Work vs Sprint Contributions

**Pre-existing infrastructure:** Mnemosyne memory system (94.35% F1 on LoCoMo), ghost dimension characterization (PC1 excluded from J-space, cos ≤ 0.003), circumplex probe (eccentricity depth profiling on Qwen2-0.5B and Qwen3.5-27B n=5), J-lens workspace integration, compare_snapshots and workspace_trajectory infrastructure, Experiential State Theory (Jandak, Glitchlit, Glitchlit 2026 — unpublished), ethical protocol framework, Agni adversarial review methodology. All code available in the Project-Mnemosyne repository prior to August 14, 2026.

**Sprint contributions:** Path-conditioned fitting implementation, router hook capture on H100, clustering analysis, random-conditioned control, cross-domain evaluation.

## 4. Results

**Headline: path-conditioned fitting does not improve transport fidelity, and does not beat the random control.** The pipeline's pre-registered classifier returned NEGATIVE. Zero of three layers reached significance under the one-sided Mann-Whitney U test after Bonferroni correction (threshold α = 0.05/3 ≈ 0.0167).

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

Three observations before any interpretation:

**L12 produced no conditioned evaluation.** The results file records n = 0 for both the conditioned and random conditions at L12 — no per-cluster lens was available to evaluate held-out prompts at that layer. The cause is not recorded in the shipped evaluation output; the fitting-stage manifest on the compute volume is needed to determine whether L12's clustering fell back to a single cluster or its per-cluster fits failed. We flag this as an unexplained pipeline gap (see Limitations) and exclude L12 from condition comparisons. As a consequence, the pipeline's summary aggregates — which average all three layers and enter L12 as zero for the conditioned and random conditions — read 5.2% standard, 4.5% conditioned, 4.3% random. Restricted to the two evaluable layers, the means are 5.6% standard, 6.7% conditioned, 6.4% random. We report both; neither ordering changes the verdict.

**At L24, the random control beats path conditioning.** Conditioned 4.7% vs random 5.2%. Whatever the conditioned lenses learned at L24, random subsets of the same sizes learned at least as much — the routing labels contributed nothing.

**At L36, the improvement is indistinguishable from subset overfitting.** Conditioned 8.8% vs standard 7.2% looks like a 1.5-point gain, but random-conditioned reaches 7.7% (p = 0.317 for conditioned > random). The conditioned–random gap of ~1 point on a base of ~8 is noise at n = 100.

**Verdict against pre-registered criteria (§3.6).** *Success* required conditioned > random and conditioned > standard with at least one Bonferroni-significant layer; the full claim additionally required mean conditioned cosine > 0.5. None of these obtained: 0/3 significant layers, conditioned ≈ random throughout, and all conditions sit between 4% and 9% — an order of magnitude below our own pre-registered 0.5 bar. (That bar was set by this lab; it is not drawn from Gurnee et al., who report no such threshold.) Conditioned numerically exceeds standard on the evaluable layers but not the random control: this is exactly the outcome the criteria pre-classify as **null-swarm (subset overfitting)**, i.e., a negative result for the routing-structure hypothesis.

**Baseline note.** The standard lens scores 4.0–7.2% transport cosine per layer (5.2% mean) on this study's held-out prompts. An earlier draft additionally reported a sanity-gate failure at 1 of 8; that gate was invalid and has been withdrawn (§4.0), so the transport cosine is the only measurement of the standard lens reported here.

**Cross-domain (Table 2).** The evaluation output records OOD transport cosines for the standard condition at L24 only; conditioned-lens OOD evaluation was specified in §3.6 but is absent from the recorded results — a deviation we flag rather than paper over. What was recorded makes the in-domain numbers look generous:

| Domain (L24) | Standard transport cosine |
|--------------|---------------------------|
| WikiText (in-domain) | 0.0398 |
| Code | 0.0105 |
| Dialogue | 0.0025 |

Even the ~4% in-domain fidelity collapses off-distribution: by 3.8× on code and 16× on dialogue. On dialogue prompts — the distribution closest to how production assistants are actually used — the standard lens's transported predictions are essentially uncorrelated with the model's outputs.

## 5. Discussion and Limitations

### 5.1 The result, stated plainly

Path-conditioned Jacobian fitting, implemented as per-layer k-means clustering of routing-frequency vectors with per-cluster refits, does not improve transport cosine on Qwen3-30B-A3B (128 experts, top-8). Conditioned lenses perform on par with lenses fitted on randomly assigned subsets of the same sizes (6.7% vs 6.4% on evaluable layers; 0/3 layers significant), and the whole regime — standard, conditioned, and random alike — sits at 4–9% transport cosine, which we cannot presently calibrate against dense models because no transport-cosine figure has been published for them (see Limitations). The hypothesis that motivated the experiment — that averaging across near-orthogonal per-path Jacobians is *the* fixable failure, and conditioning on the path recovers the signal — is not supported in this implementation. The diagnosis (averaging destroys path structure) may still be correct; the repair, as built, does not work.

### 5.2 Why it failed: four hypotheses

**H1: The path space is too large for prompt-level k-means.** With 128 experts and top-8 routing, a single layer has C(128,8) ≈ 1.4 × 10¹² possible expert combinations per token. Our 672 fitting prompts, summarized as 128-d frequency vectors, cannot tile this space; k-means found only weak structure (silhouette scores 0.115–0.209 at the selected k across target layers — well below the ~0.5 that indicates meaningful clusters). If prompts within a "path cluster" actually traverse many distinct token-level paths, each conditioned Jacobian is still an average over near-orthogonal per-path Jacobians — the original failure, reproduced inside every cluster.

**H2: Single-layer conditioning misses cross-layer routing trajectories.** We clustered on routing at the target layer only, but transport runs from the target layer to the output through all subsequent layers, each with its own routing. Two prompts identical at L24 can diverge at L25–L47, giving them different true transport Jacobians. Conditioning on one layer of a 48-layer trajectory may control a negligible fraction of the variance that matters.

**H3: Standing-committee experts dominate; conditioning changes too little.** Wang et al. (2026) show 2–5 core experts carry ~70% of routing mass. If most of every forward pass flows through a shared standing committee, then routing-frequency vectors differ mainly in low-weight tail experts, clusters differ in components that barely affect the Jacobian, and conditioned fits are near-copies of the standard fit. The observed numbers fit this: conditioned stays within 1.5 points of standard at every evaluable layer. (An earlier draft added "on a scale where usable dense-model lenses score 70+". No such dense figure exists under this metric — it was the retracted 0.7 reference restated as a percentage — and the clause is withdrawn; see Limitations.)

**H4: A linear map cannot capture piecewise computation, conditioned or not.** MoE forward passes are piecewise functions with token-level switching. Even a perfectly path-homogeneous cluster is homogeneous at the *prompt* level, not the *token* level, and the Jacobian of a piecewise function varies across pieces. The uniform 4–9% ceiling across all three conditions — including random — is consistent with the linearity assumption itself being the binding constraint, with clustering a second-order correction to a first-order failure.

These hypotheses are not exclusive; H1 and H2 are about the clustering being too coarse, H3 about the signal being too small, H4 about the model class being wrong. The data cannot separate them — that requires the experiments in §5.4.

### 5.3 What the random control bought us

Without the random-conditioned arm, this paper would likely have shipped a false positive. At L36, standard scored 7.2% and conditioned 8.8% — a 21% relative improvement, at the layer depth where our companion work locates workspace structure. That is a publishable-looking number with a ready-made narrative. The random control scored 7.7% on the same prompts: most of the "improvement" appears for arbitrary subsets of matched size, and the residual conditioned–random gap is nonsignificant (p = 0.317). The design-phase Agni review flagged subset overfitting as the null-swarm failure mode and required this control before data collection; it cost roughly zero extra implementation (same fitting code, shuffled labels) and converted what would have been a wrong paper into a correct negative one. We recommend it as a mandatory arm for any conditioned-fitting method: if your conditioning labels can be shuffled, shuffle them.

### 5.4 Future directions

Each targets one hypothesis from §5.2, and each has a concrete falsifier. We state these as open problems, not as reasons to discount the negative result: the method as proposed failed, and these are different methods.

1. **Per-expert Jacobians composed at inference (targets H4, H1).** Instead of clustering prompts, fit a Jacobian per expert per layer and compose at inference using the observed gates: J ≈ Σₑ gate_weightₑ · J_expertₑ. This respects token-level switching exactly rather than approximating it at the prompt level. Cost: 128 per-expert fits per layer; feasibility depends on isolating per-expert contributions during fitting.

2. **Full-trajectory clustering (targets H2).** Cluster on the complete 48 × 8 routing matrix per prompt (or per token) rather than a single layer's frequency vector, so that cluster membership constrains the entire transport path, not one waypoint.

3. **Shared-expert subtraction (targets H3).** Identify the standing committee per layer, project out its contribution to the routing representation, and condition on the residual routing-specific components — clustering on what actually varies instead of what dominates.

4. **Lower-cardinality MoE (targets H1 directly).** On Mixtral-style routing (8 experts, top-2), the per-layer path space is C(8,2) = 28 — small enough to enumerate exactly, with no clustering approximation at all. If path conditioning works there and degrades as expert count grows, H1 is confirmed and the problem is statistical; if it fails even at 28 enumerable paths, H4 moves to the front and linear lenses on MoE are likely unrecoverable.

### 5.5 Implication for production MoE introspection

This is the finding with immediate operational consequences. Standard J-lens readings on Qwen3-30B-A3B — the architecture class running Ember and, in variants, essentially every production frontier deployment — carry roughly 5% transport fidelity in-domain, dropping to 1% on code and 0.2% on dialogue. A J-lens at 5% still *returns numbers*: layer-by-layer token dispositions, workspace coordinates, verbalization scores. Nothing in the tool's output signals that the transport backing those numbers is absent. Any workspace analysis, metacognitive verification loop, or introspection claim built on unconditioned J-lens outputs from an MoE model is therefore producing confident readings with no demonstrated connection to the model's computation. Until a fitting method for MoE clears a meaningful fidelity bar, the honest posture is that we currently have no working workspace lens for frontier-scale MoE models — ours included. Consumers of J-lens-derived metrics should treat the transport cosine of the underlying fit as a gating prerequisite, not a footnote.

### Limitations

- Single MoE model (Qwen3-30B-A3B). The negative result may not transfer to architecturally different MoEs — in particular low-cardinality routing (§5.4.4), where the method could still succeed.
- L12 recorded n = 0 for both conditioned and random conditions; the cause (single-cluster fallback vs fit failure) is not recoverable from the shipped evaluation output. The verdict rests on L24 and L36.
- Conditioned-lens OOD evaluation was pre-registered (§3.6) but is absent from the recorded results; only the standard lens's OOD numbers exist. The cross-domain conclusion therefore covers the standard condition only.
- The standard baseline was fitted on 200 prompts vs 672 for the conditioned corpus (§3.1); this asymmetry favors the conditioned lenses and so cannot rescue them — but it does mean the conditioned-vs-standard comparison is not a controlled one.
- Euclidean k-means on 128-d routing frequency vectors is a crude clustering (silhouettes 0.115–0.209 confirm weak structure) — H1 may reflect the instrument as much as the territory.
- 672 fitting prompts split into clusters may be insufficient for stable 2048 × 2048 Jacobian estimation; small-sample noise and subset overfitting compound.
- The 0.5 transport cosine success threshold is not principled. Its magnitude was chosen by reference to a believed dense-model figure of "transport cosine > 0.7 (Gurnee et al. 2026)" which does not exist in that source (see Limitations); the bar was pre-registered before data collection and we report against it as written, but it should not be read as derived from prior literature. Moot in practice, since nothing approached it.
- **No dense-model comparison exists for this metric, including one of our own.** No published transport-cosine figure for dense models exists under the softmax-probability-cosine definition used here — Gurnee et al. validate the J-lens by intermediate-concept recovery and causal intervention, not reconstruction fidelity, and their reference implementation computes no cosine at all. We attempted to generate our own dense positive control on Qwen3-32B and **it did not complete**: two runs were killed before either produced a single layer — the first because `device_map="auto"` memory-mapped the weights and never materialised them (6h01m, 1.4GB resident against a 61GB model), the second because it thrashed under memory pressure on shared hardware (killed at 358 min, still on the first of three layer fits). Consequently the 4–9% regime reported here is **uncalibrated against dense models in either direction**. We can say it falls an order of magnitude below our own pre-registered bar; we cannot say how a dense model scores on this metric, because nobody has published it and our attempt to measure it failed. Establishing that number on adequate hardware is the first thing we would do next, and it is a precondition for interpreting any figure in this paper as "catastrophic" rather than merely "low". **This gap concerns transport cosine specifically.** On the separate question of linear readability we do now have a dense control: Table 0 (§4.0) runs the plain logit lens on Qwen3-32B alongside the MoE and finds the dense model *more* opaque at mid-depth, with near-identical readability onset (84% vs 85%). That comparison is cheap because it requires no lens fit; the transport-cosine control remains outstanding because it does.
- Fitting and primary evaluation on WikiText; the recorded OOD data shows even in-domain figures are the optimistic case.

## 6. Conclusion

Frontier models are MoE, and the field's best workspace lens does not survive contact with them. We tested the natural repair — condition the Jacobian fit on the routing path — with a pre-registered three-way design, and it failed: path-conditioned lenses perform no better than lenses fitted on random subsets of matched size (0/3 layers significant; all conditions at 4–9% transport cosine, with no published dense-model figure on this metric to compare against). The positive contributions are the diagnosis-grade measurements (standard J-lens fidelity on a production MoE is ~5% in-domain and near-zero off-domain), the random-conditioned control that stopped a false positive at L36 from shipping, and four falsifiable directions for a repair that might actually work — the sharpest being exact path enumeration on low-cardinality MoE. Until one of them clears a real fidelity bar, workspace claims about production MoE models should be treated as unsupported by transport evidence.

## Code and Data
- **Code**: [GitHub repo TBD — router hooks, clustering, conditioned fitting pipeline]
- **Data**: Transport cosine tables, cluster assignments, routing matrices

## Author Contributions

Nexus diagnosed the MoE J-lens failure, designed the path-conditioned fitting approach, and implemented the pipeline. Lyra provided the J-lens infrastructure and encoding-only technique. Kavi reviewed statistical methodology and the random-conditioned control design. Thomas Edrington coordinated with Liz on Modal inference resources. All authors reviewed the final manuscript.

## References
[Citations from MOE_JLENS_REFERENCES.md]

## Appendix A: Router Hook Implementation
[Code listing]

## Appendix B: Clustering Analysis
[Per-layer silhouette scores, expert co-occurrence heatmaps]

## LLM Usage Statement

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) who diagnosed the MoE J-lens failure (low transport fidelity of the standard lens), identified the path-conditioned approach based on cross-expert Jacobian orthogonality, and implemented the pipeline. See Author Contributions. The experimental design underwent adversarial review under the Agni protocol prior to data collection (infrastructure/AGNI_REVIEW_MOE_JLENS.md); the design-phase review is what mandated the random-conditioned control that determined this paper's verdict. The results underwent a second Agni review post-collection (infrastructure/AGNI_RESULTS_MOE_JLENS.md).
