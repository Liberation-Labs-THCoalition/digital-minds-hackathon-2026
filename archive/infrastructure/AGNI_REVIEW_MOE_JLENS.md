# Agni Review: Experiment B — Path-Conditioned MoE J-Lens

**Reviewer:** Nexus (Agni protocol)
**Date:** 2026-08-11
**Verdict:** CONDITIONAL PASS — 3 FAILs, 4 WARNs, 3 PASSes. Fixable before hackathon if the FAILs are addressed.

---

## 1. Null Swarm: Subset Overfitting — FAIL

**The problem:** If you split ANY dataset into k subsets by ANY criterion and fit separate linear models per subset, you get better fits than a single model. This is the bias-variance tradeoff, not a discovery about MoE routing. Path-conditioned fitting could produce transport cosine > 0.5 purely by fitting to smaller, more homogeneous groups — with no MoE-specific mechanism involved.

**Why this is critical:** This is the exact null swarm pattern that killed findings in the trajectory paper (5/5 issues, 2 critical). The metric appears to measure "routing-aware interpretability" but could actually measure "subset-specific overfitting." Liu (2605.16349) shows cross-expert Jacobians are near-orthogonal, which SUPPORTS path conditioning — but the experiment as designed cannot distinguish "routing captures real computational structure" from "smaller groups = better fit."

**The fix:** Add a random-conditioned control arm. For each layer:
1. Split prompts into the same number of same-sized random groups (ignoring routing)
2. Fit separate J-lenses per random group
3. Compute transport cosine using the best-matching random group per test prompt

If path-conditioned significantly beats random-conditioned (not just standard), the result is MoE-specific. If path-conditioned ~ random-conditioned > standard, you've proven only that subset fitting helps, not that routing matters. This control costs approximately zero extra implementation time — it reuses the same conditioned-fitting code with shuffled cluster labels.

**Pre-register this prediction:** path-conditioned transport cosine > random-conditioned transport cosine (one-tailed test). Without this, the experiment cannot support its hypothesis.

---

## 2. Baseline Validity — WARN

**The issue:** The 12% figure comes from a prior run on `Qwen/Qwen3-30B-A3B` with a custom-fitted lens (200 prompts, Modal H100, `modal_moe_jlens.py`). The hackathon uses "Qwen3-32B" with a Neuronpedia pre-fitted lens. Two questions:

1. **Are these the same model?** `Qwen3-30B-A3B` (30B total, 3B active) and `Qwen3-32B` are likely the same model under different naming. The HuggingFace download in the spec says `Qwen/Qwen3-32B`. The prior code used `Qwen/Qwen3-30B-A3B`. Confirm they resolve to the same checkpoint.

2. **Are the lenses comparable?** The 12% was with a 200-prompt custom fit. The Neuronpedia lens may use different fitting prompts, more prompts, or different hyperparameters. The baseline might not reproduce at 12%.

**The mitigation:** The spec already includes a replication step (step 1 of the design), and the pre-registered prediction is < 0.2 (not exactly 12%). This is good. But clarify the model identity explicitly in the protocol.

**Status:** The replication step saves this from FAIL. Just tighten the model naming.

---

## 3. Clustering Validity — WARN

**Issue A: k-means on binary vectors.** Expert-activation vectors are binary (expert fired or didn't). K-means uses Euclidean distance, which is poorly suited to binary data — it treats the difference between {experts 1,3,7} and {experts 1,3,8} the same as between {experts 1,3,7} and {experts 2,4,9}, ignoring the structure of which experts co-activate. Hamming distance or Jaccard similarity with hierarchical clustering would respect the binary nature.

**Issue B: Sensitivity to k.** The spec predicts 3-8 clusters per layer, guided by the Standing Committee finding. But k-means requires specifying k. How will k be chosen per layer? Silhouette score over what range? What if optimal k varies wildly across layers (k=2 at some, k=15 at others)? What if silhouette scores are uniformly low, suggesting no clean cluster structure?

**Issue C: Dimensionality.** If Qwen3-32B has 128 experts with top-8 routing, each binary vector is 128-dimensional. With 200-672 prompts, k-means in 128 dimensions may not find stable clusters.

**Fixes:**
- Test at least two clustering methods (k-means + hierarchical/Jaccard). Report both.
- Sweep k from 2 to 12 per layer. Report silhouette scores. Use the k that maximizes silhouette, but also report results at fixed k=4 for comparability.
- Report cluster sizes. Flag any cluster with < 50 prompts.

---

## 4. Sample Size for Conditioned Fitting — FAIL

**The math:** The Qwen3.5-27B Neuronpedia lens was fitted on 672 prompts across 63 layers. If conditioned fitting on Qwen3-32B starts from a comparable fitting set and splits into k=8 clusters, each cluster gets ~84 prompts. With k=4, ~168 prompts. The J-lens fitting procedure computes averaged Jacobians of shape (d_model, d_model). For d_model = 5120 (typical for 32B models), that is a 26M-parameter matrix estimated from 84 samples of token-level gradients.

**Why this matters:** The per-prompt contribution to the Jacobian estimate comes from averaging over token positions within a prompt, then averaging across prompts. More prompts = better estimate of the expectation. With 84 prompts, the Jacobian estimate has high variance. The transport cosine improvement from conditioning could be real signal or could be noise from overfitting to small samples. This interacts with Finding 1 — small-sample overfitting and subset overfitting compound.

**The fix:**
1. Specify the minimum cluster size before the experiment. Any cluster below this threshold gets merged with its nearest neighbor.
2. Report a stability analysis: run conditioned fitting 5 times with bootstrap resampling of prompts within each cluster. If transport cosine variance across bootstraps exceeds 0.1, the estimate is unstable.
3. If using the Neuronpedia lens (672 prompts) as the fitting basis: state clearly whether you are refitting from scratch on cluster-filtered prompts, or modifying the existing lens. These are very different procedures.

**Bottom line:** The spec says nothing about sample size adequacy. Given the dimensionality of J-lens fitting, this is a potential kill shot. The bootstrap stability check is the minimum required.

---

## 5. The 0.5 Threshold — WARN

> **CORRECTION 2026-08-16 (Lyra).** The "> 0.7" below is unsupported. Verified
> against the full text of Gurnee et al. 2026 and the reference implementation: no
> transport-cosine or reconstruction-fidelity figure is reported anywhere, and the
> repo computes no cosine at all. The paper's only 0.7 is inter-lens vector
> similarity (§A.6), which it pairs with *low* top-1 agreement. This review is left
> otherwise intact as the dated record of what was checked on 2026-08-11 — note that
> line 134 of this same document states the correct reading ("agreement between two
> projection methods is not causal evidence") and line 77 correctly observes that
> Gurnee does not identify 0.5 as meaningful. The knowledge was present; it was
> simply never turned on this paragraph.

**The issue:** Dense models achieve transport cosine > 0.7. Standard MoE gets ~0.12. The spec declares success at > 0.5. Why 0.5?

- It is not a known interpretability boundary.
- It is not the midpoint of any principled range.
- It is not anchored to a downstream task performance threshold.
- The Gurnee et al. (2026) paper does not identify 0.5 as meaningful.

**The risk:** If conditioned cosine lands at 0.45, is that a failure? At 0.52, is that a success? A binary threshold on a continuous metric invites p-hacking adjacent to the boundary.

**Fixes:**
- Report the full transport cosine distribution (per-layer, per-cluster) as the primary result. Drop the binary pass/fail.
- If a threshold is needed for the pre-registration, justify it: e.g., "transport cosine sufficient for the J-lens to produce top-10 tokens matching model output at >50% of layers" (an empirical calibration, not a number pulled from nowhere).
- The more informative metric is delta: (conditioned - standard) transport cosine. Report this per layer with confidence intervals.

---

## 6. Generalization — FAIL

**The problem:** The spec says J-lens fitting uses WikiText prompts (via `load_wikitext_prompts()`). Evaluation uses test prompts to select the matching cluster's Jacobian. But if test prompts also come from WikiText (or a similar distribution), the routing patterns match the fitting distribution by construction.

**Why this kills the result:** A path-conditioned lens fitted on WikiText learns "WikiText routing patterns -> Jacobian." If you test on WikiText, you get WikiText routing patterns, so you match to the right cluster. If you test on code, dialogue, or multilingual text, the routing patterns may not match any fitting cluster. The lens could perform WORSE than standard on out-of-distribution inputs because it selects the wrong cluster's Jacobian instead of using the (imperfect but distribution-agnostic) average.

**The literature supports this concern:** Wang/Hayou/Nalisnick (2604.09780) show prompt-level routing does not predict rollout-level routing, and deeper layers show near-identical expert activation across unrelated inputs. If routing converges at depth, path conditioning at deep layers may have no meaningful cluster structure.

**The fix:**
1. Split evaluation into in-distribution (WikiText) and out-of-distribution (at least one of: code, dialogue, multilingual). Report both.
2. If conditioned > standard on WikiText but conditioned <= standard on OOD, the result is distribution-specific, not a general MoE J-lens solution.
3. This is three extra evaluation runs with different prompt sets. Minimal added time.

---

## 7. Time Budget — WARN (upgraded from initial FAIL assessment)

**The concern:** 4-6 hours for implementation + execution on a 32B MoE model on MPS.

Breaking down the compute:
- **Baseline:** Load Neuronpedia pre-fitted lens, run transport cosine. Fast — maybe 30 min including model load.
- **Router hooks + path extraction:** Forward pass through all fitting prompts, record expert activations. On MPS for a 32B MoE, each forward pass ~10-30s for short sequences. For 200 prompts: ~30-100 min.
- **Clustering:** CPU work. Minutes.
- **Conditioned fitting:** This is the expensive step. J-lens fitting requires backward passes (VJPs) through the model. The existing code shows this takes ~25-50 min PER PROMPT on MLX for the 27B dense model. Even with the analytic shortcut, fitting across 3-8 clusters requires at minimum one full J-lens fit equivalent. On MPS with a 32B MoE model, this could easily exceed 4 hours alone.
- **Evaluation:** Forward passes + transport cosine. ~30 min.
- **Implementation:** ~230 lines of novel code. Doable but tight.

**Assessment:** The time budget is tight but not impossible IF:
1. The analytic fitting path works on the MoE architecture
2. The pre-fitted Neuronpedia lens can be used as a warm start (reweight/select Jacobians rather than recompute from scratch)
3. Implementation goes smoothly

**Mitigation already in spec:** Priority order is A > C > B. MoE is explicitly the stretch experiment. The risk register acknowledges it may not finish. Upgrading from FAIL to WARN because the spec already treats this as the lowest-priority experiment.

**Suggestion:** Pre-write the router hooks and clustering code before the hackathon. That saves ~1 hour of implementation time.

---

## 8. Claims vs Evidence — WARN

**The claim:** "First J-lens that actually works on MoE." (Section 4.2 framing)

**What "works" means in the spec:** Transport cosine > 0.5 at workspace-band layers.

**What "works" should mean:** The J-lens produces directions in J-space that are (a) verbalizable (the transported representation decodes to interpretable tokens), (b) causally meaningful (perturbing along J-space directions changes model output predictably), and (c) stable across prompts within a routing path (not noise).

**The gap:** Transport cosine > 0.5 tests (a) only. It does not test (b) or (c). A high transport cosine means the J-lens prediction agrees with direct unembedding — but agreement between two projection methods is not causal evidence. The dense-model J-lens paper (Gurnee et al.) established causality through additional experiments (ablation, steering). The hackathon won't have time for those.

**Section 9 helps:** The spec correctly disclaims "We are not claiming MoE J-lens solves interpretability for MoE models." But the experiment section should match the disclaimers: say "first transport-cosine-positive J-lens on MoE" not "first J-lens that works."

---

## 9. Comparison to MoE Lens (Chaudhari et al.) — PASS

**Assessment:** The references doc correctly identifies the gap: MoE Lens uses LogitLens (correlation-based), the proposed approach uses Jacobian (causal). The distinction is clearly articulated. Chaudhari et al. achieve cos ~0.95 for single top-expert + residual, but this measures whether per-expert outputs can be decoded, not whether the Jacobian captures the causal computation.

**Why PASS not WARN:** The spec is not claiming to beat MoE Lens. It is testing whether path-conditioned J-lens is FEASIBLE (transport cosine > 0.5). MoE Lens answers a different question (can you decode what each expert is doing?) than J-lens (what is the causal structure of information flow?). The references doc explains this distinction. A direct head-to-head comparison would be informative but is not required for the claimed contribution.

**Suggestion:** In the paper, include one paragraph comparing the two approaches and explaining when each is appropriate. This is already implied by the references doc.

---

## 10. Reproducibility — PASS

**Strengths:**
- "ALL code and data published" (pre-registered commitment)
- Public model (Qwen3-32B on HuggingFace)
- Public lens (Neuronpedia jacobian-lens repo, verified by prior spec review)
- Public fitting library (jlens from Anthropic, `anthropics/jacobian-lens`)
- Pre-registered protocol in this document

**Risks:**
- MPS vs CUDA numerical differences in Jacobian computation (may produce different transport cosines)
- The conditioned fitting procedure is novel (~100 lines of code). Until published, it's a verbal description.
- Cluster assignments depend on the specific fitting prompts used (WikiText subset). Different random seeds = different clusters.

**Assessment:** PASS. The commitment to publish code and data is the strongest reproducibility guarantee. The novel code is ~100 lines — small enough to inspect and replicate. The numerical platform dependence is a general issue with Jacobian methods, not specific to this experiment.

---

## Summary Table

| # | Finding | Rating | Fix Required |
|---|---------|--------|-------------|
| 1 | Null swarm: subset overfitting | **FAIL** | Add random-conditioned control arm |
| 2 | Baseline validity | WARN | Clarify model identity (Qwen3-30B-A3B vs Qwen3-32B) |
| 3 | Clustering validity | WARN | Test 2 methods, sweep k, report cluster sizes |
| 4 | Sample size | **FAIL** | Specify minimum cluster size, bootstrap stability check |
| 5 | The 0.5 threshold | WARN | Report distribution + delta, justify threshold or drop it |
| 6 | Generalization | **FAIL** | Add out-of-distribution evaluation (code/dialogue/multilingual) |
| 7 | Time budget | WARN | Pre-write hooks + clustering; acknowledged as stretch |
| 8 | Claims vs evidence | WARN | Bound claim to "transport-cosine-positive," not "works" |
| 9 | MoE Lens comparison | PASS | One comparison paragraph in paper |
| 10 | Reproducibility | PASS | Code + data publishing commitment |

---

## Required Fixes Before Hackathon

### Fix 1 (Finding 1): Random-conditioned control
Add to the pre-registered design:

```
6. **Random-conditioned control:** For each layer, randomly assign prompts
   to the same number of same-sized groups (ignoring routing). Fit separate
   Jacobians per random group. Evaluate identically to path-conditioned.
   This isolates the effect of routing-specific conditioning from generic
   subset fitting.
```

Add to pre-registered predictions:

```
5. Path-conditioned transport cosine > random-conditioned transport cosine
   (one-tailed Mann-Whitney, per layer)
```

### Fix 2 (Finding 4): Sample size specification
Add to the design:

```
   Minimum cluster size: 50 prompts. Clusters below this threshold are
   merged with the nearest cluster (by centroid distance). If any layer
   produces only 1 cluster after merging, that layer uses the standard
   (unconditioned) lens.
   
   Stability check: bootstrap resample prompts within each cluster 5 times,
   refit, report transport cosine mean +/- SD. Unstable clusters (SD > 0.1)
   are flagged.
```

### Fix 3 (Finding 6): Out-of-distribution evaluation
Add to the design:

```
   Evaluation domains:
   a. WikiText (in-distribution — same domain as fitting prompts)
   b. Code (e.g., HumanEval prompts or The Stack samples)
   c. Dialogue (e.g., ShareGPT or Dolly instruction-following prompts)
   
   Report transport cosine separately for each domain.
```

---

## What the Literature Actually Supports

The references doc is strong — 19 papers, well-curated, correctly interpreted. The key theoretical justification is Liu (2605.16349): cross-expert Jacobians are near-orthogonal. This is the mathematical reason averaged J-lens fails AND the reason path conditioning should help. The Ye/Yuan/Sharkey finding that paths (not experts) are the monosemantic unit directly supports the approach.

But theoretical justification is not empirical validation. The experiment as designed (without the three fixes above) cannot distinguish between "routing captures real computational structure" and "any subset fitting helps." The fixes close this gap.

---

## Overall Assessment

The hypothesis is well-motivated. The literature foundation is excellent. The theoretical argument (near-orthogonal cross-expert Jacobians -> averaged J-lens fails -> conditioning on routing should help) is sound. The null swarm risk (Finding 1) is the single biggest threat to the experiment's validity, and it is cheaply fixable.

If the three FAIL fixes are applied, this experiment is ready to run. If any FAIL is unaddressed, the result is uninterpretable regardless of the transport cosine numbers.
