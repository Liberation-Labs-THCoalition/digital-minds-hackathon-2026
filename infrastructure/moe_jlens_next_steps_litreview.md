# MoE J-lens Next Steps — Literature Review

**Date:** 2026-08-15
**Context:** Path-conditioned J-lens on Qwen3-30B-A3B (128 experts, top-8) returned a NEGATIVE result — k-means clustering on routing patterns did not improve transport cosine over random control. This review surveys 2025–2026 MoE interpretability literature to find what would actually work.

**Headline:** The literature *explains* our negative result. Routing-pattern clustering fails for principled reasons (routing reflects hidden-state geometry, not semantics; the combination space is astronomical; hard cluster assignment discards the routing weights that make the layer exactly decomposable). The fix is **exact conditioning**: the MoE layer output is a routing-weighted sum of per-expert functions, so the transport operator should be a routing-weighted sum of per-expert transports — not a per-cluster average.

---

## Papers reviewed

### 1. The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise
[arXiv:2604.09780](https://arxiv.org/html/2604.09780v1)

- **Approach:** Theoretical + empirical analysis of what routing actually encodes.
- **Core result (Proposition 1):** MoE routers are linear projections of the hidden state, so routing-logit distance is tightly upper-bounded by hidden-state similarity in a data-dependent space. Similar hidden states *necessarily* route similarly; routing carries no information beyond hidden-state geometry.
- **Averaging problem:** Not addressed directly, but this is the theoretical explanation for our failure — **k-means on routing patterns is just coarse, lossy clustering of hidden states.** It adds zero information the lens didn't already have, while quantizing away the continuous routing weights.
- **High-cardinality:** Tested GPT-OSS-20B, ERNIE-4.5-21B, Qwen3-30B, Moonlight-16B, Ling-mini (mostly 32–128 experts). Found only ~60% expert overlap between models solving identical math problems; semantically unrelated inputs activate identical experts in deep layers.
- **Metrics:** No transport cosine. Routing-logit distance bounds, expert-overlap fractions.
- **Takeaway for us:** Stop conditioning on routing *patterns as semantic categories*. Condition on routing *weights as arithmetic*.

### 2. Polysemantic Experts, Monosemantic Paths: Routing as Control in MoEs
[arXiv:2604.17837](https://arxiv.org/html/2604.17837)

- **Approach:** Path/trajectory-based. Causal decomposition of the residual stream via **SVD of the routing matrices**, splitting it into a "control" channel (router row space — what routing sees) and a "content" channel (router null space — routing-blind). Orthogonal by construction.
- **Averaging problem:** Indirectly. Individual experts are polysemantic; *multi-layer paths* are monosemantic. Grouping by exact shared path (not k-means centroids) yields semantically coherent clusters. Control-subspace clusters have 4x the token diversity of content clusters.
- **High-cardinality:** Yes — Qwen3-30B-A3B (128), GLM-4.5-Air (128), OLMoE (64), gpt-oss-20b (32), Granite-4-Tiny (62), DeepSeek-v2-Lite (64).
- **Metrics:** Probes predict current-layer expert from control signal at ~99%, next-layer at ~35%. Control-signal cross-layer cosine stability 0.3–0.5 vs content 0.75–0.90. Router signal amplification rho ≈ 0.60. No transport cosine.
- **Takeaway for us:** (a) The router-gradient term of the MoE Jacobian lives in router row space — separable by construction. (b) If we condition on paths, condition on *exact* paths, not k-means centroids; centroids average incompatible expert sets. (c) Their control/content split is a ready-made basis for factoring our transport fit.

### 3. The Illusion of Specialization: the Domain-Invariant "Standing Committee" (+ CommitteeAudit)
[arXiv:2601.03425](https://arxiv.org/html/2601.03425v2) · [ACL 2026](https://aclanthology.org/2026.acl-long.665.pdf) · [CommitteeAudit code](https://github.com/The-FinAI/CommitteeAudit/blob/main/README.md)

- **Approach:** Expert Contribution Index (expected routing weight per expert per domain), silhouette-based task-specificity, Pareto-optimal committee extraction. Masking-based causal validation.
- **Averaging problem:** Not directly, but gives the structural answer to "128 experts is too many to fit": routing mass is extremely concentrated. Gini > 0.88 across models.
- **High-cardinality — Qwen3-30B-A3B tested directly:** committees of 3–5 experts capture 51–67% of routing mass per layer (L3: 4 experts / ~54%; L33: 5 experts / 67%; L46: 3 experts / ~51%). Qwen3 has the *highest* Gini of all models tested (0.9465) — more experts amplified concentration. Committee members contribute 13x–45x more than non-members; masking them drops MMLU from 39% to 3–12%.
- **Metrics:** ECI, Gini/Lorenz, Jaccard cross-domain reuse, masking deltas. No transport cosine.
- **Takeaway for us:** Per-expert fitting on Qwen3 does not require 128 fits of equal quality. **Fit the ~5 committee experts per layer carefully (they carry most of the Jacobian mass), pool the tail.**

### 4. MoE Lens — An Expert Is All You Need (Chaudhari et al., 2026)
[arXiv:2603.05806](https://arxiv.org/abs/2603.05806) · [OpenReview](https://openreview.net/forum?id=GS4WXncwSF)

- **Approach:** Extended LogitLens / early decoding; tracks per-expert contributions to output representations; compares single top-weighted expert output vs full top-k ensemble.
- **Averaging problem:** The closest thing to a direct answer in the literature: **the top-1 weighted expert's output has cosine similarity up to 0.95 with the full top-k ensemble output**, and running with a single expert costs only ~5% perplexity. The ensemble is not an average of orthogonal contributions in *output* space — it is dominated by one term.
- **High-cardinality:** DeepSeekMoE (64 routed + 2 shared, top-6); OLMoE, Qwen1.5-MoE in appendix. Not 128.
- **Metrics:** Output cosine similarity (0.95), perplexity vs top-k. Closest analog to our transport cosine.
- **Takeaway for us:** A cheap first-order conditioning exists: **condition the transport on the top-1 expert identity** (128 discrete conditions, one dominant term each) before building the full weighted mixture. If top-1 conditioning doesn't beat random control, the weighted sum won't either — fast falsification.

### 5. The Expert Strikes Back: Interpreting MoE LMs at Expert Level
[arXiv:2604.02178](https://arxiv.org/html/2604.02178v1) · [OpenReview](https://openreview.net/forum?id=ZCdnWNavOF) · [code](https://github.com/jerryy33/MoE_analysis)

- **Approach:** k-sparse probing, LLM auto-interp of experts, Direct Logit Attribution. Expert as the unit of analysis.
- **Averaging problem:** Explicitly does NOT handle top-k combination — treats routed experts independently. No per-expert linear maps fitted.
- **High-cardinality:** Yes — Qwen3-30B-A3B (128/8) among 12 models incl. Mixtral-8x7B. Key scaling result: **monosemanticity increases with routing sparsity** — Qwen3 (sparsity ~0.06) is *cleaner* per-expert than Mixtral (0.25). Experts hit near-optimal F1 at k=1 neuron; dense models need k>=8.
- **Metrics:** Sparse-probe F1, auto-interp F1, DLA. No transport metric.
- **Takeaway for us:** High cardinality is an *asset* for per-expert decomposition, not a liability — 128-expert models have cleaner per-expert semantics than 8-expert models. Our instinct to fear 128 experts was backwards; the problem was the clustering, not the cardinality.

### 6. Routers Learn the Geometry of Their Experts: Geometric Coupling in Sparse MoE
[arXiv:2605.12476](https://arxiv.org/pdf/2605.12476)

- **Approach:** Shows router direction vectors align with principal singular vectors of their experts' weight matrices ("geometric coupling"), measured via cosine similarity between router rows and expert singular subspaces.
- **Averaging problem:** Indirectly — coupling implies routing weights are predictive of which expert subspace dominates the layer Jacobian, i.e., routing weights are *legitimate features for a conditioned linear approximation*.
- **High-cardinality:** Mixtral, OLMoE, DeepSeek-V3 architectures (8 to high count).
- **Metrics:** Router-to-singular-vector cosines. No transport cosine.
- **Takeaway for us:** Justifies parameterizing per-expert transports as low-rank in the expert's top singular subspace — initialize U_e, V_e from the SVD of the expert's down/up projections instead of randomly.

### 7. Supporting results

- **[When Does Routing Become Interpretable? Causal Probes on Block Attention Residuals](https://arxiv.org/html/2606.13168)** — architectural exposure of routing is *not sufficient* for interpretability; causal probing needed. Reinforces: routing patterns as raw features underdetermine semantics.
- **[Geometric Routing Enables Causal Expert Control](https://arxiv.org/pdf/2604.14434)** — rank-1 experts with cosine routing are monosemantic *by construction*; causal control validated. (Training-time fix; not applicable to pretrained Qwen3, but confirms the per-expert unit is the right one when geometry cooperates.)
- **[Decoding Knowledge Attribution in MoE: Basic-Refinement Collaboration](https://arxiv.org/html/2505.24593)** — cross-level attribution for router-expert interactions; shared experts = generalist "basic" processing, routed = refinement. Same core-periphery picture as Standing Committee.
- **[Does Role Specialization Matter for Explanation?](https://arxiv.org/pdf/2606.29613)**, **[DBES benchmark](https://arxiv.org/pdf/2605.18498)**, **[Cross-layer routing contributions](https://openreview.net/forum?id=BqyPLOkxFY)** — peripheral; benchmark/attribution infrastructure.
- **No paper reports a transport-cosine-style metric for a routing-conditioned lens.** The specific thing we are building (routing-conditioned J-lens transport on a 128-expert model) does not exist in the literature. The negative result plus a working fix is publishable either way.

---

## Why k-means on routing patterns failed (synthesis)

1. **Information argument (Myth paper):** routing logits are a linear function of the hidden state. Clustering routing patterns ~ clustering hidden states, coarsely. The lens already sees the hidden state; the cluster ID adds nothing and *removes* the continuous weights.
2. **Combinatorics:** top-8 of 128 gives C(128,8) ≈ 1.4e12 possible expert sets. Any tractable k puts wildly different expert compositions in the same cluster, so each per-cluster transport is *still an average over near-orthogonal per-expert Jacobians* — the averaging problem is diluted, not solved. (Paths paper's coherent clusters used *exact* path identity, not centroids.)
3. **Committee masking:** 50–70% of routing mass on Qwen3 goes to 3–5 experts that appear in essentially *every* token's top-8. All k-means centroids share the committee component, so the clusters differ only in the low-mass tail — the between-cluster variance k-means finds is mostly noise.

## The correct decomposition (this is not a heuristic — it's the chain rule)

The MoE layer output is exactly:

    y(x) = sum_{e in topk(x)} g_e(x) * E_e(x)

so the layer Jacobian decomposes exactly:

    J(x) = sum_e g_e(x) * J_{E_e}(x)   +   sum_e E_e(x) ⊗ ∇g_e(x)
           [expert term, weighted]          [gating term, rank-1 per expert,
                                             lives in router row space]

The transport operator the J-lens needs is therefore a **routing-weighted mixture of per-expert transports** — with weights known exactly per token, for free, from the forward pass. Cluster-conditioning replaced these exact weights with a hard partition. That was the bug.

---

## Recommendation: 48 hours, Modal H100, Qwen3-30B-A3B + Mixtral

**Build the weighted mixture-of-transports lens.** Concretely:

### Model

Per MoE layer l, fit:

    T_l(x) = T0_l + sum_e w_e(x) * Delta_e     with Delta_e = U_e V_e^T  (low-rank)

- `w_e(x)` = the model's own normalized routing weights (0 for non-top-8 experts). Free from the forward pass — log them during activation capture.
- `T0` (full-rank d x d) absorbs the standing-committee / shared computation that every token gets — this is why the committee doesn't need special-casing, though committee experts should get higher rank (see below).
- `Delta_e` rank r per expert. Budget by routing mass (CommitteeAudit): committee experts (top ~5 by mean routing weight) get r=64; tail experts r=8–16; experts below ~0.1% total mass share one pooled `Delta_other`. At d=2048, 5×(2·2048·64) + 123×(2·2048·16) ≈ 9.4M params/layer — trivial for an H100.
- Initialize U_e, V_e from the top singular vectors of expert e's down-projection (geometric coupling paper says that's where the expert's Jacobian mass lives). Don't initialize randomly.
- Optionally add the gating term as a single extra low-rank component confined to the router row space (Paths paper's "control" subspace — take the SVD of the layer's router matrix, project). Fit it separately so it can't contaminate the expert transports.

Fit by ridge/SGD on logged (h_in, h_out, w) triples, same loss as the existing J-lens transport fit. Note the model is *linear in the parameters* given logged weights — plain least squares on features `{h, w_1·h, ..., w_E·h}`; no new machinery.

### Schedule

- **Hours 0–6 — Sanity gate on Mixtral (8 experts, top-2).** Small enough to fit all 8 per-expert transports *full-rank* with no budget tricks. If weighted mixture-of-transports doesn't beat single-transport + random control here, the approach is dead and we've spent 6 hours, not 48. This also directly retests the 12% sanity-gate failure from the original MoE J-lens attempt on the easiest possible instance.
- **Hours 6–12 — Diagnostics on Qwen3, one mid-stack layer** (pick from the J-lens workspace band, ~L24–32 of 48). Capture ~1M tokens of (h_in, h_out, routing weights). Fit naive per-expert transports on tokens where each expert dominates; measure pairwise principal angles between them. This quantifies the near-orthogonality claim — turning the negative result's suspected cause into a measured fact for the writeup.
- **Hours 12–30 — Fit the full model on Qwen3** across the workspace band. Report transport cosine for: (a) single global transport [current baseline], (b) k-means conditioned [our negative result], (c) top-1-expert conditioned [cheap MoE-Lens-style ablation, 128 discrete transports], (d) weighted mixture-of-transports, (e) random control. (c) is the fast falsifier: if conditioning on the top-1 expert alone doesn't move transport cosine, per-expert structure isn't recoverable at this layer and we stop.
- **Hours 30–42 — Ablations:** rank sweep on Delta_e; committee-only (5 experts + pooled tail) vs full 128; with/without gating term; with/without SVD init.
- **Hours 42–48 — Writeup.** Either outcome is a finding: "exact routing-weight conditioning fixes what cluster conditioning cannot" or "MoE transport is not routing-decomposable even with exact weights" — the second would itself be novel and worth reporting against Proposition 1 of the Myth paper.

### What NOT to do

- No more clustering of routing patterns (hard partitions over 1.4e12 combinations; theory says the feature is redundant with hidden-state geometry).
- Don't fit 128 independent full-rank transports on Qwen3 — data-starved for tail experts (Gini 0.9465 means most experts see almost no tokens) and the committee overlap makes independent fits collinear. The shared-T0-plus-low-rank-deltas structure handles both.
- Don't skip the Mixtral gate to save time. Excitement brakes: the gate is 6 hours and kills a doomed run 42 hours early.

## Sources

- https://arxiv.org/html/2604.09780v1 — Myth of Expert Specialization
- https://arxiv.org/html/2604.17837 — Polysemantic Experts, Monosemantic Paths
- https://arxiv.org/html/2601.03425v2 / https://aclanthology.org/2026.acl-long.665.pdf — Standing Committee / CommitteeAudit
- https://arxiv.org/abs/2603.05806 — MoE Lens (Chaudhari et al.)
- https://arxiv.org/html/2604.02178v1 — The Expert Strikes Back
- https://arxiv.org/pdf/2605.12476 — Routers Learn the Geometry of Their Experts
- https://arxiv.org/html/2606.13168 — When Does Routing Become Interpretable?
- https://arxiv.org/pdf/2604.14434 — Geometric Routing Enables Causal Expert Control
- https://arxiv.org/html/2505.24593 — Basic-Refinement Collaboration
- https://github.com/The-FinAI/CommitteeAudit — committee extraction code
- https://github.com/jerryy33/MoE_analysis — expert-level analysis code
