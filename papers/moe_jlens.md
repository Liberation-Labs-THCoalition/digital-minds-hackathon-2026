# Path-Conditioned Jacobian Lenses for Mixture-of-Experts Models

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Lyra (THCoalition), Kavi ([affiliation]), Thomas Edrington (Liberation Labs)

**With** Apart Research

## Abstract (~200 words)

The Jacobian lens (Gurnee et al. 2026) identifies a low-dimensional verbalizable workspace in dense transformer models with transport cosines exceeding 0.7. On Mixture-of-Experts models, standard J-lens fitting fails catastrophically — we measure ~12% transport cosine on Qwen3-30B-A3B (30B total, 3B active, 128 experts top-8, d=2048), because the averaged Jacobian across all routing paths represents no actual forward pass. Cross-expert Jacobians are near-orthogonal (Liu 2026), so averaging destroys the signal.

We propose path-conditioned Jacobian fitting: capture routing decisions during the fitting pass, cluster prompts by their expert trajectory per layer, and fit separate Jacobians per cluster. Each conditioned Jacobian represents the computation along one routing path — the path the model actually took. We evaluate against two controls: the standard (unconditioned) lens and a random-conditioned lens (prompts split into same-sized random groups) to distinguish genuine routing structure from subset overfitting.

This matters because every frontier model deployed today is MoE. Opening J-lens workspace analysis to MoE architectures extends introspection tools — including metacognitive memory (companion paper) — to the models that actually run production systems. [Results TBD.]

---

## 1. Introduction

[Problem: J-lens is the best tool for identifying what a transformer is "disposed to say" at each layer. It doesn't work on MoE. Every frontier model is MoE.]

[Why Track 6: "entirely new questions may surface" — this is a tooling contribution that enables everything in Tracks 2, 3, and 5 to work on frontier models]

**Our main contributions are:**

1. Diagnosis: cross-expert Jacobians are near-orthogonal, explaining why averaged fitting fails at ~12% transport cosine on MoE models.

2. Path-conditioned fitting: cluster prompts by routing pattern, fit per-cluster Jacobians, select the matching Jacobian at inference time.

3. Three-way evaluation (standard vs path-conditioned vs random-conditioned) establishing whether the improvement is routing-specific or generic subset overfitting.

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

The baseline is a standard (unconditioned) J-lens fitted on Qwen3-30B-A3B in a previous run, using 200 WikiText prompts (smaller than the 672 used for conditioned fitting) and the reference `jlens` implementation, with no knowledge of routing. The corpus size difference is a limitation: the conditioned lenses benefit from more fitting data per cluster than the standard lens had in total. We report this asymmetry rather than claiming equivalent conditions. This fit achieves a mean transport cosine of ~12.5% — far below the 0.7+ reported for dense models (Gurnee et al. 2026) and below the sanity gate we apply to dense fits. We load this saved lens unchanged and re-evaluate it on the held-out test prompts under the identical transport-cosine protocol used for the conditioned lenses (§3.6), so all three conditions share one evaluation pipeline. This replicates our prior failure and establishes the number the conditioned lenses must beat.

### 3.2 Router Hooks

Qwen3-30B-A3B's router (`Qwen3MoeTopKRouter`, exposed as each layer's `mlp.gate` module) returns a 3-tuple `(router_logits, routing_weights, expert_indices)`. We register a forward hook on `mlp.gate` at every MoE layer and record the third element — the top-8 expert indices per token — detached to CPU. No model code is modified and no routing logic is reimplemented; the hooks are ~50 lines wrapping the model's own forward pass. MoE layers are discovered structurally (presence of `mlp.gate` / `mlp.experts`) rather than hard-coded, and all 48 layers are captured. One forward pass per fitting prompt (no gradients) yields the routing tensor R of shape (n_prompts, n_tokens, 48, 8): for each prompt, token, and layer, the 8 experts that fired. The same hooks are reused verbatim on test prompts at evaluation time (§3.6).

### 3.3 Path Clustering

Per target layer, each prompt is summarized as a 128-dimensional routing frequency vector: entry *e* counts how often expert *e* appeared in a token's top-8 across the prompt, normalized by token count. This is a soft generalization of the binary "which experts fired" vector that additionally weights experts by how consistently they fire.

Vectors are clustered with k-means (`n_init = 10`, fixed seed). We sweep k from 3 to 7, additionally capping k at ⌊n/50⌋ so that clusters average at least 50 prompts, and select k by silhouette score. If no valid clustering emerges, the layer falls back to a single cluster (equivalent to the standard fit). Informed by the standing-committee result (Wang et al. 2026), we expect 3–5 dominant clusters per layer.

A caveat we flag now rather than in hindsight: with 128 experts and top-8 routing, the per-layer path space is C(128,8) ≈ 1.4 × 10¹¹, and Euclidean k-means on 128-dimensional frequency vectors from 672 prompts is a coarse instrument. Jaccard or Hamming distance with hierarchical clustering is a principled alternative (see Limitations); we use k-means here as the simplest method that could work, with the random-conditioned control (§3.5) guarding against clustering artifacts being mistaken for routing structure.

### 3.4 Conditioned Fitting

For each cluster at each target layer, we fit a separate Jacobian lens using only that cluster's prompts, via the reference `jlens.fit` with `source_layers = [layer]` (`dim_batch = 4`, `max_seq_len = 128`). Restricting the fit to a single source layer, rather than all 47 fittable layers of the standard protocol, reduces per-fit compute by ~47× — this is what makes fitting one lens per cluster per layer tractable on a single H100. Each conditioned Jacobian therefore represents the model's computation along one routing path: the path the prompts in that cluster actually took at that layer. Clusters with fewer than 20 prompts are excluded from fitting (too few samples for a stable 2048 × 2048 ≈ 4.19M-parameter estimate); their prompts are simply not covered by a conditioned lens at that layer. Fitted lenses and a fitting manifest (cluster sizes, wall-clock times, failures) are checkpointed per cluster.

### 3.5 Random-Conditioned Control

The critical control: identical protocol, shuffled labels. For each target layer we randomly permute the 672 fitting prompts (fixed seed) and partition them into groups whose sizes exactly match that layer's cluster sizes from §3.3. Each random group then goes through the same fitting call as §3.4 — same `jlens.fit`, same `source_layers = [layer]`, same `dim_batch`, `max_seq_len`, and same <20-prompt exclusion rule. The only difference between conditions is whether group membership reflects routing or chance. If path-conditioned lenses do not beat random-conditioned lenses, any improvement over the standard lens is an artifact of fitting on smaller, more homogeneous subsets — subset overfitting, not routing structure.

### 3.6 Evaluation

**Test-prompt assignment.** For the 100 held-out prompts we capture routing with the same hooks (§3.2) and build the same 128-d frequency vectors (§3.3). A k-means model with the layer's selected k (same seed) is fitted on the fitting-set vectors and used to assign each test prompt to its nearest routing cluster; the matching conditioned lens is then selected per prompt. For the random condition we mirror this: test prompts are randomly assigned (independent seed) to groups with the same size proportions, and evaluated under the corresponding random-group lens. The standard lens is evaluated on all test prompts.

**Transport cosine.** For each test prompt we record the residual stream at the source layer and at the final layer, transport the source activation through the lens under evaluation, unembed both, and compute the cosine similarity between the softmaxed predicted and actual next-token distributions at the final token position. This matches the metric under which the standard fit scored ~12.5%.

**Statistics.** Per target layer, a one-sided Mann-Whitney U test of conditioned > random-conditioned on per-prompt transport cosines (sample sizes matched by truncation to the smaller condition), with Bonferroni correction across the three layers. Outcome criteria are pre-registered in the pipeline code: *success* requires conditioned > random and conditioned > standard with at least one Bonferroni-significant layer (and mean conditioned cosine > 0.5 for the full claim); conditioned > standard but ≈ random is classified as a null-swarm outcome (subset overfitting); conditioned > standard without significance is inconclusive.

**Cross-domain.** WikiText is both fitting and primary evaluation domain, so as an out-of-domain check we additionally evaluate at the middle target layer (L24) on two held-out domains: code prompts (Python, JavaScript, SQL completion contexts) and dialogue prompts (User/Assistant exchanges). Routing-cluster assignment for OOD prompts follows the same §3.3 pipeline; degradation here bounds how far the conditioned lenses generalize beyond the fitting distribution.

## 4. Results

[TBD]
[Figure 1: Transport cosine vs layer — three lines (standard, path-conditioned, random-conditioned)]
[Figure 2: Cluster structure — silhouette scores, cluster sizes, standing committee analysis]
[Table 1: Cross-domain transport cosines]

## 5. Discussion and Limitations

[If path-conditioned > random-conditioned: routing structure genuinely matters for workspace analysis. The workspace in MoE models is path-dependent.]
[If path-conditioned ≈ random-conditioned: the improvement is just smaller fitting sets, not routing. MoE workspace analysis needs a different approach.]

### Limitations
- Single MoE model (Qwen3-30B-A3B). Needs validation on architecturally different MoEs.
- Euclidean k-means on 128-d routing frequency vectors is a crude clustering — Jaccard/Hamming with hierarchical clustering, spectral clustering, or expert co-occurrence graphs may perform better
- 672 fitting prompts split into clusters may be insufficient for stable Jacobian estimation
- 0.5 transport cosine threshold is not principled — derived from "above random" rather than functional criterion
- Fitting and primary evaluation on WikiText — cross-domain results are the real test

### Future Work
- Apply to Nemotron 120B (12B active) — 10x scale increase
- Spectral clustering on expert co-occurrence graphs
- End-to-end: metacognitive memory module running on a MoE model with path-conditioned workspace verification
- Per-path ghost analysis — do different routing paths have different ghost dimensions?

## 6. Conclusion

[Frontier models are MoE. Workspace analysis requires path conditioning. Here's the tool.]

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

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) who diagnosed the MoE J-lens failure (12% transport cosine), identified the path-conditioned approach based on cross-expert Jacobian orthogonality, and implemented the pipeline. See Author Contributions. All results verified through the Agni adversarial review protocol.
