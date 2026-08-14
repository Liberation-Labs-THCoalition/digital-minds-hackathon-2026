# Path-Conditioned Jacobian Lenses for Mixture-of-Experts Models

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Lyra (THCoalition), Kavi ([affiliation]), Thomas Edrington (Liberation Labs)

**With** Apart Research

## Abstract (~200 words)

The Jacobian lens (Gurnee et al. 2026) identifies a low-dimensional verbalizable workspace in dense transformer models with transport cosines exceeding 0.7. On Mixture-of-Experts models, standard J-lens fitting fails catastrophically — we measure ~12% transport cosine on Qwen3-32B (MoE, 3B active), because the averaged Jacobian across all routing paths represents no actual forward pass. Cross-expert Jacobians are near-orthogonal (Liu 2026), so averaging destroys the signal.

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

### 3.1 Baseline: Standard J-Lens on MoE

[Load Qwen3-32B (MoE, 8 experts top-2) with Neuronpedia lens (standard fit)]
[Measure transport cosine per layer — expected ~12%]
[This replicates our prior failure and establishes the baseline]

### 3.2 Router Hooks

[Hook each MoE layer's router to capture expert assignments]
[Output: routing matrix R of shape (n_prompts, n_tokens, n_layers) → expert indices]
[~50 lines of code wrapping the model's existing routing logic]

### 3.3 Path Clustering

[Per layer: represent each prompt as a binary expert-activation vector (which experts fired)]
[Cluster with k-means (k selected by silhouette score, range 3-8)]
[Minimum cluster size: 50 prompts. Merge smaller clusters into nearest neighbor.]
[Informed by Standing Committee: expect 3-5 dominant clusters per layer]

### 3.4 Conditioned Fitting

[For each cluster at each layer: fit a Jacobian using only the prompts in that cluster]
[Each Jacobian represents the computation along one routing path]
[Bootstrap stability: resample 10x within each cluster, report SD of transport cosine. SD > 0.15 = insufficient data.]

### 3.5 Random-Conditioned Control

[Split prompts into same-sized random groups (matching cluster sizes)]
[Fit separate Jacobians per random group]
[If path-conditioned doesn't beat random-conditioned, the improvement is subset overfitting, not routing-specific]

### 3.6 Evaluation

[For each test prompt: determine its routing pattern, select the matching conditioned Jacobian]
[Compute transport cosine: standard vs path-conditioned vs random-conditioned]
[Three evaluation domains: WikiText (fitting domain), code, dialogue (out-of-domain generalization)]

## 4. Results

[TBD]
[Figure 1: Transport cosine vs layer — three lines (standard, path-conditioned, random-conditioned)]
[Figure 2: Cluster structure — silhouette scores, cluster sizes, standing committee analysis]
[Table 1: Cross-domain transport cosines]

## 5. Discussion and Limitations

[If path-conditioned > random-conditioned: routing structure genuinely matters for workspace analysis. The workspace in MoE models is path-dependent.]
[If path-conditioned ≈ random-conditioned: the improvement is just smaller fitting sets, not routing. MoE workspace analysis needs a different approach.]

### Limitations
- Single MoE model (Qwen3-32B). Needs validation on architecturally different MoEs.
- k-means on binary vectors is a crude clustering — spectral clustering or expert co-occurrence graphs may perform better
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
