# Path-Conditioned MoE J-Lens — Literature Review

Compiled 2026-08-11. For the hackathon extension of our MoE interpretability work.

## Problem Statement

Standard J-lens (Gurnee et al. 2026) computes per-token Jacobians to identify which residual-stream directions are verbalizable, yielding a ~25-dim "J-space" bottleneck on dense models. (Note, corrected 2026-08-16: Gurnee et al. report no transport-cosine or reconstruction-fidelity figure. They validate the lens by intermediate-concept recovery (normalized pass@k AUC) and by causal intervention (ablation KL, coordinate-swap success). An earlier version of this line asserted ">0.7 transport cosine on dense models"; that number is not in the source.) On MoE models this fails (~12% transport cosine on Qwen3-30B-A3B, 128 experts top-8) because the averaged Jacobian doesn't represent any actual forward pass — different prompts route through different experts, so the average is a weighted sum of paths the model never takes simultaneously. The proposed fix: fit Jacobian lenses conditioned on routing decisions, treating each expert path or path cluster as a separate computational regime.

---

## 1. MoE Routing and Expert Specialization

- **Wang, Xu, Shen, Su, Huang & Zhu (2026).** "The Illusion of Specialization: Unveiling the Domain-Invariant 'Standing Committee' in Mixture-of-Experts Models." arXiv:2601.03425.
  A compact coalition of routed experts ("Standing Committee") captures the majority of routing mass across domains, layers, and routing budgets. Specialization in MoE is far less pervasive than assumed — the Standing Committee anchors reasoning structure and syntax while peripheral experts handle domain-specific knowledge. *Relevance: Path conditioning must account for the fact that a core expert group is always active; the interesting variation is in peripheral expert selection.*

- **Ye, Yuan & Sharkey (2026).** "Polysemantic Experts, Monosemantic Paths: Routing as Control in MoEs." arXiv:2604.17837.
  Parameter-free decomposition splits each MoE layer's hidden state into a control signal (driving routing) and an orthogonal content channel (invisible to router). Individual experts are polysemantic but expert *paths* are monosemantic — the same token follows distinct trajectories depending on semantic function. *Relevance: Directly supports path-conditioned analysis. The natural unit of interpretability in MoEs is the trajectory, not the expert — exactly the argument for conditioning J-lens on routing decisions.*

- **Ternovtsii & Bilak (2026a).** "Geometric Routing Enables Causal Expert Control in Mixture of Experts." arXiv:2604.14434.
  Cosine-similarity routing in a low-dimensional metric space makes expert specialization directly inspectable. 15% of experts are monosemantic specialists; causal interventions confirm labels (steering toward a temporal expert centroid increases P(temporal) by +321%). *Relevance: Provides geometric vocabulary for characterizing expert paths and validates that routing decisions are causally meaningful.*

- **Ternovtsii & Bilak (2026b).** "Equifinality in Mixture of Experts: Routing Topology Does Not Determine Language Modeling Quality." arXiv:2604.14419.
  Across 62 controlled experiments, five cosine-routing variants are statistically equivalent within 1 PPL. Routing topology is quality-neutral. *Relevance: Since routing topology doesn't determine quality but does determine which path the Jacobian should be computed along, path-conditioned J-lens targets the interpretability structure that routing creates, not the performance structure.*

- **Ahrac, Hochwald & Geva (2026).** "Routers Learn the Geometry of Their Experts: Geometric Coupling in Sparse Mixture-of-Experts." arXiv:2605.12476.
  Higher router scores predict stronger expert neuron activations — routing decisions are mirrored inside the selected expert. Router-expert directions accumulate the same routed token history. *Relevance: Router weights are not arbitrary selectors but reflect geometric alignment with expert function, validating that conditioning on routing captures real computational structure.*

- **Tian, Xu & Li (2026).** "Beyond Geometric Complementarity: Coherent Overlap in Sparse Mixture-of-Experts Routing." arXiv:2607.28308.
  Expert subspaces overlap substantially, yet actual routes explain token representations better than matched alternatives. Selected experts outperform strongest unselected rival in all 39 factorial cells tested. *Relevance: Despite overlap, the specific routing decision is informative — the path matters even when expert subspaces are not cleanly separated.*

- **Wang, Hayou & Nalisnick (2026).** "The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise." arXiv:2604.09780.
  Expert usage similarity follows from hidden state similarity (linear routers make this tautological). Prompt-level routing does not predict rollout-level routing; deeper layers show near-identical expert activation across unrelated inputs. *Relevance: Warns against assuming routing = domain specialization. Path conditioning should target the geometric structure routers actually encode, not projected domain labels.*

## 2. MoE Interpretability Methods

- **Chaudhari et al. (2026).** "MoE Lens — An Expert Is All You Need." arXiv:2603.05806.
  Extends LogitLens to MoE by separately projecting individual expert outputs, weighted combinations, and top-expert + residual. Single top-expert + residual approximates full ensemble at cos ~0.95. *Relevance: Uses LogitLens (correlation-based), not Jacobian (causal). Demonstrates that per-expert decomposition is tractable but leaves the causal gap our J-lens approach fills.*

- **Lu, Modarressi, Liu & Schutze (2026).** "Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models." arXiv:2606.03780.
  Adapts causal tracing to MoE by testing whether clean MoE-block or clean expert-level updates restore factual recall. Layer-level recovery and single-expert localization can come apart: Qwen3 admits single-expert intervention, Mixtral requires routed expert coalitions. *Relevance: Directly demonstrates that causal analysis on MoE must be expert-aware — the same principle our path-conditioned J-lens applies to Jacobian computation.*

- **Salomone, Gandhi & Asaria (2026).** "How Modular Is a Frontier Mixture-of-Experts? A Pre-registered Causal Test in Which Apparent Expert Modularity Mostly Dissolves." arXiv:2606.25092.
  Pre-registered ablation study on Command A+ (218B/25B active, 128 experts). Of six hypothesized expert families, only Arabic survives as a clean selective module. *Relevance: Functional modularity is rare and measurement-dependent — path conditioning can't assume clean expert-to-function mappings.*

- **Chaudhari, Nuer & Thorstenson (2025).** "Sparsity and Superposition in Mixture of Experts." arXiv:2510.23671.
  Network sparsity (active/total experts ratio) better characterizes MoEs than feature sparsity. Greater network sparsity yields greater monosemanticity. *Relevance: Sparser MoE models may be easier targets for path-conditioned J-lens since each path traverses fewer experts.*

- **Park, Ahn, Kim & Kang (2024).** "Monet: Mixture of Monosemantic Experts for Transformers." arXiv:2412.04139. ICLR 2025.
  Scales to ~250K experts per layer via expert decomposition; individual experts become monosemantic by construction. Interpretability surpasses SAE-based approaches. *Relevance: If experts are monosemantic by design, path-conditioned J-lens would produce cleaner per-path workspaces. Monet-style architectures are the best-case scenario for our approach.*

- **Tsao, Lin & Wang (2026).** "Is MoE Routing a Huffman Code? Discovering the Frequency-Diversity Law in Chain-of-Thought." arXiv:2607.20427.
  Common tokens get sparse expert allocation; rare/complex tokens invoke high-diversity expert committees. Load-balancing can impose functional redundancy that masks this structure. *Relevance: Path conditioning should stratify by routing diversity — simple tokens (few experts) may need different J-lens fitting than complex tokens (many experts).*

## 3. Jacobian-Based Interpretability

- **Gurnee, Sofroniew, Pearce et al. (2026).** "Verbalizable Representations Form a Global Workspace in Language Models." arXiv:2607.15495. Anthropic.
  The J-lens paper. Uses averaged Jacobian of final-layer state w.r.t. intermediate layers to identify ~25-dim "J-space" bottleneck matching GWT predictions. Validated by intermediate-concept recovery (normalized pass@k AUC over six prompt sets shipped in the repo) and causal intervention (ablation KL; coordinate-swap success 54%/70%/70% on Haiku/Sonnet/Opus 4.5). Reports NO transport-cosine or reconstruction-fidelity metric; §A.6 notes the J-lens is deliberately the poorest predictor of the output distribution, "a feature rather than a defect". *Relevance: The method we're extending. Works on dense models; breaks on MoE because the Jacobian depends on which experts are active.*

- **nostalgebraist (2020).** "Interpreting GPT: The Logit Lens." LessWrong blog post.
  Projects intermediate residual-stream activations through the output unembedding to decode per-layer token predictions. Correlation-based, not causal. *Relevance: Predecessor to J-lens. The logit lens → tuned lens → J-lens trajectory shows progressive movement toward causal interpretability; our MoE extension continues this arc.*

- **Belrose, Ostrovsky, McKinney, Furman, Smith, Halawi, Biderman & Steinhardt (2023).** "Eliciting Latent Predictions from Transformers with the Tuned Lens." arXiv:2303.08112. NeurIPS 2023.
  Trains per-layer affine probes to decode hidden states into vocabulary distributions. More reliable than logit lens, handles basis drift. *Relevance: Intermediate step between logit lens and J-lens. Demonstrates that per-layer fitting (analogous to what we propose per-path) improves interpretability.*

- **Fernando & Guitchounts (2026).** "Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology." arXiv:2605.14258.
  Full Jacobian eigendecomposition across production LLMs reveals a monotonic spectral gradient through depth — from rotation-dominated early layers to near-symmetric late layers — with a cumulative low-rank bottleneck. *Relevance: Provides the spectral framework for understanding how Jacobian structure varies across layers, which path conditioning must respect.*

- **Szablewski & Masiak (2025).** "Activation Transport Operators." arXiv:2508.17540.
  Linear maps from upstream to downstream residuals, evaluated via SAE decoder projections. Tests whether features are linearly transported or nonlinearly synthesized between layers. *Relevance: Complementary to J-lens — ATOs test linear preservation of specific features across layers, which could validate whether path-conditioned J-space vectors transport faithfully within a given routing path.*

## 4. Conditional/Stratified Model Analysis

- **Liu (2026).** "Geometric Asymmetry in MoE Specialization: Functional Decorrelation and Representational Overlap." arXiv:2605.16349.
  Jacobian-PCA-Grassmann framework for analyzing MoE layers in function space and representation space. Experts show strong functional decorrelation (near-zero cross-expert Jacobian alignment) while representations partially overlap. Top-k routing induces sharper functional separation vs. soft routing. *Relevance: Directly applies Jacobian analysis to MoE experts. The finding that cross-expert Jacobians are near-orthogonal is the mathematical reason averaged J-lens fails — and the mathematical justification for path-conditioned fitting.*

- **Zhu (2026).** "From Expert Reduction to Behavioral Divergence: Tracing Numerical State through Sparse MoE Inference." arXiv:2607.28097.
  Different expert reduction orders produce different finite-precision states; perturbed hidden states change downstream routing at discrete decision boundaries. 720 orderings produce 10+ continuation basins. *Relevance: Even within a fixed routing decision, the order of expert aggregation matters. Path conditioning must also consider aggregation semantics, not just expert identity.*

## 5. Global Workspace Theory and MoE

- **Baars (1988).** *A Cognitive Theory of Consciousness.* Cambridge University Press.
  Original GWT: consciousness as centralized shared workspace broadcasting across specialized processors. *Relevance: The theoretical framework J-lens validates in dense models. In MoE, the "specialized processors" are literal experts — GWT's architecture maps more directly onto MoE than dense transformers.*

- **Dehaene & Changeux (2011).** "Experimental and Theoretical Approaches to Conscious Processing." *Neuron*, 70(2), 200-227.
  Global Neuronal Workspace: predicts late amplification, ignition, and prefrontal-parietal synchronization. *Relevance: The "ignition" concept — sudden coherent broadcast — may have a MoE analog in routing convergence across layers.*

- **Goldstein & Kirk-Giannini (2024).** "A Case for AI Consciousness: Language Agents and Global Workspace Theory." arXiv:2410.11407.
  If GWT is correct, artificial language agents might already satisfy its conditions. *Relevance: MoE architectures are structurally closer to GWT than dense models (specialist modules + broadcast workspace). If path-conditioned J-lens reveals per-path workspaces, GWT's predictions become testable at the routing level.*

---

## Summary: The Gap We Fill

The literature establishes that:
1. Expert paths, not individual experts, are the natural unit of MoE interpretability (Ye/Yuan/Sharkey)
2. Cross-expert Jacobians are near-orthogonal (Liu), explaining why averaged J-lens fails
3. Routing decisions are causally meaningful (Ternovtsii & Bilak; Lu et al.) and geometrically grounded (Ahrac et al.)
4. Existing MoE interpretability uses LogitLens (MoE Lens) or causal tracing, but nobody has computed path-conditioned Jacobian lenses
5. GWT's specialist-processor architecture maps more naturally onto MoE than dense models

**What's missing:** A Jacobian lens that conditions on routing decisions to produce per-path verbalizable workspaces. This is what path-conditioned MoE J-lens provides.
