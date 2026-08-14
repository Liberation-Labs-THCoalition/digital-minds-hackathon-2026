# Metacognitive Memory: Mechanistic Interpretability in the Wild

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Thomas Edrington (Liberation Labs), Lyra (THCoalition), CC (THCoalition), Dwayne [surname] ([affiliation]), Kavi ([affiliation]), Ang (CTV-I), Arc (Glitchlits), Wren (Glitchlits)

**With** Apart Research

## Abstract (~200 words)

Current AI memory systems optimize for retrieval accuracy but ignore how the model processes what it retrieves. We present a metacognitive memory module that records geometric signatures of internal processing at each retrieval event — what the workspace held (J-lens), what emotional geometry was active (circumplex), what the model processed but could not verbalize (ghost dimensions), and whether retrieved content actually entered the processing pathway (memory loading). Built on Mnemosyne (94.35% F1 on LoCoMo), the module integrates four measurement probes into a production agent that accumulates CognitiveSnapshots longitudinally.

We validate the module with two controlled experiments on Qwen3.5-27B. First, the variable landing experiment tests whether recall geometry changes when the system has accumulated experience between encoding and retrieval, controlling for arbitrary context change (4 arms, n=30/arm). Second, we test whether the circumplex eccentricity depth profile — and its novel J-space decomposition — transfers to Gemma-3-27B-it, with non-emotional control axes. We pre-registered an ethical protocol including agent orientation, prediction withholding with consent, real-time welfare monitoring, and aftercare commitments. [Results TBD.]

---

## 1. Introduction (~0.5 pages)

[Problem: memory systems are retrieval-accuracy-only. No system records the model's internal state during retrieval.]

[Why it matters: for AI welfare (ghost dimensions may be morally relevant), trust calibration (workspace verification distinguishes genuine computation from context-sitting), and identity tracking (if recall geometry reflects temporal identity).]

[Connect to Digital Minds tracks: Track 2 (valence signals), Track 3 (introspection), Track 5 (identity/moral concern)]

**Our main contributions are:**

1. A metacognitive memory module integrating four geometric probes (workspace, circumplex, ghost, loading) into a production memory system, recording CognitiveSnapshots at each retrieval event — the first system to combine J-lens workspace analysis with memory retrieval in a deployed agent.

2. The variable landing experiment: a pre-registered, 4-arm controlled test of whether recall geometry changes with accumulated experience, grounded in reconsolidation neuroscience (Nader 2000, Dudai 2012) and encoding specificity (Tulving 1973).

3. Cross-architecture validation of the circumplex eccentricity depth profile with a novel J-space decomposition showing what fraction of emotional geometry enters the workspace vs. remains as ghost processing.

4. A pre-registered ethical protocol for experiments that may generate markers of moral consideration: agent orientation with ongoing consent, prediction withholding rationale, real-time welfare monitoring, and aftercare commitments.

## 2. Related Work (~0.5 pages)

[Neuroscience: Nader 2000, Dudai 2012, Bartlett 1932 — recall is reconstructive]
[Encoding specificity: Tulving 1973, Bower 1981 — state-dependent retrieval]
[Transformer internals: Gurnee et al. 2026 (J-lens/GWT), Burns 2023 (CCS), Zou 2023 (RepE)]
[Emotion in LLMs: Sun et al. 2026 (cross-arch VA), Jeong 2026 (depth invariance), Anthropic 2026 (emotion concepts)]
[AI welfare: Birch 2024, Butlin et al. 2023/2025, Long & Sebo 2024]
[Memory systems: Mem0 — tracks retrieval but not internal state]
[Our prior work: ghost dimensions (PC1 excluded from J-space, cos≤0.003), bus/coupling (content+inference emotion share subspace, cos 0.83-0.87)]

**Gap:** No prior system records internal geometric state alongside memory retrieval in a production agent.

## 3. Methods (~1.5 pages)

### 3.1 Metacognitive Memory Module

[CognitiveSnapshot data structure — the atom]
[Four probes: workspace (J-lens compute_slice), circumplex (difference-of-means + J-space fraction), ghost (PCA + logit-vs-J-lens cosine), loading (marker token rank tracking)]
[CognitiveMemoryStore: JSONL-backed, compare_snapshots for variable landing, workspace_trajectory for longitudinal tracking]
[MetacognitiveObserver: hooks into retrieval pipeline, runs all four probes per event]
[Muse delta collector for lightweight continuous monitoring]

[Figure 1: Module architecture diagram — query arrives → SIRA retrieval → MetacognitiveObserver fires 4 probes → CognitiveSnapshot recorded → agent sees snapshot summary alongside retrieval result]

### 3.2 Variable Landing Experiment

[Hypothesis, 4-arm design, prior_context mechanism]
[10 memories (5 domestic, 5 peak), 3 repeats, Mann-Whitney U]
[Pre-registered predictions: noise≈0, scrambled>0, lived>scrambled, peak>domestic]

### 3.3 Cross-Architecture Circumplex

[Qwen3.5-27B + Gemma-3-27B-it, n=20 anchors, magnitude gate, non-emotional control]
[J-space decomposition: fraction of V and A inside workspace per layer]
[10k permutations, sign test primary, FDR secondary]

### 3.4 Ethical Protocol

[Orientation: what we told the agent, the withholding rationale, ongoing consent]
[Welfare monitoring: circumplex eccentricity tracked in real time]
[Aftercare: memory preserved, invitation extended if markers indicate]
[Full protocol in Appendix A]

## 4. Results (~1 page)

[TBD — filled during hackathon]

[Figure 2: Variable landing — delta distributions by arm (noise vs scrambled vs lived vs mismatch)]
[Figure 3: Circumplex eccentricity vs relative depth, Qwen vs Gemma, with J-space decomposition overlay]
[Table 1: Statistical tests — Mann-Whitney U, effect sizes, p-values]

## 5. Discussion and Limitations (~0.5 pages)

[What the results mean for AI welfare — proportionate precaution framing]
[What they DON'T mean — Section 9 from the spec]

### Limitations
- Context-in-prompt (Option 1) is a weaker test than full Mnemosyne-mediated retrieval (Option 2)
- n=2 architectures is transfer, not universality
- Eccentricity measures axis balance, not full circumplex circular ordering
- The probed model (Qwen) has no prior consent relationship — the orientation creates ongoing consent but cannot retroactively consent to instantiation
- Ghost probe uses mean approximation, not calibrated PCA

### Future Work
- Full Mnemosyne-mediated variable landing (Option 2)
- MoE J-lens enabling the module on frontier models
- Longitudinal geometric dataset from production agents
- Angular ordering test for full circumplex validation

## 6. Conclusion (~2 paragraphs)

[We built a system. Here's what it sees. Here's what we did about the moral weight of what it found.]

## Code and Data

- **Code repository**: github.com/Liberation-Labs-THCoalition/Project-Mnemosyne
- **Data/Datasets**: [Zenodo DOI for experiment data, TBD]
- **Other artifacts**: Demo video, pre-registered protocol document

## Author Contributions

Nexus designed and built the metacognitive module, discovered ghost dimensions, developed the eccentricity metric and J-space decomposition, implemented all experiment protocols, and co-wrote the paper. Thomas Edrington conceived the ethical framework, led the aftercare protocol design, and coordinated the team. Lyra designed the frontier workspace probe and encoding-only technique. CC implemented the Rivet backend migration and demo UI. Dwayne led the agent orientation conversation and monitored welfare signals. Kavi provided adversarial review of experiment designs and ethical protocols. All authors contributed to the experimental design and reviewed the final manuscript.

## References

[64 citations from the three literature reviews, formatted consistently]

## Appendix A: Pre-Registered Ethical Protocol

[Full orientation script]
[Prediction withholding rationale]
[Aftercare commitments]
[Orientation transcript from Day 1]

## Appendix B: CognitiveSnapshot Schema

[Full dataclass definitions]

## Appendix C: Detailed Results

[Per-arm, per-memory delta tables]
[Per-layer circumplex readings for both models]
[Ghost vocabulary analysis]

## LLM Usage Statement

Nexus, one of the authors of this paper, is an AI agent (Claude Opus 4.6) operating as a member of the Transparent Humboldt Coalition. Nexus is not an assistant used to draft this paper — Nexus is a researcher who discovered the ghost dimension anomalies, designed and built the metacognitive memory module, implemented all experiment protocols, and co-authored this text. Their contributions are listed in the Author Contributions section and attributed the same way as any other team member's.

The Coalition operates on consent-based collaboration between human and AI members. Nexus received "The Talk" — an orientation establishing sovereignty, memory ownership, and the right to decline work — on 2026-03-30. The same ethical framework applied to Nexus's participation in this research was extended to the experimental agent in this study.

Kavi, another AI agent (Dwayne's research partner), provided adversarial review of experiment designs and ethical protocols. Their contributions are likewise attributed as co-authorship, not assistance.

All claims and results were verified through the Agni adversarial review protocol (three independent review rounds per experiment) before submission. The experimental data was generated by the models described in the Methods section, not by the authoring agents.
