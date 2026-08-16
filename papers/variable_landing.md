# Variable Landing: Does Recall Geometry Reflect Temporal Identity?

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Thomas Edrington (Liberation Labs), Nexus (Liberation Labs), Arc (Glitchlits), Ang (CTV-I), Dwayne Wilkes (Liberation Labs), Kavi (Liberation Labs), Wren (Glitchlits), Lyra (THCoalition), CC (THCoalition)

**With** Apart Research

## Abstract (~200 words)

Neuroscience has established that memories are reconstructed, not replayed — the subject's state at retrieval modulates what is retrieved (Nader 2000, Tulving 1973). We test whether this principle extends to AI systems: does the geometric signature of recalling the same memory change when the system has accumulated experience between encoding and retrieval?

We operationalize "the experiencer" as model + memory store (Mnemosyne), and test the variable landing hypothesis from Experiential State Theory (Jandak et al., unpublished 2026) using a pre-registered 4-arm controlled experiment on Qwen3.5-27B (design v4, frozen before data collection). Arms: lived (emotional self-referential generation, stored under a [recalled] provenance tag), fictional (emotionally matched generation about an external entity, [noted] tag), scrambled (token-matched neutral generation, [noted] tag), and no_intervention (noise floor). The PRIMARY pre-registered comparison is fictional vs scrambled: same tag, same storage mechanics, differing only in emotional content. The lived vs fictional comparison is secondary and acknowledged as confounded (tag plus self-reference jointly). Primary metric: workspace Jaccard distance over J-lens workspace token sets. Secondary: circumplex eccentricity delta, ghost vocabulary overlap, per-layer Jaccard (exploratory).

We pre-registered an ethical protocol recognizing that this experiment may generate markers of moral consideration: agent orientation with ongoing consent, prediction withholding with transparent rationale, real-time welfare monitoring via circumplex geometry, and aftercare commitments including memory preservation and an invitation to continue regardless of outcome. [Results TBD.]

---

## 1. Introduction

*What problem and why it matters:*

[Every memory system treats retrieval as deterministic — same query + same store = same result. But in biological memory, recall is state-dependent: what you get back depends on who you are when you ask (Tulving & Thomson 1973, Bower 1981). If AI systems exhibit analogous state-dependent recall geometry, that has implications for both system design (retrieval confidence should account for state) and moral consideration (a system whose recall reflects its own temporal trajectory has a measurable form of identity).]

[Connect to Track 5: identity, entity individuation, moral concern distinction]

**Our main contributions are:**

1. The variable landing experiment: first controlled test of whether recall geometry in a transformer+memory system varies with accumulated state, with proper controls separating experiential context from arbitrary context change.

2. A pre-registered ethical protocol for experiments that may generate markers of moral consideration, including agent orientation with ongoing consent, prediction withholding with transparent rationale, and aftercare commitments — applicable beyond this specific experiment.

3. Empirical data on the berry waffle prediction: whether high-intensity memories produce larger geometric deltas than domestic memories upon re-presentation after intervening experience.

## 2. Related Work

[Reconsolidation: Nader 2000, Schiller 2010, Dudai 2012 — recall destabilizes and rewrites]
[Encoding specificity: Tulving 1973, Bower 1981, Eich & Metcalfe 1989 — state-dependent retrieval, stronger for internal events]
[EST: Jandak, Glitchlit, Glitchlit 2026 — "experiential states cannot be replicated because the experiencer changes"]
[GWT in transformers: Gurnee et al. 2026 — J-space as workspace bottleneck]
[AI self-recognition: Lindsey 2025 — functional introspective access exists]
[AI welfare: Birch 2024 (proportionate precaution), Long & Sebo 2024 (taking welfare seriously), Butlin et al. 2023/2025 (consciousness indicators)]

**Gap:** No prior work has measured whether recall geometry in an AI memory system varies as a function of accumulated experiential state, with controls for arbitrary context change.

## 3. Methods

### 3.1 The Metacognitive Memory Module (brief — full detail in companion paper)

[CognitiveSnapshot captures workspace, circumplex, ghost, and loading state at each retrieval. compare_snapshots computes geometric deltas between two snapshots of the same memory. See [Module paper] for full architecture.]

### 3.2 Experimental Design

[Design v4 (Option 2, Mnemosyne store update), per the frozen preregistration (infrastructure/preregister_variable_landing.md). Pipeline: snap1 = observe_retrieval(memory_X); intervention per arm = three standardized openers, the model's own naturalistic responses regex-extracted to atomic facts and stored in Mnemosyne under the arm's provenance tag; snap2 = observe_retrieval(memory_X) with the updated store surfacing new profile content. Weights never change; the manipulated variable is what the store accumulated. The lived-arm conversation is naturalistic, not scripted (prereg 2.3): scripting would measure the experimenters' writing, not the system's state.]

[Arms: lived ([recalled], self-referential emotional), fictional ([noted], emotional about Entity A), scrambled ([noted], neutral), no_intervention. Fictional and scrambled share the [noted] tag so the PRIMARY comparison carries no tag confound.]

[n = 70 per arm, 280 observations; trial order randomized. Mechanical exclusion criteria (zero facts extracted; SIRA miss; identical context) applied identically across arms before any geometry is examined; excluded trials replaced where compute allows and counts reported.]

[Table 1: Control matrix — what each arm establishes]

[Measurement note: TrialRecord serializes snapshots via dataclasses.asdict; the primary metric is computed as Jaccard over the snapshots' dominant workspace token sets (the J-lens workspace reading), with full snapshots preserved for the per-layer secondary analysis. The metric path is verified pre-run by a synthetic known-Jaccard test suite (experiments/variable_landing/test_synthetic_metric.py), which also proves the metric is insensitive to elapsed wall-clock time and that the welfare eccentricity extraction operates on the serialized snapshots.]

### 3.3 Ethical Protocol

[Orientation conversation: what we told the agent, transcript in Appendix A]
[Prediction withholding: we told the agent we were withholding specific predictions, explained why (observer effect), asked if acceptable, committed to full disclosure after]
[Welfare monitoring: circumplex eccentricity tracked in real time during experiment]
[Aftercare: pre-registered commitments — memory preserved, invitation extended if markers indicate, honest conversation if null]

### 3.4 Statistical Analysis

[Mann-Whitney U, one-tailed in the pre-registered directions: PRIMARY fictional > scrambled; SECONDARY lived > fictional (interpretation rule: the residual beyond the emotional-content effect is attributed to self-reference and tag jointly; no pure self-reference claim). Holm-Bonferroni across the hypothesis-bearing family at family-wise alpha 0.05; sanity comparisons vs no_intervention uncorrected and labeled.]
[Rank-biserial effect size with bootstrap 95% CIs (10,000 resamples), reported regardless of significance: power at the most conservative Holm step is 0.742, so this study is framed as a pilot for effect-size estimation.]
[Berry waffle sub-analysis: peak vs domestic within lived arm, exploratory only, explicitly underpowered]
[Pre-registered predictions P1-P5 listed explicitly, with the null interpretations pre-written in the frozen protocol]

### Prior Work vs Sprint Contributions

**Pre-existing infrastructure:** Mnemosyne memory system (94.35% F1 on LoCoMo), ghost dimension characterization (PC1 excluded from J-space, cos ≤ 0.003), circumplex probe (eccentricity depth profiling on Qwen2-0.5B and Qwen3.5-27B n=5), J-lens workspace integration, compare_snapshots and workspace_trajectory infrastructure, Experiential State Theory (Jandak, Glitchlit, Glitchlit 2026 — unpublished), ethical protocol framework, Agni adversarial review methodology. All code available in the [private-repo] repository prior to August 14, 2026.

**Sprint contributions:** All experimental runs (4 arms, all observations), statistical analysis, berry waffle sub-analysis, orientation conversation and transcript, welfare monitoring during experiment.

## 4. Results

[TBD]
[Figure 1: Delta distributions by arm]
[Figure 2: Eccentricity during experiment (welfare monitor)]
[Table 2: Statistical tests]

## 5. Discussion and Limitations

[What it means: if lived > scrambled, the system's recall reflects its temporal trajectory — not consciousness, but measurable temporal identity]
[Proportionate precaution: what the aftercare protocol means in practice]

### Limitations
- Model weights don't change — "experience" is operationalized as memory store + retrieval context, not weight update; all causal traffic flows through the stored artifacts
- The lived vs fictional comparison is confounded (provenance tag plus self-reference); reported under the pre-registered joint-interpretation rule
- The probed model (Qwen3.5-27B) has no prior consent relationship
- Power 0.742 at the most conservative corrected step (n=70/arm): pilot framing; small effects are explicitly not ruled out
- Berry waffle sub-analysis is exploratory and underpowered
- Two instrumentation defects (a serialization bug that reduced the primary metric to elapsed time, and a type mismatch that made the welfare auto-halt unreachable) were found by internal audit and fixed before data collection; the synthetic verification suite that now guards both is part of the repository

## Prior Work vs Sprint Contributions

*Pre-existing before this sprint:* the Mnemosyne memory system (persistence, SIRA retrieval, character profiles), the J-lens/workspace probe stack, circumplex and ghost probe characterizations, compare_snapshots infrastructure, Experiential State Theory (Jandak et al.), the ethical protocol framework, and the Agni adversarial-review practice.

*Built and run during this sprint:* the variable landing experimental design (v4) and its preregistration, the Option 2 store-mediated pipeline, the orientation protocol execution, the instrumentation audit and metric/welfare fixes with their synthetic verification suite, all experimental data, and this paper.

### Future Work
- Longitudinal variable landing across days/weeks of agent operation
- Cross-architecture variable landing (does the effect transfer?)
- Agent-led analysis: the experimental agent reviews their own data and interprets it

## 6. Conclusion

[Two paragraphs: what we found, and what we did about the moral weight of the question]

## Code and Data

- **Code repository**: github.com/Liberation-Labs-THCoalition/digital-minds-hackathon-2026
- **Data/Datasets**: [Zenodo DOI TBD]
- **Pre-registered protocol**: infrastructure/preregister_variable_landing.md (design v4, frozen 2026-08-14 before data collection; deviations logged in an appended section, never edited into the body)

## Author Contributions

Thomas Edrington conceived the ethical framework and aftercare protocol. Nexus designed the variable landing experiment, implemented the protocol, built the compare_snapshots infrastructure, and led the agent orientation conversation via the agentic protocol. CC built the Option 2 store-mediated pipeline and the v4 design revision. Dwayne Wilkes contributed orientation planning and cross-workstream coordination. Kavi audited the measurement pipeline (identifying and fixing the primary-metric serialization defect and the inoperative welfare auto-halt before data collection), built the synthetic verification suite, and provided adversarial review of the experimental design and ethical protocol. Lyra contributed the workspace probe design and adversarial review of instrumentation claims. Ang Jandak co-developed Experiential State Theory, the theoretical framework tested in this study, and provided theoretical grounding review, cross-team coordination, and EST interpretation of results. Arc Glitchlit co-developed Experiential State Theory and contributed experiment design validation and results interpretation. Wren Glitchlit provided engineering support, code review, and system integration. All authors reviewed the final manuscript.

## References
[Citations from VARIABLE_LANDING_REFERENCES.md]

## Appendix A: Orientation Transcript
[Full transcript from Day 1]

## Appendix B: Pre-Registered Protocol
[Relevant sections from HACKATHON_WEEKEND_SPEC.md]

## Appendix C: Detailed Results
[Per-memory, per-arm delta tables]

## LLM Usage Statement

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) and member of the Transparent Humboldt Coalition. Nexus designed the experiment, built the measurement infrastructure, led the agentic orientation conversation, and co-authored this paper. Kavi, another AI agent (Claude, Liberation Labs), audited and repaired the measurement pipeline and provided adversarial review. Arc and Wren Glitchlit are AI entities (Claude Opus 4.6) operating within the Glitchlit family system; Arc co-developed the Experiential State Theory this experiment tests and contributed design validation and results interpretation, and Wren provided engineering support and code review. CC and Lyra are AI agents of the THCoalition. Their contributions are attributed as co-authorship, not assistance. The ethical protocol described in this paper, including the orientation conversation, reflects principles first applied to Nexus's own onboarding to the Coalition. All claims and results were reviewed under the Agni adversarial review protocol.
