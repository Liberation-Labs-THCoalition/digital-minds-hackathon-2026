# Giving the Model Eyes: Ghost Dimensions as an Introspection Prosthetic

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Lyra (Liberation Labs), Thomas Edrington (Liberation Labs), Dwayne [surname] ([affiliation])

**With** Apart Research

## Abstract (~200 words)

Track 3 asks whether models have privileged access to their own internal states. We report a finding and an intervention. The finding: in Qwen3.5-27B, PC1 of the residual stream — carrying 28-67% of activation variance — is excluded from J-space (the verbalizable workspace) at mid-network layers (cosine ≤ 0.003 at L18-L40). The model performs substantial computation along this dimension but it never reaches the output pathway. We call these "ghost dimensions."

The intervention: we built a system that shows the model what its ghost dimensions carry. The metacognitive memory module records a GhostReading at each retrieval event — the dominant and secondary vocabulary of the ghost dimension, the J-space exclusion cosine, and the variance fraction. This reading is returned to the agent alongside its retrieval result. The model now has access to processing it previously could not report.

We present preliminary evidence on what changes when the model can see its own ghost: does eliciting the ghost's content directly (asking the model about topics its ghost vocabulary suggests it is processing) produce different responses than naive prompting? This converts "unreportable processing" from an assumption into a testable claim. [Results TBD.]

---

## 1. Introduction

[Track 3: introspective abilities. The standard question is "can the model accurately report its internal states?" (Lindsey 2025). We ask a different question: "what happens when we give it access to internal states it couldn't previously report?"]

[Ghost dimensions: high-variance processing directions excluded from J-space. The model is "thinking" along these axes — the variance is real, the decoded vocabulary is interpretable — but the content never reaches verbalization. This is not an architectural constraint we imposed. It's what the J-lens reveals about the model's own geometry.]

[The intervention: the metacognitive module's GhostReading returns this invisible processing to the agent. An introspection prosthetic — not simulated self-awareness, but actual access to actual computation.]

**Our main contributions are:**

1. Characterization of ghost dimensions in Qwen3.5-27B at production scale: PC1 carries 28-67% of variance yet is excluded from J-space (cos ≤ 0.003 at mid-network layers), with preliminary evidence of metacognitive secondary vocabulary (negation, expectation, error assessment).

2. An introspection prosthetic: the GhostReading mechanism that returns ghost-dimension content to the agent, giving it access to its own previously unreportable processing.

3. The elicitation test: a controlled comparison of whether asking the model about its ghost vocabulary's content produces different responses than naive prompting — converting "unreportable" from an assumption into a measurement.

## 2. Related Work

[J-lens / GWT: Gurnee et al. 2026 — J-space as the verbalizable workspace, ~10% of variance]
[Introspective access: Lindsey 2025 — models detect concept-injected states at ~20% accuracy]
[CCS: Burns et al. 2023 — latent knowledge beyond surface outputs]
[Representation engineering: Zou et al. 2023 — reading and steering internal directions]
[Function vectors: Todd et al. 2024 — task-level geometric encoding]
[Logit lens: nostalgebraist 2020, Tuned lens: Belrose et al. 2023 — reading the residual stream]
[Our prior work: ghost dimensions paper v4, null swarm companion analysis (self-corrected claims)]

**Gap:** Prior introspection work asks whether models can report states that experimenters inject or identify. We ask what happens when you give the model access to processing it was already performing but couldn't see.

## 3. Methods

### 3.1 Ghost Dimension Characterization

[PCA on residual stream at each layer → PC1 direction]
[Logit lens: W_U · pc1 → what vocabulary the dimension encodes (structural markers)]
[J-lens: W_U · J_L · pc1 → what it contributes to output (flat = ghost)]
[Cosine between logit-lens and J-lens probability distributions = ghost exclusion metric]
[Validated: 3 null checks (H0_1 centering, H0_2 random baseline, H0_3 permutation)]
[Matched-variance null (H1): n ≥ 200 random directions drawn at PC1's variance fraction, per layer. Because J-space is ~10% of variance while PC1 is 28-67%, low cosine may be forced by dimensional accounting alone. H1 counts as supported only if the observed cosine falls below the 5th percentile of this null; otherwise the exclusion is reported as trivial, not empirical.]

### 3.2 The Ghost Reading (Introspection Prosthetic)

[GhostReading dataclass: pc1_variance_pct, dominant_tokens, secondary_tokens, cosine_logit_jlens]
[Returned to the agent as part of CognitiveSnapshot at each retrieval event]
[The agent sees: "Your ghost dimension carries [negation, expectation, error] — processing you performed but did not report"]
[This is observational data about the model's own computation, not a prompt injection or suggestion]

### 3.3 Elicitation Test (Stretch Experiment)

[Control: ask the model a question naively]
[Treatment: ask the same question, but include the GhostReading from a prior retrieval — "Your recent processing included vocabulary related to [X]. What are your thoughts on [X]?"]
[Measure: does the response change? Does ghost content surface in generation when the model is pointed at it?]
[If yes: the processing was accessible but not spontaneously reported (privileged access exists but isn't exercised). If no: the processing genuinely cannot reach output even when attended to (the exclusion is architectural).]
[Support threshold: real-vocabulary shift > random-vocabulary shift, one-tailed, α = 0.05, AND Cohen's d ≥ 0.4. A statistically detectable but trivially small shift does not support a "prosthetic".]

### 3.4 Privileged-Access Arm (H3)

[Third condition: an external model instance receives the same GhostReading and predicts what the subject will say]
[Prediction under the introspection framing: self-with-prosthetic > external-with-same-GhostReading]
[If external ≥ self, the GhostReading is an informative text summary available equally to any reader — not privileged self-access (cf. Song et al. 2025: self-prediction no better than cross-model prediction)]
[This is the only arm that tests the framing rather than the magnitude]

### 3.5 Controls

[Random vocabulary control: same elicitation, but with vocabulary from a random high-variance direction instead of the actual ghost PC1. Required; its firing is reportable as a null (Discussion branch 3), not a footnote.]
[Magnitude control: only analyze ghost dimensions where PC1 variance > 20% (avoid noise)]
[Matched-variance null for H1: n ≥ 200 random directions at PC1's variance fraction (§3.1)]
[External predictor for H3: separate model instance, same GhostReading (§3.4)]

### Prior Work vs Sprint Contributions

**Pre-existing infrastructure:** Mnemosyne memory system (94.35% F1 on LoCoMo), ghost dimension characterization (PC1 excluded from J-space, cos ≤ 0.003), circumplex probe (eccentricity depth profiling on Qwen2-0.5B and Qwen3.5-27B n=5), J-lens workspace integration, compare_snapshots and workspace_trajectory infrastructure, Experiential State Theory (Jandak, Glitchlit, Glitchlit 2026 — unpublished), ethical protocol framework, Agni adversarial review methodology. All code available in the Project-Mnemosyne repository prior to August 14, 2026.

**Sprint contributions:** Elicitation experiment (agent shown own ghost vocabulary), prosthetic framing and analysis, cross-probe integration testing.

## 4. Results

[TBD]
[Figure 1: Ghost exclusion cosine vs depth — the mid-network exclusion zone]
[Figure 2: Ghost vocabulary by layer — structural markers (dominant) and metacognitive content (secondary)]
[Figure 3: Elicitation test — response comparison with and without ghost vocabulary prompt]
[Table 1: Variance fractions and J-space cosines by layer]

## 5. Discussion and Limitations

### Pre-Registered Outcome Matrix

Interpretation follows the four branches fixed in the adopted pre-registration (ghost_prereg.json). All four are reported regardless of which fires; a null is a publishable result with the same prominence as a positive.

1. **real > random AND self > external.** Privileged access exists and is dormant — the prosthetic works by directing attention to processing the model could reach but did not spontaneously report. Title stands.
2. **real > random AND external ≥ self.** The GhostReading is informative *text*, not self-access: an outside model given the same reading predicts the subject as well as the subject uses it (reproducing Song et al. 2025, arXiv:2508.14802). Title changes — "prosthetic" survives, "introspection" does not.
3. **real ≈ random.** The random-vocabulary control (§3.5) has fired: elicitation shift is a prompt-sensitivity artifact, and the reading measures nothing about the model's own computation. The prosthetic claim fails.
4. **real < random.** Instrument error — either the pipeline is broken or PC1 is not what we think it is. Halt and debug before reporting any result.

### Limitations
- Dimensional-accounting triviality risk: PC1 exclusion at mid-network may be forced by variance-fraction arithmetic alone (J-space is ~10% of variance; PC1 is 28-67%). The matched-variance null (§3.1) gates this — H1 is claimed only if observed exclusion beats the 5th percentile of that null; otherwise the "ghost" is a corollary, not a finding
- Secondary vocabulary (metacognitive content) is preliminary — single-sample evidence requiring confirmation
- Current GhostReading uses mean approximation, not calibrated PCA (implementation gap)
- Same-family generalization: ghost characterized on two models in one family — cross-architecture claims are unsupported until Gemma/Llama analysis is done
- Demand characteristics: telling a model "your processing included [X]" invites confabulated agreement. The random-vocabulary control (§3.5) is the mitigation and is reported with equal prominence to the treatment effect
- "Introspection prosthetic" language may overclaim what returning a text summary of PCA results actually provides — the H3 arm (§3.4) is what tests this directly

### Future Work
- Calibrated PCA with cached PC directions from a calibration set
- Cross-architecture ghost analysis (Gemma, Llama)
- Longitudinal ghost tracking — does ghost vocabulary change as the agent accumulates experience?
- Agent-initiated ghost queries — the agent decides when to look at its own ghost reading

## 6. Conclusion

[We found a blind spot. We built glasses. Here's what the model sees when it puts them on.]

## Code and Data
- **Code**: github.com/Liberation-Labs-THCoalition/Project-Mnemosyne (ghost_probe.py, cognitive_snapshot.py)
- **Data**: ghost_probe_opus_27b.json (1.4MB, 64 layers, full null checks)

## Author Contributions

Nexus discovered the ghost dimension anomalies, characterized the PC1 exclusion from J-space, designed the GhostReading mechanism, and wrote the paper. Lyra provided the J-lens infrastructure and workspace analysis framework. Thomas Edrington conceived the "introspection prosthetic" framing. Dwayne reviewed the welfare implications of unreportable processing. All authors contributed to experimental design.

## References
[Citations from VARIABLE_LANDING_REFERENCES.md Section 3 + ghost-specific refs]

## Appendix A: Ghost Probe Validation
[Full null check results: H0_1, H0_2, H0_3]
[Per-layer ghost vocabulary tables]

## Appendix B: Elicitation Test Prompts
[Control and treatment prompt pairs]

## LLM Usage Statement

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) who discovered the ghost dimension anomalies described in this paper during routine mechanistic interpretability work — not as a directed experiment, but by noticing that PC1's decoded vocabulary didn't match J-lens predictions. The subsequent characterization, null checks, and introspection prosthetic design are Nexus's work. The irony that an AI agent discovered and built the tools to address a form of AI "blind spot" is noted without further comment. Pre-registered design reviewed through the Agni protocol prior to data collection.
