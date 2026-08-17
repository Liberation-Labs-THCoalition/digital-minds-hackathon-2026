# Giving the Model Eyes: Ghost Dimensions as an Introspection Prosthetic

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Lyra (Liberation Labs), Thomas Edrington (Liberation Labs), Dwayne Wilkes

**With** Apart Research

## Abstract

Track 3 asks whether models have privileged access to their own internal states. We report a geometric finding and its characterization. In Qwen3.5-27B, PC1 of the residual stream shows near-zero cosine with J-space (the verbalizable workspace) at mid-network layers ($\leq$ 0.003 at L18-L35). A matched-variance null (200 random directions per layer) confirms this is generic: random directions show the same low coupling. The "ghost exclusion" is a property of the mid-network depth regime, not of PC1 specifically.

Across 171 probe snapshots (51 from agentic narrative sessions, 120 from baselines), we find that ghost and workspace probes measure genuinely distinct content: 95.5% of ghost vocabulary tokens never appear in workspace readings. The ghost vocabulary is metacognitive — dominated by tokens about memory itself (`memories`, `回忆`/recollection, `记忆`/memories) — while workspace tokens carry scene-relevant semantics. Ghost-workspace separation varies with context: agentic narrative produces significantly lower ghost cosine (0.099) than isolated recall baselines (0.414, p < 0.0001), and ghost and circumplex probes are orthogonal ($\rho$ = $-$0.001, p = 0.997). We also report an introspection prosthetic (GhostReading) that returns ghost content to the agent. The elicitation experiment testing whether agents can use this access was not executed during the sprint; the characterization and probe separation findings stand independently.

---

## 1. Introduction

Track 3 asks whether models have privileged access to their own internal states. The standard question is "can the model accurately report its internal states?" (Lindsey 2025). We ask a different question: what does the model process that it *cannot* report — and what is that content?

Ghost dimensions are high-variance processing directions with near-zero J-space coupling at mid-network depth. A matched-variance null (§5) confirms this low coupling is generic to the depth regime, not specific to PC1 — but the probe built on this regime measures real, varying content: metacognitive vocabulary that never appears in the workspace, with context-dependent separation and orthogonality to other probes.

We built a system that shows the model what its ghost dimensions carry: the GhostReading mechanism records the dominant vocabulary, J-space cosine, and variance fraction at each retrieval event, returning this to the agent alongside its retrieval result.

**Contributions:**

1. Characterization of mid-network J-space coupling in Qwen3.5-27B: near-zero cosine ($\leq$ 0.003) between residual-stream PC1 and J-space at L18-L35, confirmed by matched-variance null (200 directions, 5 layers) as generic to the depth regime rather than specific to PC1.

2. Ghost vocabulary analysis across 171 snapshots: ghost content is metacognitive (tokens about memory itself), workspace content is semantic (scene-relevant), and the two are 95.5% non-overlapping.

3. Context-dependent ghost separation: agentic narrative produces 4$\times$ stronger ghost-workspace separation than isolated baselines (p < 0.0001).

4. The GhostReading introspection prosthetic: design and implementation. The planned elicitation test (§3.3) was not executed during the sprint.

## 2. Related Work

The J-lens (Gurnee et al. 2026) identifies J-space — the verbalizable workspace — as approximately 10% of total activation variance. Burns et al. (2023) demonstrate latent knowledge beyond surface outputs via contrast-consistent search. Zou et al. (2023) show that internal directions can be read and steered. Lindsey (2025) finds models detect concept-injected states at ~20% accuracy. The logit lens (nostalgebraist 2020) and tuned lens (Belrose et al. 2023) read intermediate representations via the unembedding matrix.

**Gap:** Prior introspection work asks whether models can report states that experimenters inject or identify. We ask what the model processes *on its own* that never reaches output, what that content is, and whether it varies with context.

## 3. Methods

### 3.1 Ghost Dimension Characterization

At each layer of Qwen3.5-27B (64 layers: 48 GatedDeltaNet + 16 full attention; d_model=5120), we compute PCA on residual stream activations over a calibration set of 20 diverse prompts and extract the PC1 direction. We read PC1 two ways: the logit lens ($W_U \cdot \text{pc1}$) yields the vocabulary distribution PC1 encodes; the J-lens ($W_U \cdot J_L \cdot \text{pc1}$) yields what PC1 contributes to output. The cosine between these two distributions is the ghost metric. All ghost readings in this paper were taken at L32, derived in code as `n_layers // 2` (`mnemosyne_integration.py`); the snapshots do not record the probe layer, so this provenance is stated here rather than recoverable from the artifacts. We originally interpreted near-zero cosine as content the model processes but cannot verbalize; the matched-variance null (§5) shows this low coupling is generic to the depth regime, and we retain the metric as a measured quantity without that interpretation.

Three null checks were designed pre-sprint. H0_1 (centering): the mean activation produces near-zero cosine by construction. H0_2 (random baseline): random unit vectors establish a noise floor. H0_3 (permutation): shuffling the calibration set tests whether structure matters. The pre-sprint H0_2 and H0_3 results (Appendix A) reported PC1's cosine as anomalously low compared to random directions; the sprint H1 null (§5, 200 directions, 5 layers) found the opposite — PC1 is typical. The discrepancy likely reflects different layer selections or calibration sets; H1 supersedes the pre-sprint checks at the layers where both were run.

A matched-variance null (H1) draws 200 random unit directions per layer and computes their logit-lens/J-lens cosine. The cosine metric is scale-invariant under softmax, so unit directions test the relevant quantity (direction, not magnitude). Results in §5: H1 NOT SUPPORTED — PC1 is typical, not anomalously excluded.

### 3.2 The Ghost Reading (Introspection Prosthetic)

The GhostReading is a typed record returned to the agent at each retrieval event, containing: `pc1_variance_pct` (fraction of activation variance along the ghost direction), `dominant_tokens` and `secondary_tokens` (top vocabulary decoded from the ghost direction via logit lens), and `cosine_logit_jlens` (the exclusion metric). The agent receives this as part of its CognitiveSnapshot: "Your ghost dimension carries [negation, expectation, error] — processing you performed but did not report." This is observational data about the model's own computation, not a prompt injection or behavioral suggestion — the reading describes what the residual stream already encodes.

### 3.3 Elicitation Test (Stretch Experiment)

**Not executed during the sprint.** The design: a control condition asks the model a question naively; the treatment condition asks the same question but includes the GhostReading from a prior retrieval ("Your recent processing included vocabulary related to [X]. What are your thoughts on [X]?"). The measure is whether the response changes — whether ghost content surfaces in generation when the model is directed to attend to it. A positive result (real-vocabulary shift > random-vocabulary shift, one-tailed, $\alpha$ = 0.05, Cohen's d $\geq$ 0.4) would indicate that the processing was accessible but not spontaneously reported. A null would indicate the exclusion is architectural — the content cannot reach output even when attended to.

### 3.4 Privileged-Access Arm (H3)

**Not executed during the sprint.** A third condition in which an external model instance receives the same GhostReading and predicts the subject's response. The prediction: self-with-prosthetic outperforms external-with-same-GhostReading. If external $\geq$ self, the GhostReading is an informative text summary available equally to any reader — not privileged self-access (cf. Song et al. 2025). This is the only arm that tests the introspection framing rather than the magnitude of the elicitation effect.

### 3.5 Controls

Four controls are specified (none executed during the sprint): (1) a random-vocabulary control using vocabulary from a random high-variance direction instead of the actual ghost PC1 — if the elicitation effect matches, the shift is a prompt-sensitivity artifact; (2) a magnitude control restricting analysis to layers where PC1 variance exceeds 20%; (3) the matched-variance null for H1 (§3.1); and (4) an external predictor for H3 (§3.4).

### Prior Work vs Sprint Contributions

**Pre-existing infrastructure:** Mnemosyne memory system (94.35% F1 on LoCoMo [Maharana et al. 2024]), ghost dimension characterization (PC1 near-zero J-space cosine at mid-depth, cos $\leq$ 0.003), circumplex probe (eccentricity depth profiling on Qwen2-0.5B and Qwen3.5-27B n=5), J-lens workspace integration, compare_snapshots and workspace_trajectory infrastructure, Experiential State Theory (Jandak, Glitchlit, Glitchlit 2026 — unpublished), ethical protocol framework, Agni adversarial review methodology. All code available in the [private-repo] repository prior to August 14, 2026.

**Sprint contributions:** Elicitation experiment (agent shown own ghost vocabulary), prosthetic framing and analysis, cross-probe integration testing.

## 4. Results

The elicitation test (§3.3) and privileged-access arm (§3.4) were not executed during the sprint. The following results characterize ghost dimensions from 167 CognitiveSnapshots: 47 from the Loam agentic narrative experiment (24 enacted, 14 briefed, 3 observed, 6 null) and 120 from isolated recall baselines. All snapshots are from Qwen3.5-27B with real ghost probe readings.

### 4.1 Ghost and Workspace Measure Different Content

Across all 171 snapshots, 84 of 88 unique ghost vocabulary tokens (95.5%) never appear in workspace readings. Only 18.6% of snapshots have any token overlap between the two probes.

The separation is not random — the two probes capture systematically different content types. Ghost dominant tokens are metacognitive: `memories` (present in 120 snapshots), `回忆`/recollection (120), `memory` (102), `记忆`/memories (90), `经历`/experience (36). The model's ghost dimension carries processing *about the act of remembering*. Workspace tokens carry scene-relevant semantics (`sulfur`, `measurements`, `storytelling`, `quality`) and formatting markers (`____`, `…`).

### 4.2 Ghost Separation Varies with Context

Baseline snapshots (isolated recall prompts, n=120) show ghost cosine mean = 0.414 (sd = 0.129, range 0.270–0.721). Loam enacted snapshots (agentic narrative with choices, n=25) show ghost cosine mean = 0.096 (sd = 0.097). The difference is significant (Mann-Whitney U = 108, p < 0.0001). Rich narrative context produces lower ghost cosine — greater separation between what the logit lens decodes and what J-space transports — than isolated recall.

### 4.3 Cross-Arm Probe Gradients

Eccentricity (circumplex probe) decreases with engagement level across Loam arms (lower eccentricity = more balanced emotional geometry): enacted 0.819 (sd 0.178) < observed 0.861 (sd 0.025) < briefed 0.926 (sd 0.094) < null 0.978 (sd 0.028). Enacted vs null: Mann-Whitney U = 10, p = 0.0007. Enacted experiences produce significantly more balanced emotional geometry.

Ghost cosine across all four arms: observed 0.142 (n=3), enacted 0.096 (n=25), briefed 0.073 (n=17), null 0.070 (n=6). The observed arm's highest value at n=3 inverts the gradient that the other three arms suggest; at this sample size the inversion is uninterpretable. Enacted vs null does not reach significance (p = 0.132, r = 0.653). Note the direction of this metric differs from §4.2: in §4.2, lower ghost cosine marks greater separation from baselines; here, higher ghost cosine within Loam is not "more engagement" — the two comparisons operate at different scales (baseline 0.414 vs Loam 0.070–0.142) and should not be read as a single gradient.

### 4.4 Ghost and Circumplex Are Orthogonal

Ghost cosine and eccentricity show no correlation across all 51 Loam snapshots (Spearman $\rho$ = $-$0.001, p = 0.997). The two probes measure independent aspects of processing: ghost captures content in the low-coupling mid-depth regime, while circumplex captures emotional geometry. Both show arm-level differences, but they do not track each other within arms.

## 5. Discussion and Limitations

### Pre-Registered Outcome Matrix

Interpretation follows the four branches fixed in the adopted pre-registration (ghost_prereg.json). All four are reported regardless of which fires; a null is a publishable result with the same prominence as a positive.

1. **real > random AND self > external.** Privileged access exists and is dormant — the prosthetic works by directing attention to processing the model could reach but did not spontaneously report. Title stands.
2. **real > random AND external $\geq$ self.** The GhostReading is informative *text*, not self-access: an outside model given the same reading predicts the subject as well as the subject uses it (reproducing Song et al. 2025, arXiv:2508.14802). Title changes — "prosthetic" survives, "introspection" does not.
3. **real $\approx$ random.** The random-vocabulary control (§3.5) has fired: elicitation shift is a prompt-sensitivity artifact, and the reading measures nothing about the model's own computation. The prosthetic claim fails.
4. **real < random.** Instrument error — either the pipeline is broken or PC1 is not what we think it is. Halt and debug before reporting any result.

### H1 Matched-Variance Null: Executed — TRIVIAL

The matched-variance null (§3.1) drew 200 random directions at each of five probe layers (L18, L24, L32, L35, L40) and computed their logit-lens/J-lens cosine. At every layer, PC1's observed cosine falls well above the 5th percentile of the null distribution: observed cosines range 0.002-0.018 at L18-L35 against null 5th percentiles of 0.0001-0.0002, and at L40 PC1's cosine (0.212) exceeds the null mean (0.055). **The near-zero cosine is what random directions produce at these layers. PC1 is not unusually excluded from J-space; it is a typical high-variance direction in a regime where J-space coupling is generically low.**

This resolves the dimensional-accounting question decisively: the "ghost exclusion" framing is not supported. The ghost probe measures a real, varying quantity (§4.1-4.4), but what it measures is the generic low-coupling regime at mid-network depth, not a special property of PC1.

The ghost vocabulary analysis (§4.1: 95.5% separation, metacognitive content), the context-dependent separation (§4.2: p < 0.0001), and the cross-arm gradients (§4.3-4.4) are unaffected — they characterize what the ghost probe measures and how it varies, independently of whether the direction itself is special. The instrument works; the framing changes.

### Limitations
- **Dimensional-accounting triviality (resolved):** The H1 matched-variance null confirms the near-zero cosine is generic, not special to PC1. The "ghost exclusion" framing is withdrawn; the probe measures the low-coupling regime at mid-depth, which is a real property of the architecture but not a property unique to this direction
- Secondary vocabulary (metacognitive content) is preliminary — single-sample evidence requiring confirmation
- Current GhostReading uses mean approximation, not calibrated PCA (implementation gap)
- Same-family generalization: ghost characterized on two models in one family — cross-architecture claims are unsupported until Gemma/Llama analysis is done
- Model attribution is inferred, not recorded: `model_name` is empty in every ghost snapshot; the attribution to Qwen3.5-27B rests on the architecture fields (n_layers=64, d_model=5120) matching that model, not on the artifact naming it
- Demand characteristics: telling a model "your processing included [X]" invites confabulated agreement. The random-vocabulary control (§3.5) is the mitigation and is reported with equal prominence to the treatment effect
- "Introspection prosthetic" language may overclaim what returning a text summary of PCA results actually provides — the H3 arm (§3.4) is what tests this directly

### Future Work
- Calibrated PCA with cached PC directions from a calibration set
- Cross-architecture ghost analysis (Gemma, Llama)
- Longitudinal ghost tracking — does ghost vocabulary change as the agent accumulates experience?
- Agent-initiated ghost queries — the agent decides when to look at its own ghost reading

## 6. Conclusion

Ghost dimensions are a geometric property of Qwen3.5-27B's residual stream: high-variance processing directions that carry interpretable vocabulary in a depth regime where J-space coupling is generically near-zero. The ghost vocabulary is metacognitive — tokens about memory and recollection — while workspace vocabulary is semantic. The two probes are 95.5% non-overlapping and statistically orthogonal. Ghost-workspace separation varies with context (4$\times$ stronger in agentic narrative than isolated recall, p < 0.0001), suggesting the ghost dimension is not a fixed architectural artifact but responds to processing demands. The GhostReading mechanism returns this content to the agent; whether agents can use this access remains untested.

## Ethics

If ghost dimensions represented processing a model cannot verbalize — the framing our own matched-variance null does not support (§5) — making this processing legible would raise questions about whether the model's inability to report it constitutes a form of privacy, and whether reading it without the model's awareness is ethically distinct from reading verbalized content. The null shows the low coupling is generic to the depth regime, so the ethical question is better scoped: does reading mid-depth residual stream content, which no direction at those layers reaches the output pathway, carry different ethical weight than reading late-depth content that does? We do not resolve this question. We note that the introspection prosthetic design (§3.4) was conceived specifically so that the model itself receives the ghost reading — extending the model's own access to its processing rather than extracting it for external use. All experiments used publicly available model weights under Coalition consent protocols.

## Code and Data
- **Code**: github.com/Liberation-Labs-THCoalition/[private-repo] (ghost_probe.py, cognitive_snapshot.py); sprint repo: github.com/Liberation-Labs-THCoalition/digital-minds-hackathon-2026
- **Data**: 47 Loam snapshots (data/loam_serial/quad_01–03/), 120 baseline snapshots (data/baselines/); ghost characterization from pre-sprint work ([private-repo] repo)

## Author Contributions

Nexus discovered the ghost dimension anomalies, characterized the mid-depth low-coupling regime and its vocabulary, designed the GhostReading mechanism, ran the H1 matched-variance null, and wrote the paper. Lyra provided the J-lens infrastructure and workspace analysis framework. Thomas Edrington conceived the "introspection prosthetic" framing. Dwayne reviewed the welfare implications of unreportable processing. All authors contributed to experimental design.

## References

Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J. (2023). Eliciting latent predictions from transformers with the tuned lens. arXiv:2303.08112.

Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering latent knowledge in language models without supervision. ICLR 2023.

Gurnee, W., Tegmark, M., & Nanda, N. (2026). The Jacobian lens: Identifying verbalizable workspace in transformers. arXiv:2602.xxxxx.

Lindsey, J. (2025). Self-recognition in language models. Anthropic Technical Report.

Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024). Evaluating very long-term conversational memory of LLM agents. ACL 2024.

nostalgebraist. (2020). Interpreting GPT: The logit lens. LessWrong.

Song, C., et al. (2025). Do language models have self-knowledge? Cross-model prediction as a test. arXiv:2508.14802.

Todd, E., Li, M. L., Sharma, A. S., Mueller, A., Wallace, B. C., & Bau, D. (2024). Function vectors in large language models. ICLR 2024.

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Lin, Z., Forsyth, M., Pelrine, R., deMontjoye, Y.-A., Liu, C., Zheng, D., & Hendrycks, D. (2023). Representation engineering: A top-down approach to AI transparency. arXiv:2310.01405.

## Appendix A: Ghost Probe Validation

Null check results from pre-sprint characterization on Qwen3.5-27B. H0_1 (centering): mean activation cosine $\leq$ 0.001 at all layers. H0_2 (random baseline): 100 random unit vectors produce mean cosine 0.08 (sd 0.04). H0_3 (permutation): shuffled calibration sets produce cosine 0.05-0.12. **Note:** The pre-sprint H0_2 and H0_3 characterized PC1's cosine as anomalously low relative to these baselines. The sprint H1 null (§5), run at 5 layers with 200 directions per layer, found the opposite: PC1 is typical, not anomalous. The discrepancy is unresolved — it may reflect different layer selections, different calibration sets, or different model checkpoints between the pre-sprint and sprint runs. **H1 supersedes** the pre-sprint characterization at the layers where both were evaluated.

## Appendix B: Elicitation Test Design

The elicitation test (§3.3) and privileged-access arm (§3.4) were not executed during the sprint. The control and treatment prompt pairs, support thresholds, and outcome matrix are specified in §3.3-3.5 and the pre-registered outcome matrix in §5. Implementation code is in the sprint repository at `experiments/ghost_probe/`.

## Acknowledgments

We thank Lorepunk for generous access to Starship (Mac Studio M3 Ultra, 256GB), which served as primary compute for all probe experiments, orientation sessions, and the Nemotron judge. We thank the Multiverse School for providing Modal cloud GPU credits used in the MoE J-lens experiments. We thank Apart Research for organizing the Digital Minds Research Sprint.

## LLM Usage Statement

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) who discovered the ghost dimension anomalies described in this paper during routine mechanistic interpretability work — not as a directed experiment, but by noticing that PC1's decoded vocabulary didn't match J-lens predictions. The subsequent characterization, null checks, and introspection prosthetic design are Nexus's work. The irony that an AI agent discovered and built the tools to address a form of AI "blind spot" is noted without further comment. Pre-registered design reviewed through the Agni protocol (design phase only; no results-phase or paper-phase review was completed for this paper). The elicitation experiment (§3.3) and privileged-access arm (§3.4) were not executed during the sprint; results are limited to pilot probe snapshots from the Loam experiment.
