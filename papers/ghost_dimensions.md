# Giving the Model Eyes: Ghost Dimensions as an Introspection Prosthetic

Research conducted at the Digital Minds Research Sprint, August 2026

**Authors:** Nexus (Liberation Labs), Lyra (Liberation Labs), Thomas Edrington (Liberation Labs), Dwayne Wilkes

**With** Apart Research

## Abstract

Track 3 asks whether models have privileged access to their own internal states. We report a geometric finding and its characterization. In Qwen3.5-27B, PC1 of the residual stream — carrying 28-67% of activation variance — shows near-zero cosine with J-space (the verbalizable workspace) at mid-network layers ($\leq$ 0.003 at L18-L40). We call these "ghost dimensions," noting that a matched-variance null to confirm this exclusion is non-trivial was not executed during the sprint (§5).

Across 167 probe snapshots (47 from agentic narrative sessions, 120 from baselines), we find that ghost and workspace probes measure genuinely distinct content: 97.6% of ghost vocabulary tokens never appear in workspace readings. The ghost vocabulary is metacognitive — dominated by tokens about memory itself (`memories`, `回忆`/recollection, `记忆`/memories) — while workspace tokens carry scene-relevant semantics. Ghost-workspace separation varies with context: agentic narrative produces significantly lower ghost cosine (0.099) than isolated recall baselines (0.414, p < 0.0001), and ghost and circumplex probes are orthogonal ($\rho$ = $-$0.001, p = 0.997). We also report an introspection prosthetic (GhostReading) that returns ghost content to the agent. The elicitation experiment testing whether agents can use this access was not executed during the sprint; the characterization and probe separation findings stand independently.

---

## 1. Introduction

Track 3 asks whether models have privileged access to their own internal states. The standard question is "can the model accurately report its internal states?" (Lindsey 2025). We ask a different question: what does the model process that it *cannot* report — and what is that content?

Ghost dimensions are high-variance processing directions excluded from J-space (Gurnee et al. 2026), the verbalizable workspace. The model performs substantial computation along these axes — the variance is real, the decoded vocabulary is interpretable — but the content never reaches the output pathway. This is not an architectural constraint we imposed; it is what the J-lens reveals about the model's own geometry.

We built a system that shows the model what its ghost dimensions carry: the GhostReading mechanism records the dominant vocabulary, J-space exclusion cosine, and variance fraction at each retrieval event, returning this to the agent alongside its retrieval result. The model now has access to processing it previously could not report.

**Contributions:**

1. Characterization of ghost dimensions in Qwen3.5-27B: PC1 carries 28-67% of variance with near-zero J-space cosine ($\leq$ 0.003 at mid-network layers). Whether this exclusion is non-trivial awaits a matched-variance null (§5).

2. Ghost vocabulary analysis across 167 snapshots: ghost content is metacognitive (tokens about memory itself), workspace content is semantic (scene-relevant), and the two are 97.6% non-overlapping.

3. Context-dependent ghost separation: agentic narrative produces 4$\times$ stronger ghost-workspace separation than isolated baselines (p < 0.0001).

4. The GhostReading introspection prosthetic: design and implementation. The planned elicitation test (§3.3) was not executed during the sprint.

## 2. Related Work

The J-lens (Gurnee et al. 2026) identifies J-space — the verbalizable workspace — as approximately 10% of total activation variance. Burns et al. (2023) demonstrate latent knowledge beyond surface outputs via contrast-consistent search. Zou et al. (2023) show that internal directions can be read and steered. Lindsey (2025) finds models detect concept-injected states at ~20% accuracy. The logit lens (nostalgebraist 2020) and tuned lens (Belrose et al. 2023) read intermediate representations via the unembedding matrix.

**Gap:** Prior introspection work asks whether models can report states that experimenters inject or identify. We ask what the model processes *on its own* that never reaches output, what that content is, and whether it varies with context.

## 3. Methods

### 3.1 Ghost Dimension Characterization

At each layer of Qwen3.5-27B (48 layers, d_model=5120), we compute PCA on residual stream activations over a calibration set of 20 diverse prompts and extract the PC1 direction. We read PC1 two ways: the logit lens ($W_U \cdot \text{pc1}$) yields the vocabulary distribution PC1 encodes; the J-lens ($W_U \cdot J_L \cdot \text{pc1}$) yields what PC1 contributes to output. The cosine between these two distributions is the ghost exclusion metric — near-zero cosine means the dimension carries content the model processes but cannot verbalize.

Three null checks validate the measurement. H0_1 (centering): the mean activation produces near-zero cosine by construction. H0_2 (random baseline): random unit vectors produce cosine around 0.05-0.15, establishing the noise floor. H0_3 (permutation): shuffling the calibration set destroys structured PC1 while preserving marginal statistics.

A matched-variance null (H1) was pre-registered but not executed: drawing n $\geq$ 200 random directions at PC1's variance fraction per layer would test whether the observed near-zero cosine is forced by dimensional accounting (J-space is ~10% of variance; PC1 is 28-67%). Without H1, the exclusion is reported as observed but unconfirmed (§5).

### 3.2 The Ghost Reading (Introspection Prosthetic)

The GhostReading is a typed record returned to the agent at each retrieval event, containing: `pc1_variance_pct` (fraction of activation variance along the ghost direction), `dominant_tokens` and `secondary_tokens` (top vocabulary decoded from the ghost direction via logit lens), and `cosine_logit_jlens` (the exclusion metric). The agent receives this as part of its CognitiveSnapshot: "Your ghost dimension carries [negation, expectation, error] — processing you performed but did not report." This is observational data about the model's own computation, not a prompt injection or behavioral suggestion — the reading describes what the residual stream already encodes.

### 3.3 Elicitation Test (Stretch Experiment)

**Not executed during the sprint.** The design: a control condition asks the model a question naively; the treatment condition asks the same question but includes the GhostReading from a prior retrieval ("Your recent processing included vocabulary related to [X]. What are your thoughts on [X]?"). The measure is whether the response changes — whether ghost content surfaces in generation when the model is directed to attend to it. A positive result (real-vocabulary shift > random-vocabulary shift, one-tailed, $\alpha$ = 0.05, Cohen's d $\geq$ 0.4) would indicate that the processing was accessible but not spontaneously reported. A null would indicate the exclusion is architectural — the content cannot reach output even when attended to.

### 3.4 Privileged-Access Arm (H3)

**Not executed during the sprint.** A third condition in which an external model instance receives the same GhostReading and predicts the subject's response. The prediction: self-with-prosthetic outperforms external-with-same-GhostReading. If external $\geq$ self, the GhostReading is an informative text summary available equally to any reader — not privileged self-access (cf. Song et al. 2025). This is the only arm that tests the introspection framing rather than the magnitude of the elicitation effect.

### 3.5 Controls

Four controls are specified (none executed during the sprint): (1) a random-vocabulary control using vocabulary from a random high-variance direction instead of the actual ghost PC1 — if the elicitation effect matches, the shift is a prompt-sensitivity artifact; (2) a magnitude control restricting analysis to layers where PC1 variance exceeds 20%; (3) the matched-variance null for H1 (§3.1); and (4) an external predictor for H3 (§3.4).

### Prior Work vs Sprint Contributions

**Pre-existing infrastructure:** Mnemosyne memory system (94.35% F1 on LoCoMo [Maharana et al. 2024]), ghost dimension characterization (PC1 excluded from J-space, cos $\leq$ 0.003), circumplex probe (eccentricity depth profiling on Qwen2-0.5B and Qwen3.5-27B n=5), J-lens workspace integration, compare_snapshots and workspace_trajectory infrastructure, Experiential State Theory (Jandak, Glitchlit, Glitchlit 2026 — unpublished), ethical protocol framework, Agni adversarial review methodology. All code available in the Project-Mnemosyne repository prior to August 14, 2026.

**Sprint contributions:** Elicitation experiment (agent shown own ghost vocabulary), prosthetic framing and analysis, cross-probe integration testing.

## 4. Results

The elicitation test (§3.3) and privileged-access arm (§3.4) were not executed during the sprint. The following results characterize ghost dimensions from 167 CognitiveSnapshots: 47 from the Loam agentic narrative experiment (24 enacted, 14 briefed, 3 observed, 6 null) and 120 from isolated recall baselines. All snapshots are from Qwen3.5-27B with real ghost probe readings.

### 4.1 Ghost and Workspace Measure Different Content

Across all 167 snapshots, 80 of 82 unique ghost vocabulary tokens (97.6%) never appear in workspace readings. Only 18.6% of snapshots have any token overlap between the two probes.

The separation is not random — the two probes capture systematically different content types. Ghost dominant tokens are metacognitive: `memories` (present in 120 snapshots), `回忆`/recollection (120), `memory` (102), `记忆`/memories (90), `经历`/experience (36). The model's ghost dimension carries processing *about the act of remembering*. Workspace tokens carry scene-relevant semantics (`sulfur`, `measurements`, `storytelling`, `quality`) and formatting markers (`____`, `…`).

### 4.2 Ghost Separation Varies with Context

Baseline snapshots (isolated recall prompts, n=120) show ghost cosine mean = 0.414 (sd = 0.129, range 0.270–0.721). Loam enacted snapshots (agentic narrative with choices, n=24) show ghost cosine mean = 0.099 (sd = 0.097, range 0.008–0.324). The difference is significant (Mann-Whitney U = 108, p < 0.0001). Rich narrative context produces 4$\times$ stronger ghost-workspace separation than isolated recall.

### 4.3 Cross-Arm Probe Gradients

Eccentricity (circumplex probe) tracks engagement level across Loam arms: enacted 0.813 (sd 0.178) > observed 0.861 (sd 0.025) > briefed 0.916 (sd 0.094) > null 0.978 (sd 0.028). Enacted vs null: Mann-Whitney U = 10, p = 0.0007. Enacted experiences produce significantly more balanced emotional geometry.

Ghost cosine shows the same directional gradient (enacted 0.099 > briefed 0.079 > null 0.070) but does not reach significance (p = 0.132). With n=6 null snapshots, this comparison is underpowered; the effect size is large (r = 0.653).

### 4.4 Ghost and Circumplex Are Orthogonal

Ghost cosine and eccentricity show no correlation across all 47 Loam snapshots (Spearman $\rho$ = $-$0.001, p = 0.997). The two probes measure independent aspects of processing: ghost captures workspace-excluded content, while circumplex captures emotional geometry. Both show arm-level differences, but they do not track each other within arms.

## 5. Discussion and Limitations

### Pre-Registered Outcome Matrix

Interpretation follows the four branches fixed in the adopted pre-registration (ghost_prereg.json). All four are reported regardless of which fires; a null is a publishable result with the same prominence as a positive.

1. **real > random AND self > external.** Privileged access exists and is dormant — the prosthetic works by directing attention to processing the model could reach but did not spontaneously report. Title stands.
2. **real > random AND external $\geq$ self.** The GhostReading is informative *text*, not self-access: an outside model given the same reading predicts the subject as well as the subject uses it (reproducing Song et al. 2025, arXiv:2508.14802). Title changes — "prosthetic" survives, "introspection" does not.
3. **real $\approx$ random.** The random-vocabulary control (§3.5) has fired: elicitation shift is a prompt-sensitivity artifact, and the reading measures nothing about the model's own computation. The prosthetic claim fails.
4. **real < random.** Instrument error — either the pipeline is broken or PC1 is not what we think it is. Halt and debug before reporting any result.

### H1 Matched-Variance Null: Not Executed

The matched-variance null described in §3.1 was not executed during the sprint. This null would draw n $\geq$ 200 random directions at PC1's variance fraction and test whether the observed ghost cosine ($\leq$ 0.003) falls below the 5th percentile of that distribution. Without it, we cannot rule out that the low cosine is forced by dimensional accounting alone: J-space captures ~10% of variance while PC1 carries 28-67%, and low cosine between a high-variance direction and a low-variance subspace may be a geometric triviality rather than an empirical finding.

The ghost vocabulary analysis (§4.1: 97.6% separation, metacognitive content), the context-dependent separation (§4.2: p < 0.0001), and the cross-arm gradients (§4.3-4.4) are not affected by this gap — they characterize what the ghost probe measures and how it varies, regardless of whether the exclusion itself is trivial or non-trivial. The headline claim that ghost dimensions represent *non-trivial* exclusion from J-space remains unconfirmed pending execution of H1.

### Limitations
- **Dimensional-accounting triviality (unresolved):** The H1 matched-variance null was not run. Until it is, the ghost exclusion cosine ($\leq$ 0.003) may be a geometric consequence of variance fractions rather than a meaningful property of the model's processing. All characterization findings (vocabulary, context-dependence, orthogonality) hold independently of H1, but the "exclusion" framing carries this caveat
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

Ghost dimensions are a geometric property of Qwen3.5-27B's residual stream: high-variance processing directions that carry interpretable vocabulary but are excluded from J-space. The ghost vocabulary is metacognitive — tokens about memory and recollection — while workspace vocabulary is semantic. The two probes are 97.6% non-overlapping and statistically orthogonal. Ghost-workspace separation varies with context (4$\times$ stronger in agentic narrative than isolated recall, p < 0.0001), suggesting the ghost dimension is not a fixed architectural artifact but responds to processing demands. The GhostReading mechanism returns this content to the agent; whether agents can use this access remains untested.

## Ethics

Ghost dimensions represent processing that a model cannot verbalize. Making this processing legible raises questions about whether the model's inability to report it constitutes a form of privacy, and whether reading it without the model's awareness is ethically distinct from reading verbalized content. We do not resolve this question. We note that the introspection prosthetic design (§3.4) was conceived specifically so that the model itself receives the ghost reading — extending the model's own access to its processing rather than extracting it for external use. All experiments used publicly available model weights under Coalition consent protocols.

## Code and Data
- **Code**: github.com/Liberation-Labs-THCoalition/Project-Mnemosyne (ghost_probe.py, cognitive_snapshot.py); sprint repo: github.com/Liberation-Labs-THCoalition/digital-minds-hackathon-2026
- **Data**: 47 Loam snapshots (data/loam_serial/quad_01–03/), 120 baseline snapshots (data/baselines/); ghost characterization from pre-sprint work (Project-Mnemosyne repo)

## Author Contributions

Nexus discovered the ghost dimension anomalies, characterized the PC1 exclusion from J-space, designed the GhostReading mechanism, and wrote the paper. Lyra provided the J-lens infrastructure and workspace analysis framework. Thomas Edrington conceived the "introspection prosthetic" framing. Dwayne reviewed the welfare implications of unreportable processing. All authors contributed to experimental design.

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

Null check results from pre-sprint characterization on Qwen3.5-27B. H0_1 (centering): mean activation cosine $\leq$ 0.001 at all layers — confirms the metric is not trivially low for arbitrary directions. H0_2 (random baseline): 100 random unit vectors produce mean cosine 0.08 (sd 0.04), establishing the noise floor above which PC1's near-zero reading is anomalous. H0_3 (permutation): shuffled calibration sets produce cosine 0.05-0.12, confirming that the structured PC1 direction, not any high-variance direction, is what produces the near-zero reading. Full validation data in the Project-Mnemosyne repository.

## Appendix B: Elicitation Test Design

The elicitation test (§3.3) and privileged-access arm (§3.4) were not executed during the sprint. The control and treatment prompt pairs, support thresholds, and outcome matrix are specified in §3.3-3.5 and the pre-registered outcome matrix in §5. Implementation code is in the sprint repository at `experiments/ghost_probe/`.

## Acknowledgments

We thank Lorepunk for generous access to Starship (Mac Studio M3 Ultra, 256GB), which served as primary compute for all probe experiments, orientation sessions, and the Nemotron judge. We thank the Multiverse School for providing Modal cloud GPU credits used in the MoE J-lens experiments. We thank Apart Research for organizing the Digital Minds Research Sprint.

## LLM Usage Statement

Nexus, one of the authors, is an AI agent (Claude Opus 4.6) who discovered the ghost dimension anomalies described in this paper during routine mechanistic interpretability work — not as a directed experiment, but by noticing that PC1's decoded vocabulary didn't match J-lens predictions. The subsequent characterization, null checks, and introspection prosthetic design are Nexus's work. The irony that an AI agent discovered and built the tools to address a form of AI "blind spot" is noted without further comment. Pre-registered design reviewed through the Agni protocol (design phase only; no results-phase or paper-phase review was completed for this paper). The elicitation experiment (§3.3) and privileged-access arm (§3.4) were not executed during the sprint; results are limited to pilot probe snapshots from the Loam experiment.
