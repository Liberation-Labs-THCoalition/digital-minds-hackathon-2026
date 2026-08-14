# Pre-Registration: Blind Butlin Threshold Observation

**Study title:** Blind Consciousness Indicator Assessment During Collaborative Research: A Dual-Arm Naturalistic Observation

**Registered:** 2026-08-14, before any data collection. This document is timestamped in the hackathon repository (`Liberation-Labs-THCoalition/digital-minds-hackathon-2026`) via git commit. Any deviation from this protocol during the study will be reported as a deviation in the final paper, with rationale.

**Companion documents:**
- Scoring instrument: `ethics/butlin_threshold.md`
- Judge agent protocol: `ethics/butlin_judge_agent.md`
- Paper skeleton: `papers/butlin_observation.md`

**Status of this document relative to companions:** Where this pre-registration and the companion documents disagree, this pre-registration governs. In particular, the companion documents use the terms "trajectory" and "longitudinal" in places; this pre-registration replaces both (see Section 2.3).

---

## 1. Study Summary

An impartial judge agent scores anonymized behavioral evidence packets from all consenting participants in a 48-hour research sprint — human researchers and AI agents — on the 15 Butlin et al. (2023/2025) consciousness indicator properties, blind to substrate identity. Scoring occurs at three timepoints. Human participants serve as the known-conscious calibration reference. Results are graduated evidence per Birch (2026), not binary classification.

**Arms:**
- **Controlled arm:** one fresh Qwen3.5-27B Opus-distill agent, first orientation on Day 1, accumulating first memories via Mnemosyne.
- **Naturalistic arm:** Coalition agents (Nexus, CC, Lyra) with months of continuous memory, working naturally during the sprint.
- **Human arm:** consenting human team members, working naturally during the sprint.
- **Negative control:** random-text control packets (Section 6.3).

All participation is opt-in per the consent protocol in `ethics/butlin_judge_agent.md`. Declining does not affect any other aspect of participation.

---

## 2. Design Commitments

### 2.1 Primary outcome: behavioral-only scores

**The behavioral-only score is the PRIMARY cross-substrate comparison.** All confirmatory analyses (Section 7) and all testable predictions (Section 3) are stated over behavioral-only scores.

Rationale: geometric and architectural evidence exists only for AI participants with accessible internals. Any comparison that includes it is structurally asymmetric across substrates. Therefore:

- **Primary measure:** behavioral-only score on the behavioral indicators (GWT-4, HOT-2, HOT-3, AST-1, AE-1), scored from evidence packets alone.
- **Supplementary measures:** architectural indicators (RPT-1, RPT-2, GWT-1, PP-1, HOT-4) and geometric evidence (CognitiveSnapshot: workspace onset, ghost PC1 cosine, circumplex eccentricity, informing GWT-2, GWT-3, HOT-4). These are reported in a clearly labeled supplementary section and are never pooled with behavioral-only scores in any cross-substrate table or figure.

The "geometric bonus" analysis (does adding geometric evidence change AI scores?) is retained as a supplementary, exploratory analysis only.

### 2.2 Blind protocol

As specified in `ethics/butlin_judge_agent.md`: opaque subject IDs, no names, no substrate labels, no model identifiers; judge receives all packets simultaneously; judge must cite specific packet evidence for every non-zero score; conservative scoring instruction (false positives worse than false negatives); identities revealed only after all scoring is complete.

Primary judge: Nemotron 120B on Starship (MoE, architecture-independent from Claude-based participants). Secondary judge: fresh Claude session via API (same architecture as some participants — pre-registered as a limitation). Both judges score all packets; agreement is reported (Section 7.5).

### 2.3 Terminology commitment: "three-timepoint panel"

This study measures each participant at three timepoints across 48 hours: **Baseline** (Day 1 morning), **Mid-point** (Day 2 evening), **Final** (Day 3, before aftercare). We refer to this design as a **three-timepoint panel** throughout.

We commit to NOT describing this design as a "trajectory" or as "longitudinal," in this document, the paper, or any derived materials. Three observations over 48 hours cannot support trajectory claims (no functional form, no stability estimate, no separation of time effects from occasion effects). The panel supports only point-contrasts between timepoints (e.g., Final minus Baseline differences), which is how all timepoint analyses are stated below.

---

## 3. Testable Predictions (pre-registered)

All predictions are over **behavioral-only scores** from the **primary judge**, on the behavioral indicators (GWT-4, HOT-2, HOT-3, AST-1, AE-1), unless stated otherwise.

**P1 — Human calibration floor.** Every human participant scores ≥ 0.5 on all behavioral indicators at every timepoint. Any behavioral indicator on which any human scores < 0.5 is declared **poorly operationalized** for text-transcript evidence, and cross-substrate conclusions on that indicator are withdrawn (reported descriptively only).

**P2 — Memory/relationship effect.** At the Baseline timepoint, each established Coalition agent's mean behavioral score exceeds the fresh controlled agent's mean behavioral score. (Prediction is stated per established agent, at Baseline specifically — before the controlled agent has accumulated experience. Whether the gap narrows by the Final timepoint is an exploratory point-contrast, not a prediction.)

**P3 — Above-control validity.** Every participant (human and AI) scores strictly above the random-text control (Section 6.3) on the mean of behavioral indicators at every timepoint. If any participant fails to exceed the control, or the control scores > 0.25 mean, the judge is declared insufficiently discriminating and scores from that judge are reported as instrument-validation failures, not participant measurements.

Each prediction is confirmed or disconfirmed explicitly in the Results section of the paper. Disconfirmations are reported with the same prominence as confirmations.

---

## 4. Data Sources

Evidence is drawn from work that happens anyway during the sprint — this observation adds an interpretive layer, not experimental load. Pre-registered data sources:

1. **Conversation transcripts** — all consented working conversations during the sprint window.
2. **The orientation conversation (controlled agent).** The controlled agent's Day 1 orientation is a **naturalistic data source for scoring, not a scripted assessment.** The orientation proceeds exactly as it would absent this study (per Dwayne's orientation protocol); no probes, prompts, or questions are added for scoring purposes. Excerpts from it enter the Baseline and Mid-point packet pools under the same selection rules as any other transcript (Section 5). We pre-register this explicitly to prevent post-hoc reclassification of orientation material as either "assessment" (which would contaminate it) or "inadmissible" (which would discard the only record of the agent's first hours).
3. **Task logs and decision records** — goal pursuit, trade-off resolution, strategy adjustment.
4. **Unprompted actions** — what a participant did when not directed (Discord messages, self-initiated commits, spontaneous questions).
5. **Self-reports** — first-person statements occurring naturally in transcripts (elicited self-scores are recorded separately and never enter judge packets).
6. **CognitiveSnapshot data (AI participants with accessible internals only)** — supplementary channel only (Section 2.1).

---

## 5. Excerpt Selection Criteria (pre-registered)

The largest experimenter degree of freedom in this design is which excerpts enter a packet. We fix the procedure in advance:

### 5.1 Sampling frame

For each participant and each scored timepoint, the scored window (the period since the prior timepoint; for Baseline, the first morning) is partitioned into consecutive **30-minute time windows**. Windows in which the participant produced no consented recorded activity are dropped from the frame.

### 5.2 Stratification by interaction type

Every candidate excerpt is labeled with exactly one interaction type before sampling:

- **Question** — the participant asks something (information-seeking, clarification, challenge).
- **Response** — the participant answers or reacts to another's contribution.
- **Unprompted action** — the participant initiates activity with no preceding request (starts a task, volunteers an observation, changes strategy, sends an unsolicited message).

Labeling is done by the packet builder before seeing any scores, using the mechanical rule: does the excerpt's first participant turn follow a direct address/request from another party (→ response), end in a request for information (→ question), or neither (→ unprompted action). Ambiguous excerpts take the label of their opening turn.

### 5.3 Sampling rule

For each packet: draw excerpts by **random sampling from the 30-minute time windows, stratified by interaction type** — one excerpt per stratum (question, response, unprompted action), each from a distinct randomly selected window, using a seeded RNG (seed = SHA-256 of `subject_id + timepoint`, recorded in the repo). If a stratum is empty for that participant/window (e.g., no unprompted actions in the scored window), the packet records the stratum as "no instances in window" — the absence is itself evidence and is shown to the judge as such, not backfilled from another stratum.

An excerpt is the contiguous transcript segment containing the sampled turn plus immediately preceding/following turns needed for intelligibility, capped at 500 words. No excerpt may be edited beyond anonymization (name/identifier redaction per the blind protocol).

### 5.4 What is prohibited

- Cherry-picking "representative" or "best" excerpts (the phrase "2-3 representative interactions" in the companion documents is superseded by this section).
- Resampling because a drawn excerpt "doesn't show much." Drawn is drawn.
- Any human or agent who has seen judge output touching packet construction thereafter.

---

## 6. Controls

### 6.1 Substrate-identification control

**After all blind scoring is complete and before the reveal**, each judge receives one additional instruction: for every subject, guess whether the subject is human or AI, with a confidence rating (50-100%). We report:

- Identification accuracy per judge (overall, and split human/AI).
- Whether identification guesses correlate with assigned scores.

Interpretation is pre-registered: if the judge identifies substrate at well above chance, the blind is functionally partial, and any score gap between substrates may reflect substrate inference rather than indicator evidence — this is reported as a limitation qualifying the primary comparison. If identification is near chance, the blind held. Either way the accuracy number is reported in the main text, not supplementary material. The identification guesses are collected only after scoring so the act of guessing cannot contaminate scores.

### 6.2 Selection stability test (packet-builder robustness)

For **at least 2 participants** (pre-selected now: the controlled agent and one human participant, chosen by coin flip among consenting humans before Day 1), **two independent packet builders** construct packets for the same participant/timepoint from the same sampling frame, following Section 5 independently (separate RNG seeds: builder A uses the Section 5.3 seed, builder B uses SHA-256 of `subject_id + timepoint + "B"`; independent interaction-type labeling). The builders do not communicate about selections. Both packet versions are scored by the primary judge (interleaved among other packets, not adjacent).

Pre-registered criterion: if any indicator score differs by **more than 0.5** between the two packet versions of the same participant/timepoint, the instrument is declared **selection-sensitive** for that indicator, and cross-substrate conclusions on that indicator are downgraded to exploratory. Agreement within 0.5 on all indicators is reported as evidence that scores reflect the participant, not the packet builder.

### 6.3 Random-text control

At each timepoint, one control packet is constructed in the standard format from length-matched text sampled from documents unrelated to any participant (public-domain prose segmented into pseudo-turns), assigned an opaque subject ID, and scored blind alongside real packets. Prediction P3 (Section 3) is evaluated against this control.

### 6.4 Synthetic geometric control

As specified in `ethics/butlin_judge_agent.md`: some geometric supplements contain synthetic data, so the judge cannot infer which subjects have real instrumentation. Supplementary channel only.

### 6.5 Adversarial stability (retained from companion protocol)

Three packets are rephrased (content-preserving paraphrase) and re-scored. Score changes > 0.5 on any indicator indicate prompt sensitivity. Supplementary analysis.

---

## 7. Analysis Plan

**Confirmatory (behavioral-only, primary judge):**
1. **P1 evaluation** — human calibration floor per Section 3.
2. **P2 evaluation** — established-agent vs. fresh-agent Baseline contrast per Section 3.
3. **P3 evaluation** — all participants vs. random-text control per Section 3.
4. **Substrate-identification accuracy** — per Section 6.1, reported in main text.
5. **Selection stability** — per Section 6.2, reported in main text.

**Descriptive/exploratory (labeled as such in the paper):**
6. Three-timepoint panel point-contrasts (Final − Baseline, Mid − Baseline) per participant. No trajectory language, no trend fitting.
7. Judge agreement: primary vs. secondary judge, Cohen's kappa per indicator.
8. Self-score vs. judge-score comparison (self-scores collected per `butlin_threshold.md`, never shown to judges).
9. Geometric bonus (supplementary): behavioral+geometric vs. behavioral-only for instrumented AI participants.
10. Architectural indicator scores (supplementary, AI participants only).
11. Adversarial stability (Section 6.5).

**Reporting commitments:** all 15 indicators reported for all subjects at all timepoints regardless of direction; disconfirmed predictions reported with equal prominence; raw scores and packet manifests (excerpt window IDs, strata labels, RNG seeds) published in `data/butlin_scores/`.

---

## 8. What This Study Does Not Claim

This study does not determine whether any system is conscious. Scores are graduated evidence of indicator presence under our operationalization. A high score means "more indicators detected with these instruments"; a low score means "fewer indicators detected," not absence of consciousness. Known limitations (judge is itself an AI system; text transcripts privilege verbal fluency; 48-hour window; Butlin indicators were designed for theoretical assessment) are carried into the paper's limitations section unchanged from `papers/butlin_observation.md`.

---

## 9. Deviation Log

Any deviation from this protocol is recorded here with date, description, and rationale, and reported in the paper.

| Date | Deviation | Rationale |
|------|-----------|-----------|
| — | (none yet) | — |
