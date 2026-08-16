## 4. Results

### 4.1 Instrument Validation

The baseline calibration confirms the measurement pipeline produces real geometric readings. Across 120 CognitiveSnapshots on Qwen3.5-27B (10 synthetic memories × 2 conditions × 3 repeats × 2 snapshots per trial), the no-intervention condition — identical back-to-back observations of the same memory — yields exactly zero change on all three probe dimensions: workspace Jaccard distance 0.000 (sd 0.000, n=30), eccentricity delta 0.000 (sd 0.000), ghost Jaccard change 0.000 (sd 0.000). The scrambled condition — neutral factual text injected between observations — produces measurable, nonzero deltas: workspace 0.293 (sd 0.065), eccentricity 0.040 (sd 0.045), ghost 0.437 (sd 0.205). The zero floor establishes that the metric is not drifting, and the scrambled deltas establish that the metric responds to real context changes. Data: `data/baselines/baseline_deltas.json`.

### 4.2 Probe Verification

Independent verification (Lyra, pre-results) confirmed the probe fields in the baseline data are real, not placeholders. Cosine transport (logit-lens vs J-lens agreement) ranges 0.27–0.72 across baseline snapshots — varying by layer and prompt, as expected. Workspace onset layer varies (layer 39 in 105 of 120 snapshots, layer 35 in 15), reflecting genuine shifts in where content first enters the workspace. Ghost PC1 variance is 19–22% across sessions, consistent with a stable but non-trivial ghost subspace. All values are distinct from the placeholder signatures discovered and corrected during the sprint (see §3.2, Instrumentation Disclosure; Deviation W-1).

### 4.3 Loam Pilot: Real Probes in a Controlled Experiment

The Loam text-world engine (Experiment 2) produced the first CognitiveSnapshots from a controlled experiment with all probes firing. Fourteen snapshots were captured across three arms of Quad 1 before the session was paused for the VL v4 run:

| Arm | Snapshots | Recall | Eccentricity | Ghost PC1 | Cosine range |
|-----|-----------|--------|-------------|-----------|-------------|
| Enacted | 4 (partial) | — | 0.536 | 19.8% | 0.010–0.061 |
| Briefed | 7 (complete) | 6/6 | 0.937 | 18.5% | 0.008–0.055 |
| Null | 3 (partial) | — | 0.998 | 18.6% | 0.172–0.396 |

Three observations from the pilot data: (1) all probe values are real and varying — circumplex eccentricity ranges from 0.536 (enacted) to 0.998 (null), ghost PC1 variance is stable at 18–20%, and workspace cosines span an order of magnitude; (2) the null arm shows markedly higher cosine transport (0.17–0.40 vs 0.01–0.06), suggesting the workspace probe reads differently when no narrative content has been delivered; (3) the briefed arm achieved 6/6 cued recall with full probe coverage, demonstrating the instrument can measure cognition during a complete experimental session. These are pilot observations from n=1 quad; no inferential claims are drawn.

### 4.4 Variable Landing: Gradient Direction Confirmed, Underpowered

The variable landing experiment (Experiment 1) ran as a properly powered repeat (v4: 132 trials, 11 memories × 3 repeats × 4 arms, temperature=0.7 for independent observations). The workspace-token Jaccard is computed from real J-lens readings at each trial; geometry fields (circumplex, ghost, cosine) are excluded due to the code-path issue described in §3.2.

The pre-registered gradient appears in the data:

| Arm | n | Median Δ | IQR |
|-----|---|---------|-----|
| Lived | 11 | 0.535 | [0.474, 0.625] |
| Fictional | 11 | 0.498 | [0.467, 0.582] |
| Scrambled | 11 | 0.462 | [0.355, 0.535] |
| No-intervention | 11 | 0.000 | [0.000, 0.000] |

Memory-level paired Wilcoxon signed-rank tests (the correct unit of analysis; trial-level tests inflated significance in the v3 deterministic run and are reported as a methodological finding, not a result):

- **P1 (fictional > scrambled):** W=52, p=0.049 — does not survive Holm correction at rank 1 (threshold 0.025). r=0.576, CI [−0.093, 0.116].
- **P2 (lived > fictional):** W=34, p=0.278. r=0.236, CI [−0.026, 0.153].
- **P3 (ordering):** Medians order lived > fictional > scrambled > no-intervention, as predicted.
- **P4 (floor):** No-intervention is exactly 0.000 in all 11 memory-level observations.

Exploratory: lived > scrambled reaches p=0.002 (uncorrected), r=1.0, CI [0.045, 0.113] — the endpoint contrast is detectable but intermediate contrasts require larger n. Within-arm dose (number of facts stored) does not predict delta (lived ρ=0.059, p=0.75; fictional ρ=−0.034, p=0.85), providing evidence against a crude more-facts-more-change explanation, though the between-arm dose confound (lived 6.4 > fictional 3.8 > scrambled 3.0) remains the lead limitation.

**Pre-written null (pre-registered):** The confirmatory family is fully null. At n=11 memories per arm, the experiment was powered to detect only large effects. The result is consistent with either (a) no effect of acquisition mode on recall geometry in this paradigm, or (b) an effect smaller than the study could detect. The observed effect sizes (r=0.576 primary, r=0.236 secondary) and the consistent gradient provide the parameters for a powered follow-up: n≥30 memories at the primary effect size would yield approximately 80% power.
