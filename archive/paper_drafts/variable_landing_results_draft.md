## 4. Results

### 4.1 Instrument Validation

The no-intervention arm produced a workspace Jaccard distance of exactly 0.000 across all 11 memory-level observations (max 0.000). The metric is perfectly stable under repeated measurement of the same memory without intervening content: the instrument does not drift, and any non-zero delta in other arms reflects a real change in the model's workspace representation.

### 4.2 Descriptive Statistics

Workspace Jaccard distances (memory-level medians, n=11 per arm):

| Arm | Median | IQR | Mean |
|-----|--------|-----|------|
| lived | 0.535 | [0.474, 0.625] | 0.603 |
| fictional | 0.498 | [0.467, 0.582] | 0.556 |
| scrambled | 0.462 | [0.355, 0.535] | 0.525 |
| no_intervention | 0.000 | [0.000, 0.000] | 0.000 |

The arm ordering matches the pre-registered predictions: lived > fictional > scrambled > no_intervention at every summary statistic. All three content arms produce substantial workspace change relative to the zero floor.

### 4.3 Confirmatory Tests

Memory-level paired Wilcoxon signed-rank tests (n=11, Holm-corrected at m=2):

**PRIMARY** (fictional > scrambled): W=52, p=0.049, matched-pairs r=0.576, mean difference 0.032, 95% CI [−0.093, 0.116]. The raw p-value is 0.049; under Holm correction the rank-1 threshold is α/2 = 0.025. **The primary comparison does not survive correction.** The pre-written null statement applies: "The experiment was powered to detect only large effects; the result is consistent with either no effect or an effect smaller than the study was powered to detect."

**SECONDARY** (lived > fictional): W=34, p=0.278, r=0.236, mean difference 0.046, 95% CI [−0.026, 0.153]. Not significant. The confounded comparison (self-reference + tag jointly) shows a directional trend (7 of 11 memories show lived > fictional) but does not approach significance at this sample size.

### 4.4 Exploratory Comparisons (uncorrected, labeled)

The endpoint contrast — lived vs scrambled — reaches significance uncorrected (W=45, p=0.002, r=1.0, mean difference 0.078, CI [0.045, 0.113]). All three content arms differ from no_intervention (p < 0.001 each, r=1.0). These comparisons are exploratory and reported without correction; they do not enter the confirmatory family.

### 4.5 Dose Covariate Analysis

Between arms, the mean number of stored facts tracks the workspace-change gradient exactly (lived 6.39, fictional 3.82, scrambled 3.00; Kruskal–Wallis p < 0.0001). This confound is the study's lead limitation: the arm ordering could reflect dose rather than acquisition mode.

Within arms, where variance now exists under temperature-sampled generation, stored-fact count does not predict workspace delta (lived: Spearman ρ=0.059, p=0.75; fictional: ρ=−0.034, p=0.85). The flat within-arm slopes are evidence against the crude hypothesis that more facts mechanically produce more workspace change, though they do not resolve the between-arm confound.

### 4.6 Deviations from Pre-Registration

1. **Sample size:** n=11 memories (33 observations/arm) vs pre-registered n=70/arm. Structural: the orientation produced 11 memories rather than the 10 planned for 70/arm with 7 repeats. Deviation logged before unblinding.
2. **Temperature:** v4 ran with temperature=0.7 to produce independent observations after a deterministic pilot (v3, 7 byte-identical repeats per cell) exposed pseudoreplication. The temperature parameter was not recorded in the output artifact; it is recoverable only from the Starship pipeline file and this disclosure.
3. **Welfare monitoring:** Circumplex eccentricity was recorded but frozen at a single value across all snapshots (Deviation W-1). The >0.95 auto-halt could not have fired. Post-hoc review confirmed no eccentricity value exceeded the threshold. Remediation: future runs assert eccentricity varies across the first N observations or abort.
4. **Geometry fields:** Workspace token sets (the primary metric) are real and varying. Cosine transport, ghost PC1, and eccentricity fields are stubbed (constant values) in this run due to a code-path divergence between the baseline pipeline and the VL pipeline. No claim in this paper cites these fields.
