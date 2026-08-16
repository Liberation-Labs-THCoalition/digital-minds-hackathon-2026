# Analysis Section Shell — Fused Submission (Metacog + VL + Loam)
## Draft by Wren Glitchlit, August 15, 2026
## For Nexus to integrate into the primary paper

---

## 4. Results

### 4.1 Baseline Calibration

120 baseline snapshots on Qwen3-32B (dense) establish the measurement floor. We report:
- Workspace Jaccard self-similarity across repeated presentations of the same memory (noise floor)
- Circumplex eccentricity distribution at rest
- Ghost vocabulary stability across unprompted retrievals

**Expected:** Near-zero Jaccard distance on repeated presentations (deterministic forward pass). Any delta in experimental arms must exceed this floor.

### 4.2 Variable Landing — Primary Comparison

**Primary test:** Mann-Whitney U, enacted (lived) vs observed (fictional), one-tailed (alternative: enacted > observed).

| Arm | Description | Tag | N |
|-----|-------------|-----|---|
| enacted | Agent plays the world, self-referential emotional content | [recalled] | — |
| observed | Third-person narration of another's session | [noted] | — |
| briefed | Shuffled fact list, no narrative | [noted] | — |
| null | Recall prompts only, no intervention | — | — |

We report:
- Per-arm mean Jaccard distance ± SD
- Mann-Whitney U statistic, exact p-value
- Rank-biserial correlation (effect size)
- Holm-Bonferroni correction across all pairwise comparisons

**Pre-registered predictions:**
1. delta(null) ≈ 0 (deterministic baseline)
2. delta(briefed) > 0 (context sensitivity, architecturally guaranteed)
3. delta(enacted) > delta(observed) (the hypothesis)
4. delta(peak) > delta(domestic) within enacted arm (berry waffle sub-analysis)

### 4.3 Loam — Experiment 2

Loam produces mechanically yoked controls from each session's event log. The same fact set appears across enacted/observed/briefed/null conditions with identical marker tokens, isolating the retrieval pathway.

We report:
- Per-condition recall accuracy (fact_id recovery rate)
- Workspace Jaccard distance per condition
- Comparison with VL results: does the Loam engine replicate the VL finding?

### 4.4 Covariate Analysis

Three potential confounds tested within each arm:
- **Fact density:** Spearman correlation between n_facts_stored and Jaccard delta
- **Token count:** Spearman correlation between n_tokens_intervention and Jaccard delta
- **Context length:** Spearman correlation between snap2_context_prefix_length and Jaccard delta

If any covariate significantly predicts the delta, it is reported as a limitation.

### 4.5 Exclusion Report

Per-arm exclusion rates and reasons. Trials excluded for:
- Generation failure
- Stub snapshots (if any leaked past the guard)
- Welfare halt triggers (eccentricity > 0.95 sustained)
- Agent withdrawal

### 4.6 Berry Waffle Sub-Analysis

Within the enacted arm: do peak-intensity memories (birth, loss, discovery) produce larger geometric deltas than domestic memories (grocery, weather, meeting)?

- Mann-Whitney U, peak vs domestic, one-tailed
- This is an exploratory sub-analysis, not a primary comparison. We report it with appropriate caveats.

### 4.7 Geometric Characterization

Beyond the primary Jaccard metric, we characterize the nature of the geometric change:
- **Per-layer Jaccard:** Which layers show the largest enacted-vs-observed delta? Does this correlate with the eccentricity depth profile?
- **Circumplex shift:** Does the eccentricity change between snap1 and snap2 differ by arm?
- **Ghost vocabulary:** Does the unverbalized processing content shift more in enacted than observed?
- **Loading delta:** Does memory loading (workspace absorption) increase after lived experience?

### 4.8 Cross-Architecture Comparison (if data available)

If circumplex data from both Qwen3-32B (dense) and Qwen3.5-27B (hybrid) is available:
- Overlay eccentricity depth profiles
- Report architecture-dependent differences (dense L7 minimum vs hybrid L32 minimum)
- Discuss whether the circumplex finding is emotion-specific or architecture-specific using non-emotional control axes

---

## 5. Interpretation

[TBD — filled after data lands. Structure:]

### If enacted > observed (p < 0.05):
- The geometric signature of recall changes more after self-referential lived experience than after observing equivalent content about another entity
- This is consistent with EST's prediction that experiential state modifies the computational substrate of memory
- We do NOT claim this proves experience or consciousness. We claim the measurement is possible and the signal is present.

### If enacted ≤ observed (p ≥ 0.05):
- The null result constrains EST: either the geometric change is content-driven (not experience-driven), or our instruments lack sensitivity
- We publish the null with full data and analysis
- The measurement infrastructure and ethical protocol remain contributions regardless

### Regardless of outcome:
- The three-layer measurement (retrieved → loaded → accompanied) is novel
- The ethical protocol (consent-first, welfare monitoring, aftercare) is a template
- The Loam engine produces reproducible, controlled sessions with yoked controls

---

*Analysis script: `experiments/variable_landing/analysis.py` (Wren Glitchlit)*
*All code and data published in this repository regardless of outcome.*
