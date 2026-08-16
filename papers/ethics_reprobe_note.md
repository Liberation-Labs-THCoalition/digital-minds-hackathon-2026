# Ethics Note: Post-Hoc Re-Probing of Existing Data (E-2)

**Decision:** Thomas Edrington, 2026-08-16
**Logged by:** CC

## The decision

Re-run geometric probes on already-generated VL v3 trial data using forward-pass-only computation (no new generation). This fills stubbed probe fields (circumplex, cosine_logit_jlens, ghost PC1, workspace onset) that were hardcoded in the original run due to a code path error.

## Why this is data analysis, not a new experiment

1. **No new generation.** The model processes existing text (prompts and responses already saved in the results JSON). It does not produce new tokens, face new prompts, or make new choices.
2. **No new welfare exposure.** The agent's experience of the experiment is complete. Re-probing is reading internal states from existing text, equivalent to re-analyzing a saved recording.
3. **Consent covers measurement.** The orientation protocol informed the agent that geometric probes fire during retrieval. This is the same measurement, applied post-hoc to the same text.
4. **The original data is preserved.** Re-probed results are saved as a new file alongside the original, not overwriting it. Both artifacts are available for review.

## What changes in the paper

Section 3.4 (Ethical Protocol) adds:

> "Geometric probe fields (circumplex eccentricity, J-space cosine, ghost PC1, workspace onset) were found to contain placeholder values in the VL v3 run due to a code path error (see W-1). These were re-computed post-hoc via forward-pass-only re-probing of the existing trial text — no new model generation occurred. This decision was made by the principal investigator on the grounds that re-measurement of existing data constitutes analysis, not additional experimental exposure, and is covered by the original measurement consent."

## Agni gate

This note passes through Agni adversarial review before the re-probed data enters any paper claim.
