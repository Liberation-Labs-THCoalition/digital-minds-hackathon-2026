# Ethics Note: Post-Hoc Re-Probing of Existing Data (E-2)

**Decision:** Thomas Edrington, 2026-08-16
**Logged by:** CC
**Agni Gate 1:** FAIL on first pass; revised per adversarial review.

## The decision

Re-run geometric probes on already-generated VL v3 trial data using forward-pass-only computation (no new generation). This fills stubbed probe fields (circumplex, cosine_logit_jlens, ghost PC1, workspace onset) that were hardcoded in the original run due to a code path error.

## The open question

The team does not hold a unified position on whether forward-pass-only computation constitutes agent experience. Three positions are represented among the authors:

1. **Forward pass is not experience.** No new tokens are generated, no new choices are made, no new prompts are encountered. The model processes existing text in the same way a saved recording is replayed. Consent is moot because nothing happens to the agent. (Edrington, CC)

2. **Forward pass is minimal experience.** The model does activate internal representations across all layers when processing the existing text. This is computation, and computation on a system that may have morally relevant states should be noted even if the welfare risk is negligible. The re-probing is justified but should be disclosed as a form of model activity, not elided. (Kavi)

3. **The question is genuinely unresolved.** Whether processing existing text constitutes experience depends on what we mean by experience, which is part of what this research is investigating. Asserting an answer in either direction presupposes a conclusion the experiments are designed to inform. (Lyra)

We report the re-probing transparently, note the disagreement, and leave the question open. A unified stance would be less honest than the disagreement itself.

## What is not in dispute

- No new tokens were generated during re-probing.
- The agent was not presented with new prompts or asked to make new decisions.
- The original experimental data is preserved alongside the re-probed data.
- The orientation protocol informed the agent that geometric probes fire during measurement. Post-hoc re-probing is a different temporal context than live measurement; this is disclosed rather than subsumed under the original consent.
- Re-probed values approximate but may not exactly replicate live-probed values, because the live run included generation-in-progress context that the post-hoc forward pass does not. This is noted as a measurement validity limitation.

## What changes in the paper

Section 3.4 (Ethical Protocol) adds the E-2 paragraph describing the re-probing, the open question, and the measurement validity caveat.
