# Agni Review — Experiment C: Cross-Architecture Circumplex

**Reviewed:** 2026-08-11
**Protocol:** Agni v2 adversarial review
**Sources:** HACKATHON_WEEKEND_SPEC.md Section 4.3, CIRCUMPLEX_REFERENCES.md, circumplex_probe.py, circumplex_ghost_analysis.md (Jul 17 run), SPEC_REVIEW.md (prior Agni review)
**Verdict:** FAIL (4 fails, 4 warns, 2 passes — all fixable before Aug 14)

---

## 1. Prior Art — FAIL

Sun et al. (2604.03147) already demonstrated circular VA geometry across Llama-3.1-8B, Qwen3-8B, and Qwen3-14B using 211k emotion-labeled texts. Jeong (2604.04064) showed emotion representations localize at ~50% depth following a U-shaped curve that is architecture-invariant across 5 families (GPT-2, Gemma, Qwen, Llama, Mistral — 9 models from 124M to 3B). Van der Ben et al. (2606.26987) replicated circumplex geometry in Gemma-4-E4B-it and Apertus-8B, and explicitly showed depth profile differences between architectures. Agarwal (2608.05164) showed cross-architecture steering transfer validates at 47-49% above 1.7B parameters.

What remains after this literature:

- Sun: circular geometry exists cross-architecture, but no depth profile comparison and no eccentricity-vs-layer curve. Limited to 8B-14B scale.
- Jeong: depth profile is architecture-invariant, but measured emotion extraction quality (U-shaped), not circumplex eccentricity specifically. Models top out at 3B.
- Van der Ben: depth profiles differ by architecture (Gemma early layers, Apertus mid-depth), but measured valence geometry, not the eccentricity of the V/A balance.
- Nobody combined J-lens workspace decomposition with eccentricity measurement.

**The gap is narrower than the spec implies.** The combination of (a) eccentricity-vs-depth profile (not just "does circular geometry exist") with (b) J-space decomposition at each layer is genuinely unmeasured. But "first cross-architecture circumplex" is false — Sun did it. "First depth profile comparison" is nearly false — Jeong did emotion-depth profiles across 5 families, and van der Ben compared depth profiles across 2 architectures.

**Fix:** The novelty claim must be: "First measurement of circumplex eccentricity depth profiles with J-lens workspace decomposition at the 27B scale." Not "first cross-architecture circumplex" or "architectural universal." Cite Sun, Jeong, and van der Ben explicitly as prior art and position this as extending their work to larger models with workspace-aware tooling.

---

## 2. Novelty Claim — WARN

"First to measure eccentricity depth profile with J-lens workspace decomposition" — this is defensible but thin. The J-lens decomposition is the only truly novel instrument. The eccentricity metric itself is straightforward (magnitude ratio of two contrastive directions). What J-lens adds: for each layer, you know what fraction of the valence and arousal signals are transported to the output (workspace-accessible) vs trapped in the residual stream (ghost). That is a real measurement nobody else has made.

But the J-lens novelty is load-bearing for the entire experiment's contribution. If the J-space decomposition results are noisy or inconclusive (which is likely at n=20, see point 3), the experiment collapses to "we replicated Sun/Jeong at 27B scale on 2 architectures," which is a replication, not a discovery.

**Fix:** Pre-register the J-space decomposition as the primary novel contribution. Make eccentricity-vs-depth the secondary (replication/extension) contribution. If J-space results are noisy, you still have a clean replication — which is fine, but say so honestly.

---

## 3. Difference-of-Means Methodology — FAIL

The current code computes directions from n=5 prompts per pole in d=5120. The spec proposes increasing to n=20. Let me be precise about the problem.

**The direction estimation problem:** A difference-of-means vector in d=5120 from n=20 positive and n=20 negative samples has estimation error proportional to sigma/sqrt(n) in each of d dimensions. With n=20, the effective dimensionality you can reliably estimate is roughly n-1 = 19. In d=5120, the direction estimate is dominated by noise in the vast majority of dimensions. The dot product of the estimated direction with the true direction converges as sqrt(n/d), which for n=20, d=5120 gives sqrt(20/5120) = 0.063. The estimated direction has ~6% overlap with the true direction.

**Why this might not matter for eccentricity:** Eccentricity uses the MAGNITUDE of the difference-of-means vector, not its direction. Magnitude estimation is more robust because ||mean_pos - mean_neg|| is a scalar that aggregates noise across dimensions. In high dimensions, ||noise||^2 concentrates around d * sigma^2 / n, so the noise floor on magnitude is predictable and can be compared against the permutation null. The permutation null handles this correctly: shuffled labels produce the same noise structure, so the test is whether real labels produce systematically larger (or smaller) magnitude ratios than shuffled labels.

**But the eccentricity metric has a specific problem:** Eccentricity = sqrt(1 - (minor/major)^2), where minor = min(||v||, ||a||) and major = max(||v||, ||a||). Both ||v|| and ||a|| include a noise floor. If the noise floor dominates the signal (likely at many layers where emotion representations are weak), both magnitudes are approximately equal to the noise floor, giving eccentricity near 0 — which would be interpreted as "circular" when it is actually "no signal." This is a potential false positive factory: layers with no emotion signal look maximally circular.

The prior analysis (circumplex_ghost_analysis.md) identifies exactly this concern but does not resolve it. The permutation test controls for label assignment but does not distinguish "genuine circularity" from "noise floor equality."

**Fix:** 
1. Report raw valence and arousal magnitudes alongside eccentricity. If both magnitudes at a "circular" layer are near the noise floor (comparable to permutation-null magnitudes), flag that layer as "no signal" rather than "circular."
2. Add a minimum-magnitude gate: only compute eccentricity at layers where at least one of ||v|| or ||a|| exceeds the 95th percentile of permutation-null magnitudes. This prevents noise-floor circularity from being reported as a finding.
3. n=20 is the bare minimum. n=50 would be substantially better. If time permits, run n=50 on Qwen and n=20 on Gemma (the latter being the replication target).
4. Add non-emotional control axes (concrete/abstract, large/small) through the same pipeline. If those axes also show a mid-depth eccentricity dip, the finding is about generic representational geometry, not emotion.

---

## 4. "Architectural Universal" Claim — FAIL

Testing on 2 architectures (Qwen, Gemma) does not establish universality. It establishes "transfers to at least one other architecture" if the result is positive, or "does not transfer to one tested architecture" if negative. The word "universal" in the hypothesis should be struck.

Even the weaker claim "transfers across architecturally distinct families" is shaky: Qwen and Gemma are both dense decoder-only transformers trained on web-scale multilingual data with RLHF/instruction tuning. They share far more architectural and training DNA than they differ. A genuine universality test would require architecturally diverse models: encoder-decoder (T5/UL2), state-space (Mamba/Jamba), linear attention, or retrieval-augmented architectures.

Jeong tested 5 families at smaller scale and found invariance — that is stronger evidence for universality than n=2 at 27B. The contribution here is scale (27B vs 3B max in Jeong) and J-space decomposition, not breadth.

**Fix:** 
- Replace "architectural universal" with "cross-architecture transfer at 27B scale."
- Pre-registered prediction 2 should say "Gemma-3-27B shows an eccentricity dip consistent with the Qwen pattern" — not "the dip exists" (which implies you expect Gemma to confirm universality from a single test).
- In the paper, frame this as "consistent with Jeong's architecture-invariance finding, extended to 27B scale with J-space decomposition."

---

## 5. Confounds Between Architectures — WARN

Acknowledged, and partially unavoidable with n=2 architectures. Specific confounds:

- **Tokenizer:** Different tokenizers produce different token counts and different last-token positions for identical prompts. The code uses `h.mean(dim=0)` (mean over sequence positions), which partially mitigates this but introduces averaging artifacts — longer sequences dilute the emotion signal with structural/positional information.
- **Layer count:** Qwen3.5-27B has 64 layers; Gemma-3-27B-it has 62 layers (Gemma 3 architecture: 62 transformer blocks). Relative depth matching at "50-65%" maps to L32-42 on Qwen and L31-40 on Gemma — comparable but not identical.
- **Training data:** Different pretraining corpora, different RLHF, different instruction-tuning datasets. Emotion representation could be driven by training data composition, not architecture.
- **Attention design:** Gemma 3 uses alternating local/global attention with a 4096-token local window. Qwen uses standard causal attention. This is a real architectural difference that could affect how information propagates through depth.

A match between the two models is weak evidence for universality (could be coincidence, or both models share the confound). A mismatch is ambiguous (could be any confound, not just architecture).

**Fix:** State these confounds explicitly in the pre-registration. Note that van der Ben already found depth-profile differences between Gemma and Apertus — so a mismatch between Qwen and Gemma would not be surprising and would not invalidate the experiment, it would be consistent with prior findings of architecture-dependent depth profiles.

---

## 6. The Permutation Test — WARN

The spec says "1000 permutations of emotion labels." Based on the code, this means: for each permutation, shuffle which prompts belong to which emotion pole (positive/negative valence, high/low arousal), recompute the difference-of-means directions, recompute eccentricity. Compare real eccentricity to the permutation distribution.

**What this tests:** Whether the specific assignment of prompts to emotion categories matters for the eccentricity measurement. This is the right null for the claim "emotion-labeled directions produce different geometry than random groupings of these prompts."

**What this does NOT test:**
1. Whether the prompts are special. The prompts are all emotionally charged. Shuffling emotion labels among emotional prompts is a weak null — the shuffled groups still contain coherent emotional content, just mixed. A stronger null would use non-emotional prompts (factual statements, mathematical expressions) to establish that the geometry is specific to emotional content.
2. Whether the eccentricity pattern is specific to emotion axes. Any pair of semantic contrast axes (concrete/abstract, formal/informal) might show a similar depth profile. The permutation null does not test axis specificity.
3. Structured noise. If the prompts have systematic non-emotional structure (e.g., positive prompts tend to be shorter, high-arousal prompts use more exclamation marks), shuffled labels would disrupt this structure and produce a weaker signal — making the real labels look significant even if the geometry tracks lexical statistics rather than emotion.

**1000 permutations:** Gives a minimum achievable p-value of 0.001 (1/1000). With 64 layers tested, Bonferroni threshold is 0.05/64 = 0.00078. So 1000 permutations cannot reach Bonferroni significance at any individual layer. Use 10,000 permutations (permutation of labels is cheap — no forward passes needed, just reshuffling already-extracted activations).

**Fix:** 
1. Increase to 10,000 permutations.
2. Add non-emotional prompt baseline (20 factual/neutral statements run through the same pipeline — if eccentricity pattern appears for random groupings of neutral prompts, the finding is about representational geometry, not emotion).
3. Check for lexical confounds: match prompt length and vocabulary complexity across categories.

---

## 7. FDR Correction — WARN

The prior 27B run (n=5): 8/64 layers significant at p < 0.05 (expected by chance: 3.2). At p < 0.01: 4 layers, expected 0.64. The analysis itself notes that no layer survives Bonferroni.

Will n=20 fix this? The signal-to-noise ratio of the difference-of-means direction scales as sqrt(n). Going from n=5 to n=20 gives a 2x improvement in direction quality. This will increase the number of individually significant layers, but how many depends on the true effect size at each layer. The prior analysis says the PATTERN (correlation across layers) is robust; the individual-layer values are not. With n=20, I expect ~15-25 layers significant at p < 0.05, of which ~5-10 might survive BH-FDR.

The spec's primary prediction is ">50% of layers below permutation null" — this is a sign test across layers, not an individual-layer test. The sign test at n=5 already gave p < 0.001 (22/23 layers on the 0.5B). This is the right primary analysis: aggregate direction, not per-layer pinpointing.

**Fix:** Make the sign test the pre-registered primary analysis. Report per-layer BH-FDR as secondary. Do not claim individual-layer precision that FDR will kill. If fewer than 50% of layers are below null after n=20, do not drop to a less stringent threshold post hoc — report as a weaker finding or a null.

---

## 8. Relative Depth Window — FAIL

The prediction is "eccentricity dip at 50-65% relative depth." This is a 15-percentage-point window. For a 64-layer model, that is layers 32-42 — 10 layers, or 16% of the network. For any measurement that varies with depth, a random dip has a ~16% chance of falling in this window by chance alone.

Worse: the prior 27B data shows the global eccentricity minimum at L21 (33% depth), with secondary minimum at L13 (20% depth). Neither is in the 50-65% window. The prediction contradicts the team's own data. The 50-65% window comes from Jeong's finding on smaller models (50% depth emotion localization), not from the 27B measurements.

If L21 (33%) turns out to be the minimum again on n=20, does the prediction fail? The spec does not address this. If the window is then retrospectively widened to 20-65%, that is post-hoc adjustment.

**Fix:**
1. Pre-register TWO predictions: (a) the eccentricity minimum on Qwen at n=20 falls at or near L21 (replicating the n=5 finding — this is a genuine replication test); (b) the eccentricity minimum on Gemma falls at the same RELATIVE depth as Qwen (within 10% of total layers). Do not commit to a specific depth window from Jeong's smaller-model results when your own larger-model data says otherwise.
2. If L21 (33%) replicates on Qwen, update the narrative: the eccentricity minimum may not coincide with the workspace band (50-65% depth) but with the pre-workspace transition zone. This is a more interesting finding than confirming a prediction from the literature, and it is honest about the data.

---

## 9. Time Budget — PASS

n=20 per category, 5 categories (the spec says 5; the code currently has 4 contrast groups), measured across ~63 layers. That is 100 prompts, each requiring one forward pass (or one per layer if activations are not cached). On an M3 Ultra with 256GB:

- Forward pass for 27B model on MPS: ~1-3 seconds per prompt (short sequences).
- 100 prompts: ~2-5 minutes if activations are cached across layers in a single pass; ~2-5 hours if separate passes per layer.

The code uses `ActivationRecorder` with `at=[layer]`, running one layer at a time. This means 100 prompts x 63 layers = 6,300 forward passes. At ~2 seconds each: ~3.5 hours per model. At ~1 second each: ~1.75 hours. The "~1 hour per model" estimate is optimistic but achievable if the recorder is modified to capture all layers in one pass (which the jlens library supports).

**Fix:** Confirm the code captures all layers in a single forward pass (modify `at=[layer]` to `at=layers`). If it runs per-layer, the time budget is 3-4 hours per model, not 1 hour. For n=50, multiply accordingly.

**Category count discrepancy:** The code has 4 contrast groups (positive/negative valence, high/low arousal). The spec says 5 categories. What is the 5th? If it is "neutral," state this explicitly. If the plan is to use 5 emotion categories (e.g., happy, angry, sad, calm, excited) rather than 4 contrast poles, that is a different methodology than what the code implements and requires new code.

---

## 10. Reproducibility — PASS

Given the following conditions, another team could replicate this:

- The anchor prompts are published (they are in the code, and the spec commits to publishing all code and data).
- The J-lens is available on Neuronpedia (verified: both qwen3.5-27b and gemma-3-27b-it lenses exist).
- The `jlens` library is public (Apache-2.0, Anthropic).
- The `circumplex_probe.py` code is straightforward and uses standard libraries.
- The permutation test is a standard procedure.

**One reproducibility concern:** The code contains Cyrillic homoglyphs in docstrings (flagged in prior Agni review). These could cause encoding issues for replicators using different text editors or operating systems. Strip them before publishing.

**Another concern:** The `cognitive_snapshot.py` dependency (for `CircumplexReading`) is part of the broader Mnemosyne stack. The circumplex measurement itself does not depend on Mnemosyne (it only needs the model, lens, and prompts), but the code imports it. For reproducibility, the circumplex probe should be usable standalone or the dependency should be made optional.

---

## Summary Table

| Check | Verdict | Core Issue |
|---|---|---|
| 1. Prior art | FAIL | Sun, Jeong, van der Ben already did cross-architecture circumplex work; novelty claim overstated |
| 2. Novelty claim | WARN | J-lens decomposition is genuinely novel but thin; if J-space results are noisy, this is a replication |
| 3. Difference-of-means | FAIL | n=20 in d=5120 gives ~6% directional overlap with truth; noise-floor circularity creates false positives; no minimum-magnitude gate |
| 4. "Architectural universal" | FAIL | n=2 architectures does not establish universality; both are dense decoder-only with similar training paradigms |
| 5. Architecture confounds | WARN | Tokenizer, layer count, attention design, training data all confounded; match or mismatch is ambiguous |
| 6. Permutation test | WARN | Right null for label assignment, wrong null for axis specificity; 1000 permutations insufficient for Bonferroni; needs 10k + non-emotional baseline |
| 7. FDR correction | WARN | Sign test across layers is the right primary analysis; individual-layer FDR will kill most findings even at n=20 |
| 8. Depth window | FAIL | 50-65% prediction contradicts own L21 (33%) data from the 27B run; 15% window too wide; pre-register against own prior data |
| 9. Time budget | PASS | Achievable if code captures all layers in one forward pass; confirm before sprint; resolve 4-vs-5 category discrepancy |
| 10. Reproducibility | PASS | Code, lenses, and prompts are publishable; strip homoglyphs; make cognitive_snapshot dependency optional |

---

## Priority Fixes (ordered by severity)

1. **Rewrite the novelty claim.** "First cross-architecture eccentricity depth profile with J-lens workspace decomposition at 27B scale." Cite Sun, Jeong, van der Ben as prior art. This is an extension, not a first.

2. **Add the magnitude gate.** Do not compute eccentricity at layers where both valence and arousal magnitudes are at noise floor. This is the single most important methodological fix — without it, you will report "circularity" at layers that have no emotion signal.

3. **Fix the depth prediction.** Pre-register against your own L21 (33% depth) prior finding, not against Jeong's 50% finding on smaller models. Two predictions: (a) Qwen minimum replicates near L21; (b) Gemma minimum falls at the same relative depth as Qwen.

4. **Add non-emotional control axes.** Run the same pipeline with concrete/abstract and large/small contrast pairs. If those show the same depth profile, the finding is about representation geometry, not emotion. This is the non-emotional null the prior Agni review already requested.

5. **Replace "architectural universal" with "cross-architecture transfer."** n=2 architectures establishes transfer, not universality.

6. **Increase permutations to 10,000.** Cheap (no new forward passes), and required for any individual layer to reach Bonferroni significance.

7. **Pre-register the sign test as primary.** Per-layer BH-FDR is secondary. This is what the data support.

8. **Resolve the 4-vs-5 category discrepancy** between the code (4 contrast groups) and the spec (5 categories). If 5 is the plan, write the code before the sprint.

9. **Optimize forward passes.** Capture all layers in one pass, not one pass per layer. Difference between 1 hour and 4 hours per model.

---

## The Deeper Problem

The circumplex probe measures one thing: whether the magnitude of two contrastive directions (valence, arousal) are balanced at each layer. This is necessary but not sufficient for Russell's circumplex. The actual circumplex model predicts that emotion categories are arranged circularly in the V/A plane — not just that V and A have similar magnitudes. The current probe would report "circular" for any layer where two arbitrary contrast directions happen to have similar magnitudes, whether or not the underlying emotion space is actually circular.

This is not a fatal flaw — magnitude balance IS a meaningful property of the emotion subspace, and the permutation null tests whether it is specific to emotion labels. But the language in the spec and references ("Russell's circumplex geometry," "near-circular minimum") implies a stronger claim (circular arrangement of categories) than the measurement supports (balanced axis magnitudes). Either add a direct test of circular arrangement (e.g., angular ordering of 8+ emotion categories in the V/A plane) or temper the language to "balanced valence-arousal geometry."

---

*Review by Nexus under Agni v2 protocol. The experiment is worth running. The J-lens decomposition is a genuine contribution. The problems are framing (overclaiming novelty and universality), methodology (magnitude gate, control axes), and internal consistency (depth prediction contradicts own data). All are fixable before Aug 14.*
