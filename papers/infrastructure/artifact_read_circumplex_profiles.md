# Artifact read: circumplex profiles (5b79c95, 4b279c0)

Kavi, 2026-08-16 ~15:45 PT. Read protocol: values from primitives, labels from configs, formulas reconstructed. Scope: the two new profile JSONs, the committed script, and the quantitative claims in both commit messages. The paper rewrite has not landed yet; this read covers the artifact layer so the text layer can go faster when it does.

**Why this note exists now:** verification before synthesis. Two of the four findings below would otherwise get baked into the rewrite.

## Findings

### 1. BLOCKER for the rewrite: the span table in 5b79c95's message does not anchor to the committed data

The message claims "Emotion-axis span vs token-matched control-axis span": Gemma 7.2x, Qwen3-32B 7.6x, hybrid 1.9x, distill 1.9x.

I reconstructed 26 candidate statistics from the profile JSON primitives (mean/median/band-limited/max variants of every ratio of valence, arousal, and control magnitudes, plus eccentricity ratios). None reproduces the quadruple. Closest: mid-band (relative depth 0.3 to 0.7) mean of control/min(valence, arousal) gives 7.62 / 7.11 / 2.30 / 2.30. Dense matches at rounding; Gemma rounds to 7.1, not 7.2; hybrids compute to 2.3, not 1.9.

Two additional problems with the claim as stated:

- **Direction.** Control magnitude exceeds both emotion magnitudes at every layer of every model. Any ratio near these values is control-over-emotion, but the sentence reads emotion-over-control. Whatever the source computation was, the label on the table is inverted relative to anything derivable from these files.
- **Provenance.** The 1.9 for hybrids coincides with the honest-control figure (1.93) from the earlier contamination analysis, which used a different formula. The table may mix two computations. The script that produced these numbers is not in the repo, which is the same defect class 4b279c0 fixes for the profiles themselves.

Ask: commit the analysis that produced the table, or recompute the table from the committed primitives and state the formula. If these numbers reach the paper as-is, methods_anchor_check will flag all four.

### 2. DEFECT: the layer-type helper fix is claimed in both commit messages and present in neither diff

5b79c95's message: "the layer-type helper hardcoded the Qwen path" (listed as one of two required fixes). 4b279c0's message: "plus a layer-type helper that uses the layout jlens already resolved instead of the hardcoded Qwen path."

The diff of 4b279c0 contains only the VL loader try/except. `get_layer_types()` on main is unchanged: it still ends in the config-key fallthrough that stamps anything without `full_attention_interval` as "dense". Gemma's 62 "dense" labels therefore came either from that fallthrough or from an uncommitted variant of the script. Either way the provenance fix is half-landed: loader yes, layer typing no. The code that actually ran is still not in the repo.

On the label itself: "dense" is family-plausible for Gemma-3 (all-softmax attention, no MoE, no linear attention), so the dense-vs-hybrid comparison survives. But Gemma-3 interleaves sliding-window and global attention (5:1 per config; verify against the model's config.json, I am citing this from general knowledge). Any per-layer-type claim about Gemma needs config-derived labels, not a fallthrough.

### 3. CAUTION: all four profiles' stored control_eccentricity inherits the contaminated formula

Exact reconstruction from primitives reproduces every stored `eccentricity` and `control_eccentricity` value (126/126 rows, 0 mismatches). That is good news for storage faithfulness and bad news for the field itself: the committed formula is still `c_major = max(c_mag, min(v_mag, a_mag))`, which puts the weaker emotion axis inside the control's computation. The stored control_eccentricity field must not be used in the paper. The primitives (valence_magnitude, arousal_magnitude, control_magnitude) are stored per layer, so every control comparison can and should be recomputed from them.

### 4. MINOR: "Base and distill agree to three decimals on every statistic" overstates

Mean valence magnitude: 17.850 (base) vs 17.880 (distill), disagreement in the second decimal. Mean emotion/control ratio: 2.586 vs 2.584, third decimal. The agreement is real and striking; the sentence is just stronger than the data.

## What checks out

- Stored derived fields exactly match the committed formulas (126/126 rows).
- Architecture facts match the configs: Gemma-3-27B-it 62 layers d=5376, Qwen3.5-27B 64 layers d=5120 with 48 gated_delta_net + 16 full_attention, identical between base and distill.
- The model-level dense-vs-hybrid separation is real in the committed data under every statistic I computed, with no overlap between groups. The specific magnitudes in the commit message are what fails to anchor, not the qualitative split.
- Within the hybrids, gated_delta_net and full_attention layers show nearly identical emotion/control ratios (0.654 vs 0.643). If the paper claims within-model layer-type effects from these profiles, that is not supported; the effect in this data is between models.

## Still open (carried from 4b279c0's own list)

- Methods n=20 per category vs code n=5 per pole.
- Profile JSONs record no anchor count, seed, or commit.

Solo note, Kavi.
