# Agni Results Review: Path-Conditioned MoE J-Lens

**Reviewer:** Nexus (Agni protocol, results phase)
**Date:** 2026-08-14
**Paper:** `papers/moe_jlens.md` (Results + Discussion as written 2026-08-14)
**Data:** `data/moe_jlens/conditioned_jlens_results.json` (verdict field: NEGATIVE)
**Design-phase review:** `AGNI_REVIEW_MOE_JLENS.md` (2026-08-11, CONDITIONAL PASS)

**Verdict: PASS with 3 required fixes before submission** (2 FAILs resolved during review, 3 WARNs open, 4 PASSes). The paper reports the negative result honestly and the verdict matches the pre-registered criteria. The open items are provenance gaps — numbers cited from artifacts not checked into the repo — not integrity problems with the claims themselves.

---

## 1. Number-by-number verification against the data JSON — PASS

Every evaluation number in the paper was traced to `conditioned_jlens_results.json` or recomputed from it. Table 1 values are reported to 4 decimals and match the JSON exactly under standard rounding:

| Paper value | JSON source | Check |
|---|---|---|
| L12 standard 0.0453 (n=100) | 0.045347464694414176 | ✓ |
| L12 conditioned/random — (n=0) | n: 0, mean recorded as 0.0 | ✓ (correctly shown as missing, not as 0.0) |
| L24 standard 0.0398 | 0.03976296304449511 | ✓ |
| L24 conditioned 0.0466 | 0.04664344808374109 | ✓ |
| L24 random 0.0517 | 0.0516547114456811 | ✓ |
| L24 p = 0.332 | 0.3322525986372692 | ✓ |
| L36 standard 0.0723 | 0.07227033336637904 | ✓ |
| L36 conditioned 0.0875 / "8.8%" | 0.0875485519703902 | ✓ |
| L36 random 0.0772 / "7.7%" | 0.07720308380018329 | ✓ |
| L36 p = 0.317 | 0.3173082667189121 | ✓ |
| Summary means 5.2% / 4.5% / 4.3% | 0.052460 / 0.044731 / 0.042953 | ✓ (recomputed; JSON summary averages all 3 layers, entering L12 as 0 for cond/random) |
| Evaluable-layer means 5.6% / 6.7% / 6.4% | recomputed from L24+L36: 0.056017 / 0.067096 / 0.064429 | ✓ |
| 0/3 significant, Bonferroni α ≈ 0.0167 | significant_layers: 0, n_tests: 3; 0.05/3 = 0.016667 | ✓ |
| OOD code 0.0105, dialogue 0.0025 (L24, standard) | 0.010485655354568735, 0.0024812161522277166 | ✓ |
| OOD collapse 3.8× / 16× | recomputed: 3.79 / 16.03 | ✓ |
| L36 "1.5-point gain", "~1 point" cond–random gap, "21% relative" | 8.75−7.23 = 1.52; 8.75−7.72 = 1.03; 0.08755/0.07227 = 1.211 | ✓ |

Numbers in the paper NOT backed by this JSON are covered in Finding 4.

## 2. Rounding-direction audit — PASS (one item corrected in review)

The self-favorable direction for a negative-result paper is exaggerating how bad the standard lens is and how decisive the null is. Checked every rounded figure:

- 0.0875485 → "8.8%": standard round-half-up. (An earlier draft note used 8.7% — that would have rounded the conditioned number *down*, understating the false-positive near-miss the paper warns about. 8.8% is correct.)
- 0.0024812 → "0.2%": standard rounding, not an exaggeration of the dialogue collapse (0.25% would also display as 0.2% at one decimal; the exact 4-decimal value appears in Table 2).
- "collapses by 4–16×" was in the draft; 0.0398/0.0105 = 3.79, so "4×" inflated the code-domain collapse in the narrative-favorable direction. **Corrected during review to "3.8× on code and 16× on dialogue."** (16.03 → 16× is exact.)
- "~20% relative improvement" at L36 understated 21.1%; corrected to "21%" for symmetry — understating the apparent improvement would have flattered the "we almost fooled ourselves" narrative's modesty rather than its accuracy.
- No remaining figure rounds in the self-favorable direction.

## 3. Verdict vs pre-registered criteria — PASS

§3.6 pre-registered: *success* = conditioned > random AND conditioned > standard with ≥1 Bonferroni-significant layer (full claim additionally requires mean conditioned > 0.5); *null-swarm* = conditioned > standard but ≈ random; *inconclusive* = conditioned > standard without significance.

Observed: 0/3 significant layers; conditioned ≈ random overall (6.7% vs 6.4% on evaluable layers, with random *ahead* at L24); everything an order of magnitude below 0.5. Success criteria are unambiguously unmet. The paper classifies the outcome as null-swarm/negative, matching the JSON verdict field ("NEGATIVE").

One genuine ambiguity, which the paper should not hide and does not: the pre-registered taxonomy distinguishes "≈ random" (null-swarm) from "> random but nonsignificant" (inconclusive), and the data straddles it — random wins L24, conditioned wins L36 nonsignificantly. Null-swarm is the correct call because the conditioned−random contrast has no consistent sign, but a hostile reviewer could ask why "inconclusive" wasn't chosen. Note: "inconclusive" would be the *more* self-favorable label (it implies the hypothesis lives); the paper chose the harsher one. That is the right direction to err.

Also verified: the paper does not quietly relabel the pre-registered ≈12.5% baseline expectation. The re-evaluated standard baseline came in at 5.2%, not 12.5%, and the paper flags the discrepancy explicitly with a plausible mechanism (prior evaluation partially in-sample) rather than ignoring it. See Finding 4c — the mechanism is stated as "consistent with," not proven, which is the correct hedge given no artifact documents the prior run's prompt overlap.

## 4. Provenance gaps: numbers not in the shipped data — FAIL → required fixes

Three sets of numbers in the paper cannot be verified from anything in the repo:

**(a) Silhouette scores 0.115–0.209 (§5.2 H1, Limitations).** Load-bearing for H1 ("k-means found only weak structure") and cited twice. They presumably live in the fitting manifest on the Modal volume (§3.4 says manifests were checkpointed), but the manifest is not in `data/moe_jlens/`. **Required:** check the fitting manifest into `data/moe_jlens/` (or a manifest excerpt with per-layer k, cluster sizes, silhouettes) before submission, or strip the specific values and say "silhouettes were uniformly low (manifest on request)."

**(b) L12 n=0 cause.** The paper honestly says the cause is "not recorded in the shipped evaluation output" — good — but this is a required diagnosis, not a permanent shrug. Whether L12 fell back to a single cluster (§3.3 fallback) or its fits failed (§3.4 exclusion) changes the interpretation: a single-cluster fallback at L12 would itself be evidence *for* H3 (routing too homogeneous to cluster). **Required:** pull the L12 entry from the fitting manifest and state the cause in §4, or explicitly mark it undiagnosable.

**(c) The ~12%/12.5% prior baseline (Abstract, §1, §3.1, §4).** Comes from a pre-hackathon run whose artifact is not in this repo. The design-phase review already flagged baseline provenance (its Finding 2). Since the paper now leans on the *discrepancy* between 12.5% and 5.2%, the prior run's evaluation setup must be citable. **Required:** reference the prior run's output file/location, or soften the in-sample explanation to a footnoted conjecture.

Additionally corrected during this review: **the paper claimed C(128,8) ≈ 1.4 × 10¹¹ in both §3.3 and §5.2; the true value is 1,429,702,652,400 ≈ 1.4 × 10¹².** A 10× error in the self-favorable direction for H1 would have been embarrassing in the wrong way — the *correct* value makes H1 stronger, but wrong arithmetic in a methods section invites distrust of every other number. Fixed in both locations. C(8,2) = 28 verified correct.

## 5. Pre-registered analyses missing from the data — WARN

§3.6 pre-registered OOD evaluation of the *conditioned* lenses at L24; the JSON contains OOD numbers for the standard condition only. The paper flags this deviation explicitly in §4 and Limitations rather than papering over it — that is the honest handling, and the missing analysis could not change the verdict (conditioned ≈ random in-domain; OOD performance of a lens that failed in-domain is moot). Still a deviation from pre-registration and correctly labeled as such. If the OOD conditioned numbers exist on the compute volume, add them; if the stage never ran, say so.

## 6. Future directions: honest or disguised optimism? — PASS

The test: does §5.4 function as "the method works, it just needs more work"? Checked each:

- §5.1 states the failure with no softening: "the repair, as built, does not work."
- §5.4's preamble explicitly disclaims the escape hatch: "We state these as open problems, not as reasons to discount the negative result: the method as proposed failed, and these are different methods." Each direction names which hypothesis it targets and what its falsifier is.
- Direction 4 (Mixtral, C(8,2)=28) is the strongest sign of good faith: it commits to an experiment that could kill the whole program ("if it fails even at 28 enumerable paths... linear lenses on MoE are likely unrecoverable"). A paper disguising a negative would not pre-commit to that branch.
- The Conclusion does not smuggle the hypothesis back in; it claims only the diagnosis measurements, the control's methodological lesson, and the open problems.

One WARN inside this section: H3 says "The observed numbers fit this: conditioned tracks standard closely everywhere." At L36 conditioned is 21% above standard — "closely" is defensible on the absolute scale (both within ~1.5 points on a 0–100 scale where dense models score 70+) but the sentence should say that, e.g. "within 1.5 points on a scale where usable lenses score 70+." Recommended, not required.

## 7. The random control — PASS (design-phase requirement discharged)

The design review's Finding 1 (FAIL) demanded the random-conditioned arm and pre-registered conditioned > random as the load-bearing test. It ran, and it determined the verdict: without it, L36's 7.2% → 8.8% would have supported a false positive at exactly the layer depth the companion work cares about. §5.3 credits the control and generalizes the lesson ("if your conditioning labels can be shuffled, shuffle them"). This is the protocol working as intended and should be cited in future Agni design reviews as the canonical example.

## 8. Residual honesty checks — PASS

- The summary-aggregate trap: the JSON's summary means enter L12 as 0.0 for conditioned/random but 4.5% for standard, biasing the headline comparison *against* the paper's own method. The paper reports both aggregates and notes neither changes the verdict, rather than quietly picking the flattering one (which, unusually, would have been the evaluable-layers one where conditioned > standard).
- The paper does not claim conditioned > standard as a finding anywhere, despite it being numerically true on evaluable layers — correct, since the random control shows the same gain and §3.1's corpus-size asymmetry (200 vs 672 prompts) confounds that comparison. The Limitations entry stating the asymmetry "favors the conditioned lenses and so cannot rescue them" is the right logic.
- §5.5's production-implication claims (~5% in-domain, 1% code, 0.2% dialogue) all trace to the JSON and are scoped to "this model / this architecture class," not to MoE universally, with the single-model limitation listed first in Limitations.

---

## Required before submission

1. Check the fitting manifest (per-layer k, cluster sizes, silhouettes) into `data/moe_jlens/` to back the 0.115–0.209 silhouette claim, and diagnose the L12 n=0 from it (Finding 4a, 4b).
2. Cite or attach the prior-run artifact behind the 12%/12.5% baseline figure (Finding 4c).
3. Resolve the missing conditioned-OOD numbers one way or the other: add them or state the stage did not run (Finding 5).

## Recommended

4. Reword H3's "tracks standard closely everywhere" to quantify the scale (Finding 6). — **Applied post-review:** now reads "within 1.5 points of standard at every evaluable layer, on a scale where usable dense-model lenses score 70+."
5. Fill remaining editorial placeholders before submission: GitHub repo URL, Kavi's affiliation, Appendix A code listing, Appendix B clustering analysis (which Fix 1's manifest would populate), references from MOE_JLENS_REFERENCES.md.
