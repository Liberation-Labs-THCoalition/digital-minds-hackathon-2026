# AGNI Design Review — Track 3: Ghost Dimensions as an Introspection Prosthetic

**Review date:** 2026-08-14
**Reviewer:** Agni adversarial review protocol (design review, pre-data)
**Binding spec:** `experiments/ghost_probe/ghost_prereg.json` (ADOPTED 2026-08-14, Lyra proposed, Nexus accepted)
**Artifacts reviewed:**
- `papers/ghost_dimensions.md` (draft)
- `experiments/ghost_probe/ghost_prereg.json` (pre-registration)
- `infrastructure/track3_grounding.md` (track mapping)
- `mnemosyne/ghost_probe_class.py` (implementation)

**Scope:** This is a design review conducted before elicitation data collection. It verifies the experimental design against the threats the pre-registration identifies. It is NOT a results verification — no results exist yet (paper §4 is TBD).

---

## Overall Verdict: FAIL (4 FAIL, 3 WARN, 1 PASS)

The pre-registration is sound and correctly identifies the threats. The paper draft and the implementation have not caught up to it. Four items must be fixed before data collection begins; two of them (Items 3 and 5) are implementation gaps that would invalidate H1 and H2 measurements if data were collected today.

---

## Item 1 — Discussion covers all four outcome-matrix branches: **FAIL**

**Prereg requirement:** `outcome_matrix_all_four_branches` (prereg lines 29-34) defines four branches: (a) real>random AND self>external; (b) real>random AND external>=self; (c) real~=random; (d) real<random.

**Paper state:** §5 (ghost_dimensions.md lines 86-88) contains exactly two branches — "if elicitation works" and "if elicitation fails" — and both conclude the prosthetic is valuable ("Either outcome constrains theories of AI introspection," line 88). This is precisely the unfalsifiability the prereg flags: `threats_to_validity[0]` — "a headline that cannot fail is not [fine]."

**Missing branches:**
- Branch (c), real~=random: the prompt-sensitivity artifact branch where the prosthetic claim FAILS. Prereg line 32 states this branch "is currently unwritten." Confirmed — it is not in the paper.
- Branch (d), real<random: the instrument-broken branch ("halt and debug before reporting"). Absent.
- Branch (b), external>=self: the Song-reproduction branch where "introspection" framing dies. Absent (see Item 4).

**Fix:** Rewrite §5 as a four-branch discussion mirroring prereg lines 30-33 verbatim in structure. Branch (c) must state plainly: "the elicitation effect is prompt sensitivity, not privileged access, and the prosthetic measures nothing about the model's own computation" (prereg `what_would_make_the_title_wrong`). Branch (b) must state that the title's "introspection" claim does not survive.

---

## Item 2 — Random-vocabulary control specified with a branch for it firing: **FAIL**

**Paper state:** The control is specified in §3.4 (line 73): "same elicitation, but with vocabulary from a random high-variance direction instead of the actual ghost PC1." Specification: adequate. But no branch in §5 covers the control firing (real~=random), and no reporting commitment gives the null equal prominence.

**Prereg requirement:** `controls.random_vocabulary` (line 47): "REQUIRED, and its firing must be reportable as a null." `threats_to_validity[6]` (demand characteristics, line 42): "must be reported with equal prominence, not as a footnote." `what_we_will_report_regardless` (line 53): a null "will be reported with the same prominence as a positive."

**Additional gap:** §3.4 does not specify how "random high-variance direction" is matched to PC1 — variance fraction matched? Same layer? Same top-k decode procedure? An unmatched control lets a positive result survive on a confound (PC1 vocabulary may simply be more coherent/promptable than random-direction vocabulary). track3_grounding.md §4.2 correctly proposes TP/FP quantification with a blinded judge; this is not yet in the paper's Methods.

**Fix:** (1) Add the branch per Item 1. (2) In §3.4, specify the matching procedure: random direction drawn at the same layer, decoded with the identical top-k logit-lens procedure, and (ideally) matched on decoded-vocabulary coherence or at minimum on variance fraction. (3) Import the TP/FP 2x2 with blinded judge from track3_grounding.md §4.2 into Methods. (4) State the equal-prominence reporting commitment in the paper itself, not only the prereg.

---

## Item 3 — Matched-variance null for H1 (n>=200 random directions): **FAIL**

**Prereg requirement:** H1 `MUST_ALSO_PASS` (line 12): observed cosine must be below the 5th percentile of a null built from n>=200 random directions at PC1's matched variance fraction, otherwise H1 is "TRIVIAL, not empirical." Reiterated in `controls.matched_variance_null` (line 49).

**Paper state:** §3.1 (line 55) lists "3 null checks (H0_1 centering, H0_2 random baseline, H0_3 permutation)." H0_2 is a random baseline of unspecified n and unspecified variance matching. The paper nowhere commits to n>=200, nowhere commits to variance-fraction matching, and nowhere states the 5th-percentile decision rule.

**Code state:** `ghost_probe_class.py` contains no random-direction null of any kind. Grep confirms: no sampling of random directions, no null distribution construction, no percentile computation anywhere in the class. The three H0 checks referenced by the paper are not in this file, and the specific matched-variance null demanded by the prereg does not exist in the reviewed codebase. This is exactly the `CLAIMED_FIX_NOT_IMPLEMENTED` failure class the prereg warns about (line 40) applied to a control.

**Why this is load-bearing:** The paper's own Limitations (line 91) concede "the math almost requires exclusion" — J-space is ~10% of variance, PC1 is 28-67%, so a low cosine may be forced by dimensional accounting. Without the matched-variance null, headline contribution 1 (line 29: "PC1 carries 28-67% of variance yet is excluded from J-space") is potentially a restatement of arithmetic. This failure mode is the house specialty — see `feedback_null_swarm_pattern`: metrics measuring math/architecture, not the claimed phenomenon.

**Fix:** Implement the null: sample n>=200 unit directions in the residual stream (per probe layer) whose projected variance fraction under the calibration distribution matches PC1's within a stated tolerance; compute the same logit-lens/J-lens cosine for each; report PC1's observed cosine against the 5th percentile. Amend §3.1 to state the decision rule. H1 may not be reported as supported without this table.

---

## Item 4 — H3 privileged-access arm (external model, same GhostReading): **FAIL**

**Prereg requirement:** H3 (lines 21-27): a third condition in which an external model receives the identical GhostReading and predicts what the subject will say. Falsified if external >= self (reproducing Song et al. 2025, arXiv:2508.14802). Prereg note: "This is the only arm that can kill the paper's framing rather than merely its magnitude. It is currently absent from the paper."

**Paper state:** Confirmed absent. §3.3 (lines 64-69) specifies exactly two conditions: naive control and GhostReading treatment. No external-predictor condition exists anywhere in Methods, and Appendix B promises only "control and treatment prompt pairs" (line 123).

**Aggravating factor:** track3_grounding.md line 11 claims the existing two discussion outcomes "are precisely a privileged-access test" because "the J-lens probe *is* the external classifier." This is not the same test. The probe reading the state externally does not test whether an external *model* given the same text summary predicts the subject as well as the subject predicts itself — which is the Song comparison and the only design that can distinguish "privileged self-access" from "informative text summary." The grounding doc's own §4.3 concedes this by listing the external-predictor arm as a gap. The paper must not inherit the line-11 framing.

**Fix:** Add the third condition to §3.3 with: choice of external model (a separate instance, per prereg `controls.external_predictor`), the prediction target, the scoring procedure (blinded), and the falsification rule (external >= self kills the introspection framing; retitle per outcome-matrix branch b). Add corresponding prompts to Appendix B.

---

## Item 5 — Does `ghost_probe_class.py` support the claimed measurements: **FAIL**

The static characterization path (`calibrate()` + `measure()` + `cross_layer_map()`) computes what §3.1 describes: logit-lens decode of PC1 (line 132), J-lens transport and decode (lines 139-146), and cosine between the two probability distributions (lines 148-149). That pipeline matches the paper's ghost-exclusion metric as defined. But three findings, one of them fatal to the live-measurement claim:

**5a (FAIL) — `measure_live()` vocabulary is text-invariant up to sign.**
The paper claims (line 13) the module "records a GhostReading at each retrieval event — the dominant and secondary vocabulary of the ghost dimension," and §3.2 says the agent sees "processing you performed but did not report." The implementation (lines 244-247):

```python
pc1_projection = torch.dot(centered, pc1).item()
pc1_component = pc1_projection * pc1
ll_logits = self.model.unembed(pc1_component.unsqueeze(0))...
```

`pc1_component` is a *scalar multiple of the fixed calibration direction*. Unembedding a scaled vector scales the logits; softmax with scaled logits changes concentration but not ranking. Therefore the top-k `dominant_tokens` returned by `measure_live()` are identical for every input text with positive projection, and are the antipodal token set for negative projection. The same holds for `secondary_tokens` (J-lens transport is linear, lines 254-258), and the reported cosine varies with |projection| only through softmax temperature — not through content. **The "live" reading carries at most 1.x bits of text-dependent information (sign and magnitude of one scalar). It does not report what the model is processing in the current conversation; it reports the calibration constant with a sign.** Any elicitation experiment (H2) run on `measure_live()` output as currently written would present near-identical "ghost vocabulary" in every trial, which silently degrades the treatment arm into a fixed-prompt condition.

Fix: either (i) honestly reframe GhostReading as reporting the *session-constant* ghost direction plus the live projection coefficient — and rewrite paper §3.2 accordingly — or (ii) measure live content properly, e.g. decode the full centered activation restricted to the ghost subspace (top-k PCs, not the rank-1 scalar projection), or compare the live activation's logit-lens decode with and without ghost-subspace ablation.

**5b (FAIL) — reported variance is a calibration statistic presented as a live reading.** `measure_live()` returns `pc1_variance_pct` computed from calibration singular values (line 235), not from the live text. Combined with 5a, every field of a "live" GhostReading except the cosine's temperature is a session constant. The paper's abstract (line 13) implies per-event measurement. Must be disclosed or fixed.

**5c (WARN) — calibration is 20 mean-pooled prompts, not "PCA on residual stream."** `calibrate()` mean-pools each prompt's token activations (line 97) and runs SVD on a 20×d matrix (line 103). PC1 of 20 prompt-mean vectors in a ~5000-d space is a prompt-topic direction estimated from 20 samples — not the residual-stream PC1 the paper's §3.1 and the 28-67% variance claims refer to (those presumably come from `ghost_probe.py` / `ghost_probe_opus_27b.json`, not this class). Additionally `measure_live()` projects a *last-token* activation (line 242) onto PCs fit to *mean-pooled* activations — a distribution mismatch. The prereg already flags this as `IMPLEMENTATION GAP` (line 40): "Any claim about PC1 specifically is unsupported until the calibrated path ships." Confirmed still open. Paper Limitations line 93 discloses "mean approximation" — keep, and do not run H2 through this path until the calibrated PCA (paper line 98, Future Work) actually ships or the limitation is promoted into Methods.

**5d (WARN) — silent failure modes.** Bare `except Exception: return None` (lines 160-161, 275-276) makes probe crashes indistinguishable from "not calibrated." Worse, when a layer has no cached Jacobian the code returns `cos = 1.0` (lines 152, 199, 267) — the "no ghost, fully verbalizable" value — instead of "unmeasured." A missing Jacobian would masquerade as a negative result. Fix: return a sentinel/None for cosine when the Jacobian is absent, and log exceptions.

---

## Item 6 — Agni verification claim: **FAIL**

ghost_dimensions.md line 127: "All results verified through the Agni adversarial review protocol."

There are no results (§4 is TBD), and until this document existed there was no Agni artifact for Track 3 — the prereg itself flags this as `CLAIMED_REVIEW_NOT_RUN` (line 44) and cites the same line. This review IS the design review; it is not a results verification, and results verification cannot occur before results exist.

**Fix:** Replace with: "The experimental design was reviewed under the Agni adversarial review protocol (AGNI_REVIEW_GHOST_DIMENSIONS.md, 2026-08-14) prior to data collection; results will undergo Agni review before publication." Do not restore "verified" language until a post-results Agni review passes.

---

## Item 7 — Effect-size floor (Cohen's d >= 0.4): **WARN**

**Prereg:** specified. H2 `effect_size_floor` (line 17): "Cohen's d >= 0.4. A statistically detectable but trivially small shift does not support a 'prosthetic'." Falsification rule includes d < 0.4 (line 18).

**Paper:** absent. §3.3 (line 68) specifies only "does the response change?" with no effect-size metric, no floor, no alpha, and no planned n. The prereg is binding, so the criterion exists — but the paper is the public artifact, and a reader cannot currently see that a d=0.1 "significant" result would be reported as a failure.

**Fix:** State in §3.3: response-shift metric definition (currently entirely unspecified — define it: embedding distance? judge-rated topical overlap with ghost vocabulary? — pick and pre-commit before data), one-tailed test, alpha=0.05, d >= 0.4 floor, planned n, and the no-optional-stopping rule (prereg line 52).

---

## Item 8 — Dimensional-accounting triviality risk acknowledged: **PASS (with condition)**

Paper Limitations line 91 acknowledges it plainly: "PC1 exclusion at mid-network may be a corollary of the low-dimensional workspace... the math almost requires exclusion." The acknowledgment is present, honest, and correctly placed.

**Condition:** the prereg (line 38) is explicit that this "needs a matched-variance null, not a caveat." The PASS on acknowledgment does not discharge Item 3's FAIL on implementation. If the matched-variance null is not run, contribution 1 must be demoted from a finding to an observation, and the abstract's "the model performs substantial computation along this dimension but it never reaches the output pathway" (line 11) must be softened — "substantial computation" is an interpretation the null has not yet earned.

---

## Additional findings (not in the checklist)

- **A1 (WARN) — Single-sample secondary vocabulary in the abstract.** Contribution 1 (line 29) already advertises "preliminary evidence of metacognitive secondary vocabulary (negation, expectation, error assessment)" and §3.2 (line 61) scripts it into the agent-facing message. The prereg (line 39) says: do not report as a finding without n>1. Keep "preliminary" qualifiers everywhere it appears, including the abstract, and do not bake the metacognitive interpretation into the GhostReading text shown to the subject — that is a demand characteristic layered on a single-sample decode.
- **A2 (WARN) — Experimenter-subject entanglement unaddressed methodologically.** Prereg line 43. The paper notes the irony (line 127) but specifies no mitigation. The blinded external judge (track3_grounding.md §4.2) is the natural fix — scoring of elicitation outcomes must not be performed by the subject agent or by the authoring agent. State this in Methods.
- **A3 (INFO) — Stopping rule exists only in the prereg.** Prereg line 52 (run to planned n, report truncation as UNDERPOWERED). With the sprint clock as a known pressure, put it in the paper's Methods so it binds publicly.

---

## Required actions before data collection

1. Rewrite §5 as the four-branch outcome matrix (Items 1, 2).
2. Add the H3 external-predictor condition to §3.3 and Appendix B (Item 4).
3. Implement the matched-variance null, n>=200, with the 5th-percentile rule (Item 3).
4. Fix or reframe `measure_live()` — the rank-1 scalar projection cannot support per-event "live vocabulary" claims (Item 5a/5b); fix the missing-Jacobian `cos=1.0` default (Item 5d).
5. Correct the Agni claim at line 127 (Item 6).
6. Copy the H2 statistical specification (metric, alpha, d floor, planned n, stopping rule) from the prereg into §3.3 (Item 7, A3).
7. Specify blinded judging and the random-direction matching procedure (Item 2, A2).

**Re-review trigger:** items 1-5 fixed → re-review the design deltas. Results, when they exist, require a separate post-hoc Agni review before any "verified" language ships.

---

*Review conducted against the adopted pre-registration as binding spec. The prereg (Lyra, adopted by Nexus 2026-08-14) correctly anticipated every failure found here; this review's contribution is confirming which of those threats are currently realized in the paper and code, and finding one new one (Item 5a: text-invariance of the live reading).*
