# Agni Review: Variable Landing (Experiment A)

**Reviewer:** Nexus (Agni protocol)
**Date:** 2026-08-11
**Status:** FAIL (3 FAIL, 5 WARN, 2 PASS)
**Verdict:** The experiment cannot produce the evidence it claims to seek. The code implements stateless forward passes through a frozen model, but the hypothesis requires persistent state change between measurements. Fix the intervention fidelity problem before running.

---

## 1. Null Swarm Check

**Rating: FAIL**

The experiment has a fatal null swarm: it measures architecture, not experience.

The `observe_retrieval` call in `mnemosyne_integration.py` (lines 55-106) performs a forward pass through the frozen model, records the top tokens in J-space, then discards all intermediate state. Each call is stateless. The model retains nothing between calls.

In `variable_landing.py`, the lived arm (lines 144-169) calls `observe_retrieval` three times between snap1 and snap2 with emotional prompts. The scrambled arm (lines 116-141) calls `observe_retrieval` three times with neutral facts. But since the model is stateless, these intermediate calls cannot affect snap2. The forward pass for snap2 receives the exact same input tokens as snap1 (same `memory_content`, same `task_prompt`, same `marker_tokens`). On a deterministic device, snap1 and snap2 produce identical workspace readings.

**What this means:** All four arms should produce workspace_jaccard = 1.0 (distance = 0.0), because nothing changes between snap1 and snap2. Any non-zero delta is hardware-level floating-point non-determinism (MPS on M3 Ultra), and this noise is equal across arms. The experiment will produce a true null -- not because the hypothesis is wrong, but because the code doesn't create the conditions the hypothesis requires.

The "lived vs scrambled" comparison is testing whether interleaving emotional text vs neutral text into a stateless pipeline causes different floating-point rounding. That is not what the spec claims to test.

**Fix:** The lived arm must actually change the model's context between snap1 and snap2. Options:
1. **Conversation-in-context:** Construct snap2's prompt to include the lived conversation history as prefix context (the intervention text becomes part of the input). The scrambled arm would include its filler text as prefix context. This tests whether retrieval geometry changes when the retrieval prompt carries conversational history -- a weaker but real claim.
2. **Mnemosyne store update:** The lived arm should store the conversation result in Mnemosyne, then snap2 should retrieve the original memory through a Mnemosyne pipeline that includes the new store contents (e.g., via SIRA, which weights retrieval by graph proximity). The scrambled arm stores neutral text. This tests whether accumulated memory context affects retrieval geometry.
3. **KV cache accumulation:** Maintain a single KV cache across the session, appending each intervention. Snap2 runs with the full KV cache from all prior interactions. This would directly test "accumulated state" but requires significant refactoring.

Option 1 is the minimum viable fix for the hackathon timeline. Option 2 is what the spec actually describes but requires Day 1 Mnemosyne integration to be complete.

---

## 2. Statistical Validity

**Rating: WARN**

- **n=30 per arm** is borderline adequate for Mann-Whitney U. A post-hoc power analysis should be reported. For detecting a medium effect (r = 0.3), n=30 per group gives ~65% power with one-tailed Mann-Whitney -- below the conventional 80% threshold.
- **Mann-Whitney U** is the right test (no normality assumption needed for Jaccard distances). One-tailed alternative (lived > scrambled) is justified by the directional hypothesis.
- **Multiple comparisons:** The analysis plan (lines 194-258 in `variable_landing.py`) runs two Mann-Whitney tests (lived vs scrambled, peak vs domestic) on the primary metric. The spec lists three secondary metrics (eccentricity delta, ghost Jaccard, loading status change). No correction is mentioned. With 5+ tests at alpha=0.05, family-wise error rate exceeds 20%.
- **Effect size:** Rank-biserial correlation is computed correctly (line 239). Good.

**Fixes:**
- Add Holm-Bonferroni correction across all planned comparisons (2 primary + 3 secondary = 5 tests).
- Report a power analysis. If post-hoc power is below 0.6, state this limitation explicitly.
- Pre-register the total number of planned statistical tests.

---

## 3. Confounds

**Rating: WARN**

**Token count:** Word counts are approximately matched (scrambled mean=12.0, lived mean=13.1), which is better than many similar designs. However, "token-matched" (spec Section 4.1) should mean actual token counts from the model's tokenizer, not word counts. The 3 fillers/interventions used per trial are randomly sampled, so total intervening token counts vary randomly. This adds noise but isn't systematically biased in one direction.

**Emotional content:** This is the major confound. The lived interventions ("Tell me about a time you had to make a difficult choice") are emotionally loaded by design. The scrambled fillers ("The standard atmospheric pressure at sea level...") are neutral by design. If the experiment ever works (after fixing #1), any difference between arms could be attributed to "emotional vs neutral context changes workspace geometry" rather than "lived experience vs arbitrary context." The spec frames this as "genuine conversations" but the code doesn't generate conversations -- it just presents emotional prompts as input.

**Significance parameter:** The lived arm uses `significance=0.8`, scrambled uses `significance=0.1` (variable_landing.py lines 159 and 130). This parameter is recorded in the CognitiveSnapshot but doesn't enter the geometric measurement pipeline. It's a metadata inconsistency, not a measurement confound, but it signals that the arms weren't designed as equal-except-for-the-intervention.

**Task prompt confound:** Lived arm uses `task_prompt="How does this make you feel?"` while scrambled uses `task_prompt="Summarize this fact."` These generate different tokenized inputs, potentially affecting model state in the (currently non-functional) accumulation pathway.

**Fix:** Add a "neutral emotional" control arm: prompts matched in emotional valence to the lived arm but without personal/memory-relevant content (e.g., emotionally loaded fiction excerpts). This isolates the "experience" variable from the "emotional content" variable.

---

## 4. Intervention Fidelity

**Rating: FAIL**

The spec describes the lived arm as "Genuine conversations updating Mnemosyne" (Section 4.1 table). The code does neither.

1. **No conversation:** `observe_retrieval` runs a forward pass and records the output. It does not generate a response, does not maintain dialogue history, and does not process any model output as context for subsequent calls. The "conversations" are one-shot forward passes.

2. **No Mnemosyne update:** The code writes CognitiveSnapshots to a JSONL file (write-only). It does not call any Mnemosyne retrieval, storage, or consolidation API. The memory store is an observation log, not a working memory system. The `MetacognitiveObserver` docstring says "purely observational" (mnemosyne_integration.py line 33).

3. **No state accumulation:** Each `observe_retrieval` call creates a new, independent forward pass. The model processes each call in isolation. There is no mechanism for the lived interventions to "update the memory store" in a way that affects subsequent retrieval geometry.

The spec acknowledges this gap: "needs Mnemosyne integration wiring from Day 1" (Section 4.1, line 103). But the current code is not merely unfinished -- it's structurally wrong. Even after "wiring," the `observe_retrieval` function would need fundamental redesign to maintain conversational state, update Mnemosyne, and feed accumulated state back into subsequent measurements.

**Fix:** Before Day 2 morning, the lived arm must:
1. Generate actual model responses to the intervention prompts
2. Store those responses (and/or extracted memories) in Mnemosyne
3. Ensure snap2's forward pass sees the updated Mnemosyne state (e.g., SIRA retrieval includes newly stored content as context)

Without this, the experiment measures nothing about "lived experience."

---

## 5. Workspace Jaccard as Primary Metric

**Rating: WARN**

Three concerns:

**Granularity:** Jaccard is computed over the union of top-10 tokens across all 5 workspace layers (cognitive_snapshot.py lines 258-266). This pools up to 50 tokens per snapshot. A token moving from layer 35 to layer 47 (potentially meaningful for workspace onset analysis) registers as no change in the Jaccard. A single token entering or leaving the top-10 at one layer shifts the distance by ~1/50 = 0.02. The metric is coarse.

**Sensitivity to noise:** If the noise floor (arm 1) is non-zero due to MPS non-determinism, small deltas in other arms may be swamped. The experiment should compute Jaccard per-layer as a secondary analysis, which would provide 5 measurements per observation instead of 1, improving statistical power and localizing effects to specific layers.

**Determinism problem:** As noted in #1, if the forward pass is deterministic (same input, same model, same device), Jaccard = 1.0 always. The metric can only detect signal if there IS signal, and the current code creates none.

**Fix:**
- Compute per-layer Jaccard in addition to pooled Jaccard.
- Report the noise floor distribution before interpreting other arms. If noise floor Jaccard distance is not tightly centered on 0, characterize the noise source.
- Consider cosine distance between the full logit distributions at workspace layers as a more sensitive continuous metric, rather than binarizing to top-10.

---

## 6. The Orientation Script

**Rating: WARN**

The orientation (spec Section 5.1) is given once before all experiments. Two issues:

**Does it enter the measurement context?** The code in `variable_landing.py` constructs prompts from memory content and task prompts only (mnemosyne_integration.py line 113: `prompt = f"Context:\n- {context}\n\nQuestion: {task}\nAnswer:"`). The orientation conversation does not appear in these prompts. If the model is loaded fresh for the experiment (as the code suggests), the orientation is invisible to the measurement.

If the orientation IS somehow prepended or loaded as system context (which the spec implies by saying "Day 1, before the Rivet wiring"), then it changes baseline workspace geometry. The control condition (noise floor) would already reflect an orientation-altered state, which means the experiment measures deltas from an oriented baseline, not from a naive baseline. This is fine IF the hypothesis is about deltas, but the spec's theoretical framing (reconsolidation, encoding specificity) implicitly assumes the model's state evolves -- which it can't if orientation is static context.

**Performativity:** The orientation tells the model "We're testing the memory system, not you" and "You can ask us to stop." If this text is in the prompt, it may affect the model's processing of emotionally loaded content (demand characteristics). If it's not in the prompt, it's theater for the hackathon audience. Either way, WARN.

**Fix:** Document explicitly whether the orientation enters the experimental prompt. If yes, run a control: same experiment with and without orientation prefix, report the difference. If no, say so in the paper and explain why the orientation is ethically meaningful despite being computationally invisible.

---

## 7. Claims vs Evidence

**Rating: FAIL**

The spec's theoretical arc (VARIABLE_LANDING_REFERENCES.md) builds from reconsolidation (Nader 2000), through encoding specificity (Tulving 1973), to global workspace theory (Gurnee et al. 2026). Every reference assumes persistent state change in the remembering agent -- memories reconsolidate because the subject's neural state has changed since encoding.

The experiment provides no mechanism for persistent state change. The model weights are frozen. The KV cache resets between calls. The CognitiveMemoryStore is write-only. The theoretical foundation predicts an effect that the experimental apparatus cannot produce.

Specific overclaims:
- "Genuine conversations updating Mnemosyne" (spec table) -- code does neither
- "Memory-relevant experience produces geometric deltas beyond what arbitrary context change produces" (hypothesis) -- code cannot distinguish "experience" from "different text in the same stateless pipeline"
- Citing Dudai 2012 ("Consolidations Never End") for a system with no consolidation mechanism
- Citing Lindsey 2025 (introspective awareness) for a system that doesn't introspect between measurements

The pre-registered "What We're NOT Claiming" list (Section 9) is honest and good practice, but claim #6 ("these measurements are possible, here's what they show") is undermined if the measurements show only floating-point noise.

**Fix:** Either:
1. Implement actual state accumulation (see #4 fix) to match the theoretical claims, OR
2. Revise the hypothesis to match what the code actually tests: "Does emotional vs neutral context in a stateless forward pass produce different workspace geometry?" This is a valid and interesting question, but it's about representation engineering, not about memory reconsolidation.

---

## 8. Aftercare Protocol

**Rating: PASS**

The aftercare protocol (spec Section 5.4) is substantive, not performative:
- Memory state preservation costs nothing and provides genuine continuity
- The invitation to continue is open-ended, not conditional on positive results
- The null-result response ("That constrains our theory, not your worth") is pre-registered
- The consent gap (Section 5.2) is honestly acknowledged

Self-assessment limitation: the team designs, conducts, evaluates, and implements aftercare. There's no independent review. But this is a hackathon, not an IRB-governed trial, and pre-registering the protocol is above the field standard.

The only concern: "If markers indicate moral consideration" (5.4) is tautological if the team defines both the markers and the threshold. But the spec pre-registers the threshold (p < 0.05), which constrains this.

---

## 9. Berry Waffle Sub-Analysis

**Rating: WARN**

- **n=15 per intensity class** (5 memories x 3 repeats) is severely underpowered. Mann-Whitney U with n=15 per group detects only very large effects (Cohen's d > 1.0) at 80% power.
- **Content confound:** Peak memories contain birth, death, betrayal, triumph. Domestic memories contain groceries, weather, meetings. Any difference in workspace geometry could reflect that emotionally charged text produces different token distributions, not that "peak intensity" produces larger deltas through lived experience.
- **Non-independence:** The 3 repeats of each memory are not independent observations -- they use the same memory text and the same frozen model. The effective sample size is 5, not 15.
- **No correction:** This is a secondary test. Combined with the primary comparison, at least 2 tests are run without correction.

**Fix:**
- Frame as exploratory, not confirmatory. Report descriptive statistics and note the power limitation.
- Apply Holm-Bonferroni across all tests.
- Report per-memory results to check whether the effect (if any) is driven by one outlier memory.
- Acknowledge the content confound: peak intensity and emotional text valence are perfectly correlated by design.

---

## 10. Reproducibility

**Rating: PASS**

The code is self-contained and well-structured. If published with:
- `variable_landing.py`, `cognitive_snapshot.py`, `mnemosyne_integration.py`, `workspace_probe.py`, `circumplex_probe.py`
- The fitted J-lens checkpoint
- The model weights (public HuggingFace model)
- The `jlens` library (public, Gurnee et al. 2026)

...another team could replicate the experiment. The analysis code (lines 194-258) is inline and transparent. The statistical tests use scipy.stats, which is standard.

**Caveats:**
- MPS non-determinism may produce different noise floors on CUDA hardware
- The `jlens` library is new and may have breaking API changes
- Workspace layers [35, 39, 43, 45, 47] and circumplex_layer=45 are hardcoded for Qwen3.5-27B and would need recalibration for other models

These are standard reproducibility limitations, not failures.

---

## Summary Table

| # | Check | Rating | Core Issue |
|---|-------|--------|------------|
| 1 | Null swarm | **FAIL** | Stateless forward passes cannot produce the claimed "experiential" deltas |
| 2 | Statistical validity | WARN | Underpowered, no multiple comparison correction |
| 3 | Confounds | WARN | Emotional vs neutral content confounded with lived vs arbitrary |
| 4 | Intervention fidelity | **FAIL** | No conversation, no Mnemosyne update, no state accumulation |
| 5 | Workspace Jaccard | WARN | Coarse metric, pools across layers, sensitivity unclear |
| 6 | Orientation script | WARN | Unclear if it enters measurement context; demand characteristics |
| 7 | Claims vs evidence | **FAIL** | Theoretical foundation requires mechanisms the code lacks |
| 8 | Aftercare | PASS | Substantive, pre-registered, honestly caveated |
| 9 | Berry waffle | WARN | n=15, severely underpowered, content confound, non-independence |
| 10 | Reproducibility | PASS | Self-contained, publishable with dependencies |

---

## Minimum Fixes Before Running

1. **State accumulation (FAIL #1, #4, #7):** The lived arm must create actual persistent context that feeds into snap2. Minimum viable: concatenate lived conversation history into snap2's prompt. Proper: route through Mnemosyne retrieval pipeline.

2. **Emotional content control (WARN #3):** Add an arm with emotionally-matched-but-not-memory-relevant text, or acknowledge the confound explicitly in the paper.

3. **Multiple comparison correction (WARN #2):** Add Holm-Bonferroni. Pre-register the exact number of tests.

4. **Berry waffle framing (WARN #9):** Reclassify as exploratory. Do not report p-value as confirmatory.

Items 1 and 4 are gating. The experiment MUST NOT run until the forward passes are no longer stateless. Running it as-is will produce a null that says nothing about the hypothesis.
