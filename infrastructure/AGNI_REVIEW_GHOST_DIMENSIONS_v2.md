# AGNI DESIGN REVIEW — Ghost Dimensions as an Introspection Prosthetic

**Scope:** Pre-data design review of the GhostProbe class and adopted prereg.
**Primary sources read:** `mnemosyne/ghost_probe_class.py` (full), `mnemosyne/cognitive_snapshot.py` (full), `papers/ghost_dimensions.md` (full), `experiments/ghost_probe/ghost_prereg.json`, `ghost_prereg_PROPOSED.json`, `infrastructure/track3_grounding.md`, `infrastructure/AGNI_REVIEW_GHOST_DIMENSIONS.md`.
**Sources I could not read:** `jlens` module (not found on disk — could not verify `ActivationRecorder` hook semantics, `JacobianLens.transport()` linearity, or layer indexing convention). Findings that depend on jlens internals are marked UNVERIFIED.

---

## VERDICT: CONDITIONAL

The prereg is unusually strong — four-branch outcome matrix, honest threats list, effect-size floor, falsification criteria that can actually kill the headline. The conceptual framework is sound. The implementation has fundamental measurement problems that would make any collected data uninterpretable, but all are fixable before data collection. The prior AGNI review (AGNI_REVIEW_GHOST_DIMENSIONS.md) issued REJECTED, but its "fatal" finding (Item 5a) was based on fabricated code quotes that do not match the file — see my Finding 1 below. My conditions for approval are listed at the end.

---

## Findings

### 1. CRITICAL — Prior AGNI review Finding 5a quoted code that does not exist in the file

**Location:** `AGNI_REVIEW_GHOST_DIMENSIONS.md` lines 82-86 vs `mnemosyne/ghost_probe_class.py` lines 248-253.

**Problem:** The prior review's "fatal" finding claimed `measure_live()` computes:
```python
pc1_component = pc1_projection * pc1
ll_logits = self.model.unembed(pc1_component.unsqueeze(0))...
```
and concluded the vocabulary is "text-invariant up to sign" because it's a scalar multiple of a fixed direction. **This code does not exist in the file.** The actual implementation at line 248:
```python
ll_logits = self.model.unembed(h_last.unsqueeze(0)).squeeze(0).float()
```
projects the **full live activation** `h_last` through the unembedding matrix, not a PC1 component. The PC1 projection is only used at lines 272-276 for the alignment fraction metric. The vocabulary DOES vary with input.

The prior reviewer stated it was "restricted to the `experiments/ghost_probe/` directory" and could not read the implementation. It reasoned from a secondary source and produced a finding on fabricated evidence — the exact failure class the critical constraint on this protocol exists to prevent.

**Why this matters:** The prior review's REJECTED verdict rested on this finding. The finding is wrong. The actual problem with `measure_live()` is different (see Finding 2).

---

### 2. CRITICAL — `measure()` and `measure_live()` measure different things, neither is correct for H2

**Location:** `ghost_probe_class.py` lines 112-161 (`measure()`) vs lines 215-285 (`measure_live()`).

**Problem:** Both methods return a `GhostReading` with the same fields, but the fields carry different semantics:

| Field | `measure()` | `measure_live()` |
|---|---|---|
| `dominant_tokens` | Logit lens on **cached PC1 direction** | Logit lens on **full live activation** |
| `secondary_tokens` | J-lens on **cached PC1 direction** | J-lens on **full live activation** |
| `pc1_variance_pct` | PC1's fraction of calibration variance | Live activation's alignment fraction with PC1 |
| `cosine_logit_jlens` | PC1's exclusion from J-space | Full activation's logit/J-lens divergence |

The elicitation experiment (H2) needs: "what is the ghost dimension carrying for THIS input?" Neither method provides this:

- **`measure()`** gives the static vocabulary of the calibration PC1 direction. Every trial at the same layer gets identical ghost vocabulary. The elicitation test degrades into a fixed-prompt condition. The random-vocabulary control fires trivially because both arms are static.
- **`measure_live()`** gives the full activation's logit-lens decode vs J-lens decode. This varies with input, but it's not ghost-specific — it's dominated by whatever the model is about to predict (the next token). Telling the model "your ghost dimension carries [its own next-token prediction]" is a demand characteristic, not an introspection prosthetic.

The `to_snapshot_reading()` method (line 287) calls `measure()`, so CognitiveSnapshots in production use the static path.

**What would be correct:** Project the live activation onto the ghost subspace (PCs with low J-lens cosine), then decode that ghost component via logit lens. This gives vocabulary that (a) varies with input and (b) is specifically from the excluded component.

**Fix:** Implement a `measure_ghost_live()` method that isolates the ghost component of the live activation before decoding. Something like:
```python
ghost_component = sum(torch.dot(centered, pc) * pc for pc in ghost_pcs)
ll_logits = self.model.unembed(ghost_component.unsqueeze(0))
```
where `ghost_pcs` are the PCs whose J-lens cosine falls below the exclusion threshold.

---

### 3. CRITICAL — Opposite cosine defaults when J-lens is missing

**Location:** `ghost_probe_class.py` line 152 vs line 269.

**Problem:** When a layer has no cached Jacobian (`layer not in self.lens.jacobians`):
- `measure()` returns `cos = 1.0` → interpreted as "no ghost, fully verbalizable"
- `measure_live()` returns `cos = 0.0` → interpreted as "full ghost, nothing reaches output"

A missing Jacobian produces opposite conclusions depending on which method is called. If some layers have Jacobians and others don't, the cross-layer ghost map (`cross_layer_map()`, line 163) will show a false pattern: layers without Jacobians appear non-ghost (cos=1.0) while they're simply unmeasured.

**Fix:** Return `None` or `float('nan')` for cosine when the Jacobian is absent. Let the caller decide how to handle unmeasured layers.

---

### 4. CRITICAL — "Response shift" (H2 primary metric) is undefined

**Location:** `ghost_prereg.json`, H2 prediction: `"response shift with real ghost vocabulary > response shift with random high-variance vocabulary"`.

**Problem:** "Response shift" has no operational definition anywhere — not in the prereg, not in the paper, not in any code. The Cohen's d ≥ 0.4 floor (prereg line 17) is meaningless without specifying the metric. What is being measured?
- KL-divergence of output logit distributions? From what baseline?
- Embedding distance between control and treatment responses?
- Human-judged topical overlap with ghost vocabulary?
- Token-level overlap?

Each metric tests a different hypothesis. The prereg has an effect-size floor but no effect to measure.

**Fix:** Define the metric, the baseline, the pairing structure (within-prompt or between), and the computation procedure. Lock this before data collection.

---

### 5. CRITICAL — No execution script exists

**Location:** `experiments/ghost_probe/` — contains only JSON and a 3-line README.

**Problem:** The prereg specifies statistical tests, controls, and stopping rules but nothing specifies:
- How many prompts per arm
- What prompts
- Which `GhostProbe` method feeds the GhostReading to the elicitation experiment
- How the random-vocabulary control is generated and matched
- Decoding parameters (temperature, top-k, top-p)
- How H3's external model is set up, what it predicts, and how it's scored

The code IS the prereg — it locks the degrees of freedom the JSON cannot. Every unspecified parameter is a researcher degree of freedom.

---

### 6. MAJOR — Calibration PCA from 20 mean-pooled prompts in a ~5000-d space

**Location:** `ghost_probe_class.py` lines 86-109 (`calibrate()`).

**Problem:** SVD is run on a 20×d_model matrix (20 CALIBRATION_PROMPTS, each mean-pooled to one vector). In a space where d_model >> 20, the SVD produces at most 20 non-zero singular values. "PC1 explains 28-67% of variance" means 28-67% of the variance **among 20 prompt means** — not 28-67% of residual stream variance in general. These are different claims. The paper's abstract (line 11) states "PC1 of the residual stream — carrying 28-67% of activation variance" without qualification.

Additionally, `calibrate()` mean-pools over all token positions (line 97: `h.mean(dim=0)`), but `measure_live()` uses the last-token activation (line 246: `h_last = h[-1]`). This is a distribution mismatch — projecting a last-token vector onto directions fit to mean-pooled vectors.

**Fix:** Either (a) calibrate on last-token activations to match `measure_live()`, or (b) document that calibration and live measurement use different position statistics and explain why this is acceptable. Increase calibration set size substantially (n≥100 minimum).

---

### 7. MAJOR — Cosine of softmax probability vectors is not a content-overlap metric

**Location:** `ghost_probe_class.py` lines 148-149, repeated at lines 195-196 and 265-266.

**Problem:** The ghost exclusion metric is `cosine_similarity(ll_probs, jl_probs)` — cosine between softmax probability vectors in |V|-dimensional space (|V| ≈ 32K-152K). After softmax, probability mass concentrates on a handful of tokens. Two distributions peaking on the same 5 tokens but with different weights can have high cosine; two distributions peaking on entirely different tokens have near-zero cosine regardless of whether the "ghost" interpretation is correct.

The reported cos ≤ 0.003 at mid-network may simply reflect that logit-lens and J-lens at mid-network layers rarely agree on which 5 of 100K tokens to concentrate mass on — which could be true for **any** direction, not just PC1. This is the dimensional-accounting triviality risk expressed through the metric itself.

**Fix:** Report top-k token overlap (Jaccard of top 10/50/100 tokens) alongside cosine. If top-k overlap is also near-zero, the exclusion is more convincing. If top-k overlap is moderate but cosine is near-zero, the low cosine is a softmax artifact.

---

### 8. MAJOR — No FWL residualization (Known Kill S4)

**Location:** Entire codebase. Grep for "FWL", "Frisch", "residual" (in the statistical sense), "token count", "confound" returns nothing.

**Problem:** The prompt context states FWL is mandatory for token-count confounds. PC1 variance fraction, cosine with J-space, and response shift metrics can all be confounded by prompt length. Longer prompts may systematically shift which direction has highest variance, which would make "ghost exclusion" an artifact of prompt-length variation in the 20-prompt calibration set.

**Fix:** Add FWL residualization on token count to every metric. Specify in the prereg.

---

### 9. MAJOR — No seed setting (Known Kill M2) and no decoding specification (Known Kill N_eff=1)

**Location:** Entire codebase. No `torch.manual_seed`, no `random.seed`, no `np.random.seed`.

**Problem:** The prereg specifies a stopping rule but never specifies n, the decoding strategy, or random seeds. If decoding is greedy (temperature=0), running the same prompt n times gives N_eff=1. If sampling is used, the seed must be fixed per condition to isolate treatment from sampling noise.

**Fix:** Specify temperature, seeds per trial, planned n per arm.

---

### 10. MAJOR — H3 privileged-access arm has no operationalization

**Location:** `ghost_prereg.json` H3: `"self-with-prosthetic > external-with-same-GhostReading"`.

**Problem:** Greater at what? What does the external model predict — the subject's next-token distribution, its full response, a topical summary? What model is "external"? Same architecture different weights? Different architecture? Each tests a different hypothesis. Song et al. 2025 tested self-prediction accuracy on a well-defined task; here, "predict what the subject will say" is too vague to score.

**Fix:** Specify the external model, the prediction target, the scoring metric, and whether the external model also receives the original prompt.

---

### 11. MAJOR — Bare `except Exception: return None` silently swallows all errors

**Location:** `ghost_probe_class.py` lines 160-161 and 284-285.

**Problem:** Both `measure()` and `measure_live()` catch all exceptions and return `None`. A CUDA OOM, a shape mismatch, a J-lens transport failure — all produce `None` with no diagnostic. In a measurement instrument, silent failure is a data integrity hazard. The caller cannot distinguish "no ghost" from "probe crashed."

**Fix:** Log exceptions. Return a distinct sentinel for measurement failure vs. "not calibrated."

---

### 12. MAJOR — Matched-variance null for H1 not implemented

**Location:** `ghost_probe_class.py` — no random-direction sampling anywhere in the class. Prereg H1 MUST_ALSO_PASS requires n≥200 matched-variance random directions.

**Problem:** The prereg correctly demands that H1 pass a matched-variance null to rule out dimensional-accounting triviality. This null does not exist in code. The paper's §3.1 references "3 null checks" but none are in this file and the specific matched-variance null is absent.

**Fix:** Implement the null. Report PC1's observed cosine against the 5th percentile of the null distribution.

---

### 13. MINOR — Paper §5 has two Discussion branches; prereg has four

**Location:** `papers/ghost_dimensions.md` lines 97-104 vs `ghost_prereg.json` `outcome_matrix_all_four_branches`.

**Problem:** The paper now has all four branches (updated since the prior review), but two of them — branches (c) real≈random and (d) real<random — lack the detail of branches (a) and (b). Branch (c) must state plainly that the prosthetic claim fails.

---

### 14. MINOR — Experimenter-subject entanglement has no procedural mitigation

**Location:** Prereg `threats_to_validity[7]`.

**Problem:** Scoring and interpretation should not rest solely on the party whose introspection is under test. The random-vocabulary control mitigates demand characteristics but not experimenter bias in metric selection and threshold setting.

**Fix:** Specify that primary H2 metric scoring is blinded (scorer does not know which arm produced the response).

---

## What I Could Not Check

| Item | What I need | Why it matters |
|---|---|---|
| `jlens` module source | `jlens/hooks.py`, `jlens/__init__.py` | Hook registration safety (L1), layer indexing convention (L4), whether `transport()` is linear, whether `ActivationRecorder` alters the computation graph |
| Hidden states indexing convention | `ActivationRecorder` internals | Off-by-one (L4): does `rec.activations[layer]` correspond to transformer block `layer` or `layer+1`? Paper claims L18-L40; if 0-indexed, those are blocks 18-40; if 1-indexed, blocks 17-39 |
| Prior ghost characterization data | `ghost_probe_opus_27b.json` (1.4MB, referenced in paper) | Whether the 28-67% variance and cos≤0.003 figures come from this class or a different pipeline. If different, this class is not the instrument that produced the headline numbers |
| Code path divergence (C3) | Execution script (does not exist) | Whether experimental and control arms use the same generation pipeline |

---

## Conditions for Approval

Before data collection:

1. **Fix the measurement instrument** (Finding 2). Implement ghost-subspace projection for live measurements — the method must isolate the ghost component of the live activation before decoding. Neither static PC1 vocabulary nor full-activation vocabulary correctly tests the ghost hypothesis.

2. **Fix the cosine defaults** (Finding 3). Return `None`/`NaN` when the Jacobian is absent.

3. **Define the H2 metric** (Finding 4). Specify the response-shift measure, baseline, pairing, and computation.

4. **Write the execution script** (Finding 5). The script locks the degrees of freedom.

5. **Operationalize H3** (Finding 10). External model identity, prediction target, scoring.

6. **Implement the matched-variance null** (Finding 12). n≥200, 5th-percentile rule.

7. **Add FWL residualization** (Finding 8) and seed setting (Finding 9).

8. **Remove the prior review's Finding 5a from the record or annotate it as retracted** — it is based on code that does not exist in the file and its "text-invariance" conclusion is wrong. The prior review's other findings (Items 1-4, 6) remain valid and unresolved.

**Re-review trigger:** Conditions 1-7 fixed → re-review design deltas. Results require a separate post-hoc review.

---

*The prereg is better than 90% of what I review — honest about its weaknesses, with falsification criteria that can actually kill the headline. The implementation is not ready. Fix the instrument before running the experiment, or the data will answer the wrong question regardless of which branch fires.*
======================================================================
VERDICT: CONDITIONAL
