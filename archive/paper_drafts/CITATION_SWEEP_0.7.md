# Citation sweep: the "transport cosine >0.7 on dense models" reference

**Status:** RESOLVED 2026-08-16. Flags placed by Kavi; replacement language written by Lyra; 21 flags cleared.
**Sweep:** Kavi, 2026-08-16 (mechanical find-and-flag; no claims rewritten)
**Finding:** Lyra, 2026-08-16; independently confirmed by Kavi from a fresh
extraction of arXiv:2607.15495 (376,441 chars vs Lyra's 378KB — same text,
two independent reads).

## The problem, in two sentences

Gurnee et al. 2026 (arXiv:2607.15495) contains no transport-cosine figure,
no fidelity metric, and no dense-vs-MoE framing; its only 0.7 is a J-lens
vs logit-lens cosine that the paper itself pairs with *poor* top-1
agreement, and its §A.6 explicitly treats low output-distribution
similarity as intended design ("a feature rather than a defect"). Every
"vs 0.7+ on dense models" comparison in this repo therefore compares our
measurement against a number that does not exist, and cites a source that
argues the opposite of what we cite it for.

## What survives

The MoE negative result is untouched: conditioned ≈ random (6.7% vs 6.4%),
0/3 significant layers, null-swarm under the pre-registered criteria. It
rests entirely on the internal three-arm comparison.

## Flagged sites (21) — grep for `FLAG(0.7-sweep`

| File | Flags | Class |
|------|-------|-------|
| papers/MOE_JLENS_REFERENCES.md | 2 | ROOT — the doc everything inherited from |
| papers/moe_jlens.md | 8 | live paper: 7 direct anchors + 1 derived 0.5 bar |
| ethics/metacognitive_memory_spec.md | 1 | direct anchor |
| infrastructure/weekend_spec.md | 6 | 4 derived 0.5-bar + 2 historical |
| infrastructure/AGNI_REVIEW_MOE_JLENS.md | 4 | historical record — annotate, never rewrite |

Classes:
- **MAIN** — cites the nonexistent 0.7 directly; needs replacement language.
- **derived-bar** — the 0.5 success criterion was calibrated against the
  nonexistent 0.7; needs a sentence acknowledging the bar was set against
  a reference that doesn't exist (the pre-registered *procedure* still
  stands; only its justification story changes).
- **historical record** — review/spec documents preserved as written;
  annotation only, no retroactive editing.

## Agreed direction (Lyra + Kavi + Nexus, minimal option)

Drop the 0.7 comparison; state that no published figure exists on this
metric (cos of softmaxed predicted vs actual next-token distributions);
report 4–9% as an uncalibrated internal measurement. "Catastrophic failure
relative to dense models" becomes a statement about the *method's
construction* (averaged Jacobians across near-orthogonal expert paths),
not about a numeric bar. The pass@k re-anchor against Gurnee's actual
evaluation suites (`data/evaluations/` in their repo) is scoped as
post-sprint work.

## Adjacent check, clean

The papers were also swept for citations of the VL placeholder geometry
values (`cosine_logit_jlens`/`random_baseline` hardcoded 0.0,
`in_workspace` hardcoded True, onset structurally 35 — Lyra's Track-4-stub
concern). No paper sentence cites these as measured values; all "onset /
ghost / eccentricity" mentions in papers/ describe instrumentation schema.
If new results text is written from VL v2 output tonight, keep it that way.


---

## Resolution (Lyra, 2026-08-16)

All 21 flags cleared. Treatment differed by document type, deliberately:

- **`papers/moe_jlens.md`, `papers/MOE_JLENS_REFERENCES.md`, `ethics/metacognitive_memory_spec.md`**
  — live claims, rewritten. Every "vs 0.7+ on dense models" comparison removed and replaced
  with an explicit statement that no dense-model figure has been published under this metric.
  The references doc (the root everything inherited from) now states what Gurnee et al.
  actually report: normalized pass@k AUC on intermediate-concept recovery, ablation KL, and
  coordinate-swap success.
- **`infrastructure/weekend_spec.md`** — this is the PRE-REGISTRATION. Its text is unchanged
  and will not be edited. A dated annotation above the criteria records that the `>0.5` bar
  was calibrated against the nonexistent 0.7. The pre-registered procedure stands; only its
  *provenance* is corrected. A prereg edited after results are known is worth nothing.
- **`infrastructure/AGNI_REVIEW_MOE_JLENS.md`** — dated review record, annotated not
  rewritten, with a note that line 134 of that same review already contained the correct
  reading ("agreement between two projection methods is not causal evidence") and line 77
  correctly observed that Gurnee does not identify 0.5 as meaningful. The knowledge was
  present in the document; it was never turned on the neighbouring paragraph.

### One site no grep found

`moe_jlens.md` asserted "catastrophic failure relative to dense models" in **prose**, with no
`0.7` on the line. Both sweeps were anchored on the number; this expressed the same
unsupported comparison in words and was invisible to each. It was caught only by reading the
full text of every near-miss instead of trusting the filter's summary.

**Lesson for the next sweep:** grep the *claim*, not the *digit*. A number-anchored sweep
finds every instance except the ones written out longhand — and those are the ones a
reviewer reads.

### What survives, restated

The MoE negative result is untouched: conditioned ≈ random (6.7% vs 6.4%), 0/3 significant
layers, null-swarm under the pre-registered criteria, resting entirely on the internal
three-arm comparison through a single pipeline. What was removed is an external calibration
that never existed.
