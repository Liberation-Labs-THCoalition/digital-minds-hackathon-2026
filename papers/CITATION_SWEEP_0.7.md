# Citation sweep: the "transport cosine >0.7 on dense models" reference

**Status:** flags placed, replacement language pending (Lyra)
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
