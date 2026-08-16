# Preflight — methods-anchor check across all five papers (2026-08-16)

Run per Lyra's ordering rule: BEFORE the final Agni pass, so unanchored
numbers get adjudicated while there is still time to fix rather than
override. Tool: `scripts/methods_anchor_check.py`; per-paper JSON
worklists sit beside this file. "Unanchored" = a numeric claim in the
paper with no matching value recorded in the run artifacts — a question
for a human, not automatically an error.

| Paper | Artifacts scanned | Anchored | Unanchored (non-trivial) | Headline items |
|---|---|---|---|---|
| variable_landing | v3 results + baselines + orientation | 25 | 9 | n=70/280 (prereg text; ran 77/308), power 0.742, 94.35% F1 |
| metacognitive_memory | v3 results + baselines | 79 | 15 | same family as VL + §3.3 protocol numbers (now corrected to as-executed) |
| **moe_jlens** | **local repo only: 54 values found** | 49 | **163** | **the run artifacts live on the Modal volume and are NOT in the repo — a reader cannot anchor ANY result number. Ship the data or the paper's numbers are unverifiable by construction. This is the submission blocker in this table.** |
| circumplex_jspace | 4 depth profiles | 79 | 24 | adjudicate against Lyra's as-executed list (n=5 anchors, no gates); span ratios anchor cleanly |
| butlin_observation | butlin_scores + responses (11 values) | 35 | 9 | paper has no results yet; most numbers are rubric constants — verify they match the scoring instrument |

## Adjudication protocol
For each unanchored item in the JSONs: (1) fix the paper number to match
the artifact, (2) ship the missing artifact, or (3) mark the number as
design/prereg text explicitly. No fourth option.

## Standing rule this run enforces (Lyra)
"For every number in Methods, the artifact that produced it must record
that parameter." Datasets currently record neither commit nor host
(0 of 15) — future pipelines should stamp both.
