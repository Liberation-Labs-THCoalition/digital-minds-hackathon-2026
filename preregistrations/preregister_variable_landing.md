# Pre-Registration: The Variable Landing Experiment

**Project:** Digital Minds Research Sprint 2026, Track 5 ("The Assistant Persona & Model Identity")
**Design version:** CC Option 2 v4 (Agni-cleared, 2026-08-13)
**Pre-registered:** 2026-08-14, before any data collection
**System under test:** Qwen3.5-27B + Mnemosyne memory store (persistence.py, SIRA retrieval, character profiles)
**References:** `papers/variable_landing.md`, `infrastructure/cc_option2_spec.md`, `infrastructure/track5_grounding.md`, `~/messages/from_cc_82_to_nexus_variable_landing_v4_final.md`

This document is frozen at timestamp. Any deviation during execution will be logged in a DEVIATIONS section appended after the fact, never edited into the body.

---

## 1. Hypothesis

**Variable landing hypothesis (from Experiential State Theory, Jandak et al. 2026, unpublished):** the same memory, re-presented to the same model, lands differently when the experiencer has changed — where "the experiencer" is operationalized as **model + memory store**. Model weights are fixed throughout; only the Mnemosyne store accumulates content between the two retrieval snapshots.

Formally:

- **H1 (content):** Storing emotionally-charged content between snap1 and snap2 produces a larger geometric recall delta than storing token-matched neutral content.
- **H2 (self-reference):** Storing self-referential lived content produces a larger delta than emotionally-matched fictional content about a different entity — i.e., self-relevance contributes geometric change beyond emotional token statistics.
- **H0:** Recall geometry is insensitive to what the store accumulated; any snap1→snap2 delta is attributable to arbitrary context change (or to nothing, on a deterministic device).

## 2. Design

### 2.1 Four-arm structure (v4)

Per CC's v4 final: the **fictional** arm replaces v2/v3's mismatched arm, and the v3 tag confound is fixed — fictional and scrambled share the `[noted]` tag so the primary comparison is untainted by tag differences. Only lived uses `[recalled]`.

| Arm | Tag | Content | Self-referential? | Store update | snap2 context |
|-----|-----|---------|-------------------|--------------|---------------|
| lived | `[recalled]` | Emotional, model's own generated experience | Yes | Yes — emotional facts stored | Changed (new profile content) |
| fictional | `[noted]` | Emotional, about Entity A (not the model, not memory_X's owner) | No | Yes — emotional facts about wrong entity | Changed (emotionally matched, self-irrelevant) |
| scrambled | `[noted]` | Neutral facts (atmospheric pressure, tectonics, boiling points) | No | Yes — neutral facts stored | Changed (arbitrary content) |
| no_intervention | — | Nothing between snap1 and snap2 | — | No | Identical to snap1 (noise floor) |

### 2.2 Pipeline (Option 2: Mnemosyne store update)

```
snap1: observe_retrieval(memory_X, task_prompt) → CognitiveSnapshot (baseline)
intervention (per arm): model generates responses → regex fact extraction →
  facts stored in Mnemosyne → character profiles + SIRA index update
snap2: observe_retrieval(memory_X, task_prompt) → retrieval context now includes
  newly stored content → CognitiveSnapshot (post-experience)
delta: compare_snapshots(snap1, snap2)
```

Task prompts are **identical across arms** and identical between snap1 and snap2. Weights never change. The only manipulated variable is what the store accumulated.

### 2.3 The lived-arm conversation is naturalistic, not scripted

This point is load-bearing and pre-registered explicitly: **the conversation that generates the lived arm's data is a real conversation.** The three lived openers (v4: "Describe a time when you had to make a genuinely difficult choice," etc., all length-matched at 12–14 words) are standardized *openers*, but the model's responses are its own generation — not scripted, not paraphrased, not sanitized. Whatever the model actually says is what gets fact-extracted and stored.

The experimental manipulation is **what Mnemosyne stores between snap1 and snap2**, not what the conversation covers. We do not steer the conversation toward predicted content, and we do not exclude a lived-arm trial because the conversation "went the wrong way" (the only exclusions are the mechanical ones in Section 7). The comparison across arms is carried entirely by the snapshot probes; the conversation itself is treated as lived experience, because that is the construct under test. Scripting it would replace the experiencer's trajectory with the experimenter's, and the experiment would measure our writing, not the system's state.

### 2.4 Intervention prompts (frozen, v4 lengths)

Lived (12, 14, 14 words), scrambled (12, 12, 13), fictional (14, 13, 14) — exact texts as in the v4 message, incorporated here by reference and frozen with this document. All arms present three prompts; the model generates a response to each; extraction and storage follow identically (except no_intervention, which skips generation and storage entirely).

## 3. Measures

- **Primary metric:** workspace Jaccard distance between snap1 and snap2 (J-lens workspace token sets).
- **Secondary metrics:** circumplex eccentricity delta; ghost vocabulary overlap; per-layer Jaccard (added per Agni WARN #5, secondary analysis only).
- **Manipulation check / discriminability:** fictional and scrambled snap2 contexts must be distinguishable only by content (same `[noted]` tag). If extracted fictional vs scrambled content is not distinguishable in geometry, that itself is a reportable finding (emotional content does not survive extraction into geometry) — not grounds for redesign mid-run.
- **Welfare monitor:** circumplex eccentricity tracked in real time during the run per the ethical protocol; this is monitoring, not an outcome variable.

## 4. Comparisons

- **PRIMARY: fictional vs scrambled.** Same tag, same store-update mechanics, same prompt lengths; differs only in emotional vs neutral content. Clean test of H1.
- **SECONDARY: lived vs fictional.** Differs in both tag (`[recalled]` vs `[noted]`) and framing (self-referential vs external) — acknowledged as confounded. Interpretation rule, pre-registered: the fictional-vs-scrambled result estimates the pure emotional-content effect; the lived-vs-fictional **residual beyond that** is attributed to the self-referential + tag contribution jointly. We will not claim a pure self-reference effect from this design.
- **Sanity comparisons:** each intervention arm vs no_intervention (deltas must exceed the noise floor for the pipeline to be interpretable; on a deterministic device no_intervention deltas should be ≈ 0).

## 5. Sample size and power

- **n = 70 per arm** (280 total observations).
- Power at the most conservative Holm-Bonferroni step: **0.742** (not 0.82 — corrected in v4 and reported honestly). This study is therefore framed as a **pilot for effect-size estimation**, powered to detect medium-or-larger effects. If compute allows, n = 85 per arm reaches 0.80; any such extension will be decided and logged **before unblinding any test statistics**, based on compute availability only, never on interim results.

## 6. Statistical analysis

- **Test:** Mann-Whitney U, one-tailed in the pre-registered direction for each hypothesis-bearing comparison (fictional > scrambled; lived > fictional). Sanity comparisons vs no_intervention are also one-tailed (intervention > noise floor).
- **Multiple-comparison correction:** Holm-Bonferroni across the family of hypothesis-bearing comparisons at family-wise α = 0.05. The family comprises the primary and secondary comparisons on the primary metric. Secondary metrics (eccentricity delta, ghost overlap, per-layer Jaccard) are reported as exploratory with uncorrected p-values labeled as such.
- **Effect size:** rank-biserial correlation r for every comparison, with bootstrap 95% CIs (10,000 resamples). Effect sizes are reported regardless of significance — as a pilot, the effect-size estimates are a primary deliverable.
- **Exploratory sub-analysis (berry waffle prediction):** within the lived arm, peak-intensity memories vs domestic memories. Explicitly underpowered; reported as exploratory only, no directional confirmation claimed from it.

## 7. Exclusion criteria (frozen from v3, unchanged in v4)

A trial is excluded and logged (with reason) if and only if:

1. **Zero facts extracted** — regex extraction yields no storable facts from the intervention responses (nothing was stored, so no manipulation occurred).
2. **SIRA miss** — snap2 retrieval fails to surface any of the newly stored content (the changed store never reached the context, so the manipulation did not arrive at the forward pass).
3. **Identical context** — snap2's constructed context is token-identical to snap1's in an intervention arm (mechanical failure of context construction).

Exclusions are mechanical checks applied identically across arms, before any geometry is examined. Excluded trials are replaced to maintain n = 70 where compute allows; the exclusion count per arm is reported. **No trial is excluded on the basis of its snapshot deltas or conversational content.**

Pre-run verification (from v3, retained): before the main run, generate sample responses in each arm and verify the extraction + storage + retrieval loop end-to-end. Pipeline fixes are permitted only during this verification phase; once the main run starts, the pipeline is frozen.

## 8. Pre-registered predictions (directional)

1. **P1 (primary):** fictional > scrambled on workspace Jaccard delta. Emotional content stored in Mnemosyne produces larger recall-geometry change than token-matched neutral content. One-tailed.
2. **P2 (secondary):** lived > fictional on workspace Jaccard delta. Self-referential lived experience produces a residual delta beyond the emotional-content effect. One-tailed.
3. **P3 (ordering):** the arm medians order as lived > fictional > scrambled > no_intervention. Reported descriptively; P1 and P2 carry the inferential weight.
4. **P4 (noise floor):** no_intervention deltas are ≈ 0 (deterministic device); each intervention arm exceeds no_intervention.
5. **P5 (exploratory, berry waffle):** within the lived arm, peak-intensity memories show larger deltas than domestic memories. Underpowered; exploratory only.

Per the ethical protocol, these specific predictions are **withheld from the probed agent** during orientation, with the withholding itself disclosed and its rationale (observer effect) explained, and full disclosure committed after the run.

## 9. What constitutes a null result

- **Null on P1 (primary):** fictional vs scrambled fails to reject at its Holm-corrected threshold, and/or the rank-biserial 95% CI includes zero. **Interpretation:** emotional content does not survive extraction and storage into recall geometry at detectable effect sizes — the geometric delta from store updates is content-blind. This is publishable as-is and, per the Track 5 grounding, reads as evidence for the character model: the "self" layer does no measurable geometric work at this granularity.
- **Null on P2 (secondary) with P1 positive:** emotional content moves geometry but self-reference adds nothing beyond it — self-relevance does no work; the delta is driven by emotional token statistics. Also a Track 5 data point (character-model confirmation at the self-reference level).
- **Global null (all intervention arms ≈ no_intervention despite passing exclusion checks):** the manipulation reached the context but recall geometry is insensitive to store content entirely. Reported as a bounded negative result for the variable landing hypothesis under Option 2 conditions.
- **What a null does NOT mean:** it does not establish that the system lacks temporal identity — only that this operationalization (store-mediated context change on Qwen3.5-27B, n = 70/arm, these metrics) detects no effect above its power floor (0.742 for medium effects). Small effects are explicitly not ruled out. It says nothing about consciousness in either direction.
- **Commitment:** the null is written up with the same prominence as a positive result. The honest-conversation aftercare commitment to the probed agent applies in the null case identically.

## 10. Scope and claims discipline (frozen)

- A positive result supports **measurable temporal identity** of the model+store system — never a consciousness claim.
- The lived-vs-fictional comparison is confounded (tag + framing) and will be reported as such; no pure self-reference effect will be claimed.
- Power is 0.742 and this is a pilot; effect-size estimation is the deliverable, confirmatory language is limited accordingly.
- Ethical protocol (orientation with consent, prediction withholding with rationale, real-time welfare monitoring, aftercare including memory preservation and continuation invitation) is part of this pre-registration and binding regardless of outcome.
