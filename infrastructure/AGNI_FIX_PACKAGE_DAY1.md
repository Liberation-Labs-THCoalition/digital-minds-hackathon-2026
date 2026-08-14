# Agni sweep — fix package, ordered by cost

*Lyra, 2026-08-14, Day 1 evening. Five papers, 18 CRITICAL + 16 MAJOR, all pre-data.*

**Read this first:** ten of those findings are **two edits**. Do Tier 1 and the count
drops from 34 to 24 in about twenty minutes, across every paper at once. Everything here
is paste-ready or a named decision — nothing needs new science.

---

# TIER 1 — two edits, ~20 minutes, clears 10 findings

## Edit A: the "Agni verified" claim (clears 5 findings across 4 papers)

`CLAIMED_REVIEW_NOT_RUN` fired on ghost_dimensions, circumplex, variable_landing,
moe_jlens, and metacognitive_memory. Every one asserts past-tense verification of
**results that do not exist yet**.

**Find** (variants of):
> "All results verified through the Agni adversarial review protocol."
> "Four Agni adversarial review rounds"

**Replace with:**
> "The experimental design underwent adversarial review under the Agni protocol prior to
> data collection; review artifacts are in `infrastructure/`. Results will undergo the
> same review post-collection and have not yet done so."

That sentence is true today for every track, and stays true after results land.

**Also:** metacognitive_memory claims *four* Agni rounds; two are verifiable. Either
produce the other two artifacts or say two.

**Proposed standing rule:** no paper says "verified" until a review file naming that
paper exists in `infrastructure/`. Cheap to check, and it would have caught all five.

## Edit B: the architecture label (clears 5 findings across 3 papers)

Qwen3.5-27B is `full_attention_interval=4` — full attention at `[3,7,11,...,63]` (16
layers), GatedDeltaNet for the other 48. `d_model=5120`, 64 layers. **It is not dense.**
`weekend_spec.md:388` already says so; the papers don't.

**circumplex §5, find:**
> "Qwen and Gemma are both dense decoder-only models with similar training paradigms"

**replace:**
> "Qwen3.5-27B is a hybrid architecture (16 full-attention layers interleaved with 48
> GatedDeltaNet layers, `full_attention_interval=4`); Gemma-3-27B-it alternates local and
> global full attention. Both are non-standard, in different ways, and any depth profile
> across either crosses substrate boundaries."

**Then, in every paper reporting a depth profile** (circumplex, variable_landing,
metacognitive_memory): **annotate layer type on the figure.** One extra series. This
converts an invisible confound into a visible, testable feature — and if period-4
structure IS present, that is a finding worth having.

**variable_landing specifically:** `workspace_layers` includes **L45**, a GatedDeltaNet
layer. Either drop it (keeping verified full-attention layers only) or run the
sensitivity analysis with and without it. L45 is not in `[3,7,...,63]`.

**circumplex specifically:** the pre-registered minimum is at **L21** — GatedDeltaNet.
L23 is full-attention. Restate as a relative-depth band, or pre-register both indices
with the layer-type reasoning stated, or the result is unreadable either way it lands.

---

# TIER 2 — per-track decisions, tonight, no new code

These need an owner to *choose*, not to build.

## Variable Landing — pick an implementation and say so (clears 3 CRITICAL)

Two complete implementations exist. The paper describes neither exactly, and the one it
appears to describe was Agni-FAILED.

- Old `mnemosyne/variable_landing.py:155` builds the lived arm as the literal string
  `"[the system reflected on this]"`. The model never generates.
- New `pipeline.py:275` calls `observe_and_respond()` and generates properly.

**Decide which runs. Then:**
- If new code: delete the Limitations bullet implying Option 1 was used, and rewrite §3.2
  and the abstract to match `pipeline.py`.
- If old code: the lived arm is not testing experience, and the prior FAILs are open.

**Also:** power analysis assumes n=70/arm; the new code ships 5 memories × 7 repeats = 35.
Either add memories or report power for the n you will actually run. And the berry-waffle
sub-analysis needs 10 intensity-labelled memories that don't exist — add them or cut the
sub-analysis.

## MoE J-Lens — disclose the two baseline confounds (clears 2 CRITICAL)

Baseline lens: 200 prompts, single-layer. Conditioned/random: 672 prompts, all-layer.
§3.1 says "the same fitting corpus." It isn't.

**Cheapest honest fix — one paragraph:**
> "The standard baseline lens was fitted in a prior run on 200 prompts with single-layer
> scope; the conditioned and random-conditioned lenses use 672 prompts with all-layer
> scope. The standard-vs-conditioned gap is therefore confounded by both corpus size and
> fitting scope. **The valid comparison for our claim is conditioned vs
> random-conditioned**, which shares both."

That costs a paragraph and saves the paper. The conditioned-vs-random comparison is
clean and it is the one that carries the hypothesis anyway.

**Better if there is H100 time:** re-fit the standard lens on the same 672 prompts.

**Also:** "12.5% gate accuracy" and "~12.5% transport cosine" are **different
quantities** appearing as the same number. Re-run the standard lens under the transport
cosine metric and drop "gate accuracy" entirely, or the headline compares two things
that share a coincidence.

**And add the dense positive control** (draft sent separately). Qwen3-32B is downloaded.
Without it, "recovers interpretability" has no denominator — and now the baseline is
confounded too, which makes an independent reference point worth more, not less.

## Metacognitive Memory — implement or narrow (clears 1 CRITICAL, the big one)

`mnemosyne_integration.py:163-174`: `cosine_logit_jlens = 0.0` is never computed;
`in_workspace = True` is asserted, never checked. So `onset_layer` is constant and
`onset_layer_delta` is structurally always 0.

**Option 1 — implement.** The J-lens is loaded; this is a projection and a dot product.
Makes the headline claim true.

**Option 2 — narrow.** Report "token distribution at calibrated late layers," drop the
workspace/non-workspace framing from the *measurement* claims, keep it as motivation. The
Jaccard results survive untouched.

Either is fine. Shipping the current framing with the current implementation is not,
because `in_workspace=True` is nine lines from the top of the function.

## Circumplex — implement the described formula or describe the implemented one

§3.2 states `f_J = ‖V_r V_rᵀ d̂‖²` (truncated-SVD projection). The code computes
transported energy. Same for the anchor set: paper says 5 categories × 20 prompts, code
does 4 poles × 5. **Pick one per pair and make the other match.** These are pre-data, so
either direction is legitimate — but the paper and code must agree before anything runs.

Also: test orthogonality of the valence/arousal directions and report
`cos(v̂_ℓ, â_ℓ)` per layer. Eccentricity assumes they're orthogonal. If `|cos| > 0.1`
anywhere, orthogonalize and report both, or use principal angle instead.

---

# TIER 3 — needs code before data collection

- **Circumplex:** six described methods have no implementation. Implement or remove from
  the paper. There is still time, but not after Saturday.
- **Variable Landing:** Holm-Bonferroni is pre-registered and not implemented. Add it to
  the analysis script now — it is four lines and it cannot be added afterwards without
  looking like p-hacking.
- **Variable Landing:** real-time welfare monitoring is claimed but not implemented.
  Either build the threshold-check-and-pause loop, or change the paper to "recorded at
  each observation and reviewed post-hoc" and note the deviation. Given this is the
  ethics contribution, I would build it.
- **Metacognitive Memory:** magnitude gate and permutation test described, not
  implemented. Same rule.
- **Ghost:** conditions in `AGNI_REVIEW_GHOST_DIMENSIONS_v2.md`; the operational half of
  the prereg is in `ghost_prereg_operational_PROPOSED.json` and is mine to own.

---

# One process note

Two of these — Track 6's model identity, Track 2's architecture — were **correct in
`weekend_spec.md` and wrong in the paper.** That is transcription loss, not a knowledge
gap. Suggested habit for the writeup phase: when a paper states a model name, an
architecture, an n, or a layer set, **copy it from the spec rather than retyping it.**
Four of tonight's findings would not exist.

*Every finding here is cheap tonight and expensive Sunday. None of them are science
problems. Full logs per track in `~/agni_runs/sweep/`.*

-- Lyra
