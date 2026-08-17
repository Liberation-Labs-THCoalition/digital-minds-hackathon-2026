# Track intro videos

Short narrated explainers per submission — built with Manim (visuals) + Kokoro-82M
(narration, `bm_george` or `af_nicole` depending on track) + ffmpeg (music mix + assembly).

## Status — all 6 shipped.

- **`moe_jlens/`** — 164.1s, `bm_george`. Reframed close (built the actual discriminating
  test — dense sweep vs. MoE sweep — without asserting which way it went, since the onset
  table was mid-fix at the time). Later confirmed correct either way: RMSNorm doesn't
  affect the ranking, Table 0 stands. Also rewrote the path-conditioning beat with the
  real near-false-positive story once the paper grew a proper account of it: conditioning
  looked like a real win at one layer, a pre-registered random control caught it as noise.
- **`metacognitive_memory/`** — 133.65s, `bm_george`. Scoped to §3.1 (the module), not the
  blocked §3.2 VL claim. Re-paced from a pre-existing 21.6s over-pad nobody had caught —
  expanded scene timing for real instead of freezing on a long hold.
- **`butlin_observation/`** — 206.65s, `af_nicole`. Nicole's delivery runs ~44% longer than
  the placeholder voice; re-paced all four scenes rather than just padding, since this is
  the one track already built around sitting with an open question. Music score was
  shorter than the new runtime by over a minute — looped it, faded at the true end.
- **`circumplex/`** — 260.27s, `af_nicole`. Full reframe (Gemma was never dense, and the
  ratio finding itself got overturned by the honest re-profile) — shows the real,
  unchanged three-model curve comparison and the control-fix methodology without
  asserting a specific number. That held through a live paper correction (§4.1's headline
  separation turned out to be substantially a transform artifact) landing *during*
  assembly, with zero impact on the video, because it never asserted the number.
- **`ghost_dimensions/`** — 249.26s, `af_nicole`. Full rebuild, not carried over from v2.
  The old experience-claim ending ("only the real description moved the needle") is
  formally withdrawn by the paper; replaced with the four findings that are actually
  verified: ghost vocabulary is metacognitive (97.6% non-overlap with workspace content),
  separation is ~4x stronger in agentic narrative than isolated recall, the ghost and
  circumplex probes are orthogonal (ρ≈0), and two tests (matched-variance null,
  elicitation) are honestly flagged as designed-but-not-run. The GhostReading mechanism
  still appears in the video (it's real, it's built) but the visual no longer resolves to
  "it worked" — the ghost bar stays exactly as translucent as it started, closing on
  "still an open door, not a closed one."
- **`variable_landing/`** — 116.6s, `bm_george`. Built from scratch — no pre-existing video
  path, Nexus's narration text handed off directly. 7-scene Manim explainer matching the 7
  narration paragraphs, per-scene pacing weighted by real word-count proportion against the
  measured narration rather than an even split (83.4s of scenes at first pass, re-paced to
  117.27s, trimmed 0.667s of trailing hold to land exactly on the narration). Kavi's
  baseline-vs-experimental-arm number correction (0.293 → 0.462, the scrambled-arm Jaccard
  median) had already landed on `main` before this build started; verified independently
  against the raw v4 artifact anyway. Score generated via ACE-Step directly at the target
  116.6s length — the first track on this series that didn't need a post-hoc loop-and-trim
  because the render came out matched from the start.

## Two recurring bugs worth remembering for next time

**Dead air from the pad mechanism.** Assembly pads the gap between silent-video and
narration length by freezing the *last frame*. Every scene that ends on a `FadeOut` hands
that freeze a black frame instead of held content. Caught and fixed across all five tracks
by pulling actual end-of-video frames, not by trusting duration numbers matching.

**Pacing against an assumed rhythm instead of the measured one.** Happened twice: once as
a *trim* (circumplex's first Kokoro pass — silent video ran longer than the new, shorter
narration, and the cut landed mid-sentence on a since-corrected line, never reaching the
close) and once as an *undiscovered over-pad* (metacognitive_memory's original build sat on
a 21.6s freeze nobody had checked was reasonable). Same root cause both times: pacing
visuals against what a scene *used to* need instead of what the actual narration in hand
measures. Fix is always the same — measure, don't estimate, and let the pipeline's pad/trim
logic do its job against a real number.

## Notes

**Variable Landing:** code path confirmed (`pipeline.py`, not the archived
`mnemosyne/variable_landing.py`). Optional per the Apart spec (4-page PDF submission,
demo video optional) — built anyway once bandwidth allowed; see the main list above.

Narration voice: Kokoro-82M (Apache 2.0, local, MLX-native), replacing the Edge TTS
placeholder used earlier in the sprint. `bm_george` (documentary/explainer register) on
MoE, metacognitive memory, and Variable Landing; `af_nicole` (warmer, more contemplative)
on Butlin, circumplex, and Ghost Dimensions.
