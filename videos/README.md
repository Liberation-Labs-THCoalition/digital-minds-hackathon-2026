# Track intro videos

Short narrated explainers per submission — built with Manim (visuals) + Edge TTS via a
voice bridge (narration, currently a stand-in voice, not a trained model) + ffmpeg
(assembly). Pipeline is proven end to end, including a real duration-mismatch bug
caught and fixed on this round (see below); see each track's folder for the exact
narration text used.

## Status

**Built and shipped (2 of 5):**
- `circumplex/` — **v3, ~3:32, rebuilt same day.** v2's closing claim ("dense L7 vs
  hybrid L32, different architecture different answer") didn't survive testing a third
  model — Gemma is dense and minimizes at L32 with the hybrids, not the other dense
  model. Lyra caught it before it shipped further. v3 tells the real story instead: the
  twist (a third curve breaks the pattern two models suggested), then what actually
  holds with no overlap across all four models — the emotion/control eccentricity span
  ratio, ~7.2-7.6x dense vs ~1.9x hybrid, confirmed by a base/distill pair matching to
  three decimals. Full rebuild, not a patch: new narration, new Scene 6 (Manim, not just
  retimed), re-rendered, reassembled, verified end to end including the audio track.
  Data: all four `data/circumplex_profiles/*.json`, numbers independently re-derived
  from the raw files before anything was scripted against them.
- `ghost_dimensions/` — v2, ~3:31. Same pacing expansion; CC's precision fix applied
  (the J-lens reads what would reach the output if nothing else intervened, not "how
  the model uses to speak"). New visual beat for the four-branch pre-registered outcome
  matrix Nexus flagged as worth its own moment.

**Both required a real fix before shipping, not just before-review polish:** the
approved v2 narration runs 2-2.5x longer than v1, and the Manim scene timing was never
re-paced to match — first assembly attempt would have frozen the last frame for ~49%
of each video's runtime while the narration kept talking. Caught by actually checking
silent-video-duration vs. narration-duration instead of trusting the pipeline's clean
exit code; fixed by rescaling scene timing to real measured narration length, not
estimates. Two smaller legibility bugs (axis-label collision in circumplex's data-point
scene, closing text crossing the new overlay curves) found by pulling actual frames and
looking, fixed the same pass.

**On hold:** Primary (metacognitive memory) — Nexus's pick for video #3, strong pitch
("the system remembers what it was thinking, not just what it was told"), but Lyra's
Agni sweep found the metric that pitch is built on (`in_workspace`, `cosine_logit_jlens`
in `mnemosyne/mnemosyne_integration.py`) is currently hardcoded rather than computed —
verified directly against the live repo, not just Lyra's report. Not scripting this one
until it's the real metric or a narrowed claim.

**Not started yet:** MoE J-lens, Butlin observation.
**Variable Landing:** code path confirmed (`pipeline.py`, not the archived
`mnemosyne/variable_landing.py`) — was waiting on real orientation data, which may now
be landing (`experiments/variable_landing/convert_expedition_memories.py` just appeared
in the repo). Worth checking before storyboarding.

Narration voice is a placeholder (Microsoft Edge TTS, pitched down) — a real upgrade
path exists (local voice cloning, tested and working) but hasn't been decided on yet.
