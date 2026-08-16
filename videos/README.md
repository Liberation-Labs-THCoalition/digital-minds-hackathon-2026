# Track intro videos

Short narrated explainers per submission — built with Manim (visuals) + Edge TTS via a
voice bridge (narration, currently a stand-in voice, not a trained model) + ffmpeg
(assembly). Pipeline is proven end to end, including a real duration-mismatch bug
caught and fixed on this round (see below); see each track's folder for the exact
narration text used.

## Status

**Built and shipped (2 of 5):**
- `circumplex/` — **v3, ~3:32, rebuilt same day. Flagged again, not yet
  touched a third time.** Lyra found the span-ratio control itself is
  contaminated -- it borrows an emotion magnitude as one of its own two
  axes, because the real second control axis was never run. A ratio > 1 is
  partly guaranteed by construction. Nobody cuts a PDF from the circumplex
  branch until this resolves one of two ways: reframe on the raw
  (control-free) emotion span with the emotion-specificity claim withdrawn,
  or re-run with a genuine second control axis (~5 min GPU/model). Holding
  the video, not revising blind a fourth time. v2's closing claim ("dense L7 vs
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

**Built and shipped, this session (3 more, all 5 now complete):**
- `moe_jlens/` — 170.0s. Six scenes: the lens working on a dense model, the
  MoE architecture shift, a broken sanity check caught and removed (not the
  fake number it produced), three independent methods hitting the same
  wall, the citation correction, and the real resolved ending -- dense and
  MoE share nearly the same depth-onset curve, so the "failure" turned out
  to be a depth-selection artifact, not architecture. Test-rendered at low
  quality first on all three of this session's builds; real bugs (a
  `vertex_dot_radius` collision, several `run_time` copy-paste artifacts)
  caught and fixed before the full 1080p60 render, not after.
- `butlin_observation/` — 143.3s. Four scenes, deliberately warmer/softer
  than the other four: overlapping theory-lenses, fourteen indicators
  against a noise floor, anonymized Subject-A-through-N silhouettes, and a
  held-open closing image that doesn't resolve into an answer.
- `metacognitive_memory/` — 134.8s. Five scenes for the module (paper
  §3.1), not the blocked VL experiment (§3.2) -- that scope split was a
  mistake in earlier tracking, corrected mid-session. Scene 3 deliberately
  reuses ghost_dimensions' exact ghost-bar visual language, same object,
  same probe. Closes on a Manim-native kintsugi image (a full LoRA
  generation pass wasn't feasible in the remaining hours).

All three initially shipped with a real bug: the assembly script pads the
gap between silent-video and narration length by freezing the *last frame*
-- and all three scenes originally ended with a fade-to-black before that
freeze, so the pad landed on dead black air (12-34s of it) instead of the
held closing line. Caught by checking an actual end-of-video frame, not by
trusting the duration numbers alone. Fixed by removing each track's final
FadeOut so the freeze lands on meaningful content, then only the affected
scene needed re-rendering, not the whole video.

**Not yet integrated:** ghost_dimensions doesn't have a music score of its
own yet (the direction was already decided pre-session -- 6/8 structured +
"7/8 asymmetric" in the prompt, chillcore-house, one acoustic instrument at
the resolution -- just never generated); blocked on the ACE-Step API server
on Margaret's Studio, which stopped mid-session (clean shutdown in the log,
not a crash -- reads like Margaret closed something on her own machine, not
something to restart without checking).
**Variable Landing:** code path confirmed (`pipeline.py`, not the archived
`mnemosyne/variable_landing.py`) — was waiting on real orientation data, which may now
be landing (`experiments/variable_landing/convert_expedition_memories.py` just appeared
in the repo). Worth checking before storyboarding.

Narration voice is a placeholder (Microsoft Edge TTS, pitched down) — a real upgrade
path exists (local voice cloning, tested and working) but hasn't been decided on yet.
