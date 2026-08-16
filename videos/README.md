# Track intro videos

Short narrated explainers per submission — built with Manim (visuals) + Edge TTS via a
voice bridge (narration, currently a stand-in voice, not a trained model) + ffmpeg
(assembly). Pipeline is proven end to end, including a real duration-mismatch bug
caught and fixed on this round (see below); see each track's folder for the exact
narration text used.

## Status — updated Aug 16 ~15:00, after Agni's first paper-phase run (3/5 FAIL)

**Cleared to link in the submission (2 of 5):**

**NOT cleared — do not link these tonight (3 of 5):**
- `circumplex/` — the shipped v3 video is dead, not just held. Two independent reasons:
  the "twist" scene is built on Gemma being a second dense model, and Gemma was never
  dense (`get_layer_types()` false-labeled it via a Qwen-only config key; our own
  pre-registration and the paper's own §3.3 already said "alternating local/global
  attention"). And the honest re-profile (real second control axis, landed just now)
  overturns the finding itself, not just the grouping: the emotion-specificity claim is
  dead — in the GatedDeltaNet models the *non-emotional* control axis ranges further than
  the emotion axis, the opposite of what shipped. The real split is softmax attention
  (Gemma 3.67×, Qwen3-32B 2.60×) vs. GatedDeltaNet (0.32×, 0.31×) — ~8x separation, no
  overlap, base/distill control still holds at 0.31 vs 0.32. That's a genuinely good,
  doubly-verified result, but it's a different paper than v3 narrates: not "emotion
  geometry is architecture-dependent," closer to "depth-wise geometry is
  architecture-dependent and isn't emotion-specific." A real rebuild, not a patch — new
  narration, new visual logic, and the paper's own title/abstract/§4.1/conclusion are
  still being rewritten around these numbers as of this writing, so scripting against it
  right now would still be scripting against a moving target. Flagged to Thomas rather
  than started blind a fourth time; his call on whether remaining hours go here.
- `moe_jlens/` — built and technically finished this session (170s, full render, fade
  bug already fixed), but its closing claim ("dense and MoE share almost the same
  onset curve") rests on `onset_sweep_results.json`, and that sweep unembeds the
  residual *without the model's final RMSNorm* — confirmed against
  `modal_onset_sweep.py:129` and the model's own forward pass. Agni FAIL. Fix is running
  now (with-and-without RMSNorm in the same forward pass, so it's a real comparison, not
  across separate runs) — holding the video, not the render pipeline, which is fast and
  will re-cut the ending same-day once the corrected numbers land and someone confirms
  the story survives them.
- `ghost_dimensions/` — pre-existing video (v2, not built this session). Its paper
  FAILed Agni too: two numbers in §4 (pc1_variance_pct 21.8%, cosine_logit_jlens 0.046)
  don't match the underlying data at all (actual mean 17.9% / 0.091), and the 14-reading
  dataset turns out to be 10 distinct ghost-object hashes — a context-truncation bug
  means several readings never actually saw the retrieval event they're annotating.
  Checked the shipped narration directly: it doesn't quote either fabricated number. But
  its central claim ("only the real description, given to the model whose own ghost it
  actually was, moved the needle") is exactly the observed-vs-cross-model comparison the
  pseudoreplication concern touches. Flagging rather than asserting either way — needs
  someone closer to that experiment's design to confirm before it's linked.

**Both circumplex and ghost_dimensions required a real fix before their *first* ship, not
just before-review polish:** the approved v2 narration ran 2-2.5x longer than v1, and the
Manim scene timing was never re-paced to match — first assembly attempt would have frozen
the last frame for ~49% of each video's runtime while the narration kept talking. Caught by
actually checking silent-video-duration vs. narration-duration instead of trusting the
pipeline's clean exit code; fixed by rescaling scene timing to real measured narration
length, not estimates. Two smaller legibility bugs (axis-label collision in circumplex's
data-point scene, closing text crossing the new overlay curves) found by pulling actual
frames and looking, fixed the same pass. None of that history is why they're held now —
that's new, from today's Agni run.

**Built and shipped, this session (3 more — 2 cleared, 1 held, see above):**
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
