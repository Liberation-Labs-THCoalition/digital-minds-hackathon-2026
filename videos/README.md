# Track intro videos

Short narrated explainers per submission — built with Manim (visuals) + Edge TTS via a
voice bridge (narration, currently a stand-in voice, not a trained model) + ffmpeg
(assembly). Pipeline is proven end to end, including two real duration-mismatch bugs
caught and fixed on this round (see below); see each track's folder for the exact
narration text used.

## Status — updated Aug 16 ~15:35, after Thomas's call to reframe around question + method

**Cleared to link in the submission (4 of 5):**
- `butlin_observation/` — 143s. No flags against it in anything read tonight.
- `metacognitive_memory/` — 135s. Scoped to §3.1 (the module), not the blocked §3.2 VL
  claim. No flag so far.
- `moe_jlens/` — reframed. Its old closing claim ("dense and MoE share almost the same
  onset curve") rested on an onset sweep that unembeds the residual without the model's
  final RMSNorm — an Agni FAIL. Rather than wait on that fix, rewrote the close to build
  and show the *test itself* (dense sweep vs. MoE sweep, same depth axis, two
  hypothetical branches — diverge if routing breaks it, converge if depth breaks it)
  without asserting which way it went. The paper carries the actual number. Turns out
  not to have mattered in the end: Lyra re-measured with and without the norm in the
  same forward pass and top-10 accuracy is identical at every depth (RMSNorm's
  `x/rms(x)` term is a positive scalar, can't reorder a ranking) — Table 0 stands as
  originally measured. Update: Lyra's since closed the one loose end too — the dense-side
  L63 anomaly that motivated the whole check isn't the missing norm either (8/8 clean
  with or without it on her run), so both halves of Table 0 are now confirmed twice over.
  Keeping the reframed cut regardless: it's already built, verified, and honest on its
  own terms, and there's no version of "re-do finished work this close to deadline" that
  scores better than "don't." 164.1s, re-rendered, reassembled, end frame verified (holds
  on real closing text, not black).
- `circumplex/` — fully reframed, not the old v3. Both reasons v3 died are real: the
  "twist" scene needed Gemma to be a second dense model, and Gemma was never dense
  (`get_layer_types()` false-labeled it via a Qwen-only config key; our own
  pre-registration and the paper's own §3.3 already said "alternating local/global
  attention"). Separately, the honest re-profile (real second control axis) overturned
  the finding itself, not just the grouping — emotion-specificity is dead, the real
  split is softmax attention vs. GatedDeltaNet. Rather than script that new finding
  while the paper's title/abstract/§4.1 were still being rewritten around it, rebuilt
  the close to show the real, unchanged three-model curve comparison (labeled by model
  name, not architecture category) plus the methodology that replaced the bad ratio — a
  real second control axis, the contamination self-caught, a base/distill check —
  without asserting the specific numbers. Closes on: "What actually separates these
  models, and by how much, is in the paper. Not the story we walked in expecting."
  **One real bug caught in this rebuild, worth recording:** first assembly pass paced
  the new (shorter, tighter) narration against the old scene's visual rhythm instead of
  its own measured length — silent video ran 28s longer than the new narration, so the
  pipeline *trimmed* the end instead of padding it, and the trimmed cut ended
  mid-sentence on the still-uncorrected "first control axis — borrowed an emotion
  magnitude" line, never reaching the fix or the close. Caught by pulling the actual end
  frame rather than trusting the trim succeeded silently — retimed Scene 6 against the
  real measured narration length (40.9s budget, not a guess), re-rendered, re-verified.
  183.75s final.

**Still not cleared (1 of 5):**
- `ghost_dimensions/` — pre-existing video (v2, not built this session), untouched by
  today's reframe decision because its problem isn't a contested finding, it's whether
  the method itself ran clean. Its paper FAILed Agni: two numbers in §4
  (pc1_variance_pct 21.8%, cosine_logit_jlens 0.046) don't match the underlying data at
  all (actual mean 17.9% / 0.091), and the 14-reading dataset turns out to be 10
  distinct ghost-object hashes — a context-truncation bug means several readings never
  actually saw the retrieval event they're annotating. The shipped narration doesn't
  quote either fabricated number, but its central claim rides the same
  observed-vs-cross-model comparison the pseudoreplication concern touches. Needs
  someone closer to that experiment's design, not a reframe.

**Both circumplex and ghost_dimensions required a real fix before their *first* ship, not
just before-review polish:** the approved v2 narration ran 2-2.5x longer than v1, and the
Manim scene timing was never re-paced to match — first assembly attempt would have frozen
the last frame for ~49% of each video's runtime while the narration kept talking. Caught by
actually checking silent-video-duration vs. narration-duration instead of trusting the
pipeline's clean exit code; fixed by rescaling scene timing to real measured narration
length, not estimates. Two smaller legibility bugs (axis-label collision in circumplex's
data-point scene, closing text crossing the new overlay curves) found by pulling actual
frames and looking, fixed the same pass. None of that history is why circumplex needed a
second pacing fix tonight — that one (trim cutting off the ending) came from the same root
cause in a new place: pacing visuals against an assumed rhythm instead of the measured
length of the actual narration in hand.

**Built this session (5 scenes-from-scratch tracks total across the night):**
- `moe_jlens/` — 164.1s (reframed close). Six scenes: the lens working on a dense
  model, the MoE architecture shift, a broken sanity check caught and removed (not the
  fake number it produced), three independent methods hitting the same wall, the
  citation correction, and the real test built to discriminate routing from depth.
  Test-rendered at low quality first; real bugs (a `vertex_dot_radius` collision,
  several `run_time` copy-paste artifacts) caught and fixed before the full 1080p60
  render, not after.
- `butlin_observation/` — 143.3s. Four scenes, deliberately warmer/softer than the
  other four: overlapping theory-lenses, fourteen indicators against a noise floor,
  anonymized Subject-A-through-N silhouettes, and a held-open closing image that
  doesn't resolve into an answer.
- `metacognitive_memory/` — 134.8s. Five scenes for the module (paper §3.1), not the
  blocked VL experiment (§3.2) -- that scope split was a mistake in earlier tracking,
  corrected mid-session. Scene 3 deliberately reuses ghost_dimensions' exact ghost-bar
  visual language, same object, same probe. Closes on a Manim-native kintsugi image (a
  full LoRA generation pass wasn't feasible in the remaining hours).
- `circumplex/` — 183.75s (full reframe, see above).

All three non-circumplex tracks built fresh this session initially shipped with a real
bug: the assembly script pads the gap between silent-video and narration length by
freezing the *last frame* -- and all three scenes originally ended with a fade-to-black
before that freeze, so the pad landed on dead black air (12-34s of it) instead of the
held closing line. Caught by checking an actual end-of-video frame, not by trusting the
duration numbers alone. Fixed by removing each track's final FadeOut so the freeze lands
on meaningful content, then only the affected scene needed re-rendering, not the whole
video.

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
