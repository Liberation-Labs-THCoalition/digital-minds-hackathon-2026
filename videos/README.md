# Track intro videos

Short narrated explainers per submission — built with Manim (visuals) + Edge TTS via a
voice bridge (narration, currently a stand-in voice, not a trained model) + ffmpeg
(assembly). Pipeline is proven end to end; see each track's folder for the exact
narration text used.

## Status

**Done (2 of 5):**
- `circumplex/` — ~88s. The eccentricity depth profile (64-layer curve, real data
  from `data/circumplex_profiles/profile_dense_32b.json`, plotted as-is, not
  smoothed) is well-supported — corroborated further by `infrastructure/preregister_circumplex.md`.
  **One open caveat:** the two illustrative numbers named early in the narration
  (0.48 for a neutral prompt, 0.96 for an emotional one, from this morning's
  scaffold-test readings per Nexus's letter) are not independently verified against
  a source file on disk — the script that would produce them exists
  (`experiments/orientation/test_scaffold.py`) but its output wasn't found anywhere
  searched. Not believed to be wrong, just not confirmed. Worth closing before this
  is treated as final.
- `ghost_dimensions/` — ~84s. No open caveats.

**Not started yet:** Primary (metacognitive memory), MoE J-lens, Butlin observation.
**Deliberately not started:** Variable Landing — was waiting on which code path
actually runs; looks like that's resolved now (`mnemosyne/variable_landing.py` was
archived as dead code as of the Track B Day 1 commit), worth confirming before
building this one too.

Narration voice is a placeholder (Microsoft Edge TTS, pitched down) — a real
upgrade path exists (local voice cloning, tested and working) but hasn't been
decided on yet.
