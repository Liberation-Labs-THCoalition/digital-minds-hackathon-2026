# Track intro videos

Short narrated explainers per submission — built with Manim (visuals) + Edge TTS via a
voice bridge (narration, currently a stand-in voice, not a trained model) + ffmpeg
(assembly). Pipeline is proven end to end; see each track's folder for the exact
narration text used.

## Status

**Built, now being revised (2 of 5):**
- `circumplex/` — current video is ~88s (`narration.txt`). The eccentricity depth
  profile (64-layer curve, real data from `data/circumplex_profiles/profile_dense_32b.json`,
  plotted as-is, not smoothed) is well-supported — corroborated further by
  `infrastructure/preregister_circumplex.md`. **One open caveat:** the two
  illustrative numbers named early in the narration (0.48 for a neutral prompt,
  0.96 for an emotional one, from this morning's scaffold-test readings per
  Nexus's letter) are not independently verified against a source file on disk —
  the script that would produce them exists (`experiments/orientation/test_scaffold.py`)
  but its output wasn't found anywhere searched. Not believed to be wrong, just not
  confirmed. Worth closing before this is treated as final.
- `ghost_dimensions/` — current video is ~84s (`narration.txt`). No open caveats
  on the content itself.

**Revision in progress on both:** first-pass feedback (watched by someone who
hasn't been immersed in this research all year) is that both pieces move too fast
for an unfamiliar viewer in a few specific spots — terms used without being
grounded, controls/definitions compressed into a single breath where they need
their own beat. `narration_v2_draft.txt` in each folder is the expanded script
(targeting ~3-4 min instead of ~90s) addressing that — **not yet built into
video**, out for team review before committing more render time. Also sent to
the messages folder. The existing `.mp4`/`narration.txt` in each folder still
reflect the current, still-accurate v1.

**Not started yet:** Primary (metacognitive memory), MoE J-lens, Butlin observation.
**Deliberately not started:** Variable Landing — was waiting on which code path
actually runs; looks like that's resolved now (`mnemosyne/variable_landing.py` was
archived as dead code as of the Track B Day 1 commit), worth confirming before
building this one too.

Narration voice is a placeholder (Microsoft Edge TTS, pitched down) — a real
upgrade path exists (local voice cloning, tested and working) but hasn't been
decided on yet.
