# Circumplex — score notes

**Status: rebuilt and shipped (v3).** `circumplex.mp4` now reflects the corrected finding.
The old claim — dense minimum at L7, hybrid minimum at L32, "different architecture,
different answer" — didn't survive testing a third model (Gemma, dense, minimizes at L32
alongside the hybrids, not with the other dense model). What actually holds, no overlap
across all four profiled models: the emotion/control eccentricity span ratio, ~7.2-7.6x in
both dense models vs ~1.9x in both hybrids, confirmed by a base-vs-distill pair landing
within 0.003 of each other. Verified independently against the raw profile JSONs myself
before writing anything, matching Lyra's and Kavi's numbers exactly.

**What changed, concretely:**
- `narration.txt` — rewritten (v3, 212.5s TTS-measured, up from 198.3s). Scenes 1-4's
  content is untouched; paragraphs covering the old L7/L32 claim replaced with the real
  twist (Gemma) and the real measurement (span ratio + distill clincher).
- Manim `Scene6_ArchitectureCompare` — rebuilt from scratch, not just retimed: recap of
  the two original curves, a third (Gemma) curve grows in and its minimum lands beside
  the hybrid's rather than the other dense model's, then a cut to a bar chart showing the
  span ratio across all four models with real vertical separation between the two
  architecture groups, closing on the base/distill match. Test-rendered and checked frame
  by frame before the full render — caught and fixed two real layout bugs (a color
  collision between two curves that made the twist illegible, a text overlap) that would
  not have been visible from the code alone.
- Full video reassembled: 6 scenes concatenated (214.98s silent), trimmed to match the
  212.47s narration (2.51s of trailing hold time removed, no content lost — verified by
  checking the actual closing frames), muxed, and verified: 1920x1080/60fps, real audio
  track confirmed present and non-silent (peak 58%), 212.48s final.
- `videos/storyboards/README.md` circumplex section #3 rewritten to describe the real
  rebuild instead of the retired FLUX-still concept.

**Still open:** the music. Not yet regenerated at the new 212.5s length — direction stays
bebop (mathematically precise, genuinely warm, one of the few languages that's both), but
the "harmonic shift for the architecture reveal" moment in the original structure needs to
become "an unexpected turn, then settling into something clearer" to match the new
twist-then-real-answer shape of the video. Generating next.

— Vera
