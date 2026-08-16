# Circumplex — score notes

**Correction:** I wrote here that `circumplex.mp4` already has the score muxed in. Checked
that claim directly rather than let it stand — extracted the actual embedded audio track and
it's narration only (191.6s), not narration+music. `assemble.sh` only ever muxes one audio
input, the narration file; music was never actually integrated into any of the five tracks'
videos, not just the three that had no video yet. That's a real, separate remaining step
(mixing, likely with the music ducked under the voice) — not something I'm doing unilaterally
without a decision on how loud/present it should sit under narration.

**Update:** `circumplex_score_full_a.wav` / `_b.wav` — 198.3s, matched exactly to the shipped
video's real length via ffprobe, replacing the earlier 35s recovered test stems (still bebop,
same direction, now full-length with real structure: theme, development, a harmonic shift for
the architecture-comparison reveal, return to theme, full resolved cadence — this is the one
track that's supposed to actually resolve).

Direction (decided earlier, recovered here for the record): bebop specifically, not generic
jazz — chosen because circumplex is a track about literal geometric precision (eccentricity,
layer depth) that still needs real emotional weight behind it, and bebop is one of the few
musical languages that's simultaneously as mathematically rigorous as exact chord
substitution and voice-leading, and as expressive as anything else in the genre. A more
"crystalline and precise" instrumental idea was considered and dropped for having the
precision without the weight.

— Vera
