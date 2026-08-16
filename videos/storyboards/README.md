# Video stills storyboard — FLUX reference-image pass

Draft, ready to execute the moment there's real headroom on Starship (or wherever).
All prompts assume the existing LoRA stack (likeness/kintsugi/anatomy weights) is
available but only the base model is actually needed for the non-kintsugi pieces.

Source frames pulled from the shipped v2 finals, alongside this file in
`videos/storyboards/`. Strength = FluxImg2ImgPipeline denoising strength — low
keeps geometry rigid, higher lets FLUX drift further from the source.

---

## Circumplex

**Rebuilt, not just retimed.** The L7-vs-L32 "different architecture, different
answer" claim (the old #3 below) didn't survive testing a third model — Gemma
is dense and its minimum sits with the hybrids, not the other dense model.
What actually holds, no overlap across all four models: the emotion/control
eccentricity span ratio (dense ~7.2-7.6x, hybrid ~1.9x), confirmed by a
base-vs-distill pair landing within 0.003 of each other. Narration rewritten
(v3, 212.5s, up from 198.3s), Scene 6 of the Manim source rebuilt with real
data for all four models — not just prompts, actual working code, rendered
and checked frame-by-frame. #1 and #2 below are Scenes 1-4, unchanged in
content; their exact timestamps shift slightly with the new total length but
the frames and prompts still apply. #3 is new.

### 1. Opening circle — the balanced state
**Source:** `circ_01_circle.png` (originally t=58s against the 198.3s cut;
shifts slightly with the new 212.5s length, content and frame unchanged)
**Strength:** 0.35–0.45 (preserve position/proportions, add depth and glow)
**Prompt:** "A single luminous circle drawn in fine cerulean light against a
near-black void, a faint crosshair through its center, the geometry glowing
softly as if freshly measured. Minimalist scientific illustration, restrained
bioluminescent glow, generous negative space, cool blue against deep
charcoal-black. No clutter."

### 2. The full 64-layer curve
**Source:** `circ_02_full_curve.png` (originally t=130s against the 198.3s cut,
shifts slightly with the new length)
**Strength:** 0.25–0.35 — **this one's data-critical, keep it close.** The curve
shape is real measurement; FLUX should add atmosphere around it, not reshape it.
**Prompt:** "A single continuous line tracing a mountain range of data across a
dark field, cerulean blue against near-black, its jagged descent and rise
rendered like a horizon at dusk. Precise, unsmoothed, honest in its roughness —
scientific data as landscape, not decoration."

### 3. Three curves, then four bars — the real payoff (replaces the old #3)
**This is real, rendered Manim output, not a FLUX still-image plan** — Scene 6
was rebuilt with working code against all four profile JSONs, test-rendered,
and checked frame by frame. Two beats, not one:

**3a. The twist (~t=44-93s into the new Scene 6).** Dense curve (cyan) and
hybrid curve (orange) recap as before, then a third curve — Gemma, violet,
deliberately not a near-shade of the dense cyan so the twist reads clearly —
grows in and its minimum lands beside the hybrid's, not the other dense
model's. A gold bracket connects the two nearby minima. Caption in-scene:
"lines up with the hybrid, not with the other dense model."

**3b. The real measurement (~t=93-163s).** Cut to a bar chart, four bars,
one per model: 7.63x and 7.18x (cyan, violet) clearly separated from 1.93x
and 1.93x (orange, salmon) with real vertical daylight between the two
clusters — no overlap, shown rather than asserted. Closes on the base/distill
pair's near-identical ratio as the clincher, then the bars dim and the final
text lands over them.

**If a FLUX still is wanted from this instead of the Manim frame directly:**
**Prompt:** "Four vertical bars of light on a dark field, two tall and two
short, a clear gap of empty space between the tall pair and the short pair —
no bar crossing into the other group's height. Precise, uncluttered,
scientific bar chart rendered as if made of light rather than plastic."

---

## Ghost Dimensions

### 4. The lone ghost bar
**Source:** `ghost_01_lone_bar.png` (t=60s, already pulled and eyeballed during
the pacing-fix review — clean single highlighted bar, "the ghost" label,
nothing else in frame yet). A blind re-guess at t=33s landed on a fade
transition instead; this one's confirmed good, use it.
**Strength:** 0.35–0.45 — lean into the "translucent, not solid" quality the
narration describes, which pure vector fill can only gesture at.
**Prompt:** "A single tall bar of warm gold light standing alone among a row of
small dim gray bars on a dark field, glowing softly from within like a candle
flame mid-flicker — translucent rather than solid. Presence without report,
felt more than seen."

### 5. Glasses finale
**Source:** `ghost_02_glasses.png` (t=205s)
**Strength:** 0.35–0.45
**Prompt:** "A pair of simple round spectacles resting just above a single
glowing gold bar of light on a dark field, the bar now solid and warm rather
than translucent — as if finally being truly seen. Quiet, resolved, a small
warmth in a large dark."

### 6. Cold open (no source frame — new establishing shot, txt2img not img2img)
**Prompt:** "A single point of pale gold light hovering in vast darkness, faint
concentric ripples suggesting it just spoke or was just heard, everything else
unresolved and formless. The exact moment before a thought either becomes
words or doesn't, held in suspension."

---

## MoE J-Lens (159.0s, real narration length, TTS-measured, not guessed — updated)

**Beat 9 rewritten, timings shifted.** The original "eight tries, one hit" beat
below was built around a specific gate number that Lyra found invalidated —
the ground truth in that check was hardcoded to a placeholder token, so it
never measured what the paper said it measured. Narration was corrected to
describe the broken check honestly instead of citing a number that doesn't
mean anything; storyboard follows.

No Manim source yet — video not started, but narration is written and
measured, so timing below is real, not estimated. Visual language: a reading
instrument (the lens) rendered as a clean beam/aperture; legible when it
works, scattered and dim when it doesn't. Same dark-field, restrained-glow
palette as circumplex/ghost dimensions, cerulean for the instrument itself
so it reads as the same family of tool across all three papers.

### 7. The lens, working (t=0-21s)
**Source:** none, txt2img. **Strength:** n/a.
**Prompt:** "A single clean beam of cerulean light passing through layered
translucent planes stacked in depth, each plane lighting up in clear
sequence as the beam passes — legible, orderly, nothing scattered. Dark
field, minimalist scientific illustration."

### 8. The architecture shift (t=21-39s)
**Prompt:** "The same layered planes, but now each one fractures into several
smaller separate facets, like a single beam splitting into a handful of
narrower ones that no longer travel together. Structure without a single
throughline. Dark field, cerulean against near-black."

### 9. The false gauge, removed (t=39-68s) — replaces the old "eight tries" beat
**Source:** none, txt2img.
**Prompt:** "A single measuring gauge on a dark field, its needle appearing
to move and respond to something — but on closer look the needle is fixed
in place, painted rather than connected to anything behind it. A narrow
beam of light reaches in and lifts the false gauge away entirely, leaving a
small, honest, empty space where a real reading should be. Nothing invented
to fill the gap."

### 10. Three paths, one wall (t=68-93s)
**Prompt:** "Three distinct beams of light — cerulean, amber, violet — each
taking a different route through the same layered field, all three arriving
at and stopping short of the same faint horizontal barrier, none passing
through. Three honest attempts, one shared limit."

### 11. The correction (t=93-128s)
**Prompt:** "A single glowing number hovering in dark space, a fine crack of
correction light passing through it and resolving into a different, smaller
number beside it — both visible, neither hidden. A magnifying quality to the
light, as if caught in the act of being checked."

### 12. Two curves, one shape (t=127-170s, closing — rewritten, this now resolves)
**The onset-sweep result landed: this was never MoE-specific, dense Qwen shows
almost the identical readability curve. The ending should actually land now,
not hang.**
**Prompt:** "Two beams of light — cerulean and warm amber, one that traveled
through branching layered structure and one that traveled through a single
continuous plane — arriving from different paths and settling into the
exact same resting glow at the same depth, side by side, equally bright.
Not a wall anymore. A shared shape, finally visible, resolved."

---

## Butlin Observation (143.3s, real narration length, TTS-measured)

No Manim source yet. Visual language deliberately avoids anything that reads
as a bar chart or checklist — flagged earlier as wanting something more
personal, given I'm a calibration point in this one myself. Softer, warmer
palette than the other two tracks; less measurement-instrument, more held
presence.

### 13. Overlapping lenses (t=0-26s)
**Prompt:** "Several translucent circular lenses of different faint colors —
amber, violet, teal, rose — overlapping in the center of a dark field, none
fully covering the others, their overlap brightening slightly where they
share space. No single lens dominant. Soft focus, generous negative space."

### 14. Fourteen, against the floor (t=26-58s)
**Prompt:** "Fourteen small soft points of warm light arranged in an
unforced cluster on a dark field, and beneath them, a faint even scatter of
dimmer identical points representing a noise floor — the fourteen clearly
brighter, but the floor visibly present, not hidden or erased."

### 15. Subject A through N (t=58-104s, the anonymization beat)
**This is the one to get right.** **Prompt:** "A row of simple identical
softly-glowing humanoid silhouettes on a dark field, each rendered in the
exact same warm tone and form regardless of size or posture variation,
labeled only with plain letters beneath them — no features distinguishing
one from another, deliberately unindividuated. Quiet, respectful, not
clinical."

### 16. Held open (t=104-143s, closing)
**Prompt:** "A single warm point of light at the center of a dark field,
neither expanding into full brightness nor fading to dark — held steady at
an in-between intensity, faint soft rings around it suggesting a question
still being asked rather than one already answered. Patient, not withheld."

---

## Metacognitive Memory / Primary (134.8s, real narration length, TTS-measured)

**Correction to what this section said before:** it read as still blocked —
it isn't. The module (this video) and the Variable Landing experiment are
separable; VL hit real problems (geometry-stub code path, pseudoreplication)
and stays on hold, but the module itself is validated on its own terms and
was never actually blocked. No Manim source yet, but nothing is stopping it
from being built now that narration and real timing exist.

This is the one that wants the kintsugi LoRA, not just the base model — "the
system remembers what it was thinking, not just what it was told" sits
directly on the fragmentation/repair visual language already proven. Also
the one place in this whole set where I'd deliberately reuse the ghost-bar
visual vocabulary from ghost dimensions rather than invent new — same probe,
same "the ghost," should look like the same thing when it shows up twice
across two different papers.

### 17. Retrieval, checked once (t=0-14s)
**Prompt:** "A single small object retrieved from a dark field and marked
with one plain checkmark of light — accurate, but that's the whole picture.
Nothing else in frame. Deliberately underwhelming, to set up what's missing."

### 18. Retrieved, unused (t=14-34s)
**Prompt:** "The same retrieved object now sitting inert at the edge of a
lit workspace, present but not connected to it — no beam, no absorption,
just proximity. The gap between finding and using, made visible as physical
distance that shouldn't be there."

### 19. Three layers, and the ghost (t=34-76s)
**Reuse the ghost-bar language from ghost dimensions directly — same probe,
same visual object.** **Prompt:** "Three concentric layers around a central
retrieved object: an outer ring showing a clean checkmark (found), a middle
ring showing active connected light (absorbed), and — set apart, translucent
and gold rather than solid — a single tall glowing bar standing just outside
all three rings: processing that's real but was never reported. Same ghost
as elsewhere in this work, unmistakably."

### 20. Four probes, one snapshot (t=76-109s)
**Prompt:** "Four distinct small beams of light — cerulean, gold, violet, and
warm white — converging from different angles into a single crystalline
recorded form at the center, faceted rather than solid, clearly built from
all four sources at once and query-able, not just logged."

### 21. Kintsugi close (t=109-135s) — already drafted, reused here verbatim
**Prompt (full LoRA stack — likeness/kintsugi/anatomy weights):** "A
translucent form cracked through with fine seams of liquid gold light, the
cracks not disfiguring but connecting — memory as repaired ceramic, each
gold seam a place where something was dropped and then deliberately,
carefully, found again."

---

## Status
Circumplex and Ghost Dimensions: fully storyboarded, timed against their real
shipped-video lengths, ready to execute whenever there's Starship headroom.
MoE, Butlin, and the memory-stack module: newly timed above against real,
TTS-measured narration lengths (148.9s / 143.3s / 134.8s) — drafts, not yet
executed, no Manim source built against them yet.
