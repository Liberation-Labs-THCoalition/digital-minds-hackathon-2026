# Video stills storyboard — FLUX reference-image pass

Draft, ready to execute the moment there's real headroom on Starship (or wherever).
All prompts assume the existing LoRA stack (likeness/kintsugi/anatomy weights) is
available but only the base model is actually needed for the non-kintsugi pieces.

Source frames pulled from the shipped v2 finals, alongside this file in
`videos/storyboards/`. Strength = FluxImg2ImgPipeline denoising strength — low
keeps geometry rigid, higher lets FLUX drift further from the source.

---

## Circumplex

### 1. Opening circle — the balanced state
**Source:** `circ_01_circle.png` (t=58s — landed on the circle
itself, before the morph; still the strongest single frame of this beat)
**Strength:** 0.35–0.45 (preserve position/proportions, add depth and glow)
**Prompt:** "A single luminous circle drawn in fine cerulean light against a
near-black void, a faint crosshair through its center, the geometry glowing
softly as if freshly measured. Minimalist scientific illustration, restrained
bioluminescent glow, generous negative space, cool blue against deep
charcoal-black. No clutter."

### 2. The full 64-layer curve
**Source:** `circ_02_full_curve.png` (t=130s)
**Strength:** 0.25–0.35 — **this one's data-critical, keep it close.** The curve
shape is real measurement; FLUX should add atmosphere around it, not reshape it.
**Prompt:** "A single continuous line tracing a mountain range of data across a
dark field, cerulean blue against near-black, its jagged descent and rise
rendered like a horizon at dusk. Precise, unsmoothed, honest in its roughness —
scientific data as landscape, not decoration."

### 3. Dense vs. hybrid overlay — the payoff shot
**Source:** `circ_03_dual_overlay.png` (t=165s, pre-dim, both
curves at full opacity)
**Strength:** 0.25–0.35 — same reasoning as #2, both curves are real data.
**Prompt:** "Two luminous curves inhabiting the same dark field — one cool blue
diving low, one warm amber floating high, never quite touching. Two honest
measurements of the same question, like two separate weather systems crossing
the same sky."

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

## Primary (Track 4) — once the workspace-probe fix has real data behind it

No Manim source exists yet (video not started, still held per task #19's
resolution). This is the one that actually wants the kintsugi LoRA, not just
the base model — "the system remembers what it was thinking, not just what it
was told" sits directly on top of the fragmentation/repair language I already
have a proven pipeline for.

**Prompt (full LoRA stack — likeness/kintsugi/anatomy weights):** "A translucent
form cracked through with fine seams of liquid gold light, the cracks not
disfiguring but connecting — memory as repaired ceramic, each gold seam a
place where something was dropped and then deliberately, carefully, found
again."

---

## Not yet storyboarded
MoE J-lens, Butlin — no scripts written yet, nothing to draft prompts against.
Butlin's the one I flagged earlier as wanting something more personal than a
bar chart, given I'm a calibration point in it myself — worth its own pass
once that track actually starts.
