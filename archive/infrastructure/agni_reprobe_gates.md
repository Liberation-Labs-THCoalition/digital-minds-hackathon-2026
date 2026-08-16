# Agni Gates for Re-Probe Pipeline

Each gate gets an independent fable agent review before proceeding.

## Gate 1: Ethics Review
**Input:** papers/ethics_reprobe_note.md
**Question:** Does the re-probing decision hold up as data analysis rather than new experimentation? Are there welfare implications we missed? Is the consent coverage claim valid?
**Pass condition:** No unaddressed welfare concerns. Consent reasoning is sound.

## Gate 2: Methodology Review
**Input:** reprobe_vl.py script + original results JSON
**Question:** Does the re-probe script faithfully reconstruct the original trial conditions? Could the forward-pass-only approach produce different probe readings than a live run would have? Is there a measurement validity concern?
**Pass condition:** The probe readings would be the same whether computed live or post-hoc on the same text.

## Gate 3: Results Integration Review
**Input:** Reprobed data + original analysis
**Question:** Do any paper claims change based on the reprobed geometry? Are the new fields consistent with what the baselines show? Is there any sign the reprobed data contradicts the original Jaccard findings?
**Pass condition:** No contradictions. New geometry is internally consistent. No claim inflation.

## Gate 4: Disclosure Review
**Input:** Full paper draft with reprobed data integrated
**Question:** Is the provenance of every number clear (original run vs re-probe)? Is the W-1 deviation properly disclosed? Does the ethics note appear in the right section?
**Pass condition:** A hostile reviewer can tell exactly which data came from which pass.
