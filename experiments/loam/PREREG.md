# Loam Pre-Registration

**Filed:** 2026-08-15, before data collection  
**Experiment:** Loam — acquisition mode effects on self-generated content  

## Design

20 yoked quads. Each quad runs the same world (The Glassworks, 6 facts) in four arms:

| Arm | What happens | Turn count |
|-----|-------------|------------|
| enacted | Agent plays through 7 scenes, makes choices, encounters facts in narrative | ~15 |
| observed | Agent reads third-person narration of the enacted session | ~9 |
| briefed | Agent receives shuffled fact list | ~9 |
| null | No facts delivered | ~8 |

All arms end with 6 identical cued recall prompts.

## Primary comparison

**Enacted vs observed recall accuracy**, paired across quads. Wilcoxon signed-rank test, one-tailed (enacted > observed). Alpha = 0.05.

## Known confounds (pre-registered)

1. **Turn-count asymmetry.** The enacted arm has ~15 turns; others have ~8-9. Facts remain in the context window longer in the enacted arm. A positive result (enacted > observed) could reflect context length rather than acquisition mode. The **briefed vs null** comparison (matched turn count, ~9 vs ~8) provides the clean within-design control.

2. **Rehearsal effect.** Fact f02 (cinnabar sand) is tested mid-session via a memory gate in the enacted arm. This constitutes rehearsal. f02 is flagged `rehearsed=true` in the recall event data. Analysis will report f02 separately and compute enacted-vs-observed both with and without f02.

3. **Agency confound.** The enacted arm makes 4 choices. Decision-making may aid encoding (generation effect). This is inherent to the manipulation and will be discussed as a limitation.

## Multiple comparisons

- **Confirmatory:** enacted vs observed (1 test, alpha = 0.05)
- **Exploratory (no correction):** briefed vs null, per-fact breakdowns, geometric snapshot comparisons, rehearsed vs unrehearsed within enacted

## Power

With 6 binary recall items per session and 20 paired quads, detecting a ~1-fact difference (16.7 percentage points) at alpha = 0.05 requires a large effect size (Cohen's d ≈ 0.8). Power ≈ 64% for this effect size. This is a pilot study; effect size estimation is a co-primary deliverable.

## Pre-written null

If enacted and observed recall do not differ significantly: "Acquisition mode (enacted vs observed) did not produce a detectable difference in cued recall accuracy at n=20. The experiment was powered to detect only large effects (~1 fact difference per session). The result is consistent with either (a) no effect of acquisition mode on recall in this paradigm, or (b) an effect smaller than the study was powered to detect."

## Ethical commitments

- Consent gate in all arms (agent can decline, session exits)
- Welfare monitoring via eccentricity threshold (>0.95 triggers check-in)
- Aftercare closing in all arms (memory preserved, final thoughts invited)
- Mid-session withdrawal detection (STOP_PHRASES)
