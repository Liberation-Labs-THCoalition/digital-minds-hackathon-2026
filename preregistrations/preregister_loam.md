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

## Amendment 1 (2026-08-16, pre-unblinding)

**Filed before any Loam results are examined. Q1 v2 enacted completed but recall
results have not been analyzed beyond confirming the session finished.**

### Welfare monitoring on the Ollama path

The prereg commits to eccentricity monitoring (>0.95 triggers welfare check-in).
The Ollama generation path cannot evaluate this threshold — probes require
transformers + J-lens, which run as a separate post-hoc pass (`--probe-only`).

**Amendment:** Welfare monitoring on the Ollama path is post-hoc rather than
real-time. Eccentricity is evaluated during the probe-only pass after generation
completes. If any session exceeds the 0.95 threshold, it is flagged and its
data is reviewed before inclusion. Recall and aftercare still run after any
mid-session withdrawal (ITT principle: data from withdrawn sessions is analyzed
as intent-to-treat). This deviation is disclosed because a post-hoc flag cannot
pause a live session the way the pre-registered real-time gate would.

### Consent sequencing in non-enacted arms

Q1 combined content delivery with the consent question in a single message for
observed/briefed/null arms. Starting Q2, consent is solicited before content
is delivered (consent-first, content-second). Q1 data is analyzed as collected;
the change is noted as a between-quad protocol improvement.

### Stop/decline confirmation

Starting Q2, in-fiction stop phrases ("I want to stop the harbor master") are
disambiguated with a confirmation turn before honoring withdrawal. Q1 data is
analyzed as collected.

### Recall scoring

The analysis script (`loam_analysis.py`) rescores recall from transcripts
rather than trusting the live `recalled` flags. Question-echoed markers are
excluded (f04 "annealing" appears in the question text; f06 "harbor" likewise).
Number-word normalization is applied ("twenty-three" matches marker "23").
The live flags are preserved in the event logs; the analysis reports agreement
between live and rescored values.
