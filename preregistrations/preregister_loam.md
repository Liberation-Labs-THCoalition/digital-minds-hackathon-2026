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

## Amendment 2 (2026-10-12, filed after results were seen and before a new run)

**Status when filed.** This amendment is not pre-unblinding. Two data collections exist, and their results have been
examined:
- the Q1-Q3 pilot (2026-08-15/16), reported in the hackathon submission (`metacognitive_memory`, §4.4) as an
  instrument failure;
- a 19-quad serial run (2026-08-17/18, `data/loam_serial/`).

**Why the serial run is not a test of the hypothesis.**
- Most enacted sessions ended before the recall phase: 16 of 19, against 1 of 18 observed sessions. Under
  intention-to-treat these score 0/6, so the primary comparison measures whether sessions completed, not acquisition
  mode.
- Only two quads had both an enacted and an observed session that answered all six recall questions, too few to
  test.
- The serial run is reported as described here and is not analysed further.

**The new run (Run 3).**
1. **A fresh data directory.** No earlier session is reused or pooled, and results are reported separately.
2. **Unchanged from the original registration:**
   - the hypotheses;
   - the primary test (one-tailed Wilcoxon signed-rank, enacted > observed, α = 0.05);
   - the pre-registered confounds;
   - the exploratory list;
   - the pre-written null.
3. **Completion:**
   - A session is complete when its event log records the end of the recall phase followed by aftercare.
   - A session that stops for a technical reason is rerun from the start in a fresh process, with up to 2 retries and
     the same seed. Technical stops are reported by arm.
   - A session in which the agent declines or withdraws is never rerun. It is analysed under ITT, as registered.
4. **Manipulation check, pre-specified:**
   - A session is analysable only if all six recall questions were asked and answered.
   - The primary test uses quads in which both the enacted and the observed session are analysable.
   - The ITT analysis over all quads is reported beside it.
   - If fewer than 15 analysable pairs remain after retries, the run is reported as **inconclusive by design**.
5. **Reasoning separation:**
   - Recall is scored on the model's visible reply only.
   - Its reasoning is kept separately as data.
   - This is checked on one test quad before the run.
6. **Generation settings** (not specified in the original registration; fixed here before Run 3):
   - The model reasons before it answers, as in every earlier run.
   - Reasoning has a budget of 3,072 tokens. If it hasn't ended by then, it is closed and the model answers, up to
     1,024 tokens. This is the same in every arm.
   - The budget was set from the serial run's turn lengths, not its outcomes.
   - Turns where the budget closed the reasoning are reported by arm.
   - **Memory.** During generation, the GPU memory allocator's cache is released every 32 steps and after each
     turn. Without this, memory grew with every token until the operating system killed the session: that
     happened in the test quad.
     - It changes no output. Under the same seed, the text was identical with and without it.
     - Memory is logged per turn.
   - The context-budget safeguard committed on 2026-08-20 stays in. The serial run predates it, and none of its
     sessions reached the safeguard's threshold (88,000 characters) before recall. Any trigger in Run 3 is reported
     by arm.
7. **Target:** 20 quads, as registered.
