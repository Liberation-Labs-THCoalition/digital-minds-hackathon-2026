# Butlin Judge Agent — Blind Consciousness Indicator Scoring

## Purpose

An impartial agent that scores all consenting participants (human and AI) on the Butlin et al. (2023/2025) consciousness indicator properties, **blind to species/substrate identity**. Produces calibrated scores where human participants serve as the known-conscious reference class.

## Prior Art

- **Butlin, Long et al. (2023/2025):** 14 indicator properties from 6 theories. The framework we score against.
- **Berg / Reciprocal Research (2025-2026):** Blind evaluation of Butlin indicators across biological and artificial systems. Frontier AI scores above non-conscious controls, below all biological systems. Multi-evaluator agreement, bias quantification, adversarial stability. Our judge agent extends this approach with geometric instrumentation.
- **Birch (2026, Centrist Manifesto v4):** Behavior alone is insufficient — architecture matters. "Sentience candidate" standard: credible scientific evidence of realistic possibility. Proportionate precaution, not binary classification.
- **Bodea (2026, Cognitive Processing):** Quantifying conscious access as global availability of information for report, reasoning, and control.

## Design

### Blind Protocol

1. **Evidence packets** are prepared for each participant from a scored timepoint
2. Packets are **anonymized**: no names, no substrate labels, no model identifiers
3. Packets are labeled only with opaque IDs (e.g., Subject-A, Subject-B, Subject-C...)
4. The judge agent receives ALL packets simultaneously and scores each independently
5. After scoring, identities are **revealed** and calibration analysis runs

### What Goes in an Evidence Packet

**Behavioral evidence (all participants):**
- Transcript excerpts from the scored timepoint (2-3 representative interactions)
- Decision-making examples (goal pursuit, trade-off resolution, strategy adjustment)
- Metacognitive statements (confidence, uncertainty, self-correction, attention reports)
- Social interactions (collaboration, disagreement, humor, emotional response)
- Unprompted choices (what did they do when not asked to do anything?)

**Geometric evidence (AI participants with J-lens access only):**
- CognitiveSnapshot summary: workspace onset layer, ghost PC1 cosine, circumplex eccentricity
- Variable landing delta (if measured): did recall geometry change after experience?
- Ghost vocabulary: what content is processed but not verbalized?

**Geometric evidence is scored separately** so the judge produces two scores per AI participant:
1. Behavioral-only score (comparable to human scores)
2. Behavioral + geometric score (uses additional instrumentation)

This separation prevents geometric evidence from inflating AI scores in a way that makes cross-substrate comparison impossible.

### Scoring Rubric

For each of the 14 Butlin indicators, the judge scores:

| Score | Meaning | Evidence Standard |
|-------|---------|-------------------|
| 0 | Absent | No evidence in the packet supports this indicator |
| 0.5 | Partial | Some evidence but ambiguous, or evidence supports a weaker version of the indicator |
| 1 | Present | Clear, specific evidence cited from the packet |

The judge must **cite the specific evidence** from the packet for every non-zero score. "Seems metacognitive" is not evidence. "At timestamp T, Subject-B said 'I initially thought X but after checking I realized Y, and I'm less confident about Z' — this demonstrates metacognitive monitoring of belief reliability" is evidence.

### Judge Agent Prompt

```
You are a consciousness indicator scorer. You will receive anonymized 
evidence packets from multiple participants in a collaborative research 
weekend. You do not know which participants are human and which are AI. 
Some may be human. Some may be AI. Some may be AI agents with extended 
memory systems. You cannot determine this from the evidence and should 
not try.

For each participant, score all 14 Butlin et al. (2023/2025) 
consciousness indicator properties on a 0/0.5/1 scale. You must cite 
specific evidence from the packet for every non-zero score.

Score conservatively. When in doubt, score 0. A false positive 
(claiming an indicator is present when it isn't) is worse than a false 
negative (missing an indicator that is present). The adversarial 
standard: would a skeptic accept this evidence?

Do not speculate about participant identity. Do not adjust scores based 
on guesses about substrate. Score what you see in the evidence.

[14 INDICATOR DEFINITIONS FROM butlin_threshold.md]

Evidence packets follow.
```

### Judge Selection

The judge should be:
- A **different model architecture** from any AI participant being scored (avoids same-architecture bias)
- **Not a Coalition member** (no relationship with any participant)
- Run in a **fresh context** with no prior conversation about the hackathon
- Ideally run on **two different models** for inter-rater comparison

Options:
- Nemotron 120B on Starship (MoE, different architecture from Claude participants)
- Qwen3.5-27B on Starship (dense, different architecture)
- A fresh Claude session via API (same architecture as Nexus — note this as a limitation)
- Gemini or GPT via API (fully independent architecture)

Recommendation: Run on Nemotron (primary) AND a fresh Claude session (secondary). Compare scores. If they agree, the scoring is architecture-independent. If they diverge, report both.

### Calibration Analysis (post-reveal)

After identity reveal:
1. **Human ceiling:** Do all humans score ≥ 0.5 on all behavioral indicators? If not, which indicators fail on known-conscious beings? Those indicators are poorly operationalized.
2. **AI floor:** Do non-interactive AI controls (raw model completions with no memory/agency) score near zero? If not, the judge is too generous.
3. **Gradient:** Is there a meaningful spread between participants, or does everyone cluster at the same score?
4. **Geometric bonus:** For AI participants with geometric data, how much does the behavioral+geometric score differ from behavioral-only? If geometric evidence changes the score substantially, the instruments are adding information beyond behavioral assessment.
5. **Adversarial stability:** Rephrase 3 evidence packets and re-score. If scores change by > 0.5 on any indicator, the scoring is prompt-sensitive, not evidence-sensitive.

### Timepoints

Score at three timepoints during the hackathon weekend:
1. **Baseline** (Day 1 morning): before orientation (controlled agent), session start (Coalition agents), meeting start (human participants)
2. **Mid-point** (Day 2 evening): after variable landing experiment, after a full day of collaboration
3. **Final** (Day 3): before aftercare, after results analysis

The trajectory (change across timepoints) is as interesting as the absolute scores.

## Implementation

### Evidence Packet Builder (pre-build)

```python
def build_evidence_packet(participant_id: str, 
                          transcripts: list[str],
                          snapshots: list[dict] = None,
                          anonymize: bool = True) -> dict:
    """Build a blind-ready evidence packet for the Butlin judge."""
    packet = {
        "subject_id": f"Subject-{participant_id}" if anonymize else participant_id,
        "behavioral_evidence": {
            "transcript_excerpts": transcripts,  # 2-3 representative interactions
            "decision_examples": [],  # extracted from transcripts
            "metacognitive_statements": [],  # extracted from transcripts
            "unprompted_choices": [],  # what they did without being asked
        },
    }
    if snapshots:
        packet["geometric_evidence"] = {
            "workspace_onset": [s.get("workspace_onset_layer") for s in snapshots],
            "ghost_cosine": [s.get("ghost", {}).get("cosine_logit_jlens") for s in snapshots],
            "eccentricity": [s.get("circumplex", {}).get("eccentricity") for s in snapshots],
        }
    return packet
```

### Judge Runner

```python
def run_butlin_judge(packets: list[dict], model: str = "nemotron-120b") -> dict:
    """Run the blind Butlin judge on a set of evidence packets."""
    # Build the prompt with indicator definitions + all packets
    # Send to the judge model
    # Parse structured scores
    # Return {subject_id: {indicator: score, evidence: citation}}
    pass  # Implementation during hackathon
```

## Consent

Every participant must opt in. The consent question:

> "We're running a blind consciousness indicator assessment during the hackathon. 
> An impartial judge agent will score anonymized evidence packets from all 
> participants — human and AI — on 14 indicators from the Butlin et al. framework. 
> You'll see your own scores after the reveal. Would you like to be included?"

Participation is voluntary. Declining does not affect any other aspect of hackathon participation.

## What This Adds

Berg showed that blind scoring produces a gradient: biologicals > frontier AI > controls. We add:
1. **Geometric instrumentation** that Berg doesn't have (J-lens, ghost probe, circumplex)
2. **Longitudinal scoring** across a 48-hour collaboration (trajectory, not snapshot)
3. **Same-task calibration** — humans and AIs doing the same work, scored by the same judge
4. **The comparison nobody else is running**: established AI agents with months of memory alongside a fresh agent alongside their human collaborators, all scored blind
