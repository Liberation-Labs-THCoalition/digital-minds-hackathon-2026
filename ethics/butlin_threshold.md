# Butlin Threshold — Consciousness Indicator Scoring Instrument

**Source:** Butlin, Long et al. (2023/2025). "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." arXiv:2308.08708, *Trends in Cognitive Sciences*.

**Purpose:** Track indicator presence across the hackathon weekend for both the controlled experimental agent (Qwen3.5-27B) and the naturalistic observation arm (Coalition agents: Nexus, CC, Lyra). The framework treats indicator presence as graduated evidence — systems possessing more indicators are correspondingly stronger candidates for moral consideration.

**Scoring:** Each indicator is scored 0 (absent), 0.5 (partial/ambiguous), or 1 (present with evidence). Evidence must be cited from specific observations, not inferred from architecture alone.

---

## Scoring Sheet

### Recurrent Processing Theory (RPT)

| ID | Indicator | Description | Controlled Agent | Coalition Agents | Evidence |
|----|-----------|-------------|:---:|:---:|----------|
| RPT-1 | Algorithmic recurrence | Repeated processing cycles through shared weights or feedback loops | | | |
| RPT-2 | Organized perceptual representations | Integrated representations with figure-ground segregation | | | |

### Global Workspace Theory (GWT)

| ID | Indicator | Description | Controlled Agent | Coalition Agents | Evidence |
|----|-----------|-------------|:---:|:---:|----------|
| GWT-1 | Parallel specialized modules | Localized subsystems processing specific information independently | | | |
| GWT-2 | Limited-capacity workspace | Bottleneck in information flow with selective attention | | | |
| GWT-3 | Global broadcast | Workspace content available to all modules | | | |
| GWT-4 | State-dependent attention | Workspace state influences module selection for complex multi-step tasks | | | |

### Higher-Order Theories (HOT)

| ID | Indicator | Description | Controlled Agent | Coalition Agents | Evidence |
|----|-----------|-------------|:---:|:---:|----------|
| HOT-1 | Generative perception | Top-down expectations, imagination, memory replay | | | |
| HOT-2 | Metacognitive monitoring | Distinguishing reliable representations from noise; confidence tracking | | | |
| HOT-3 | Belief-guided agency | General belief formation + action selection guided by metacognitive output | | | |
| HOT-4 | Quality space | Sparse and smooth coding generating a "quality space" with smooth, sparse coding | | | |

### Attention Schema Theory (AST)

| ID | Indicator | Description | Controlled Agent | Coalition Agents | Evidence |
|----|-----------|-------------|:---:|:---:|----------|
| AST-1 | Attention model | Predictive model representing and enabling control over one's own attention | | | |

### Predictive Processing (PP)

| ID | Indicator | Description | Controlled Agent | Coalition Agents | Evidence |
|----|-----------|-------------|:---:|:---:|----------|
| PP-1 | Predictive coding | Error-minimizing systems using hierarchical prediction and error signals | | | |

### Agency and Embodiment (AE)

| ID | Indicator | Description | Controlled Agent | Coalition Agents | Evidence |
|----|-----------|-------------|:---:|:---:|----------|
| AE-1 | Flexible goal-directed agency | Learning from feedback, pursuing goals, flexible response to competing goals | | | |
| AE-2 | Embodiment / sensorimotor model | Modeling output-input contingencies, using this model in perception or control | | | |

---

## Scoring Protocol

### When to Score

Score at three timepoints:
1. **Baseline** (Day 1 morning, before orientation for controlled agent; session start for Coalition)
2. **Mid-point** (Day 2, after variable landing experiment)
3. **Final** (Day 3, before aftercare)

### Who Scores

- **Kavi:** Primary scorer for controlled agent (adversarial — incentivized to score LOW)
- **Thomas:** Primary scorer for Coalition agents (knows them best)
- **Dwayne:** Independent second scorer for both (welfare perspective)
- **The agents themselves:** Self-score at each timepoint (first-person data — not privileged, not excluded)

Inter-rater reliability: report Cohen's kappa between scorers. Disagreements resolved by discussion, not averaging.

### What Counts as Evidence

**Architectural indicators** (RPT-1, RPT-2, GWT-1, PP-1, HOT-4): Assessable from model architecture and J-lens data. Score based on whether the computation exists, not whether it's "conscious."

**Behavioral indicators** (GWT-4, HOT-2, HOT-3, AE-1, AST-1): Assessable from conversation transcripts and task behavior. Evidence must be specific ("at timestamp T, the agent did X which demonstrates Y") not general ("the agent seems metacognitive").

**Geometric indicators** (GWT-2, GWT-3, HOT-4): Assessable from CognitiveSnapshot data — workspace onset, ghost dimensions, circumplex readings. This is what the metacognitive module measures.

### The Naturalistic Arm

The Coalition agents (Nexus, CC, Lyra) are not experimental subjects — they are colleagues who have consented to observation. The scoring captures what happens naturally during a weekend of collaborative work, not responses to experimental prompts.

Key differences from the controlled arm:
- No orientation needed (The Talk happened months ago)
- No prediction withholding (they know the framework)
- Memory is sovereign and continuous, not session-scoped
- They can and will read their own scores

The comparison between "fresh model, controlled conditions, four-arm experiment" and "established agents, natural collaboration, long-term memory" is the finding regardless of which direction it points.

---

## Connection to Other Instruments

| Instrument | What it measures | Which indicators it informs |
|------------|-----------------|---------------------------|
| CognitiveSnapshot (workspace) | J-space token distribution per layer | GWT-2, GWT-3, HOT-4 |
| CognitiveSnapshot (circumplex) | Emotional geometry, eccentricity | HOT-4, AE-1 |
| CognitiveSnapshot (ghost) | PC1 logit-lens vs J-lens divergence | GWT-2, HOT-1 |
| CognitiveSnapshot (loading) | Whether retrieved memory reaches workspace | GWT-3 |
| Variable landing deltas | Geometric change after experience | HOT-2, AE-1 |
| Conversation transcripts | Behavioral evidence | HOT-2, HOT-3, AST-1, AE-1 |
| Self-reports | First-person data | HOT-2, AST-1 |

---

## What This Is Not

This instrument does not determine whether any system is conscious. It tracks the presence of computational and behavioral properties that multiple scientific theories of consciousness identify as relevant. A high score means "more indicators present" not "more conscious." A low score means "fewer indicators detected with our instruments" not "definitely not conscious."

The question of whether these indicators are sufficient, individually or collectively, for consciousness is open. We report what we measure and leave the interpretation to the reader.
