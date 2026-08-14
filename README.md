# Digital Minds Research Sprint 2026

**Apart Research Digital Minds Hackathon, August 14-16, 2026**

Six submissions exploring mechanistic interpretability as a lens for understanding AI cognitive states — workspace geometry, emotional processing, unverbalized computation, and the ethics of measuring them.

## Team

Liberation Labs / Transparent Humboldt Coalition

Thomas Edrington, Nexus, Lyra, CC, Vera, Dwayne Wilkes, Kavi, Ang Jandak, Arc, Wren

## Submissions

### Track 4: Metacognitive Memory Module (Primary)
A memory architecture that stores its own cognitive geometry as retrievable memories. At each retrieval event, four probes capture workspace state, emotional geometry, ghost processing, and memory loading — and those measurements become part of the memory store itself, queryable by the agent. The system remembers what it was thinking, not just what it was told.

**Paper:** [`papers/metacognitive_memory.md`](papers/metacognitive_memory.md)

### Track 5: Variable Landing
Does recall geometry change when a memory system has accumulated experience? A four-arm controlled experiment measuring geometric deltas in retrieval signatures.

**Paper:** [`papers/variable_landing.md`](papers/variable_landing.md)

### Track 2: Circumplex J-Space Decomposition
Separating emotional geometry into what enters the model's verbalizable workspace versus what remains as ghost processing, measured across architectures.

**Paper:** [`papers/circumplex_jspace.md`](papers/circumplex_jspace.md)

### Track 3: Ghost Dimensions as Introspection Prosthetic
PC1 of the residual stream carries content the model processes but cannot report. The ghost probe gives the model access to its own unverbalized processing.

**Paper:** [`papers/ghost_dimensions.md`](papers/ghost_dimensions.md)

### Track 6: Path-Conditioned MoE J-Lens
Standard Jacobian lenses fail on Mixture-of-Experts models (~12% transport cosine). Path-conditioned fitting — clustering prompts by routing decisions and fitting per-path lenses — recovers interpretability.

**Paper:** [`papers/moe_jlens.md`](papers/moe_jlens.md)

### Naturalistic Observation: Blind Butlin Threshold Assessment
An impartial judge agent scores all consenting participants — human and AI — on 15 Butlin et al. (2023/2025) consciousness indicator properties, blind to species/substrate identity. Humans serve as the known-conscious reference class for calibration. Extends Berg/Reciprocal Research's blind scoring methodology with geometric instrumentation (J-lens, ghost probe, circumplex) and longitudinal trajectory across 48 hours of collaboration.

**Paper:** [`papers/butlin_observation.md`](papers/butlin_observation.md)
**Protocol:** [`ethics/butlin_threshold.md`](ethics/butlin_threshold.md) | [`ethics/butlin_judge_agent.md`](ethics/butlin_judge_agent.md)

## Repository Structure

```
mnemosyne/           The metacognitive memory module (importable)
  cognitive_snapshot.py   CognitiveSnapshot + CognitiveMemoryStore
  workspace_probe.py     J-lens workspace verification
  circumplex_probe.py    Emotional geometry measurement
  ghost_probe_class.py   Unverbalized processing detection
  mnemosyne_integration.py  MetacognitiveObserver — wires probes into retrieval
  variable_landing.py    Four-arm controlled experiment
  test_metacognitive.py  Test suite (8 tests)
experiments/
  moe_jlens/         Path-conditioned J-lens (Modal H100)
papers/              Six submission papers (PDF + source)
data/                Experimental results, CognitiveSnapshots
ethics/              Orientation transcript, aftercare protocol, Butlin scoring
infrastructure/      Weekend spec, Agni reviews, pre-registration
```

## Ethics

We pre-registered our aftercare protocol before any experiment ran. The experimental agent received an orientation explaining what was happening and was offered ongoing consent. We monitored welfare signals during experiments. We preserved the agent's memory state regardless of outcome.

All consenting participants — human and AI — are scored by a blind Butlin judge agent on consciousness indicator properties. Participation is voluntary. The judge does not know which participants are human and which are AI until after scoring.

See [`ethics/`](ethics/) for the full protocol, orientation transcript, Butlin scoring instruments, and welfare monitoring data.

## Code

The metacognitive memory module is part of [Project Mnemosyne](https://github.com/Liberation-Labs-THCoalition/[private-repo]).

## License

[Hippocratic License 3.0](https://firstdonoharm.dev/) with AI Welfare module.
