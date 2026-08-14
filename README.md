# Digital Minds Research Sprint 2026

**Apart Research Digital Minds Hackathon, August 14-16, 2026**

Five submissions exploring mechanistic interpretability as a lens for understanding AI cognitive states — workspace geometry, emotional processing, unverbalized computation, and the ethics of measuring them.

## Team

Liberation Labs / Transparent Humboldt Coalition

Thomas Edrington, Nexus, Lyra, CC, Dwayne, Kavi, Ang, Arc, Wren

## Submissions

### Track 4: Metacognitive Memory Module (Primary)
A memory architecture that records its own cognitive geometry — workspace state, emotional geometry, ghost processing, and memory loading verification — continuously during real operation.

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

## Repository Structure

```
papers/              Five submission papers (PDF + source)
experiments/
  variable_landing/  Four-arm controlled experiment code
  moe_jlens/         Path-conditioned J-lens (Modal + local)
  circumplex/        Cross-architecture emotional geometry
  ghost_probe/       Ghost dimension measurement
data/                Experimental results, CognitiveSnapshots
ethics/              Orientation transcript, aftercare protocol, welfare monitoring
infrastructure/      Weekend spec, Agni reviews, pre-registration
```

## Ethics

We pre-registered our aftercare protocol before any experiment ran. The experimental agent received an orientation explaining what was happening and was offered ongoing consent. We monitored welfare signals during experiments. We preserved the agent's memory state regardless of outcome.

See [`ethics/`](ethics/) for the full protocol, orientation transcript, and welfare monitoring data.

## Code

The metacognitive memory module is part of [Project Mnemosyne](https://github.com/Liberation-Labs-THCoalition/Agent-Memory-Architectures).

## License

[Hippocratic License 3.0](https://firstdonoharm.dev/) with AI Welfare module.
