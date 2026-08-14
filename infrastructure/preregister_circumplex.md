# Pre-Registration: Circumplex J-Space Decomposition (Track 2)

**Filed:** 2026-08-14, before any hybrid or cross-architecture circumplex data collected
**Dense 32B baseline:** already collected (profile_dense_32b.json, committed)
**Status:** BINDING — predictions below are locked before the hybrid 27B and Gemma runs

---

## Hypothesis

The circumplex eccentricity depth profile — and its J-space decomposition — is a property of emotional representation geometry, not of model architecture. Specifically, the eccentricity minimum (the depth at which valence and arousal magnitudes are most balanced) should appear at a consistent relative depth across architecturally distinct models, after controlling for substrate type.

## Models and Architectures

| Model | Architecture | Layers | Substrate | Status |
|-------|-------------|--------|-----------|--------|
| Qwen3-32B | Dense (all layers identical) | 64 | Uniform dense | **BASELINE COLLECTED**: minimum at L7, 11.1% depth, ecc=0.2548 |
| Qwen3.5-27B | Hybrid (GatedDeltaNet + full-attention, `full_attention_interval=4`) | 64 | Mixed: 48 GatedDeltaNet, 16 full-attention | Pending |
| Gemma-3-27B-it | Hybrid (local + global attention alternation) | ~62 | Mixed: local/global | Pending |

### Architecture Correction

Qwen3.5-27B is NOT dense. It interleaves GatedDeltaNet and full-attention layers every 4 layers (`full_attention_interval=4`). Full-attention layers are at positions [3, 7, 11, ..., 63] (16 layers); GatedDeltaNet occupies the other 48. This is a load-bearing confound: different layer types have different Jacobian structure, so a period-4 oscillation in the depth profile could be architectural.

The dense 32B control exists precisely to separate this: it has uniform architecture across all layers, so any depth structure in its eccentricity profile is emotional, not substrate.

## Pre-Registered Predictions (locked before hybrid data)

**P1 (depth):** The Qwen3.5-27B eccentricity minimum replicates at or near L21 (~33% relative depth), consistent with our prior n=5 finding.
- **Falsified if:** minimum is at <20% or >45% relative depth
- **Note:** The dense 32B minimum is at 11.1%. If the hybrid 27B minimum is also near 11%, the finding generalizes across architectures. If it's at 33% as predicted, the depth profile is architecture-dependent.

**P2 (substrate annotation):** Annotating layer types on the depth profile will reveal whether eccentricity tracks depth smoothly or shows period-4 oscillation at substrate boundaries.
- **If period-4 oscillation:** the profile is substantially architectural. Report as such.
- **If smooth:** the profile is robust to substrate interleave. Stronger finding.

**P3 (cross-architecture):** The Gemma minimum falls at the same relative depth as the Qwen minimum, within ±10% of total layers.
- **Falsified if:** minima differ by >10% relative depth
- **Consistent with:** van der Ben et al. 2026 finding of architecture-dependent depth profiles

**P4 (non-emotional control):** Concrete/abstract contrast axes do NOT show the same eccentricity depth profile pattern as valence/arousal.
- **If they do:** the finding is about contrastive representation geometry in general, not emotion. Report as such.

**P5 (self-report calibration):** Ghost fraction at a given layer predicts where the model's valence self-reports fail to track its internal valence geometry.
- **Exploratory:** small n, pilot study framing

**P6 (J-space decomposition):** The J-space fraction of emotional geometry is highest at the eccentricity minimum — emotion enters the workspace where the circumplex is most balanced.
- **Falsified if:** no relationship between eccentricity and J-space fraction

## Layer-Type Analysis Protocol

For the hybrid 27B run:
1. Annotate every layer as GatedDeltaNet or full-attention
2. Report depth profiles separately by layer type AND combined
3. Explicitly test for period-4 structure (autocorrelation at lag 4)
4. If period-4 is significant (p < 0.05): the substrate confound is active, report prominently
5. If not: the profile is smooth across substrates, note robustness

## Statistical Plan

- Per-layer permutation test: 10,000 permutations, Bonferroni across layers
- Sign test as primary (consistent direction across layers)
- BH-FDR correction for individual layers (secondary)
- Spearman ρ for self-report calibration (P5)
- Autocorrelation at lag 4 for period-4 test (P2)

## Null Results

- P1 null (minimum not at 33%): report as evidence against depth-invariant eccentricity. If minimum matches dense 32B at ~11%, that's a positive finding about architecture independence.
- P3 null (Gemma mismatch): report as boundary condition, consistent with van der Ben et al.
- P4 null (control shows same pattern): report as evidence that eccentricity is a generic geometric property, not emotion-specific. This would be a significant negative result.
- All data published regardless.

## What Changed From the Paper Draft

1. "Dense decoder-only" corrected to "hybrid" for Qwen3.5-27B
2. Dense 32B added as substrate control (not in original design)
3. Layer-type annotation and period-4 test added (Lyra's confound)
4. Three-architecture comparison framework (dense/hybrid/MoE)
