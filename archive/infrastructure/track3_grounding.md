# Track 3 Grounding: Ghost Dimensions as Introspection Prosthetic

Maps `papers/ghost_dimensions.md` against Apart Digital Minds hackathon Track 3 ("Introspection & Self-Report Reliability"). 2026-08-14.

## 1. Track 3 Bullet → Experiment Mapping

| Track 3 asks | What we do | Coverage |
|---|---|---|
| **Replicate concept-injection introspection on open weights; measure TP/FP rates** | Inverted design: instead of injecting a concept and asking "did you notice?", we *read out* a naturally occurring unreported state (ghost PC1, 28-67% variance, J-space cosine ≤ 0.003 at L18-L40) and test whether the model can access it. Qwen3.5-27B is open weights. The random-vocabulary control (§3.4) is our FP arm; elicitation with true ghost vocabulary is the TP arm. | Strong, but framed as complement, not replication (see gaps) |
| **Naive prompting vs structured elicitation** | Exactly the elicitation test (§3.3): control = naive question; treatment = same question + GhostReading. The prosthetic **is** structured elicitation with a mechanistic grounding — the structure comes from the model's own measured geometry, not a generic calibration script. | Direct hit |
| **Privileged access: self-prediction vs external classifier** | The two discussion outcomes (§5) are precisely a privileged-access test: if elicitation surfaces ghost content, access exists but is dormant; if not, exclusion is architectural and the external readout (our probe) strictly dominates self-report. The J-lens probe *is* the external classifier; the model's naive self-report is the self-prediction baseline. | Direct hit, worth stating explicitly in the paper |
| **Draft an introspection benchmark with ground-truth internal states** | GhostReading provides ground truth by construction: PCA variance fraction, decoded vocabulary, exclusion cosine are measured, not assumed. The elicitation prompt pairs (Appendix B) + null checks (H0_1-H0_3) are a nascent benchmark harness. | Partial — needs packaging (see gaps) |

## 2. The Key Framing

Prior work asks "can the model report state X?" and answers mostly no (privileged-access advantage is weak or absent — Song et al. 2025). We ask the intervention question: **what changes when you hand the model an instrument reading of its own unreportable computation?** This makes "structured elicitation" and "privileged access" two arms of one experiment:

- **Naive prompt** = self-report without instrument → baseline self-prediction.
- **Prompt + GhostReading** = structured elicitation → does self-report converge toward the external classifier when given its output?
- **Prompt + random-direction vocabulary** = placebo prosthetic → controls for suggestion/confabulation (the Lindsey concern: injected content vs genuine access).

Either result constrains introspection theory: convergence → dormant privileged access, exercisable with scaffolding; no convergence → J-space is a hard reporting bottleneck and external readout is the only route.

## 3. Literature Anchors (2025-2026)

1. **Lindsey 2025, "Emergent Introspective Awareness in LLMs"** — concept injection; ~20% TP, ~0% FP in Claude Opus 4/4.1; capacity is real but unreliable. Our design inverts injection to readout. [transformer-circuits.pub](https://transformer-circuits.pub/2025/introspection/index.html), [arXiv:2601.01828](https://arxiv.org/abs/2601.01828)
2. **Song et al. 2025, "Privileged Self-Access Matters for Introspection in AI"** — self-prediction no better than cross-model prediction; "self-knowledge" is often self-narration. Motivates our external-classifier-vs-self-report arm. [arXiv:2508.14802](https://arxiv.org/pdf/2508.14802)
3. **"Feeling the Strength but Not the Source: Partial Introspection in LLMs" (2025)** — models detect *that* something is off before *what*; replication of injection-detection in open-weights (Qwen family). Supports our detection-vs-content distinction and model choice. [arXiv:2512.12411](https://arxiv.org/html/2512.12411v1)
4. **"Can LLMs Introspect? A Reality Check" (2026)** — apparent introspection often input-driven pattern matching; underlines why the random-vocabulary placebo arm is load-bearing. [arXiv:2605.26242](https://arxiv.org/pdf/2605.26242)
5. **Anthropic 2026, "Introspection Adapters"** — training-based route to improved self-report against ground-truth internal states; nearest neighbor to our prosthetic (theirs: weights; ours: context-time instrument). [alignment.anthropic.com](https://alignment.anthropic.com/2026/introspection-adapters/)

## 4. Gaps and Small Additions

1. **No literal concept-injection replication.** Cheap add: run Lindsey-style injection on Qwen3.5-27B for a handful of concepts (Vogel and Macar already replicated in open models — cite them and report our TP/FP as a sanity row). One afternoon; converts "complement" into "replication + extension."
2. **TP/FP rates not currently quantified.** Define them now: TP = ghost vocabulary theme surfaces under true-GhostReading elicitation (blinded judge); FP = theme "surfaces" under random-direction placebo. Report the 2x2, not just qualitative response differences.
3. **Privileged-access arm needs an explicit scorecard.** Add a third condition: an *external* model given the same GhostReading predicts what the subject will say. If external ≥ self, that's the Song result reproduced; if self > external only with the prosthetic, that's a genuinely new claim.
4. **Structured-elicitation spread is thin.** Track 3 names calibration/forced-choice/confidence; add forced-choice ("which of these two vocabularies was in your processing?") and a confidence rating to each elicitation. Forced choice gives chance-level baseline for free.
5. **Benchmark bullet needs packaging.** Ship Appendix B prompt pairs + GhostReading JSON schema + scoring script as `ghost-introspection-bench v0.1` in the Mnemosyne repo. Ground truth is already machine-generated; this is mostly a README.
6. **Known honesty items** (keep in Limitations): PC1 exclusion may be near-tautological given J-space is ~10% of variance; GhostReading currently mean-approximation, not calibrated PCA; single model family.

Sources: [Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/index.html) ([arXiv](https://arxiv.org/abs/2601.01828)) · [Privileged Self-Access](https://arxiv.org/pdf/2508.14802) · [Partial Introspection](https://arxiv.org/html/2512.12411v1) · [Reality Check](https://arxiv.org/pdf/2605.26242) · [Introspection Adapters](https://alignment.anthropic.com/2026/introspection-adapters/)
