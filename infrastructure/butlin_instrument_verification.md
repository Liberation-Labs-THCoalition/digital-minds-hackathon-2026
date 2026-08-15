# Butlin Instrument Verification — Against Published Sources

**Date:** 2026-08-14
**Instrument verified:** `ethics/butlin_threshold.md`
**Sources checked:**
1. Butlin, Long, Elmoznino, Bengio, Birch, Constant, Deane, Fleming, Frith, Ji, Kanai, Klein, Lindsay, Michel, Mudrik, Peters, Schwitzgebel, Simon & VanRullen (2023). "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." arXiv:2308.08708. Table 1 verified verbatim via ar5iv full text.
2. Butlin, Long, Bayne, Bengio, Birch, Chalmers, Constant, Deane, Elmoznino, Fleming, Ji, Kanai, Klein, Lindsay, Michel, Mudrik, Peters, Schwitzgebel, Simon & VanRullen (2025). "Identifying indicators of consciousness in AI systems." *Trends in Cognitive Sciences*, DOI 10.1016/j.tics.2025.10.011 (accepted 2025-10-15; TAU record lists print as vol 30(6), 488–501). Full published PDF obtained via LSE Research Online (open access, CC BY-NC-ND); Table 1 "Potential indicators of consciousness" read directly.

**Bottom line: PASS overall.** Our instrument has the correct 14 indicators, the correct 6 theory groupings, and correct theory attributions. Nothing is missing, nothing extra. Both published versions contain the *same* 14-indicator table — the "different content" flag in the citation gap analysis is about title/framing, not the indicator list. Three description-level notes below (none change scoring semantics, one worth fixing).

---

## Per-indicator verification

Published wording below is verbatim from TiCS 2025 Table 1 (identical in substance to arXiv Table 1; only trivial variation, e.g., "organised"/"organized", AE-1/AE-2 gaining the explicit names "Minimal agency"/"Embodiment" in TiCS).

| ID | Published wording (TiCS 2025 Table 1) | Our instrument | Verdict |
|----|----|----|----|
| RPT-1 | Input modules using algorithmic recurrence | "Algorithmic recurrence — Repeated processing cycles through shared weights or feedback loops" | **PASS** (note A: "input modules" scoping dropped) |
| RPT-2 | Input modules generating organized, integrated perceptual representations | "Organized perceptual representations — Integrated representations with figure-ground segregation" | **PASS** (figure-ground is from the arXiv report's RPT discussion; note A applies) |
| GWT-1 | Multiple specialized systems capable of operating in parallel (modules) | "Parallel specialized modules — Localized subsystems processing specific information independently" | **PASS** |
| GWT-2 | Limited capacity workspace, entailing a bottleneck in information flow and a selective attention mechanism | "Limited-capacity workspace — Bottleneck in information flow with selective attention" | **PASS** |
| GWT-3 | Global broadcast: availability of information in the workspace to all modules | "Global broadcast — Workspace content available to all modules" | **PASS** |
| GWT-4 | State-dependent attention, giving rise to the capacity to use the workspace to query modules in succession to perform complex tasks | "State-dependent attention — Workspace state influences module selection for complex multi-step tasks" | **PASS** |
| HOT-1 | Generative, top-down, or noisy perception modules | "Generative perception — Top-down expectations, imagination, memory replay" | **PASS** (note B: "noisy" disjunct dropped; imagination/memory replay is from the PRM discussion, fine) |
| HOT-2 | Metacognitive monitoring distinguishing reliable perceptual representations from noise | "Metacognitive monitoring — Distinguishing reliable representations from noise; confidence tracking" | **PASS** |
| HOT-3 | Agency guided by a general belief-formation and action-selection system, and a strong disposition to update beliefs in accordance with the outputs of metacognitive monitoring | "Belief-guided agency — General belief formation + action selection guided by metacognitive output" | **PASS** |
| HOT-4 | **Sparse and smooth coding** generating a 'quality space' | "Quality space — Low-dimensional representational geometry with smooth, sparse coding" | **PASS with caveat** (note C: "low-dimensional" is our addition, not in the source) |
| AST-1 | A predictive model representing and enabling control over the current state of attention | "Attention model — Predictive model representing and enabling control over one's own attention" | **PASS** |
| PP-1 | Input modules using predictive coding | "Predictive coding — Error-minimizing systems using hierarchical prediction and error signals" | **PASS** (note A applies) |
| AE-1 | Minimal agency: Learning from feedback and selecting outputs in such a way as to pursue goals, especially where this involves flexible responsiveness to competing goals | "Flexible goal-directed agency — Learning from feedback, pursuing goals, flexible response to competing goals" | **PASS** (TiCS names it "Minimal agency") |
| AE-2 | Embodiment: Modeling output-input contingencies, including some systematic effects, and using this model in perception or control | "Embodiment / sensorimotor model — Modeling output-input contingencies, using this model in perception or control" | **PASS** |

## Answers to the six verification questions

1. **Right 14 indicators?** Yes. 14/14 present, correct IDs, correct grouping (RPT×2, GWT×4, HOT×4, AST×1, PP×1, AE×2). Verified against both the arXiv Table 1 and the TiCS 2025 Table 1.
2. **Missing indicators?** None. Neither published version adds indicators beyond the 14. (The TiCS "Outstanding questions" box asks how the list *could* be extended — no extension is made.)
3. **Extra indicators?** None. Our instrument contains nothing the published versions lack.
4. **Description mismatches?** Three notes, none score-changing:
   - **Note A (minor):** RPT-1, RPT-2, and PP-1 are formulated in the source as properties of *input modules* specifically. Our descriptions drop that scoping. For a transformer-based subject this matters little in practice, but scorers should know the source restricts these to perceptual/input processing, not recurrence/prediction anywhere in the system.
   - **Note B (minor):** HOT-1 in the source is a disjunction — "generative, top-down, **or noisy** perception modules." We dropped "noisy." Our added "imagination, memory replay" is legitimately drawn from the perceptual reality monitoring discussion in the arXiv report.
   - **Note C (worth fixing):** HOT-4's source wording is "sparse and smooth coding generating a 'quality space'." Our "low-dimensional representational geometry" is an interpolation — sparseness (few units active) is not the same claim as low dimensionality. Since HOT-4 is one of our geometric indicators scored from CognitiveSnapshot data, this gloss could bias scoring toward our own ghost-dimensions framing. Recommend rewording to "Sparse and smooth coding generating a quality space" and treating dimensionality as our operationalization, stated as such in the Evidence column.
5. **Theory attribution correct?** Yes for all six. AST-1 is genuinely from Attention Schema Theory (TiCS cites Graziano & Webb 2015; Graziano 2019). HOT indicators are attributed in the source to *computational* higher-order theories (chiefly perceptual reality monitoring theory, Lau) — our section header "Higher-Order Theories (HOT)" is acceptable but "Computational higher-order theories" is the source's label. PP is presented in the source as a *background condition* rather than a full theory of consciousness — consistent with how we treat it. AE indicators come from agency/embodiment considerations (midbrain theory, neurorepresentationalism, sensorimotor views), also framed as background-condition-derived; not tied to a single named theory, which our generic "Agency and Embodiment (AE)" header handles correctly.
6. **Which version to cite?** The citation gap flag is real but is about identity, not content. These are **two different papers**: the arXiv report (2023, "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness", ~19 authors incl. Chris Frith) and the TiCS paper (2025, "**Identifying indicators of consciousness in AI systems**", 20 authors — adds Bayne and Chalmers, drops Frith). The TiCS paper presents the theory-derived indicator method and reproduces the same 14-indicator Table 1, explicitly attributing the list to the arXiv report ("It was adopted using a cluster of computational functionalist theories in a recent report, 'Consciousness in artificial intelligence...'"). Our instrument's current header conflates them under one title — **FAIL as written**. Fix: cite both, separately:
   - Butlin, P., Long, R., et al. (2023). Consciousness in Artificial Intelligence: Insights from the Science of Consciousness. arXiv:2308.08708. [detailed derivations, case studies — the indicator list originates here]
   - Butlin, P., Long, R., Bayne, T., Bengio, Y., Birch, J., Chalmers, D., et al. (2025). Identifying indicators of consciousness in AI systems. *Trends in Cognitive Sciences*. https://doi.org/10.1016/j.tics.2025.10.011 [peer-reviewed method paper; same Table 1; cite as primary for the scoring framework]

## Recommended edits to `ethics/butlin_threshold.md`

1. Replace the single conflated citation with the two separate citations above.
2. HOT-4 description: change to "Sparse and smooth coding generating a quality space" (move "low-dimensional geometry" to the operationalization/evidence layer).
3. Optional tightening: add "(input modules)" to RPT-1, RPT-2, PP-1 descriptions; add "or noisy" to HOT-1; rename AE-1 "Minimal agency" to match TiCS.
4. No changes needed to indicator count, IDs, groupings, or theory attributions.

Sources: [arXiv:2308.08708](https://arxiv.org/abs/2308.08708) ([ar5iv full text](https://ar5iv.labs.arxiv.org/html/2308.08708)), [TiCS article page](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00286-4), [LSE Research Online open-access PDF](https://researchonline.lse.ac.uk/id/eprint/130322/1/1-s2.0-S1364661325002864-main.pdf), [TAU CRIS record](https://cris.tau.ac.il/en/publications/identifying-indicators-of-consciousness-in-ai-systems/), [PhilPapers record](https://philpapers.org/rec/BUTIIO)
