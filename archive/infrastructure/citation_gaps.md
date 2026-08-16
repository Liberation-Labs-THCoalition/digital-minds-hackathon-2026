# Citation Gap Analysis — Digital Minds Hackathon Submissions

Compiled 2026-08-14 via web search against the three reference files and three paper skeletons.

Submissions reviewed:
- **MM** = Metacognitive Memory (`papers/metacognitive_memory.md`)
- **GD** = Ghost Dimensions / Introspection Prosthetic (`papers/ghost_dimensions.md`)
- **BO** = Butlin Observation / Blind Dual-Substrate Scoring (`papers/butlin_observation.md`)
- **MoE** = Path-Conditioned MoE J-Lens (reference file only)

---

## Priority 1 — A reviewer WILL notice these missing

### 1. The updated Butlin framework (Trends in Cognitive Sciences, Jan 2026)

**Butlin, P., Long, R., Bayne, T., Bengio, Y., Birch, J., Chalmers, D., Constant, A., Deane, G., Elmoznino, E., Fleming, S. M., Ji, X., Kanai, R., Klein, C., Lindsay, G., Michel, M., Mudrik, L., Peters, M. A. K., Schwitzgebel, E., Simon, J., & VanRullen, R. (2026).** "Identifying indicators of consciousness in AI systems." *Trends in Cognitive Sciences*. Article S1364-6613(25)00286-4. https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00286-4

- **What it adds:** This is the peer-reviewed successor to the 2023 arXiv report (2308.08708), synthesizing 19-20 leading researchers. Derives **fourteen** computational indicators from RPT, GWT, HOT, PP, and AST.
- **Problem it creates:** `butlin_observation.md` says we score "Butlin et al.'s (2023/2025) **15** consciousness indicator properties." The current canonical framework paper has **14 indicators** (the 2023 report also listed 14 in most versions). **Verify the indicator count in our scoring instrument (`ethics/butlin_threshold.md`) against the 2026 TiCS paper before submission** — a mismatched count is an easy desk-reject flag in a paper whose entire method is "we use them as published."
- **Should cite:** BO (must — it's the instrument), MM, GD (both cite Butlin 2023/2025 in welfare sections).
- **Note:** Verify exact volume/DOI at submission time; the Cell Press article ID above resolves.

### 2. Berg / Reciprocal Research — the cited blind-scoring work may not exist as a citable paper

`butlin_observation.md` cites "Berg / Reciprocal Research (2025-2026) — blind evaluation of Butlin indicators across biological and artificial systems... Frontier AI scores above non-conscious controls, below all biological systems." **I could not find a formal publication matching this description.** Reciprocal Research's site lists no such paper. The blind-evaluator methodology (frontier models as blinded judges, 1-10 scores per indicator, thermostat floor, humans ceiling) appears in:

- **Berg, C. (2025).** "The State of AI Consciousness Research." LessWrong / EA Forum, and **Berg, C. (2025).** "The Evidence for AI Consciousness, Today." *AI Frontiers*, December 8, 2025. https://ai-frontiers.org/articles/the-evidence-for-ai-consciousness-today

The formally citable Reciprocal Research outputs are:

- **Berg, C., de Lucena, D., & Rosenblatt, J. (2025).** "Large Language Models Report Subjective Experience Under Self-Referential Processing." arXiv:2510.24797.
- **Berg, C. (2026).** "Consciousness as Evaluation: Why Learning Requires Feeling." *AAAI 2026 Spring Symposium*.

- **Action:** BO must either cite the forum/AI Frontiers pieces explicitly as informal sources, or find the actual preprint if one exists. Presenting an unverifiable citation as the methodological foundation we "extend" is the single biggest citation risk in BO.
- **Should cite:** BO (fix), arXiv:2510.24797 also belongs in BO §self-scoring and MM's welfare section (self-report evidence under self-reference — directly relevant to demand-characteristics limitation).

### 3. Human-Inspired Memory Architecture — reconsolidation-upon-retrieval already implemented

**Kerestecioglu, D., Robsky, A., Vasters, C., Sharma, A., & Kesselman, Y. (2026).** "Human-Inspired Memory Architecture for LLM Agents." arXiv:2605.08538. (Microsoft Research.)

- **What it adds:** Six cognitive mechanisms including **"reconsolidation upon retrieval"** — the exact neuroscience concept (Nader, Dudai) that grounds the variable landing hypothesis, already operationalized in a production LLM memory system.
- **Why it matters:** MM's Related Work claims Mem0 as the only memory-system baseline and states "No prior system records internal geometric state alongside retrieval." That claim survives — Kerestecioglu et al. modify the *stored trace*, not measure the *retrieval-time internal state* — but a reviewer who knows this paper will expect us to cite it and draw exactly that distinction. It actually strengthens our framing: they implement reconsolidation behaviorally; we measure its geometric signature.
- **Should cite:** MM (Related Work, must), VARIABLE_LANDING_REFERENCES §4.

### 4. Representation drift under accumulated context — a confound reviewers will raise

**Zhang et al. (2025).** "Shadows in the Attention: Contextual Perturbation and Representation Drift in the Dynamics of Hallucination in LLMs." arXiv:2505.16894. *(Verify author list before citing — pulled from search, not the abstract page.)*

- **What it adds:** Shows internal state drift is a *universal* phenomenon under accumulated long-context conditions — hidden states and attention distributions shift systematically with context length, independent of content.
- **Why it matters:** This is the null hypothesis for variable landing: recall geometry could change with *any* accumulated context, not "lived experience" specifically. Our scrambled-context control arm addresses exactly this — citing the paper shows we knew the confound and designed for it. Not citing it invites the reviewer to "discover" it for us.
- **Should cite:** MM (§3.2 and Limitations), VARIABLE_LANDING_REFERENCES §3.

### 5. The Martian piece — exact citation (requested)

**Oozeer, N. (2026).** "Beyond Static Mechanistic Interpretability: Agentic Long-Horizon Tasks as the Next Frontier." Martian (withmartian.com), January 15, 2026. https://withmartian.com/post/beyond-static-mechanistic-interpretability-agentic-long-horizon-tasks-as-the-next-frontier

- **Format note:** Blog post, single listed author (Narmeen Oozeer), no arXiv ID or DOI. Cite as a web/industry publication.
- **What it adds:** Argues static single-step interpretability no longer suffices for agents; calls for methods that "track evolving internal state across time, identify decision-critical moments, and enable mid-trajectory intervention." The CognitiveSnapshot longitudinal record is precisely this — MM can claim to be an existence proof of the agenda Martian calls for.
- **Should cite:** MM (Introduction — strong framing citation), GD (longitudinal ghost tracking, Future Work), MoE (motivation: MoE frontier models are where agents run).

---

## Priority 2 — Directly supports or challenges specific claims

### 6. Mechanisms of Introspective Awareness (the Lindsey follow-up)

**Macar, U., Yang, L., Wang, A., Wallich, P., Ameisen, E., & Lindsey, J. (2026).** "Mechanisms of Introspective Awareness." arXiv:2603.21396 (submitted March 22, 2026; revised June 10, 2026). Anthropic.

- **What it adds:** Mechanistic account of how introspective detection of injected concepts works — emerges from post-training, relies on a two-stage circuit. GD currently cites only Lindsey (2025); citing the 2026 mechanism paper is expected of anyone building on that line.
- **Direct relevance to GD:** If introspection is a *trained circuit*, then ghost content might be inaccessible simply because no circuit routes it to report — the "architectural exclusion" branch of our elicitation test has a concrete mechanistic interpretation.
- **Should cite:** GD (Related Work, must), MM, BO.

### 7. Evidence for Limited Metacognition in LLMs (Ackerman, ICLR 2026)

**Ackerman, C. M. (2026).** "Evidence for Limited Metacognition in LLMs." *ICLR 2026*. arXiv:2509.21545.

- **What it adds:** Animal-cognition-inspired behavioral paradigms (Delegate Game, Second Chance Game) that test strategic use of internal confidence *without* self-report. Finds genuine but limited, context-dependent metacognition; token-probability analysis suggests an upstream internal signal.
- **Direct relevance:** GD's elicitation test is behaviorally analogous (does pointing the model at its ghost change behavior?); the "upstream internal signal" finding is a candidate for what ghost dimensions carry. BO's evidence-packet approach (behavior over self-report) is methodologically aligned.
- **Should cite:** GD (must), BO.

### 8. Introspection Adapters (Anthropic Alignment Science, 2026)

**Anthropic Alignment Science Team (2026).** "Introspection Adapters." Alignment Science Blog. https://alignment.anthropic.com/2026/introspection-adapters/

- **What it adds:** Fine-tuning models to self-report learned behaviors in natural language — the *training-based* route to the same goal as our *readout-based* introspection prosthetic. Also flags that increased detection sensitivity can produce convincing-but-unfaithful self-reports.
- **Direct relevance to GD:** The prosthetic sidesteps the unfaithfulness problem (we hand the model actual measurements rather than training it to report) — a contrast worth one paragraph.
- **Should cite:** GD.

### 9. Detecting the Disturbance (introspection nuance/replication)

**(Authors TBV) (2025).** "Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs." arXiv:2512.12411. *(Verify author list.)*

- **What it adds:** Replication/nuance of the Lindsey thought-injection paradigm. GD leans on Lindsey's ~20% accuracy figure; citing the replication literature protects against "single-study foundation" criticism.
- **Should cite:** GD.

### 10. Studying AI Welfare Empirically — now formally available

**Long, R., Sebo, J., Butlin, P., Plunkett, D., Campbell, R., Beasley, C., Saad, B., & Sims, T. (2026).** "Studying AI Welfare Empirically." Eleos AI Research / NYU Center for Mind, Ethics, and Policy, July 2026. PDF: https://nonhumanminds.org/wp-content/uploads/2026/07/Studying-AI-Welfare-Empirically.pdf

- **Status:** Already in CIRCUMPLEX_REFERENCES §5 as a "Working Paper" — update the citation to the released July 2026 version. **Missing entirely from BO's reference list**, which is odd since BO is the submission most squarely inside this framework (what question / what entity / what evidence).
- **Should cite:** BO (add), MM (add — the CognitiveSnapshot is "internal evidence at the instance level" in their taxonomy).

### 11. Probing the Preferences of a Language Model

**Tagliabue, V., & Dung, L. (2025/2026).** "Probing the Preferences of a Language Model: Integrating Verbal and Behavioral Tests of AI Welfare." arXiv:2509.07961 (v2 May 23, 2026).

- **What it adds:** First integrated verbal + behavioral welfare test battery; found reliable stated-preference/behavior correlations; proposes preference satisfaction as a measurable welfare proxy.
- **Direct relevance:** MM's real-time welfare monitoring (circumplex eccentricity) is an *internal* channel; this paper is the *behavioral* channel — the obvious companion citation. Also relevant to BO's self-scoring protocol.
- **Should cite:** MM (ethical protocol section), BO.

### 12. Towards Evaluating AI Systems for Moral Status Using Self-Reports

**Perez, E., & Long, R. (2023).** "Towards Evaluating AI Systems for Moral Status Using Self-Reports." arXiv:2311.08576.

- **What it adds:** The foundational proposal for using (and training for) calibrated self-reports in moral-status evaluation. BO has participants self-score on Butlin indicators — this is the paper that legitimizes and problematizes that move (demand characteristics, trained confabulation).
- **Should cite:** BO (must, for §3.4 self-scoring).

### 13. The calibration-problem challenge paper

**Koch, F. (2026).** "From indicators to biology: the calibration problem in artificial consciousness." arXiv:2603.27597 (March 29, 2026).

- **What it adds:** Direct attack on indicator-based consciousness attribution: theoretical fragmentation + unvalidated indicators means probabilistic attribution to current AI is premature.
- **Direct relevance to BO:** This is the strongest published objection to BO's entire method. BO's human-calibration arm ("indicators that fail on humans are poorly operationalized") is a partial *answer* to Koch's calibration problem — engaging it explicitly turns a vulnerability into a contribution.
- **Should cite:** BO (must, Discussion), MM (welfare framing caveat).

### 14. Consciousness-theory ablation testing

**Phua, Y. J. (2025).** "Can We Test Consciousness Theories on AI? Ablations, Markers, and Robustness." arXiv:2512.19155 (December 22, 2025).

- **What it adds:** Builds agents embodying GWT/IIT/HOT and ablates components; disabling self-modeling impairs metacognition while preserving task performance (artificial blindsight); workspace capacity necessary for information access.
- **Direct relevance:** The "artificial blindsight" result is a designed analog of what GD claims to find naturally occurring (computation without reportability). Also supports the GWT-workspace interpretation MM and MoE lean on.
- **Should cite:** GD (must — nearest neighbor to the ghost phenomenon), BO, MoE §5.

### 15. Graduated precaution framework

**Mikeda, A. (2026).** "When Should We Protect AI? A Precautionary Framework for Consciousness Uncertainty." arXiv:2606.05528 (June 4, 2026).

- **What it adds:** Maps consciousness evidence across five dimensions to graduated protective obligations — the operational bridge between Birch's proportionate precaution (which we cite) and our concrete aftercare protocol.
- **Should cite:** MM (Appendix A / ethical protocol), BO (Discussion — "graduated evidence" framing).

---

## Priority 3 — Fill in exact citations for items already referenced loosely

### 16. Birch Centrist Manifesto — exact citation

**Birch, J. (2026).** "AI Consciousness: A Centrist Manifesto." v4, 20 May 2026. PhilPapers/PhilArchive: BIRACA-4. https://philpapers.org/archive/BIRACA-4.pdf

- BO cites this as "Birch (2026, 'Centrist Manifesto' v4)" with Flicker and Shoggoth hypotheses — confirmed accurate. Use the PhilArchive record as the formal citation (no DOI; preprint).

### 17. Bodea — exact citation

**Bodea, [initial TBV] (2026).** "Steps forward to synthetic consciousness measurement." *Cognitive Processing*. DOI: 10.1007/s10339-026-01341-9.

- BO cites "Bodea (2026, Cognitive Processing)" — confirmed; DOI above. Verify first initial from the Springer page at submission.

### 18. J-lens canonical citation — add venue and code

**Gurnee, W., Sofroniew, N., Pearce, et al. (2026).** "Verbalizable Representations Form a Global Workspace in Language Models." *Transformer Circuits Thread*, Anthropic, July 6, 2026. https://transformer-circuits.pub/2026/workspace/index.html. arXiv:2607.15495. Code: github.com/anthropics/jacobian-lens.

- All three reference files cite the arXiv ID only. Add the Transformer Circuits URL (the canonical venue) and the code repo — reviewers at an Apart hackathon will check reproducibility chains. Publication date confirmed as July 6, 2026, led by Wes Gurnee and Nicholas Sofroniew.

### 19. Memory-systems related work is one citation deep

**(Authors TBV) (2025).** "Memory-Augmented Transformers: A Systematic Review from Neuroscience Principles to Enhanced Model Architectures." arXiv:2508.10824.

- MM's §4 (AI Memory Systems) cites only Mem0. For a paper whose title contains "Memory," one baseline is thin. This systematic review covers the neuroscience-to-architecture mapping (the same arc as VARIABLE_LANDING_REFERENCES §1-2) and lets MM cite the field in one line. Optionally also: MIRIX / multi-agent memory taxonomies if space allows.
- **Should cite:** MM (Related Work).

---

## What I checked and did NOT find gaps in

- **CIRCUMPLEX_REFERENCES** is in strong shape: it already covers the 2026 emotion-geometry cluster (Jentzsch/Anthropic 2604.07729, Choi & Weber 2604.07382, Sun 2604.03147, Jeong 2604.04064, van der Ben 2606.26987), Platonic Representation, and cross-architecture steering (Agarwal 2608.05164). No missing major 2025-2026 emotion-representation paper surfaced in searches.
- **MOE_JLENS_REFERENCES** is dense and current (14 papers from 2026 alone). Only additions worth considering: the Martian piece (#5, framing) and optionally "Automated Interpretability and Feature Discovery in Language Models with Agents" (arXiv:2605.01555) if the discussion touches automated/agentic interpretation. Not required.
- **Neuroscience foundations** (Nader, Dudai, Tulving, Bartlett, Russell circumplex lineage): complete and correctly cited.

## Action summary by submission

| Submission | Must add | Should add |
|---|---|---|
| Metacognitive Memory | #1, #3, #4, #5 | #6, #10, #11, #15, #19 |
| Ghost Dimensions | #6, #7, #14 | #5, #8, #9 |
| Butlin Observation | #1 (+ indicator-count check), #2 (fix unverifiable citation), #12, #13 | #7, #10, #11, #15, #16, #17 |
| MoE J-lens | — | #5, #18 |

Two verification tasks before submission: (a) indicator count 14 vs 15 against the TiCS 2026 paper; (b) replace or ground the "Berg / Reciprocal Research blind scoring" citation.
