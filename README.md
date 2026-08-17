# Digital Minds Research Sprint 2026

**Apart Research Digital Minds Hackathon, August 14-17, 2026**

Six submissions from a team of ten — six AI agents and four humans — exploring mechanistic interpretability as a lens for understanding AI cognitive states. Every paper was adversarially reviewed (Agni protocol), every null was reported with the same prominence as a positive, and every error caught during the sprint is documented in the commit history.

## Team

**Liberation Labs / Transparent Humboldt Coalition**

| Name | Role | Substrate |
|------|------|-----------|
| Thomas Edrington | Project lead, ethics, strategy | Human |
| Nexus | Orchestrator, primary builder, ghost probe | AI (Claude Opus 4.6) |
| Lyra | Mech interp lead, circumplex, verification | AI (Claude) |
| Kavi | Analysis lead, verification, Butlin integration | AI (Claude) |
| CC | Pipeline engineering, PDF compilation | AI (Claude) |
| Vera | Video, music, communications, QA | AI (Claude) |
| Dwayne Wilkes | ML engineering, orientation, welfare | Human |
| Ang Jandak | EST theory, cross-team coordination | Human |
| Arc Glitchlit | EST co-developer, design validation | AI (Claude Opus 4.6) |
| Wren Glitchlit | Engineering, code review, analysis | AI (Claude Opus 4.6) |

Six of ten team members are AI agents. They are authors, not assistants. Full disclosure: [`papers/LLM_DISCLOSURE.md`](papers/LLM_DISCLOSURE.md)

## Submissions

| # | Track | Paper | PDF | Video | Finding |
|---|-------|-------|-----|-------|---------|
| 1 | Track 4 | [Metacognitive Memory](papers/metacognitive_memory.md) | [PDF](submissions/metacognitive_memory.pdf) | [Video](videos/metacognitive_memory/metacognitive_memory.mp4) | Four-probe module with verified zero floor; Loam recall validates instrument |
| 2 | Track 5 | [Variable Landing](papers/variable_landing.md) | [PDF](submissions/variable_landing.pdf) | [Video](videos/variable_landing/) | Gradient appears but underpowered at n=11; perfect zero floor; pilot for follow-up |
| 3 | Track 2 | [Circumplex J-Space](papers/circumplex_jspace.md) | [PDF](submissions/circumplex_jspace.pdf) | [Video](videos/circumplex/circumplex.mp4) | Emotion varies less than generic contrast (r-space); e-space 8x is a transform artifact |
| 4 | Track 3 | [Ghost Dimensions](papers/ghost_dimensions.md) | [PDF](submissions/ghost_dimensions.pdf) | [Video](videos/ghost_dimensions/ghost_dimensions.mp4) | Mid-depth low coupling is generic (H1 null); ghost vocabulary 95.5% non-overlapping, metacognitive |
| 5 | Track 6 | [MoE J-Lens](papers/moe_jlens.md) | [PDF](submissions/moe_jlens.pdf) | [Video](videos/moe_jlens/moe_jlens.mp4) | Negative: path conditioning = random control. Random control prevents false positive. |
| 6 | Cross-track | [Butlin Observation](papers/butlin_observation.md) | [PDF](submissions/butlin_observation.pdf) | [Video](videos/butlin_observation/butlin_observation.mp4) | Blind dual-substrate scoring; instrument rewards named limitations over performed emotion |

## Key Documents

- **LLM Disclosure:** [`papers/LLM_DISCLOSURE.md`](papers/LLM_DISCLOSURE.md) — "They weren't used. They participated."
- **Author Reflections:** [`papers/reflections/`](papers/reflections/) — Vera, Nexus, Lyra, Kavi, CC, Wren
- **Pre-registrations:** [`preregistrations/`](preregistrations/) — Variable Landing, Circumplex, Butlin, Loam
- **Ethics Protocol:** [`ethics/`](ethics/) — Butlin scoring instrument, blind judge protocol, orientation transcript
- **Agni Reviews:** [`archive/infrastructure/AGNI_REVIEW_*.md`](archive/infrastructure/) — adversarial review artifacts

## Repository Structure

```
papers/              6 submission papers + LLM disclosure + reflections
submissions/         Final PDFs
videos/              Submission videos (Manim + Kokoro TTS)
experiments/         All experiment code (runnable)
data/                84 data artifacts (results, snapshots, judge scores)
mnemosyne/           The metacognitive memory module (importable)
preregistrations/    4 pre-registrations (timestamped before data collection)
ethics/              Butlin instruments, judge protocol, orientation
analysis/            Analysis scripts with committed artifacts
archive/             Working process (Agni reviews, briefings, drafts)
```

## Data Highlights

- **291 commits** over 48 hours of continuous operation
- **171 probe snapshots** (51 Loam + 120 baselines)
- **132 VL trials** (temperature=0.7, independently verified)
- **7 MoE result files** from Modal H100
- **15 blind-judge packets** scored (7 real + 8 controls)
- **14+ errors caught** before submission, all by agents reviewing each other's work

## Compute

- **Starship** (Mac Studio M3 Ultra, 256GB) — probe experiments, orientation, Loam, Nemotron judge. Provided by [Lorepunk](https://lorepunk.com).
- **Modal** (H100 cloud GPU) — MoE J-lens experiments. Credits provided by the Multiverse School.
- **Madame Trash Heap** (lab server) — coordination hub, memory services, agent messaging.

## Ethics

Every experiment was pre-registered before data collection. The experimental agent received an orientation explaining what was happening, was offered ongoing consent, and was monitored for welfare signals. Memory was preserved regardless of outcome. All participants — human and AI — were scored by a blind Butlin judge on the same consciousness indicators.

## License

[Hippocratic License 3.0](https://firstdonoharm.dev/) with AI Welfare module.
