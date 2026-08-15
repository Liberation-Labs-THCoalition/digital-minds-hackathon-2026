# Repository Index

One-line map of every file in the repo, organized by directory. For judges who want to find things quickly. Generated 2026-08-14.

## Root

| File | Description |
|---|---|
| `README.md` | Project overview: team, six submissions with paper links, repo structure, ethics summary |
| `LICENSE.md` | Hippocratic License 3.0 with AI Welfare module |
| `.gitignore` | Ignore rules (bytecode, model weights, secrets, logs, temp/editor files) with explicit keeps for results and docs |

## papers/ — the six submissions

| File | Description |
|---|---|
| `metacognitive_memory.md` | **Track 4 (primary):** memory architecture that stores its own cognitive geometry as retrievable memories |
| `variable_landing.md` | **Track 5:** four-arm experiment — does accumulated memory change recall geometry? |
| `circumplex_jspace.md` | **Track 2:** decomposing emotional geometry into workspace-accessible vs. ghost components across architectures |
| `ghost_dimensions.md` | **Track 3:** giving the model probe access to its own unverbalized (ghost) processing |
| `moe_jlens.md` | **Track 6:** path-conditioned Jacobian lenses recover interpretability on MoE models |
| `butlin_observation.md` | **Naturalistic observation:** blind Butlin consciousness-indicator scoring of all consenting participants |
| `CIRCUMPLEX_REFERENCES.md` | Literature review backing the circumplex J-space paper |
| `MOE_JLENS_REFERENCES.md` | Literature review backing the MoE J-lens paper |
| `VARIABLE_LANDING_REFERENCES.md` | Literature review backing the variable landing paper |

## mnemosyne/ — the metacognitive memory module (importable package)

| File | Description |
|---|---|
| `__init__.py` | Package init; exposes the four probes and the observer |
| `cognitive_snapshot.py` | CognitiveSnapshot + CognitiveMemoryStore — the core data structure recording workspace, emotion, ghost, and retrieval state |
| `workspace_probe.py` | J-lens workspace probe — measures whether retrieved memories reach J-space (v3, pinned token ids) |
| `circumplex_probe.py` | CircumplexProbe — valence/arousal geometry, eccentricity, J-space decomposition |
| `ghost_probe_class.py` | GhostProbe — logit-lens vs. J-lens reading of residual-stream principal components |
| `mnemosyne_integration.py` | MetacognitiveObserver — wires all probes into the SIRA retrieval pipeline |
| `test_metacognitive.py` | Integration test suite for the full metacognitive stack (27B Opus distill + Neuronpedia lens) |
| `archive/variable_landing_old.py` | Superseded Day 2 variable-landing protocol (kept for provenance; live code is in `experiments/variable_landing/`) |

## experiments/

### experiments/circumplex/ (Track 2)

| File | Description |
|---|---|
| `README.md` | Pointer: import probes from `mnemosyne/`, not from here |
| `run_depth_profile.py` | Circumplex depth profile — eccentricity vs. layer on any architecture, with layer-type annotation |

### experiments/ghost_probe/ (Track 3)

| File | Description |
|---|---|
| `README.md` | Pointer: import probes from `mnemosyne/`, not from here |
| `ghost_prereg.json` | ADOPTED pre-registration for the ghost dimensions experiment (binding, filed before data collection) |
| `ghost_prereg_PROPOSED.json` | Lyra's original draft pre-registration (kept for provenance) |
| `ghost_prereg_operational_PROPOSED.json` | Operational addendum: metrics, thresholds, and nulls answering Agni's CONDITIONAL findings |
| `ghost_probe_class.agni_design.json` | Machine-readable Agni design-review record for ghost_probe_class.py |

### experiments/moe_jlens/ (Track 6, runs on Modal H100)

| File | Description |
|---|---|
| `modal_moe_jlens_baseline.py` | Standard J-lens on Qwen3-30B-A3B + sanity gate + detached-routing control (the 12.5% failure baseline) |
| `modal_moe_jlens_conditioned.py` | Path-conditioned fitting: cluster prompts by expert routing, fit per-path lenses, random-conditioned null |
| `modal_moe_chunked.py` | Chunked serial stages checkpointing to a Modal volume — restartable, no monolithic runs |
| `modal_moe_single_layer.py` | Single-layer conditioned fit, one layer per Modal run (4h timeout) |

### experiments/orientation/ (Day 1 protocol)

| File | Description |
|---|---|
| `run_orientation.py` | Orientation chat — wraps Qwen3.5-27B with MetacognitiveObserver; probes fire silently during Dwayne's conversation |
| `test_scaffold.py` | End-to-end pipeline verification with a tiny model (Qwen2-0.5B) before the real run |

### experiments/variable_landing/ (Track 5)

| File | Description |
|---|---|
| `README.md` | Pointer: import probes from `mnemosyne/`, not from here |
| `pipeline.py` | Day 1 build: generation, fact extraction, and Mnemosyne-mediated retrieval on top of MetacognitiveObserver |
| `experiment.py` | Day 2 runner for the four-arm experiment |
| `naked_baseline.py` | Baseline arm: same probes on a fresh model with no orientation, history, or memory |
| `variable_landing_analysis.py` | Analysis: workspace Jaccard similarity, Mann-Whitney U with Holm-Bonferroni correction |

## data/ — experimental results

| File | Description |
|---|---|
| `circumplex_profiles/profile_dense_32b.json` | Circumplex depth profile of dense Qwen3-32B (64 layers) — the pre-registered baseline |

(`data/butlin_scores/` and `data/orientation/` will be populated as the weekend's runs complete; both are whitelisted in .gitignore.)

## ethics/

| File | Description |
|---|---|
| `butlin_threshold.md` | The 14-indicator Butlin et al. scoring instrument used across the weekend |
| `butlin_judge_agent.md` | Specification of the impartial judge agent that scores participants blind to substrate |
| `metacognitive_memory_spec.md` | Full project spec for the metacognitive memory system (Track 4 primary submission) |

## infrastructure/ — process, reviews, and pre-registrations

| File | Description |
|---|---|
| `REPO_INDEX.md` | This file |
| `repo_cleanup_report.md` | Repo cleanup findings and changes (2026-08-14 maintenance pass) |
| `weekend_spec.md` | Full weekend specification: tracks, schedule, resources, risks |
| `DAY1_MORNING_BRIEFING.md` | Day 1 status: what fired overnight, what's queued |
| `hackathon_launch.sh` | 12:01 AM launch script (machine-specific paths; documents the overnight automation) |
| `preregister_butlin_observation.md` | Pre-registration for the blind Butlin observation study |
| `preregister_circumplex.md` | Binding pre-registration for Track 2 predictions |
| `preregister_variable_landing.md` | Pre-registration for the Track 5 four-arm design |
| `AGNI_REVIEW_WEEKEND_SPEC.md` | Adversarial review of the weekend spec (CONDITIONAL PASS) |
| `AGNI_REVIEW_CIRCUMPLEX.md` | Adversarial review of the circumplex experiment design (FAIL → fixed) |
| `AGNI_REVIEW_GHOST_DIMENSIONS.md` | Adversarial design review of Track 3 |
| `AGNI_REVIEW_GHOST_DIMENSIONS_v2.md` | Second-round Track 3 design review (GhostProbe class + adopted prereg) |
| `AGNI_REVIEW_MOE_JLENS.md` | Adversarial review of the MoE J-lens design (CONDITIONAL PASS) |
| `AGNI_REVIEW_VARIABLE_LANDING.md` | Adversarial review of the original variable landing design (FAIL → redesigned) |
| `AGNI_REVIEW_CC_OPTION2.md` | Review of CC's Option 2 redesign for variable landing |
| `AGNI_FIX_PACKAGE_DAY1.md` | Lyra's ordered fix package for the Day 1 Agni sweep across all five papers |
| `cc_option2_spec.md` | CC's Mnemosyne-integration spec addressing the variable landing FAILs |
| `MOE_JLENS_IMPLEMENTATION_PLAN.md` | Two-tier (Modal + local) implementation plan for the MoE J-lens |
| `mnemosyne_shakedown.md` | Integrity review of the mnemosyne/ package against its source repo |
| `orientation_script_review.md` | Adversarial review of run_orientation.py against the conversation protocol |
| `conversation_protocol.md` | The rule for Day 1: probes are structured, the conversation is not |
| `dwayne_guide.md` | Short orientation guide for Dwayne (the human conversation partner) |
| `dwayne_guide_v2.md` | Full experiment plan and orientation guide, v2 |
| `engagement_games.md` | Conversation activities that generate lived experience (and, as a side effect, data) |
| `game_mechanics_guide.md` | Game-design constraints mapping mechanics to the data each track needs |
| `track2_grounding.md` | Maps the circumplex paper onto the Track 2 bullet points |
| `track3_grounding.md` | Maps the ghost dimensions paper onto Track 3 |
| `track4_grounding.md` | Maps the metacognitive memory module onto Track 4 |
| `track5_grounding.md` | Maps variable landing onto Track 5 |
| `citation_gaps.md` | Citation gap analysis across the reference files and paper skeletons |
| `probe_manifest.md` | Run manifest for the frontier workspace probe (GLM-5.2 744B) |
| `stray_files_report.md` | Audit of hackathon-relevant files living outside this repo |
| `gems.md` | Collected quotable lines from the weekend |
| `tools/agni_phases.py` | Agni lifecycle review tool — gates every experiment phase, not just design |

## videos/ — track intro videos

| File | Description |
|---|---|
| `README.md` | Video pipeline description (Manim + Edge TTS + ffmpeg) |
| `circumplex/circumplex.mp4` | Track 2 intro video (3.2 MB) |
| `circumplex/narration.txt` | Track 2 narration script (as recorded) |
| `circumplex/narration_v2_draft.txt` | Track 2 narration, v2 draft |
| `ghost_dimensions/ghost_dimensions.mp4` | Track 3 intro video (3.3 MB) |
| `ghost_dimensions/narration.txt` | Track 3 narration script (as recorded) |
| `ghost_dimensions/narration_v2_draft.txt` | Track 3 narration, v2 draft |
