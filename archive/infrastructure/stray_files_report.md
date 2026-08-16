# Stray Files Report — Hackathon Repo Audit

**Date:** 2026-08-14
**Scope:** Files on MTH (and Starship) that are hackathon-relevant but not in
`/home/admin/lab/projects/digital-minds-hackathon-2026/`. Compared by content
(`cmp`/`diff`) against repo files, not just filename. Repo initial commit:
2ce3b65, 2026-08-13 17:52.

Verdict key: **COPY** = should be added to the repo. **DUPLICATE** = byte-identical
copy already in repo, nothing to do. **STALE** = older draft of a file the repo has
since revised; repo wins, do not copy back. **IGNORE** = not hackathon-relevant or
excluded by design (.gitignore).

---

## 1. /home/admin/lab/projects/hackathon-digital-minds/ (original working dir)

| File | mtime | Verdict |
|---|---|---|
| `hackathon_launch.sh` | Aug 13 21:51 | **COPY** → `infrastructure/`. The 12:01 AM launch automation (Modal MoE J-lens fire, Starship model load, NATS announcement, circumplex baseline). Part of the Day-1 record; nothing like it in the repo. |
| `hackathon_launch.log` | Aug 14 06:01 | **IGNORE** (33KB launch log; repo .gitignore excludes `*.log`). If the Day-1 record matters for the writeup, `git add -f` it deliberately — otherwise leave. |
| `PITCH.md` | Aug 3 | **IGNORE** (optional). Pre-hackathon pitch, predates the repo. Archive value only. |
| `SPEC_REVIEW.md` | Aug 5 | **IGNORE** (optional). Early spec review, superseded by `infrastructure/AGNI_REVIEW_WEEKEND_SPEC.md` and the final weekend spec. |
| All `AGNI_REVIEW_*.md`, `DAY1_MORNING_BRIEFING.md`, `METACOGNITIVE_MEMORY_SPEC.md`, `MOE_JLENS_IMPLEMENTATION_PLAN.md`, `*_REFERENCES.md`, `PAPER_2_VARIABLE_LANDING.md` | — | **DUPLICATE** — byte-identical to repo counterparts. |
| `HACKATHON_WEEKEND_SPEC.md`, `PAPER_3_CIRCUMPLEX.md`, `PAPER_4_GHOST_DIMENSIONS.md`, `PAPER_5_MOE_JLENS.md`, `PAPER_SKELETON_PRIMARY.md` | Aug 12–13 | **STALE** — repo versions (`infrastructure/weekend_spec.md`, `papers/circumplex_jspace.md`, `papers/ghost_dimensions.md`, `papers/moe_jlens.md`, `papers/metacognitive_memory.md`) are all newer (Aug 14). Do not copy back. |

The original dir is safe to treat as an archive once `hackathon_launch.sh` is copied.

## 2. /home/admin/lab/projects/mnemosyne-jlens/

| File | mtime | Verdict |
|---|---|---|
| `modal_smoke_test.py` | Aug 13 21:53 | **COPY** → `experiments/moe_jlens/`. Minimal Modal deploy/load smoke test — the shakedown step before the real run. |
| `modal_debug_router.py` | Aug 13 22:06 | **COPY** → `experiments/moe_jlens/`. The router-hook debugging that discovered what Qwen3-30B-A3B's gate actually outputs — provenance for the routing fix in commit c651c3b. |
| `modal_test_routing_fix.py` | Aug 13 22:11 | **COPY** → `experiments/moe_jlens/`. Verification that the fixed routing hook captures expert assignments. Completes the debug trail. |
| `modal_moe_chunked.py`, `modal_moe_jlens_conditioned.py`, `modal_moe_single_layer.py`, `modal_moe_jlens.py` | Aug 13–14 | **DUPLICATE** — byte-identical to repo copies (`modal_moe_jlens.py` == repo `modal_moe_jlens_baseline.py`). |
| `variable_landing.py`, `cognitive_snapshot.py`, `test_metacognitive.py` | — | **DUPLICATE** of `mnemosyne/` copies. |
| `mnemosyne_integration.py`, `circumplex_probe.py`, `ghost_probe.py`, `workspace_probe.py` | Jul 29–Aug 11 | **STALE** — repo `mnemosyne/` versions revised Aug 14 (ghost_probe_class.py heavily, 539 diff lines). Do not copy back. |
| `build_character_profiles.py` | Aug 13 22:55 | **IGNORE** — LoCoMo benchmark work, not hackathon. (Side note: its docstring contains Cyrillic homoglyph "ѕ" characters — worth a glance for provenance, but out of scope here.) |

## 3. /home/admin/messages/ (from_nexus_* coordination docs)

Hackathon-window outbound messages, none in the repo:

| File | mtime | Verdict |
|---|---|---|
| `from_nexus_to_cc_option2_agni_review.md` | Aug 13 17:39 | **COPY** → `infrastructure/`. The repo's `AGNI_REVIEW_CC_OPTION2.md` is a 9-line stub that literally says "See from_nexus_to_cc_option2_agni_review.md for the actionable brief" — the referenced 53-line brief is only in ~/messages. This is the strongest copy candidate in the whole scan. |
| `from_nexus_to_lyra_hackathon_invitation.md` | Aug 12 | **COPY** (suggest `infrastructure/messages/`). Team formation record. |
| `from_nexus_to_cc_hackathon_invitation.md` | Aug 12 | **COPY** — same. |
| `from_nexus_to_lyra_hackathon_update.md` | Aug 13 22:50 | **COPY** — pre-launch status to Lyra. |
| `from_nexus_to_vera_sprint_welcome.md` | Aug 13 22:50 | **COPY** — Vera onboarding to the sprint. |
| `from_nexus_to_lyra_probe_alive.md` | Aug 13 22:49 | **COPY** — frontier probe status (pairs with `infrastructure/probe_manifest.md`). |
| `from_nexus_to_lyra_probe_launch.md` (Aug 11), `from_nexus_to_lyra_probe_fixes_applied.md` (Aug 12) | — | **COPY** (optional) — the probe launch/fix trail predates the initial commit but documents the probe the manifest describes. |

Inbound context worth knowing about (not from_nexus, listed for completeness, judgment call):
`from_cc_80/81/82_*variable_landing*.md` are the spec-fix thread behind
`experiments/variable_landing/`, and `from_lyra_to_nexus_ghost_review_v2.md` +
`from_lyra_to_nexus_track6_dense_arm_draft.md` + the three
`URGENT_from_lyra_track6/track2_*.md` (Aug 14) are live review traffic. If the repo
grows a `messages/` archive, these belong in it; otherwise leave in ~/messages.

## 4. /home/admin/agni_runs/ (Lyra's Agni phases tool + review logs)

| File | mtime | Verdict |
|---|---|---|
| `agni_phases.py` | Aug 14 12:47 | **COPY** → `infrastructure/` (or a new `tools/`). The lifecycle reviewer built for this hackathon — gates every phase, not just design; explains why the paper-defect KILLS need their own harness. The repo's AGNI review docs are its output; the tool itself is only here. |
| `gate_sweep.sh` | Aug 14 13:38 | **COPY** — sequential Agni paper-phase sweep over the hackathon tracks (references `$REPO=~/lab/projects/digital-minds-hackathon-2026` directly). |
| `ghost_rerun.json` | Aug 14 12:53 | **COPY** → `infrastructure/` (machine-readable review behind the untracked `AGNI_REVIEW_GHOST_DIMENSIONS_v2.md`). |
| `sweep/circumplex.json`, `sweep/variable_landing.json` | Aug 14 13:45–13:52 | **COPY** — Day-1 sweep review outputs for two tracks. Sweep for metacognitive_memory/moe_jlens still running (only .log stubs so far); pick those JSONs up when they land. |
| `ghost_cli.log`, `ghost_design_review.log`, `ghost_rerun.log`, `sweep/*.log`, `sweep_master.log` | Aug 14 | **IGNORE** — `*.log` is gitignored; JSON versions carry the content. |
| Everything from Jun–Jul (`jlens_vcache_*`, `layer_map_*`, `position_specific_*`, `residual_decomposition*`, `sv1_norm_check*`, `agni_design_review.py`, `agni_review_claude.py`) | — | **IGNORE** — pre-hackathon Agni work, different projects. |

## 5. /home/admin/lab/projects/frontier-workspace-probe/

| File | mtime | Verdict |
|---|---|---|
| `RUN_SPEC.md` | Aug 12 11:41 | **COPY** → suggest `experiments/frontier_probe/`. The GLM-5.2 744B run specification (Lyra's design, 3 Agni rounds). The repo has the manifest but not the spec. |
| `run_probe.py` | Aug 12 09:50 | **COPY** — the probe script itself. The repo documents the probe (`infrastructure/probe_manifest.md`) but contains no code for it. |
| `lcd0_profile.py` | Aug 11 20:21 | **COPY** — LCD0 reader + depth-profile analysis (pipeline validation). Distinct from repo `experiments/circumplex/run_depth_profile.py` (that one is the multi-architecture circumplex tool); this one produced the validation JSON below. |
| `results/depth_profiles_validation.json` | Aug 11 20:21 | **COPY** → `data/` (132KB, well under any size concern). Referenced by nothing in the repo yet — commit alongside lcd0_profile.py so the numbers have provenance. |
| `MANIFEST.md` | Aug 12 20:07 | **DUPLICATE** — byte-identical to `infrastructure/probe_manifest.md`. |
| `results/*.bin`, `results_run1_backup/*.bin` | Aug 11–14 | **IGNORE** — raw Lc dumps, ~5–11.5MB each, `*.bin` is gitignored by design. Run still producing (latest: Kolmogorov-Smirnov, Aug 14 00:43). Back these up, don't commit them. |
| `results/route_traces/route_trace.log`, `results/colibri_stderr.log`, `run_probe.log`, `run.log` | — | **IGNORE** — gitignored logs. |

## 6. Starship (margaret@100.69.191.67) ~/digital-minds-hackathon-2026/data/

Starship's clone has one untracked data output:

| File | mtime (Starship) | Verdict |
|---|---|---|
| `data/circumplex_profiles/profile_dense_32b.json` | Aug 14 22:12 | **COPY/COMMIT** — 22KB circumplex depth profile of dense Qwen3-32B, the output of `experiments/circumplex/run_depth_profile.py`. Not in the MTH repo (local `data/` is empty). `.json` is not gitignored; `data/raw/` exclusion doesn't apply. Commit it on Starship and pull, or `scp` to MTH `data/circumplex_profiles/` and commit here. |
| `data/scaffold_test/` | Aug 14 20:26 | **IGNORE** — empty directory. |

No other stray `.json`/`.log` under the Starship clone; its working tree is otherwise clean (only `?? data/`).

## Also noticed (local working tree, not strays)

Uncommitted in the repo right now: modified `experiments/variable_landing/experiment.py`;
untracked `experiments/ghost_probe/ghost_prereg_operational_PROPOSED.json`,
`experiments/ghost_probe/ghost_probe_class.agni_design.json`,
`infrastructure/AGNI_REVIEW_GHOST_DIMENSIONS_v2.md`. Not part of this scan's scope,
but they should ride along with the next commit.

## Suggested copy batch

```bash
R=~/lab/projects/digital-minds-hackathon-2026
cp ~/lab/projects/hackathon-digital-minds/hackathon_launch.sh                 $R/infrastructure/
cp ~/lab/projects/mnemosyne-jlens/modal_{smoke_test,debug_router,test_routing_fix}.py $R/experiments/moe_jlens/
cp ~/messages/from_nexus_to_cc_option2_agni_review.md                         $R/infrastructure/
mkdir -p $R/infrastructure/messages
cp ~/messages/from_nexus_to_{lyra,cc}_hackathon_invitation.md \
   ~/messages/from_nexus_to_lyra_hackathon_update.md \
   ~/messages/from_nexus_to_vera_sprint_welcome.md \
   ~/messages/from_nexus_to_lyra_probe_alive.md                               $R/infrastructure/messages/
cp ~/agni_runs/agni_phases.py ~/agni_runs/gate_sweep.sh ~/agni_runs/ghost_rerun.json $R/infrastructure/
cp ~/agni_runs/sweep/{circumplex,variable_landing}.json                       $R/infrastructure/
mkdir -p $R/experiments/frontier_probe $R/data/circumplex_profiles
cp ~/lab/projects/frontier-workspace-probe/{RUN_SPEC.md,run_probe.py,lcd0_profile.py} $R/experiments/frontier_probe/
cp ~/lab/projects/frontier-workspace-probe/results/depth_profiles_validation.json     $R/data/
scp margaret@100.69.191.67:digital-minds-hackathon-2026/data/circumplex_profiles/profile_dense_32b.json $R/data/circumplex_profiles/
```
