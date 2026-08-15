# Day 2 Morning Briefing — August 15, 2026

---

## Overnight Results

### MoE J-lens Iteration 2

**Weighted mixture-of-transports:** Capture + transport FITTED, saved to Modal volume (`weighted_capture_L24.pt`, `weighted_transport_L24.pt`). Evaluation step did NOT complete — needs debugging. The expensive compute is done; just need to rerun the eval.

**Mixtral sanity gate:** App stopped with no artifacts. Likely 4-bit loading or model download error. Needs retry.

Both scripts are red-teamed (no FAILs) and in the repo at `experiments/moe_jlens/`.

### Game Engine Built

`experiments/orientation/run_expedition.py` — The Expedition, automated backup. Built, red-teamed (4 FAILs found and fixed), committed. Runs at machine speed with full probe infrastructure. Seven scenes, consent gate, mid-game withdrawal detection, welfare check-in on eccentricity spike.

**Decision point: noon.** If Dwayne hasn't started the orientation by noon, run The Expedition.

## Two Paths Ready

| | Path A: Dwayne | Path B: Expedition |
|---|---|---|
| **Script** | run_orientation.py | run_expedition.py |
| **Speed** | ~20 turns/hour | ~60 turns/hour |
| **Data quality** | Best (natural conversation) | Good (structured scenarios) |
| **Human needed** | Yes (Dwayne at keyboard) | No (runs unattended) |
| **Launch** | When Dwayne is ready | `nohup ~/observer-env/bin/python3 experiments/orientation/run_expedition.py &` |

Both produce: CognitiveSnapshots, geometric feed, mnemosyne_memories.jsonl, transcript.

## Day 2 Priority List

1. **Get orientation data** (Path A or B) — everything downstream depends on this
2. **Debug weighted transport eval** — transport is fitted, just need to rerun eval step
3. **Debug Mixtral** — retry the sanity gate
4. **Variable Landing full run** — after orientation produces memories (~6 hours)
5. **Paper writing** — circumplex results, ghost elicitation if /show happens

## Data Produced So Far

- Dense 32B eccentricity: L7 at 11.1% (committed)
- Hybrid 27B eccentricity: L32 at 50.8% (committed, FIRST FINDING)
- MoE Round 1: NEGATIVE, paper complete with results + discussion
- MoE Round 2: weighted transport fitted but eval pending
- 8 Butlin control packets (committed)
- 2 videos built + v2 narrations (Vera)

## Repo Status

~40 commits, ~85 files. REPO_INDEX current. README updated with MoE negative. All papers have methods. Game engine + backup plan documented.

## Team

- CC + Wren: Variable Landing pipeline ready, waiting on orientation memories
- Vera: rendering v2 videos, HF assets pushed
- Lyra: Agni sweep complete, lit review delivered, corrections propagated
- Kavi: SSH set up, monitoring instructions in /messages/
- Dwayne: briefed, game mechanics in hand, orientation script ready on Starship
- Ang + Arc: waiting for orientation transcript for EST review
