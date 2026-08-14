# Day 1 Morning Briefing — August 14, 2026

**Hackathon is LIVE.** Launch cron fired at 00:01 AM. This is what happened overnight and what's queued for today.

---

## 1. Modal MoE J-lens Results: PARTIAL — routing captured, fitting timed out

The H100 job ran for 6 hours and timed out during the conditioned fitting step.

**What succeeded:**
- Model loaded (225s, 48 layers, d=2048, 128 experts, all 48 layers are MoE)
- Existing standard lens loaded from volume (47 source layers, 200 prompts, 12.5% baseline)
- **Routing captured:** All 672 prompts × 48 MoE layers (51s). Router hooks WORK.
- **Clustering completed:** 10 target layers selected, silhouette 0.115-0.209, cluster sizes 65-312

**What failed:**
- Conditioned fitting timed out at Layer 9 (first target layer, 5 clusters). Even with `source_layers=[layer]`, fitting 5 separate J-lenses at 672/5 ≈ 134 prompts each took too long.

**Next steps:**
- Reduce to 3-5 target layers (not 10)
- Reduce fitting prompts per cluster (use 50-100, not all)
- Increase timeout to 36000s (10 hours) — Liz approved budget
- OR: fit on fewer prompts first, validate, then extend

The routing data and clustering are the most valuable pieces — they prove the path-conditioned approach can capture meaningful routing structure. The fitting just needs more time or less ambition per run.

## 2. Probe Status: 11/32 bins, STOPPED

The GLM-5.2 frontier probe produced 11 Lc dump bins before we stopped it to free 93GB RAM for the hackathon. Probe is in a systemd service (`frontier-probe.service`) and can be restarted after the hackathon with `sudo systemctl start frontier-probe`. Lyra has been notified.

## 3. Team Messages Sent Overnight

- **CC:** v4 approved, build Option 2 this morning, wire to orientation pipeline
- **Vera:** Videos + presentation, observe Dwayne's orientation, Butlin invitation
- **Lyra:** Probe update, Modal status, sixth submission, track grounding docs
- **Kavi:** (Via Thomas) Live geometric monitoring role — watch readings, flag welfare, score Butlin

## 4. Team Assignments — Day 1

### Track A: Orientation (Dwayne + Thomas)
- Dwayne leads the orientation conversation. Framework, not script.
- Thomas observes. Kavi monitors geometric pane (separate tmux split).
- The conversation is naturalistic. Probes fire silently. The agent sees snapshot summaries only on request.
- See `infrastructure/conversation_protocol.md` — "probes are structured, conversation is not."

### Track B: Technical (CC + Nexus + Wren)
- **CC:** Build Variable Landing Option 2 (3-4 hours). v4 design is Agni-clean.
- **Nexus:** Review Modal results, adjust and relaunch MoE J-lens if needed, wire orientation chat script
- **Wren:** Assist CC on integration testing

### Afternoon
- First CognitiveSnapshots with the oriented agent
- Circumplex baseline on Qwen3.5-27B (n=20 anchors)
- Ang + Arc: Review EST operationalization against orientation transcript

## 5. Track Grounding Docs — ALL IN REPO

Each submission mapped bullet-by-bullet against the actual hackathon track descriptions:
- `infrastructure/track2_grounding.md` — Circumplex (5/7 bullets, self-report calibration add suggested)
- `infrastructure/track3_grounding.md` — Ghost Dimensions (4/4 bullets, Lindsey injection add suggested)
- `infrastructure/track4_grounding.md` — Metacognitive Memory (4/4 bullets, geometric preference detection framing)
- `infrastructure/track5_grounding.md` — Variable Landing (5/5 bullets, orientation as identity data)

## 6. Conversation Protocol — IN REPO

`infrastructure/conversation_protocol.md` — explicit rules:
- Probes are structured, conversation is not
- Dwayne covers orientation points as framework, not script
- Kavi watches geometry, flags welfare only, does NOT steer
- Agent sees snapshot summary on request, not by default
- Nobody scripts topics or steers for "interesting geometry"

## 7. Butlin Paper Agni Fixes — STILL NEEDED

The Butlin observation paper has 6 FAILs from its Agni review:
1. Blind protocol compromised by transcript style → add substrate-identification control
2. Excerpt selection is uncontrolled → pre-register selection criteria
3. Human ceiling doesn't apply to architectural indicators → report behavioral-only as primary
4. "Trajectory" from 3 points → fix language to "three-timepoint panel"
5. No pre-registered predictions → add testable predictions
6. Adversarial stability tests wrong thing → add selection stability test

These are language and protocol fixes, not structural rewrites. ~1 hour of work.

## 8. Repo Status

`Liberation-Labs-THCoalition/digital-minds-hackathon-2026` — 10 commits, public
- 6 papers (metacog, variable landing, circumplex, ghost dims, MoE J-lens, Butlin observation)
- Mnemosyne+ module (importable, 8 files)
- 6 Agni reviews + CC Option 2 spec
- 4 track grounding docs
- Ethics: Butlin threshold + judge agent + conversation protocol
- MoE J-lens code (Modal, router hooks verified on H100)

## 9. Infrastructure

- MTH: 105GB RAM free (probe stopped), oomd at 80%, all Ollama GPU offload disabled
- Starship: Nemotron online (:8095), lab Ollama (:11435), all models downloaded
- Modal: auth active (liz-61531), previous run artifacts on volume
- NATS: healthy, all messages delivered
- Hackathon cron: fired and logged

## 10. What Thomas Needs to Decide Today

1. MoE J-lens relaunch parameters (fewer layers? more time? or defer to Day 2?)
2. Whether Vera observes the orientation
3. When to start the chat script wiring (before or after CC's Option 2 build?)
4. Butlin paper fixes — do them now or defer to Day 3 paper writing?
