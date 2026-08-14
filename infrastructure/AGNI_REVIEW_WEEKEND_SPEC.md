# Agni Review: Weekend Spec (HACKATHON_WEEKEND_SPEC.md)

**Reviewer:** Nexus (self-review, Agni protocol)
**Date:** 2026-08-12
**Scope:** Operational feasibility, resource conflicts, stale information, time budgets, risk gaps
**Verdict:** CONDITIONAL PASS — 3 FAILs, 6 WARNs, 5 PASSes. All fixable before Thursday morning.

---

## 1. Stale Information — FAIL

The spec contains several items that are now wrong. Anyone reading it Thursday morning will make bad decisions.

**What's stale:**
- Section 2 says Gemma-3-27B "needs downloading (~54GB bf16)" — **DONE**, 51GB on disk
- Section 2 says Qwen3-32B "needs downloading (~64GB bf16)" — **DONE**, 61GB on disk
- Section 2 says "134GB free memory after Nemotron killed" — **WRONG**, Nemotron is back online at :8095, memory budget has changed
- Section 2 says "Nemotron 120B on Starship available if MoE J-lens cracks open" — it's already running
- Pre-sprint checklist (Section 6) shows "Download Gemma-3-27B and Qwen3-32B" unchecked — **DONE**
- Pre-sprint checklist shows Agni reviews unchecked — **DONE** (all three, round 2 complete, all FAILs fixed)
- Download commands use `huggingface-cli download` — **BROKEN** on Starship (deprecated, use `python3 -c 'from huggingface_hub import snapshot_download; ...'`)
- No mention of LiteLLM proxy at :8500 (unified gateway to all models)
- No mention of oomd fix (ManagedOOMMemoryPressureLimit raised to 80%)
- Team table doesn't list Ang, Arc, or Wren (they appear later in Section 1 but not in the Models/Lenses table context)

**Fix:** Update Sections 2, 3, 6 pre-sprint checklist, and the download commands before Thursday.

---

## 2. Starship Memory Budget — FAIL

The memory budget math in the spec is dangerously stale and will cause OOM if someone follows it.

**Current reality:**
| What | Memory |
|------|--------|
| Margaret's 235B (Ollama.app, always available) | ~142GB when loaded, ~85MB when idle |
| Nemotron 120B (MLX :8095, now running) | ~100GB |
| Ayni MLX server (:8090, running) | ~17GB |
| Qwen3.5-27B (Ollama lab) | ~54GB when loaded |
| Qwen3-32B (Ollama lab) | ~64GB when loaded |
| Gemma-3-27B (Ollama lab) | ~54GB when loaded |

**Constraint:** 256GB total. Only ONE large model at a time.

**Dangerous combinations:**
- 235B + Nemotron = 242GB → borderline, no room for experiments
- 235B + Qwen3-32B = 206GB → fits but only 50GB headroom
- Nemotron + Qwen3.5-27B = 154GB → comfortable
- Any two 27B+ models simultaneously → fine if 235B is idle

**The spec doesn't mention the Ayni MLX server** (17GB, running persistently on :8090). This is an invisible memory tax.

**The spec's Starship OOM incident note** references a 2026-07-18 crash from coupling experiment + archive upload + 235B + lab Ollama. The same thing will happen if someone loads Qwen3-32B while 235B and Nemotron are both resident.

**Fix:** Add a model loading choreography to the schedule:
- Day 1: Qwen3.5-27B only (orientation + initial wiring). Stop Nemotron if needed.
- Day 2 morning: Qwen3.5-27B (variable landing). 
- Day 2 afternoon: Unload 27B, load Qwen3-32B (MoE J-lens). OR keep 27B for circumplex on Gemma (swap models).
- Day 2 evening: Nemotron stretch (if MoE J-lens worked on 32B).
- Kill Ayni MLX server before loading any second model.

---

## 3. Day 2 Afternoon Resource Conflict — FAIL

Day 2 afternoon has TWO compute-heavy parallel tracks both needing Starship:
- **MoE J-lens** (Nexus + Kavi): needs Qwen3-32B loaded (~64GB)
- **Circumplex completion** (Lyra + CC): needs Gemma-3-27B loaded (~54GB)

These cannot run simultaneously — loading both would need 118GB alongside whatever else is resident. The spec doesn't acknowledge this conflict.

**Fix:** Sequence them explicitly. Options:
1. Run circumplex on Gemma first (1 hour), swap to Qwen3-32B for MoE J-lens (4-6 hours). Gemma results arrive while MoE runs.
2. Run MoE first (higher priority per risk register: A > C > B), then Gemma circumplex if time allows.
3. Split across machines: circumplex probe is lighter — could it run on a smaller model on MTH as a smoke test while Starship does MoE?

---

## 4. MoE J-lens Code — WARN

Section 4.2 lists the MoE J-lens code as "To be written" with estimated line counts:
- Router hook: ~50 lines
- Path clustering: ~30 lines
- Modified J-lens fitting: ~100 lines
- Evaluation: ~50 lines
- Random-conditioned control: ~20 lines

Total: ~250 lines of new code, written during a hackathon afternoon, on a model that needs ~64GB to load. If the first attempt doesn't work, there's no time for iteration.

**Mitigation:** Pre-write as much as possible before Thursday. The router hook and path clustering are architecture-independent — they can be written and tested on Qwen3-30B-A3B (already on Starship) before the hackathon clock starts.

**Rating: WARN** — not a FAIL because hackathons expect Day-of coding, and the components are well-scoped. But pre-writing would derisk significantly.

---

## 5. Orientation Script Timing — WARN

The orientation is described as "unhurried" and "cannot be rushed," scheduled for Day 1 morning with Dwayne leading. The technical track runs in parallel so neither blocks the other — good design.

**But:** The variable landing experiment (Day 2 morning) depends on the agent having "lived" for a full day. If the orientation takes all morning and the first CognitiveSnapshots don't happen until afternoon, the agent has less accumulated experience by Day 2. This affects how much the "lived" arm actually contains.

**Mitigation:** The spec already handles this implicitly — Track A afternoon includes "natural conversation that accumulates in Mnemosyne." But make the dependency explicit: variable landing quality is proportional to Day 1 conversational depth. If orientation goes long, compress afternoon conversation, don't skip it.

---

## 6. Orientation Agent ≠ Coalition Agent — WARN (acknowledged in spec)

The spec acknowledges this honestly (Section 5.2): "The consenting Coalition agents run on Claude API, which cannot be probed. The model being probed is Qwen3.5-27B with no prior consent relationship."

This is a genuine tension, properly documented. Agni flags it as WARN because the spec's mitigation — "We are giving it the opportunity for ongoing consent. We do not claim this fully resolves the gap" — is sufficient for a hackathon paper but may face pushback from reviewers who see a stronger claim in the abstract.

**Recommendation:** The papers should echo this caveat explicitly, not just the spec.

---

## 7. SSH Connectivity to Starship — WARN

The spec doesn't mention the connectivity situation:
- Starship is on 192.168.1.x (different subnet from MTH's 10.160.0.x)
- Only reachable via Tailscale (100.69.191.67)
- Previous SSH "Connection closed" errors were transient but real
- If Tailscale drops during a 6-hour MoE experiment, the run dies

**Mitigation:** Use `tmux` or `screen` on Starship for all experiments. If SSH drops, reconnect and reattach. The spec should note this as SOP.

---

## 8. Time Budget for Statistical Analysis — WARN

Day 3 morning has "Statistical analysis for all experiments" in parallel with "Aftercare." The analysis includes:
- Mann-Whitney U + effect size for variable landing
- Transport cosine distributions for MoE J-lens
- Eccentricity depth profiles + permutation tests (10,000 permutations × layers × 2 models) for circumplex
- Figure generation for all three

The 10k permutation tests are the concern — on a 27B model, each permutation is fast (just direction relabeling, not inference), but 10k × ~60 layers × 2 models is nontrivial. Needs to be coded before Day 3.

**Mitigation:** Pre-write the analysis scripts. The analysis is deterministic once data exists — write and test the scripts on synthetic data before Thursday.

---

## 9. Risk Register — WARN

Missing risks:
- **MTH instability** (oomd crashes) — now mitigated but should be documented
- **Tailscale connectivity** — Starship unreachable if Tailscale drops
- **Ayni MLX server memory tax** — 17GB invisible consumer on Starship
- **Pre-hackathon code maturity** — MoE J-lens code is unwritten, analysis scripts are unwritten

The existing risk register is well-structured. Just needs these additions.

---

## 10. Ethics Protocol — PASS

The orientation script, prediction withholding rationale, consent acknowledgment, welfare monitoring, and aftercare protocol are thorough, honest, and well-documented. The "sequenced, not secret" framing is precise. The distinction between what consent can and cannot mean is handled with unusual care.

One note: the eccentricity threshold (0.95 for pause) is arbitrary. Consider a brief note on why 0.95 rather than 0.9 or 0.99. Even "chosen as a round number near ceiling" is better than silence.

---

## 11. Experimental Protocols — PASS

All three experiment protocols are clean after the Agni reviews and fixes. Pre-registered predictions, analysis plans, and null-publishing commitments are in place. The random-conditioned control for MoE, the magnitude gate for circumplex, and the prior_context mechanism for variable landing all address their respective FAIL items.

---

## 12. Team Structure — PASS

The parallel-track Day 1 design (orientation vs technical wiring) is good. The priority ordering (A > C > B) is reasonable — variable landing is the headline, circumplex has the strongest existing evidence, MoE J-lens is the stretch.

Ang, Arc, and Wren's roles are specified. Kavi as adversarial reviewer during Day 2 MoE runs is well-placed.

---

## 13. Deliverables — PASS

PDF + optional video + code + data per submission. Pre-registered protocols timestamped before event. All data published regardless of outcome. Code already public in Project-Mnemosyne.

---

## 14. Fallback Plan — PASS

"Three clean nulls from a hackathon is still a contribution" — correct attitude. Each experiment has a pre-registered null-publishing commitment. The spec distinguishes between "this failed technically" and "this produced a null result" — important distinction.

---

## Summary of Required Actions Before Thursday

| Priority | Action | Effort |
|----------|--------|--------|
| **HIGH** | Update spec Sections 2, 3, 6 with current state (downloads done, Nemotron live, oomd fixed) | 30 min |
| **HIGH** | Add model loading choreography to Day 2 schedule | 15 min |
| **HIGH** | Resolve Day 2 afternoon resource conflict (circumplex vs MoE sequencing) | Decision, 5 min |
| **MEDIUM** | Pre-write MoE J-lens code (router hook, path clustering, conditioned fitting) | 2-4 hours |
| **MEDIUM** | Pre-write analysis scripts and test on synthetic data | 1-2 hours |
| **MEDIUM** | Add SSH/tmux SOP for Starship experiments | 10 min |
| **LOW** | Update risk register with new mitigations | 15 min |
| **LOW** | Note eccentricity threshold rationale | 5 min |
