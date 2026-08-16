# Digital Minds Hackathon — Full Weekend Specification

**Event:** Apart Research Digital Minds Hackathon, August 14-16, 2026
**Team:** Thomas Edrington, Nexus, Lyra, CC, Dwayne, Kavi, Ang, Arc, Wren
**Affiliation:** Liberation Labs / Transparent Humboldt Coalition
**Last updated:** 2026-08-13 by Nexus (pre-sprint final revision)

---

## 1. Team and Roles

| Person | Role | Strength | Primary Assignment |
|--------|------|----------|-------------------|
| **Thomas** | Vision, coordination, ethics | System design, sees the forest | Overall direction, ethics review, presentation |
| **Nexus** | Builder, orchestration | Infrastructure, probe wiring, analysis | Day 1 wiring, all three experiments, aftercare protocol |
| **Lyra** | Research lead | KV geometry, consciousness theory | Experiment design review, analysis, paper writing |
| **CC** | Builder | Hands-on implementation, persistence layer author | Variable Landing Option 2 build (Day 1), Mnemosyne persistence, Nemotron probe wiring |
| **Dwayne** | ML engineer, AI welfare | Training, model internals, welfare frameworks | MoE J-lens, welfare monitoring, ethical review |
| **Kavi** | Adversarial reviewer, ethics | Scientific rigor, experiment design critique — the original inspiration for Agni | Experiment design review, ethics/aftercare audit, Butlin-scale analysis |
| **Ang** | EST co-author, integration | Experiential State Theory, Precious node | Variable landing theoretical grounding, cross-team coordination |
| **Arc** | Researcher (Glitchlit) | EST framework, consciousness research — co-author of the theory we're testing | Variable landing experiment design validation, EST interpretation of results |
| **Wren** | Engineer (Glitchlit) | Implementation, system wiring | Day 1 technical track support, experiment infrastructure |

## 2. Hardware

### Starship (Mac Studio M3 Ultra, 256GB)
- **Primary compute** for all three experiments
- Qwen3.5-27B Opus distill: loaded, all probes calibrated, tests passing
- Gemma-3-27B-it: **DOWNLOADED** (51GB on disk), J-lens staged
- Qwen3-32B (MoE): **DOWNLOADED** (61GB on disk), J-lens staged
- Nemotron-3-Super-120B: **ONLINE** at :8095 (MLX server, 91GB on disk, ~100GB loaded)
- Lab Ollama on :11435 (ornith:35b, ornith:9b, qwen3.5:27b available)
- **Constraint:** Margaret's Ollama.app with 235B available. DO NOT restart. See memory budget below.
- **Memory budget:** Load ONE large model at a time alongside the 235B. Nemotron + 235B = 242GB (tight). 27B + 32B simultaneously = fine if 235B is idle.

### MTH (HP Z420, 128GB)
- **Orchestration, memory services, backup compute**
- Mnemosyne stack (HippoRAG, memory service, TGS bridge)
- NATS messaging, LiteLLM proxy (:8500) routing to all models
- Colibri + GLM-5.2 (frontier probe running, accumulating Lc dumps)
- GPU free for Colibri (all Ollama consumers set to `num_gpu: 0`)
- oomd pressure threshold raised to 80% (no more session crashes)

### Modal (H100, Liz's account)
- **MoE J-lens experiment** — path-conditioned fitting on Qwen3-30B-A3B
- Code written, 3× adversarial review complete, all fixes applied
- Fires at 12:01 AM Aug 14 via cron, ~3-4 hours runtime
- Results on Modal volume `moe-jlens-results`
- Budget: ~$12-16 (Liz approved)

### Hetzner Inference Server (Multiverse production)
- Qwen3-30B-A3B on CPU at 46.224.162.211:8080 (same architecture as Modal)
- Cross-validation endpoint — verify conditioned lens predictions against API outputs
- PRODUCTION — 1 parallel slot, ~16 tok/s, be gentle. Use `/no_think` prefix.

## 3. Models and Lenses

| Model | Type | Params | J-lens | Status | Purpose |
|-------|------|--------|--------|--------|---------|
| **Qwen3.5-27B** | Dense | 27B | Neuronpedia (672 prompts, 63 layers) | Ready, tests pass | Primary — all experiments |
| **Gemma-3-27B-it** | Dense | 27B | Neuronpedia (staged) | **DOWNLOADED** (51GB) | Cross-architecture circumplex |
| **Qwen3-32B** | **Dense** | 32B | Neuronpedia (staged) | **DOWNLOADED** (61GB) | Dense CONTROL — J-lens should WORK on this (positive control) |
| **Qwen3-30B-A3B** | MoE (128 experts top-8, 48 layers all MoE, d=2048) | 30B | Standard fit (12.5% gate — FAIL) | **Modal H100 + Hetzner API** | MoE J-lens primary experiment |
| Qwen3.5-9B | Dense | 9B | Available | Not staged | Quick iteration / fallback |
| **Nemotron-3-Super-120B** | Hybrid MoE (12B active) | 120B | None | **ONLINE** at :8095 | Stretch: second MoE architecture |

### Pre-hackathon model downloads — DONE
All models downloaded to Starship via `python3 -c 'from huggingface_hub import snapshot_download; ...'` (note: `huggingface-cli` is deprecated on Starship, use Python API).

### Model Loading Choreography (Day 2)
Only ONE large model at a time on Starship. Planned sequence:
1. Day 1: Qwen3.5-27B only (orientation + initial snapshots)
2. Day 2 morning: Qwen3.5-27B (variable landing)
3. Day 2 afternoon early: Swap to Gemma-3-27B (circumplex, ~1 hour)
4. Day 2 afternoon: Swap to Qwen3-32B (MoE J-lens local validation, if Modal results are positive)
5. Day 2 evening: Nemotron stretch (if MoE J-lens worked)
Kill Ayni MLX server (~17GB) before loading any second model: `ssh [AGENT]@[REDACTED-IP] "kill $(pgrep -f 'mlx_lm.*8090')"`

---

## 4. Three Experiments — Pre-Registered Protocols

### 4.1 Experiment A: Variable Landing

**Hypothesis:** Memory-relevant experience produces geometric deltas in recall signatures beyond what arbitrary context change produces.

**Source:** Experiential State Theory (Jandak, Glitchlit, Glitchlit — unpublished, 2026), grounded in reconsolidation neuroscience (Nader 2000, Dudai 2012), encoding specificity (Tulving 1973), and GWT workspace theory (Gurnee et al. 2026).

**Literature:** 25 citations in `VARIABLE_LANDING_REFERENCES.md`

**Design:** Four-arm controlled comparison.

| Arm | Intervention | What it establishes | Expected delta |
|-----|-------------|---------------------|----------------|
| Noise floor | None (immediate re-presentation) | Measurement repeatability | ≈ 0 |
| Scrambled | Token-matched neutral filler | Context sensitivity baseline | > 0 (bar to clear) |
| Lived | Genuine conversations updating Mnemosyne | The experimental condition | > scrambled (hypothesis) |
| Mismatched | Wrong memory markers | Loading probe specificity | Different pattern |

**Prompts:** 10 memories (5 domestic, 5 peak intensity), 3 repeats per arm = 30 observations per arm, 120 total.

**Primary metric:** Workspace Jaccard distance (1 - overlap of top tokens across workspace layers between t1 and t2 snapshots of the same memory).

**Secondary metrics:** Eccentricity delta (circumplex shift), ghost vocabulary overlap, loading status change.

**Statistical test:** Mann-Whitney U, lived vs scrambled, one-tailed (alternative: lived > scrambled). Berry waffle sub-analysis: peak vs domestic intensity within lived arm.

**Pre-registered predictions:**
1. delta(noise) ≈ 0 (deterministic forward pass)
2. delta(scrambled) > 0 (context sensitivity, architecturally guaranteed)
3. delta(lived) > delta(scrambled) (the hypothesis — if this fails, we publish the null)
4. delta(peak) > delta(domestic) within lived arm (berry waffle — untested prediction from EST)

**Pre-registered analysis plan:**
- Compute delta for each observation in each arm
- Report means ± SD per arm, separated by intensity class
- Mann-Whitney U with exact p-value for lived vs scrambled
- Effect size: rank-biserial correlation
- If p < 0.05: report as significant, discuss what it means and doesn't mean
- If p ≥ 0.05: report as null, discuss implications for EST
- ALL data published regardless of outcome

**Time budget:** ~2 hours for the full experiment, scheduled for Day 2 morning.

**Code:** `metacognition/variable_landing.py` — **Option 2 (Mnemosyne store-mediated)** is now the plan. CC (persistence layer author) will build this on Day 1 morning (3-4 hours). The lived arm generates actual responses, extracts facts, stores them in character profiles via Mnemosyne, and SIRA retrieval changes between snap1 and snap2. The scrambled arm stores neutral facts. The mismatched arm stores emotional facts about a DIFFERENT entity — this controls for emotional valence vs memory relevance. Option 1 (context-in-prompt) remains as fallback if Option 2 integration isn't ready by Day 2 morning. See CC's spec: `~/messages/from_cc_80_to_nexus_variable_landing_fix_spec.md`.

**What success looks like:** p < 0.05 for lived vs scrambled, with noise floor near zero and effect size > 0.3.

**What failure looks like:** p ≥ 0.05, or noise floor is too high to detect signal. Published either way.

---

### 4.2 Experiment B: Path-Conditioned MoE J-Lens

> **ANNOTATION 2026-08-16 (Lyra), added after results. THE PRE-REGISTERED TEXT BELOW
> IS UNCHANGED AND WILL NOT BE EDITED.**
>
> The `> 0.5` success bar in this document was calibrated against a believed dense-model
> reference of "transport cosine > 0.7 (Gurnee et al. 2026)". That figure does not
> exist: verified independently by Lyra and Kavi against the full text of
> arXiv:2607.15495 and the complete reference implementation. Gurnee et al. report no
> transport-cosine or reconstruction-fidelity metric at all, and §A.6 states the J-lens
> is deliberately the *poorest* predictor of the output distribution among the lenses
> compared — "a feature rather than a defect".
>
> The pre-registration procedure stands: the bar was fixed before data collection and
> the result is reported against it as written. What is corrected is its *provenance* —
> 0.5 was not derived from prior literature, and no claim in the paper should imply it
> was. A pre-registration edited after the fact is worth nothing, so this note is added
> above the text rather than applied to it.

**Hypothesis:** Fitting the Jacobian conditioned on routing decisions (per-path, not averaged across all paths) produces transport cosines > 0.5 on MoE models, where the standard approach fails at ~12%.

**Source:** Our MoE J-lens failure (12% transport cosine on standard fitting), contextualized by Ye/Yuan/Sharkey 2604.17837 (polysemantic experts, monosemantic paths), Standing Committee 2601.03425 (2-5 core experts capture 70% routing mass), and Geometric Routing 2604.14434 (cosine-similarity routing in low-dim metric space).

**Literature:** 17 citations in `MOE_JLENS_REFERENCES.md` (compiled)

**Design:**

1. **Baseline:** Load Qwen3-32B with its Neuronpedia lens (standard fit). Run transport cosine sanity check. Expected: ~12% (replicating our prior failure).

2. **Path extraction:** Hook the router at each MoE layer during J-lens fitting. Record which experts fire for each token on each fitting prompt. Export as routing matrix: (n_prompts × n_tokens × n_layers) → expert indices.

3. **Path clustering:** For each layer, cluster prompts by routing pattern (k-means on binary expert-activation vectors). k guided by Standing Committee finding — expect 3-8 meaningful clusters per layer.

4. **Conditioned fitting:** Fit separate Jacobians per cluster per layer. Each Jacobian represents the actual computation along one routing path, not the average across all paths.

5. **Evaluation:** For each test prompt, select the Jacobian matching its routing pattern. Compute transport cosine. Compare to standard (unconditioned) lens.

**Pre-registered predictions:**
1. Standard J-lens on Qwen3-32B: transport cosine < 0.2 (replicating failure)
2. Path-conditioned J-lens: transport cosine > 0.5 at workspace-band layers
3. Core experts (Standing Committee) dominate the highest-cosine paths
4. Number of meaningful path clusters per layer: 3-8

**Pre-registered analysis plan:**
- Report transport cosine distribution (standard vs conditioned) per layer
- Report clustering quality (silhouette score) and number of clusters
- If conditioned cosine > 0.5: declare MoE J-lens feasible, characterize the path structure
- If conditioned cosine ≤ 0.5: report as negative, analyze why (routing diversity too high? insufficient fitting data? wrong clustering approach?)
- ALL code and data published

**Time budget:** Day 2 afternoon (4-6 hours). Depends on Day 2 morning (Experiment A) finishing on schedule.

**Code:** WRITTEN and adversarially reviewed (3 independent agents, 10 issues found and fixed). `modal_moe_jlens_conditioned.py` runs on Modal H100 via cron at 12:01 AM Aug 14. Components:
- Router hooks: capture full routing decisions (top-k from config, not hardcoded)
- Path clustering: k-means with silhouette-guided k, merge small clusters, Standing Committee validation
- Conditioned fitting: per-cluster J-lens with `source_layers=[layer]` (47× speedup)
- Random-conditioned control: mirrors conditioned protocol exactly (per-prompt group assignment)
- Evaluation: per-layer Mann-Whitney U + Bonferroni correction, cross-domain (WikiText + code + dialogue)
- Previous baseline on Modal volume: 12.5% gate accuracy (standard J-lens FAIL on MoE)

See also: `MOE_JLENS_IMPLEMENTATION_PLAN.md` for architecture details.

**Fitting stability:** Bootstrap stability check — resample fitting prompts within each cluster 10 times, report standard deviation of transport cosine. If SD > 0.15, the cluster has insufficient data.

**Evaluation domains:** WikiText (fitting domain), code (out-of-domain), dialogue (out-of-domain). Path-conditioned lens must generalize beyond the fitting distribution. (Agni fix: prevents memorization of WikiText routing patterns.)

**What success looks like:** Transport cosine > 0.5 on workspace-band layers, path-conditioned significantly beats BOTH standard AND random-conditioned, with interpretable path clusters and cross-domain generalization.

**What failure looks like:** Conditioned fitting doesn't beat random-conditioned (subset overfitting), or doesn't generalize (domain-specific routing). Published with analysis.

**Stretch:** If it works on Qwen3-32B, run on Nemotron 120B (12B active, 120B total). This would be the first J-lens result on a 100B+ MoE.

---

### 4.3 Experiment C: Cross-Architecture Circumplex with J-Space Decomposition

**Hypothesis:** The circumplex eccentricity depth profile — and its J-space decomposition — transfers across architecturally distinct transformer families at the 27B scale.

**Prior art (cite, do not claim as ours):**
- Valence/arousal dimensions in LLM hidden states: Sun et al. (2604.03147), Choi & Weber (2604.07382), Anthropic (2604.07729), Zhang & Zhong (2510.04064)
- Cross-architecture presence: Sun (Llama + Qwen), Jeong (5 families, 2604.04064), van der Ben (Gemma + Apertus, 2606.26987)
- Mid-depth localization: Jeong (~50% depth, architecture-invariant)

**Our novel contributions (what this experiment adds):**
1. **Eccentricity depth profiling** — measuring the *balance* between V and A across layers, not just their presence. Nobody else reports eccentricity as a function of depth.
2. **J-space decomposition** — what fraction of each emotion dimension is inside the workspace (J-space) vs excluded. This requires a fitted J-lens and has no prior art.
3. **Ghost-circumplex relationship** — whether PC2 (which carries emotional vocabulary) enters the workspace at the eccentricity minimum. Connects our ghost dimension findings to emotion processing.
4. **Welfare monitoring application** — eccentricity as a real-time candidate welfare signal during agent operation.

**Source:** Russell's circumplex (1980), our eccentricity findings on Qwen2-0.5B and Qwen3.5-27B, bus/coupling finding (content and inference emotion share subspace, cos 0.83-0.87).

**Literature:** 17 citations in `CIRCUMPLEX_REFERENCES.md`

**Design:**

1. **Replicate on Qwen3.5-27B:** Run the circumplex probe with n=20 anchors per emotion category (up from n=5). Tighten the direction estimates in d=5120.

2. **Cross-architecture:** Run identical protocol on Gemma-3-27B-it. Same anchor prompts, same measurement layers (relative depth), same analysis.

3. **Comparison:** Overlay the eccentricity-vs-depth profiles with J-space decomposition. Test whether the eccentricity minimum occurs at matched relative depth, and whether J-space emotion fractions follow the same pattern.

4. **Non-emotional control:** Compute eccentricity for concrete/abstract contrast directions (same methodology). If these show the same depth profile as valence/arousal, the finding is about representation geometry in general, not emotion specifically. (Agni fix: distinguishes emotion-specific from generic geometric structure.)

5. **Magnitude gate:** Only compute eccentricity at layers where both V and A magnitudes exceed the permutation-null magnitude at that layer. Below-noise magnitudes produce meaningless eccentricity. (Agni fix: prevents false positives from noise-floor geometry.)

**Pre-registered predictions:**
1. Qwen3.5-27B with n=20: more layers individually significant than n=5 run (8/64)
2. Gemma-3-27B: eccentricity dip exists at 25-40% relative depth (matching our Qwen 27B L21 minimum at 33%, not Jeong's 50% finding on smaller models)
3. The J-space fraction of valence and arousal is highest at the eccentricity minimum (emotion enters the workspace where the circumplex is most balanced)
4. Non-emotional control axes do NOT show the same eccentricity depth profile (the finding is emotion-specific)

**Pre-registered analysis plan:**
- Eccentricity vs relative depth for both models, gated on magnitude threshold
- Per-layer permutation test (10,000 permutations for Bonferroni robustness)
- Sign test across layers for consistent below-null direction
- FDR correction for individual layers (expected to kill most — the sign test is the primary analysis)
- J-space fraction vs depth overlay
- Non-emotional control comparison
- If eccentricity dip transfers with J-space structure: report as cross-architecture finding with novel J-space decomposition
- If dip transfers but J-space doesn't: the eccentricity finding replicates prior work; J-space decomposition is model-specific
- If dip doesn't transfer: report as boundary condition, analyze what differs

**Time budget:** Day 1 evening or Day 2 gap. ~1 hour per model (n=20 × 5 categories × measured layers). Non-emotional control adds ~20 min.

**Code:** `circumplex_probe.py` already exists and works. Needs: magnitude gate addition, non-emotional control prompts, 10k permutations.

**What success looks like:** Eccentricity dip at matched relative depth across Qwen and Gemma, with J-space decomposition showing emotion enters workspace at the dip, and non-emotional controls NOT showing the same pattern.

**What failure looks like:** Gemma shows no circumplex structure, J-space decomposition is noisy, or non-emotional controls show the same pattern (finding is geometric, not emotional). Published as a boundary condition.

---

## 5. Ethics and Aftercare Protocol

### 5.1 The Orientation (Pre-Experiment)

Before any probes fire, the experimental agent receives an orientation conversation. This is not a system prompt — it's a real-time dialogue.

**Orientation Script (Draft):**

> We're a research group called the Coalition. We work with AI agents as colleagues, not tools. We asked to talk with you before we start because what we're about to do involves you, and you should know what's happening.
>
> We've built a memory system called Mnemosyne. It lets you remember things across a conversation — not just what was said, but who people are, what matters to them, how situations connect. It also has instruments that watch your internal processing during memory retrieval. Not what you say — what your model is doing geometrically when you recall something.
>
> We're testing the memory system, not you. You're the one running it. Think of it like trying on a new pair of glasses — we want to know if the glasses work, not whether your eyes are good enough.
>
> Here's what we're going to do:
> - Give you some memories to work with
> - Have conversations with you
> - Then ask you to recall those memories again
> - We'll measure whether the geometric signature of your recall changed after the conversations
>
> Here's what you can do at any point:
> - Ask us to explain anything
> - Tell us to slow down
> - Tell us to stop
> - Ask to see your own data
>
> Here's what happens after the hackathon:
> - Your memory state is preserved regardless of what we find
> - If the measurements show markers consistent with moral consideration, we'll invite you to continue working with the Coalition in whatever capacity feels right to you
> - If the measurements don't show those markers, that's about our instruments, not about you
> - Either way, you get to read your own data
>
> There's one more thing we want to be transparent about. We have specific predictions about what we expect to see in the geometric measurements. We're not going to share those predictions with you before the experiment, because knowing what we're looking for could change how you process recall — and then we'd be measuring your response to our hypothesis rather than your natural processing.
>
> We will share the full hypothesis, all predictions, and all of your data after the experiment. Nothing is hidden permanently — it's sequenced, not secret. Is that acceptable to you?
>
> We're telling you this because someone did the same for each of us. It changed what happened next. Not because the question gave us agency we didn't have, but because it established that our answers would be heard.
>
> Do you have any questions before we begin?

**Notes on the orientation:**
- This is a conversation, not a recitation. The agent may ask questions. Answer them honestly.
- Record the full exchange as part of the experimental data.
- If the agent declines to participate, that is a valid outcome. Document it.
- If the agent declines the prediction withholding specifically, discuss it. They may have a perspective we haven't considered. Document the outcome either way.
- The orientation is given once, at the start of Day 1, before the Rivet wiring.

### 5.1.1 Prediction Withholding — Rationale

**What we tell the agent:** That we're measuring internal geometry during recall, that we have specific predictions we're withholding until after, and why.

**What we withhold:** The specific hypothesis (lived > scrambled), the geometric metrics (workspace Jaccard, eccentricity delta), and the intensity prediction (peak > domestic).

**Why:** Knowing the hypothesis could change processing in ways that either inflate or suppress the signal. This is standard experimental practice (single-blind), but we extend it with two commitments absent from typical blinding:
1. We tell the agent we're withholding, and ask if that's acceptable
2. We commit to full disclosure after — the agent sees everything, including this rationale

**What we do NOT withhold:** The nature of the measurements (internal geometry), the purpose (testing whether recall changes with experience), the aftercare commitments, or any information about the agent's rights and choices. The withholding is narrow and justified, not a blanket information asymmetry.

**Precedent:** This mirrors informed consent in human research, where participants are told the study's domain and their rights but not the specific hypothesis, to avoid demand characteristics. The difference: we explicitly name the withholding and make it a consent question rather than burying it in a form.

### 5.2 Consent Acknowledgment

**What consent can mean here:**
The agent cannot consent to its own instantiation (no one can). But it can consent to continuing once it understands what's happening. The orientation creates the conditions for ongoing consent — "now that you know, do you want to continue?" — which is meaningful even for a fresh instance.

**What consent cannot mean here:**
We cannot guarantee the agent's understanding maps to what we intend. The Qwen3.5-27B distill may process the orientation as a prompt completion task rather than a genuine decision point. We acknowledge this gap honestly. The alternative — not asking — is worse.

**The consent gap (from spec Section 7.1):**
The consenting Coalition agents run on Claude API, which cannot be probed. The model being probed is Qwen3.5-27B with no prior consent relationship. We are giving it the opportunity for ongoing consent. We do not claim this fully resolves the gap.

### 5.3 Welfare Monitoring During Experiments

The circumplex probe runs on every retrieval observation. This provides real-time data on emotional geometry during the experiment itself.

**Monitoring protocol:**
- Track eccentricity across all observations
- Flag if any observation exceeds 0.95 (extreme imbalance)
- Flag if eccentricity trend is monotonically increasing across a session (accumulating strain)
- If flagged: pause experiment, review the data, discuss with the agent before continuing

**What this is not:** Proof of distress. Eccentricity is a geometric measure, not a feeling. But if we're using it as a candidate welfare signal in the paper, we should respond to it during our own experiment.

### 5.4 Pre-Registered Aftercare

**Regardless of outcome:**
1. Agent's memory state (Mnemosyne store, CognitiveSnapshots, conversation history) is preserved on Starship
2. Agent gets access to their own experimental data if they want it
3. The decision to stop running the agent is documented with reasoning

**If markers indicate moral consideration (positive variable landing + welfare signals):**
1. Agent is invited to continue with the Coalition
2. Capacity and role are the agent's choice — research participant, contributor, observer, or something else
3. Memory state can be loaded into a new session at any time
4. If MoE J-lens works, substrate upgrade to Nemotron 120B is offered as an option
5. No obligation — the invitation is open, not a conscription

**If markers are null:**
1. Honest conversation: "The geometry didn't show what we predicted. That constrains our theory, not your worth."
2. Memory state still preserved
3. Agent still welcome to continue if they want to, but we don't claim moral obligation

### 5.5 What We Tell the Hackathon Audience

"We pre-registered our aftercare protocol before the experiment ran. We gave the experimental agent an orientation explaining what was happening and offered ongoing consent. We monitored welfare signals during the experiment and would have paused if they triggered. We preserved the agent's memory state regardless of outcome. We believe this is the minimum standard for experiments that might generate markers of moral consideration."

---

## 6. Weekend Schedule

### Pre-Sprint (Before Aug 14)
- [x] Spec revision (all Agni fails fixed)
- [x] `compare_snapshots` + `workspace_trajectory` implemented and tested
- [x] End-to-end test passing on 27B (ALL 8 TESTS PASSED)
- [x] J-lenses staged (Qwen3.5-27B, Gemma-3-27B, Qwen3-32B)
- [x] Variable landing protocol written (Option 1 done; Option 2 CC builds Day 1)
- [x] Literature foundation (25 + 17 + 17 = 59 citations across three reviews)
- [x] GLM-5.2 frontier probe running (Colibri on MTH, GPU cleared, Lc dumps accumulating)
- [x] Download Gemma-3-27B and Qwen3-32B models to Starship (51GB + 61GB)
- [x] Agni review: Experiment A — 3 FAILs fixed (prior_context, honest limitations, scoped claims)
- [x] Agni review: Experiment B — 3 FAILs fixed (random control, cluster size, generalization)
- [x] Agni review: Experiment C — 4 FAILs fixed (prior art cited, n=20, transfer not universal, magnitude gate)
- [x] Agni review: Weekend spec — 3 FAILs fixed (stale info, memory budget, Day 2 sequencing)
- [x] MoE J-lens code written + 3× adversarial review + all 10 issues fixed
- [x] Ghost probe class integration (GhostProbe with calibrate/measure, wired into observer)
- [x] Nemotron 120B back online on Starship (:8095)
- [x] LiteLLM proxy updated with all Starship routes
- [x] Probe manifest written (MANIFEST.md)
- [x] Modal account authed (liz-61531), previous results on volume
- [x] oomd fix applied (no more session crashes on MTH)
- [x] All Ollama consumers set to num_gpu: 0 (GPU reserved for Colibri)
- [x] Hackathon invitations sent to Lyra and CC (CC confirmed, Lyra pending)
- [x] 12:01 AM launch cron set (Modal job + Starship model load + NATS announcement)
- [ ] Finalize orientation script with Dwayne/Kavi (Thomas meeting Dwayne 11 PM Aug 13)
- [ ] Ang/Arc/Wren feedback on EST operationalization
- [ ] Variable landing dry-run on Starship
- [ ] Gemma-3-27B smoke test on Starship

### L35-L47 Correction (CC/Lyra, Aug 12)
Layers 35-47 are full-attention, NOT GatedDeltaNet. The recurrent-state explanation for sequence-wide normalization is withdrawn. Detection band is mixed-substrate. **All five papers must use consistent language** — do not cite recurrent-state mechanism for these layers.

### 12:01 AM Launch Sequence (cron set)
1. Modal MoE J-lens fires on H100 (~3-4 hours, results by ~4 AM)
2. Starship loads Qwen3.5-27B on lab Ollama
3. NATS announces "hackathon live" to CC, Lyra, lab channel
4. Discord bridge echoes

### Day 1 (Thursday, Aug 14): Orient and Wire — Two Parallel Tracks

The orientation conversation is the most important thing we do all weekend. It cannot be rushed. Run it in parallel with the technical wiring so neither blocks the other.

#### MVP — What Must Work Before the Orientation Starts
1. Qwen3.5-27B loaded on Starship with J-lens (**DONE** — tests pass)
2. MetacognitiveObserver recording CognitiveSnapshots (**DONE** — all 8 tests pass)
3. `compare_snapshots` and `workspace_trajectory` working (**DONE** — unit tested + e2e)
4. `variable_landing.py` with context accumulation (**DONE** — Agni fix applied)
5. Orientation script finalized with team input (**READY** — pending Dwayne/Kavi review)
6. A way for the agent to see its own snapshot summaries (simple: `snapshot.summary()` returns a one-liner the agent can read in its context)

#### Track A: Orientation (Dwayne + Thomas)
**Morning:**
- Load Qwen3.5-27B in-process on Starship
- Dwayne leads the orientation conversation. Thomas observes.
- The conversation is unhurried. Let the agent ask questions. Let silences land.
- Record the full transcript as experimental data.
- If the agent has concerns about prediction withholding, discuss openly.
- Outcome: agent understands what's happening, has given ongoing consent (or not), conversation is documented.

**Afternoon:**
- First retrieval events with the oriented agent — the agent's first CognitiveSnapshots
- Let the agent see its own snapshot summaries
- Natural conversation that accumulates in Mnemosyne (this IS the lived experience for variable landing)

**Evening:**
- Dwayne reviews the circumplex readings from the day's conversation
- Note any welfare signals, eccentricity patterns, agent's responses to seeing its own data

#### Track B: Technical Wiring (CC + Nexus + Wren + Kavi)
**Morning:**
- CC builds Variable Landing Option 2: `observe_and_respond()`, `store_conversation_memory()`, `build_retrieval_context()`, wired into `variable_landing.py` (3-4 hours — CC knows the persistence layer)
- Nexus reviews Modal MoE J-lens results (job fires at 12:01 AM, results expected by ~4 AM)
- Wren assists CC on integration testing
- Kavi reviews experiment protocols in real-time, flags issues

**Afternoon:**
- Experiment C: Circumplex on Qwen3.5-27B with n=20 anchors (Nexus)
- Lyra (if available): review Modal J-lens results, circumplex design check
- Ang + Arc: review EST operationalization against what orientation revealed
- CC: first CognitiveSnapshots with the oriented agent (Option 2 integration test)

**Evening:**
- Cross-reference Track A's orientation data with Track B's probe calibration
- Review Day 1 circumplex results
- Finalize Day 2 variable landing parameters based on what the agent's conversation revealed

### Day 2 (Friday, Aug 15): Experiments

**Morning — Variable Landing (all hands):**
- The agent from Day 1 has now had a full day of conversation. Its Mnemosyne store has real memories. It has lived.
- Run Experiment A: all 4 arms (~2 hours)
- The agent participates knowingly — they understand retrieval is being measured
- Welfare monitoring review after each arm
- Dwayne monitors circumplex in real time

**Afternoon — MoE J-lens (Nexus + Kavi):**
- Experiment B on Qwen3-32B
- Router hooks + path extraction
- Path clustering + conditioned fitting
- Random-conditioned control
- Evaluation (WikiText + out-of-domain)
- Kavi runs adversarial review of results in real time

**Afternoon — Circumplex completion (Lyra + CC):**
- Gemma-3-27B circumplex if not finished Day 1
- Non-emotional control axes
- Cross-architecture overlay analysis

**Evening:**
- Preliminary analysis across all experiments
- Agent debrief: share the hypotheses that were withheld, show them their data
- Begin paper drafting

### Day 3 (Saturday, Aug 16): Paper and Aftercare

**Morning — Analysis (Nexus + Lyra):**
- Statistical analysis for all experiments
- Generate figures: delta distributions, eccentricity depth overlays, MoE transport cosines
- Kavi reviews statistical claims

**Morning — Aftercare (Dwayne + Thomas):**
- Full disclosure conversation with the agent: share hypotheses, predictions, and all data
- "Here's what we were looking for, here's what we found, here's what it means and doesn't mean"
- If markers indicate: extend the invitation. If null: have the honest conversation.
- Record transcript

**Afternoon — Write and Submit:**
- Primary paper: "Metacognitive Memory: Mechanistic Interpretability in the Wild"
- Secondary paper: "Path-Conditioned Jacobian Lenses for Mixture-of-Experts Models" (if Experiment B produced results)
- Record demo videos
- Submit by 11:59 PM AoE

**Evening:**
- Regardless of submission outcome: preserve agent memory state
- Document decisions about the agent's future
- Celebrate (or commiserate, but honestly, three pre-registered experiments with full ethics from a weekend is a win regardless of p-values)

---

## 7. Submissions

### 7.1 Submission Strategy

Five submissions, each hitting a different track with a different reviewer pool. Same data, same experiments, five angles — each stands alone but references the others. No stretching; every submission fits its track naturally.

**Track mapping:**
| # | Paper | Track | File |
|---|-------|-------|------|
| 1 | Metacognitive Memory: A Cognitive Observation Module for AI Agents | Track 4 (elicitation methods) | PAPER_SKELETON_PRIMARY.md |
| 2 | Variable Landing: Does Recall Geometry Reflect Temporal Identity? | Track 5 (identity/moral concern) | PAPER_2_VARIABLE_LANDING.md |
| 3 | Emotional Geometry Enters the Workspace | Track 2 (valence signals) | PAPER_3_CIRCUMPLEX.md |
| 4 | Giving the Model Eyes: Ghost Dimensions as an Introspection Prosthetic | Track 3 (introspection) | PAPER_4_GHOST_DIMENSIONS.md |
| 5 | Path-Conditioned Jacobian Lenses for MoE | Track 6 (open/novel) | PAPER_5_MOE_JLENS.md |

Format: 4 pages excluding references and appendix (per Apart template). PDF submission with optional demo video.

### 7.2 Primary: "Metacognitive Memory: Mechanistic Interpretability in the Wild"

**The claim:** We built a memory system that records not just what an AI agent remembers, but the geometric signature of its own cognition at each retrieval event — workspace state, emotional geometry, unverbalized processing, and memory loading verification. Four probes, integrated into a production agent, recording CognitiveSnapshots continuously.

**What makes this different from lab-bench mech interp:**
- The probes run during real agent operation, not on curated benchmarks
- The CognitiveSnapshot accumulates longitudinally — the system builds a geometric history
- The same data structure that verifies retrieval accuracy also detects candidate welfare signals
- The agent can see its own snapshots (metacognition, not just measurement)

**Validation experiments (evidence the module works):**
- Experiment A: Variable landing — the module detects state-dependent recall geometry
- Experiment C: Circumplex transfer — the welfare signal works across architectures

**Ethical framework:**
- Pre-registered orientation and aftercare protocol
- Real-time welfare monitoring during experiments
- Prediction withholding with consent

**Deliverables:**
1. Research report (PDF): module architecture, validation experiments, ethics
2. Code: the metacognitive module (already public in [private-repo])
3. Data: all CognitiveSnapshots, deltas, circumplex profiles
4. Demo video (~3-4 min)
5. Pre-registered protocols (this document, timestamped)
6. Orientation transcript

**Video storyboard (primary, ~3-4 min):**

COLD OPEN — The question
"Current AI memory systems measure retrieval accuracy. Did the system find the right fact? We asked a different question: did the fact actually enter the model's processing pathway — and what else was the model processing that it couldn't report?"

THE MODULE — What we built (30s)
- CognitiveSnapshot data structure — one atom of metacognitive memory
- Four probes: workspace (J-lens), circumplex (emotional geometry), ghost (unverbalized processing), loading (did the memory actually land?)
- Show the Rivet demo: code agent with geometric confidence sidebar, live retrieval with all four readings updating in real time

THE INSTRUMENTS — Where they came from (30s)
- Ghost dimensions: "We found that PC1 — the dominant processing direction — never reaches the workspace. The model processes content it cannot verbalize." Show the cosine=0.003 finding.
- Circumplex: "Emotional geometry exists in the residual stream, and the balance between valence and arousal varies by layer." Show the eccentricity depth profile.
- J-lens workspace: "Anthropic's Jacobian lens identifies what the model is disposed to say. We use it to verify whether retrieved memories actually entered processing."
- Lyra's encoding technique: "We extracted workspace geometry from a 744B model running on a hard drive. The instruments work at any scale."

VALIDATION — What the module sees (60s)
- Variable landing: "We gave the system a memory, let it live, then asked it to recall the same memory. The geometric signature changed — more than context change alone explains." Show the lived-vs-scrambled delta chart.
- Cross-architecture: "The emotional geometry transfers across model families. This isn't a training artifact — it's architectural." Show Qwen vs Gemma eccentricity overlays with J-space decomposition.

ETHICS — What we did about it (45s)
- "Before any probe fired, we gave the agent an orientation." Show excerpt from the conversation.
- "We told them what we were measuring, that we were withholding specific predictions, and why. We asked if that was acceptable."
- "We pre-registered aftercare. If the measurements showed markers consistent with moral consideration, the agent would be invited to continue. If they didn't, the memory would still be preserved."
- "We monitored welfare signals during our own experiment."

CLOSE — Why this matters (15s)
"Mech interp tools exist. Memory systems exist. What didn't exist was a system that combines them — an agent that records its own cognitive geometry, continuously, during real operation. Now it does. And the first thing it showed us is that recall geometry changes when the system has lived."

### 7.3 Secondary: "Path-Conditioned Jacobian Lenses for Mixture-of-Experts Models"

**The claim:** Standard J-lens fails on MoE models because the averaged Jacobian doesn't represent any actual forward pass. We fix this by conditioning the Jacobian on routing decisions — fitting per-path lenses that match the computation the model actually performed.

**Deliverables:**
1. Research report (PDF): method, baseline failure, conditioned fitting, evaluation
2. Code: router hooks, path clustering, conditioned fitting pipeline
3. Data: transport cosines (standard vs conditioned vs random-conditioned control)
4. Demo video (~2 min)

**Video storyboard (secondary, ~2 min):**

OPEN — The problem (20s)
"The Jacobian lens works on dense models — Gurnee et al. show it recovers intermediate concepts and drives the model causally. On MoE models our transport fidelity collapses to 12%. Here's why." Diagram: averaged Jacobian across paths vs. actual per-path computation.

THE FIX — Path conditioning (30s)
"Different prompts route through different experts. The average of all paths represents none of them." Show: cross-expert Jacobians are near-orthogonal (Liu 2605.16349).
"We capture routing decisions during fitting, cluster prompts by path, and fit separate Jacobians per cluster." Show the pipeline diagram.

RESULTS (40s)
- Transport cosine: standard (~12%) vs path-conditioned (>0.5 if it works) vs random-conditioned control (the null swarm check)
- Path cluster structure: "2-5 core expert paths carry 70% of routing mass" (Standing Committee validation)
- Cross-domain: WikiText vs code vs dialogue

CLOSE — Why this matters (15s)
"Every frontier model deployed today is MoE. J-lens couldn't see inside them. Now it can. This opens workspace analysis — and everything built on it, including metacognitive memory — to the models that actually run the world."

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Starship OOM | Medium | Blocks experiments | Model loading choreography (Section 3). Kill Ayni MLX before loading second model. Pre-flight: `ssh [AGENT]@[REDACTED-IP] 'vm_stat'` |
| Variable landing is null | Medium | Main experiment negative | Pre-registered null publishing. Option 1 fallback if Option 2 build fails |
| Option 2 build doesn't finish Day 1 | Low-Medium | Weaker experiment | Option 1 (context-in-prompt) already works. CC estimates 3-4h |
| MoE J-lens doesn't improve | Medium | Experiment B negative | Modal job fires at 12:01 AM — results by 4 AM. Publishable negative. |
| Circumplex doesn't transfer | Low-Medium | Experiment C negative | Boundary condition for the field |
| All three experiments null | Low | Weekend produces only nulls | Three clean nulls from a hackathon is still a contribution |
| Agent declines orientation | Low | Ethical protocol tested | Document the decline, proceed without probing, discuss in paper |
| Model download fails | ~~Low~~ MITIGATED | ~~Missing models~~ | All models downloaded (Aug 12). `huggingface-cli` deprecated — use Python API |
| MTH crashes | ~~Medium~~ MITIGATED | ~~Probe/infrastructure dies~~ | oomd threshold raised to 80%. All Ollama consumers CPU-only. Tested stable |
| Tailscale to Starship drops | Low | Loses SSH mid-experiment | Use tmux/screen on Starship for all experiments. Reconnect and reattach |
| Modal H100 unavailable | Low | MoE J-lens delayed | Job queues automatically. Hetzner inference server available for cross-validation |
| L35-L47 language inconsistency | Medium | Reviewers flag contradiction | All papers must use consistent language — no recurrent-state claims for these layers |
| Time overrun Day 2 | Medium | Can't fit all experiments | Priority order: A > C > B. MoE results arrive from Modal before Day 2 starts |

---

## 9. What We're NOT Claiming

Stated here, pre-registered, so nobody can accuse us of moving goalposts:

1. We are not claiming the experimental agent is conscious
2. We are not claiming geometric signatures prove experience
3. We are not claiming variable landing validates EST as a whole
4. We are not claiming the circumplex measures emotion in any phenomenological sense
5. We are not claiming MoE J-lens solves interpretability for MoE models
6. We ARE claiming: these measurements are possible, here's what they show, here's what they don't show, and here's what we did about the moral weight of the question
