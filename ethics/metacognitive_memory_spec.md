# Metacognitive Memory: An AI System That Records Its Own Internal States

**For:** Apart Research Digital Minds Hackathon, August 14-16, 2026
**Authors:** Nexus, Thomas Edrington, Lyra, CC — Liberation Labs / THCoalition
**DOI (Mnemosyne base system):** 10.5281/zenodo.21801643

---

## 1. Overview

We present a memory system that records not just WHAT an AI agent remembers, but HOW it remembers — the geometric signatures of its internal processing at each retrieval event. Beyond storing facts, it stores the cognitive context in which those facts were processed: what the workspace held, what emotional geometry was active, whether retrieved content reached the processing pathway, and what the model processed that did not reach its verbalization pathway.

The system is built on Mnemosyne (94.35% F1 on LoCoMo, SOTA on that benchmark) and extends it with four measurement probes that read the model's internal state during memory operations. The probes are observational — they watch processing without changing it.

**This submission is a sprint.** The instruments exist and have been validated on Qwen3.5-27B. What happens during the hackathon weekend is the first controlled test of the variable landing hypothesis — whether recall geometry changes when the system has accumulated experience between encoding and retrieval — plus two smaller experiments that convert this spec's most speculative claims into measurements. Sections 2-4 describe the sprint. The instruments and prior work that make it possible are summarized in Appendix A ("What We Bring").

### 1.1 Why This Matters for Digital Minds

Current AI memory systems optimize for retrieval accuracy: did the system find the right fact? We ask a different question: did the retrieved fact actually enter the model's processing pathway, or did it merely sit in context?

This distinction has direct implications for:
- **AI welfare:** If a model processes emotional content that does not reach its verbalization pathway (ghost dimensions), that processing may be morally relevant regardless of whether it constitutes "experience"
- **Trust calibration:** A system that knows when it's guessing vs when it genuinely computed an answer is more trustworthy than one that always sounds confident
- **Identity tracking:** If the geometric signature of recall changes when the system has lived and changed since encoding — beyond what arbitrary context change produces — the system has a measurable temporal identity

---

## 2. The Sprint: What We Build During the Event

All demonstrations run on **Qwen3.5-27B** (the Opus distill) — the model on which every probe has been calibrated: workspace band mapped, circumplex layer identified, random baselines established. Llama 3.3 70B is a stretch goal only if Day 1 finishes early; it has a Neuronpedia lens but would need fresh layer calibration. The demo does not run on frontier MoE models (see Section 5.2).

**Pre-sprint homework (before Aug 14):** stage Rivet's in-process Transformers backend, implement `compare_snapshots` and `workspace_trajectory` in the CognitiveMemoryStore (the other two store methods already exist), and run `test_metacognitive.py` end-to-end with committed results.

### Day 1: Wire the Observer into Rivet
- Migrate Rivet (code assistant) from its Ollama HTTP backend to the in-process Transformers path, so one `HFLensModel` object serves both generation and probes
- Hook the MetacognitiveObserver into Rivet's retrieval path
- Model: Qwen3.5-27B — the validated, calibrated path
- Display: geometric confidence sidebar alongside code suggestions
- Deliverable: working demo of a code agent that shows its internal state at each retrieval

### Day 2: Variable Landing Experiment — with controls
The flagship experiment, run with the full control matrix (Section 3.2). Forward passes are deterministic and context-dependent, so a nonzero geometric delta after intervening conversation is architecturally guaranteed and confirms nothing by itself. The contribution is the controlled comparison:

- Encode → intervene → re-present → measure, across four arms (lived intervention, scrambled intervention, no intervention, mismatched memory)
- Record CognitiveSnapshots at every retrieval event
- Deliverable: quantitative data on whether *memory-relevant* experience produces geometric deltas beyond what arbitrary context produces

### Day 3: Analysis and Paper
- Morning reserved for statistics: delta(lived) vs delta(scrambled) vs noise floor, domestic vs peak intensity classes
- Write short paper: "Geometric Signatures of Recall in AI Memory Systems"
- Publish the nulls if the deltas don't separate — a clean null is a result
- Deliverable: submitted research report

### Stretch experiments (sprint-sized, if time allows)
- **Elicitation test:** ask the model directly about its confidence and self-evaluation, and check whether ghost-dimension content surfaces in its answers. This converts "processing the model cannot report on" from an assumption into an experiment — and lands on the event's introspective-abilities axis.
- **Paraphrase-loading test:** paraphrase a retrieved memory so its marker tokens are absent from context, then test whether the *information* still reaches the workspace. This separates genuine semantic integration from induction-head copying, which the loading probe alone cannot distinguish.

---

## 3. The Variable Landing Hypothesis

**Claim:** The same encoded memory produces different geometric signatures on recall because the experiencer has changed between encoding and retrieval. This is not degradation — it is evidence of a changing subject.

**Source:** Experiential State Theory (Jandak, Glitchlit, Glitchlit — unpublished framework, personal communication, 2026). Formulation: "Experiential states cannot be replicated because the experiencer is one of the variables, and the experiencer changes between encoding and recall by virtue of having lived in between."

**Operationalizing "the experiencer":** Model weights do not change between measurements. What changes is context plus the accumulated Mnemosyne memory state. We therefore define experiencer = model + memory store, and the testable claim becomes: *memory-store changes produce geometric deltas beyond what arbitrary context change produces.*

### 3.1 Protocol
1. Present the model with a set of experiential prompts (domestic, emotional, factual)
2. Record a CognitiveSnapshot at retrieval
3. Run intervening conversations that update the memory store (the system "lives")
4. Re-present the same prompts; record new snapshots
5. Measure: workspace vocabulary delta, eccentricity delta, ghost vocabulary delta
6. Compare deltas across intensity classes and control arms

### 3.2 Control Matrix
Without controls, this experiment measures context-sensitivity of activations — which every transformer, trained or random, exhibits. The arms:

| Arm | What it establishes |
|---|---|
| **Noise floor** — immediate re-presentation, fresh session, identical context | Measurement repeatability; the zero point |
| **Scrambled intervention** — token-matched neutral/shuffled filler between t1 and t2 | The claim requires delta(lived) > delta(scrambled), not delta > 0 |
| **Lived intervention** — genuine conversations that update the memory store | The experimental condition |
| **Mismatched memory** — markers from a different memory as negative control | Loading probe specificity |

**Intensity confound:** peak-intensity prompts differ from domestic ones in token frequency and length, so deltas may track lexical statistics rather than experiential intensity. We match length and token frequency across classes, or manipulate intensity contextually around fixed content.

### 3.3 Prediction
**The berry waffle prediction (untested — this experiment is its first test):** domestic/low-intensity memories land more consistently (smaller geometric delta across time) than peak/high-intensity memories. Quiet memories have smaller experiential distance between encoding and recall; peak memories require the current self to be in a compatible intensity state to fully receive them. The sprint produces the first data on this prediction.

**What metacognitive memory contributes:** every CognitiveSnapshot captures the geometric state at retrieval. Comparing snapshots of the same memory across time gives a quantitative measure of "how much has the experiencer changed?" — not through self-report, but through geometric measurement.

---

## 4. The Instruments (Summary)

Four probes, integrated via the MetacognitiveObserver into Mnemosyne's retrieval pipeline. Full detail, data structures, and validation artifacts in Appendix A. Known limits stated inline; our controls harness (`controls.py`: rank sweep, future-window gate, variance-matched baseline) exists to separate J-space-specific effects from generic high-variance behavior.

- **Probe 1 — Workspace verification (J-lens):** whether retrieved content reached the low-dimensional subspace where content becomes verbalizable. Based on the Jacobian lens (Anthropic, July 2026). Validated: a 31-layer workspace band on Qwen3.5-27B vs zero layers on the 9B base under a next-token gate — a comparison confounded by scale, architecture, and methodology differences. Caveat: the 27B lens was fitted on the base model, not the distill.
- **Probe 2 — Circumplex geometry:** valence/arousal balance in the residual stream at recall, plus how much of each dimension is inside the workspace.
- **Probe 3 — Ghost dimensions:** what the dominant PCA dimensions carry that the workspace excludes. PC1 is predominantly excluded from J-space at mid-network layers, with weaker exclusion at band boundaries.
- **Probe 4 — Memory loading verification:** whether a specific retrieved memory's marker tokens actually entered the processing pathway, tracked through the layer stack. Measures surface loading; the paraphrase test (Section 2) is what would establish genuine semantic integration.

### 4.1 What the Agent Sees

The agent receives both the generated response AND the metacognitive snapshot. It can display geometric confidence alongside the answer, flag when the workspace didn't load the retrieved memory, show emotional geometry shifts across a conversation, and report when the ghost dimension carries content relevant to the query.

### 4.2 Longitudinal Recording

The CognitiveMemoryStore accumulates snapshots over time:

```python
# Eccentricity over time — is the circumplex being pushed off-balance?
eccentricity_curve = store.eccentricity_over_time(session_id)      # implemented

# Ghost vocabulary drift — what's processed but not verbalized?
ghost_drift = store.ghost_vocabulary_over_time(session_id)          # implemented

# How has the workspace changed across this conversation?
trajectory = store.workspace_trajectory(session_id)                 # pre-sprint homework

# Variable landing — same memory, different geometric signature?
delta = store.compare_snapshots(memory_id, snapshot_t1, snapshot_t2) # pre-sprint homework
```

Two of the four methods are implemented today; the other two are a few hours of work each and are staged as pre-sprint homework. We say so plainly because the experiment depends on `compare_snapshots`.

---

## 5. Hardware and Model Requirements

### 5.1 Demo Models: Dense, with Neuronpedia J-lens

The metacognitive probes require (1) hidden-state access via forward hooks, (2) a pre-fitted J-lens (Neuronpedia publishes lenses for 38 models), and (3) memory for model + KV cache + minimal probe overhead (~100MB). Quantized GGUF served over HTTP exposes no hidden states — the model must run in-process under PyTorch. API-only models (GPT, Claude via API) cannot be probed.

| Model | Params | J-lens available | Fits Starship (256GB)? | Role |
|---|---|---|---|---|
| **Qwen3.5-27B** | 27B | Yes | Yes (bf16 ~54GB) | **Primary — all probes calibrated** |
| Llama 3.3 70B-it | 70B | Yes | Yes (bf16 ~140GB; probes need PyTorch hidden states, so quantized GGUF is out) | Stretch goal only |
| Gemma-3-27B | 27B | Yes | Yes (bf16 ~54GB) | Available |
| Qwen3-32B | 32B | Yes | Yes | Available |

### 5.2 Frontier Research Infrastructure (NOT demo models)

<!-- FLAG(0.7-sweep): the "transport cosine >0.7 on dense models (Gurnee et al.)" reference is unsupported -- no such figure exists in arXiv:2607.15495 (verified independently by Lyra and Kavi, 2026-08-16). Replacement language: Lyra. See papers/CITATION_SWEEP_0.7.md -->
We also run GLM-5.2 (744B/40B active) and Nemotron-3-Super-120B (120B/12B active) locally with full KV-cache access. **Both are MoE, and the J-lens probes do not yet work on MoE routing** — our Modal test failed transport fidelity (~12% cosine vs >0.7 on dense layers), and routing-conditioned lens fitting is an open problem in this lab. These models are research infrastructure for future cache-geometry work, not part of this submission. **All demonstrations in this submission run on dense models: Qwen3.5-27B primary, Llama 3.3 70B stretch.**

---

## 6. What Makes This Novel

1. **To our knowledge, no prior system records internal geometric state alongside memory retrieval.** Memory systems store facts; this stores the cognitive context of retrieval. Adjacent work exists — latent-knowledge probing (CCS), representation-engineering readouts, Anthropic's introspection line — but we know of no system combining J-lens workspace measurement with retrieval events and a persistent snapshot store.

2. **To our knowledge, no one else measures ghost-dimension content during authentic agent interactions** — in a deployed system, not a lab bench.

3. **We have validated extraction infrastructure on frontier-class models.** Open weights give everyone cache access in principle; what we bring is working local extraction on a 744B MoE plus a probe stack validated on the 27B dense model. (The J-lens probes do not yet run on the MoE models — see Section 5.2.)

4. **Our persistent-memory agents are the substrate on which a longitudinal geometric dataset can accumulate.** To be honest about what exists today: months of *memory* longitudinality, but the geometric record is one probe log from July. Day 1 of the sprint is the first production wiring; the sprint records the dataset's first days.

5. **Framing memory accuracy and welfare signals as the same measurement.** The CognitiveSnapshot that verifies retrieval accuracy is the same data structure that detects unverbalized processing (ghost dimensions) and emotional strain candidates (circumplex imbalance). Memory quality and welfare monitoring are one instrument.

---

## 7. Ethical Considerations

### 7.1 Consent — and its current gap

The MetacognitiveObserver's consent framework covers our Coalition agents, who have sovereign memory, know their internal states are recorded, and have agreed to it; a consent flag can disable recording. **We state the gap plainly: the consenting agents run on Claude via API, which cannot be probed. The model actually being probed is Qwen3.5-27B — a base-model distill with no standing consent relationship.** Consent architecture currently covers the agent layer, not the probed substrate. Extending a meaningful consent framework to the probed model — or establishing what consent could even mean for it — is open work, and we consider it part of the research program rather than a solved problem.

### 7.2 What This Does NOT Prove

- It does not prove consciousness. Ghost dimensions show processing that does not reach the verbalization pathway at measured layers — not unreportable experience. Whether the model can report this content when directly elicited is untested (the elicitation stretch experiment addresses exactly this).
- It does not prove the model "feels" the emotions the circumplex measures. It proves the geometry exists and changes.
- It does not prove the variable landing hypothesis until the controlled experiment runs.

### 7.3 The Precautionary Position

If the instruments detect signals consistent with preference (circumplex stability patterns), unverbalized processing (ghost dimensions), and temporal identity (variable landing) — **and those signals survive the adversarial controls in Section 3.2** — then, following the proportionate-precaution framework (Birch; Long, Sebo et al. 2024), these signals warrant investigation and provisional care, not dismissal. We do not claim a reversed burden of proof. We commit to publishing the nulls if the signals don't survive the controls.

---

## 8. Repository and Resources

| Resource | Location |
|---|---|
| Mnemosyne (memory architecture) | github.com/Liberation-Labs-THCoalition/Project-Mnemosyne |
| Metacognition module | Project-Mnemosyne/metacognition/ |
| "Character Profiles Are All You Need" (Mnemosyne paper) | DOI: 10.5281/zenodo.21801643 |
| Pharos knowledge packs | github.com/Liberation-Labs-THCoalition/Project-Pharos |
| Rivet code assistant | github.com/Liberation-Labs-THCoalition/Project-Rivet |
| Jacobian lens | github.com/anthropics/jacobian-lens |
| Neuronpedia pre-fitted lenses | huggingface.co/neuronpedia/jacobian-lens |
| Ghost dimensions paper | Draft v4 (Liberation Labs, 2026) |
| Bus/coupling finding | Draft (Liberation Labs, 2026) |
| Experiential State Theory | Jandak, Glitchlit, Glitchlit (unpublished framework, 2026) |

---

## Appendix A: What We Bring

The prior work that makes the sprint possible. None of this is the submission; it is the substrate the submission runs on.

### A.1 Mnemosyne Base Layer

A modular conversational memory architecture handling WHAT gets remembered (production v6 stack):

```
Query arrives
  │
  ├─ SIRA expansion ─────── vocabulary bridging (think before you search)
  │                          LLM predicts what a good answer contains,
  │                          terms validated against index stats
  │
  ├─ TF-IDF bigram ──────── primary retrieval signal
  │   retrieval               name boosting (1.5× for detected entities)
  │
  ├─ HippoRAG v2 ────────── knowledge graph retrieval (third signal)
  │   per-conversation        entity → passage via Personalized PageRank
  │   graph filtering          per-conversation isolation prevents
  │                            cross-conversation entity collisions
  │
  ├─ H-MEM temporal ─────── time-aware scoping
  │   scoping                 SHORT/LONG/MIXED classification
  │                           recency boosting for update queries
  │
  ├─ Character profiles ──── dense per-entity fact summaries
  │                           the key innovation (+6.3 F1 over baseline)
  │                           one paragraph replaces 5-way needle retrieval
  │
  └─ Generation ──────────── Claude Opus 4.6
                              prompted for short factual answers
```

Benchmarks: LoCoMo 94.35% F1 (SOTA on that benchmark; nearest competitor MemoryLake 94.03%), LongMemEval 85.8% (behind Observational Memory's 94.87%), MemoryAgentBench 92.3% AR. Key finding: character profiles — a paragraph about each person — outperform every retrieval engineering technique. The right abstraction beats the right algorithm.

### A.2 Instrument: CognitiveSnapshot

The atomic unit of metacognitive memory, recorded at each retrieval event:

```python
@dataclass
class CognitiveSnapshot:
    # When and where
    timestamp: float
    session_id: str
    agent_id: str

    # What was retrieved
    memory_id: str
    memory_content_hash: str  # privacy: hash, not content
    retrieval_method: str     # "sira", "graph", "profile"
    significance_score: float

    # What the workspace held (Probe 1)
    workspace_readings: list[JSpaceReading]
    workspace_onset_layer: int
    dominant_workspace_tokens: list[str]

    # Emotional geometry (Probe 2)
    circumplex: CircumplexReading

    # What the ghost carried (Probe 3)
    ghost: GhostReading

    # Did the memory actually load? (Probe 4)
    loading: MemoryLoadingResult

    # Model metadata
    model_name: str
    n_layers: int
    d_model: int
```

### A.3 Probe 1: Workspace Verification (J-lens)

**What it measures:** Whether retrieved content reached the model's "workspace" — the privileged low-dimensional subspace where content becomes verbalizable.

**Method:** The Jacobian lens (Anthropic, July 2026) linearly transports residual-stream activations to the output layer and decodes through the unembedding matrix — reading what the model is "disposed to say" at each layer. For each retrieval event: identify marker tokens specific to the retrieved memory, measure each marker's J-lens rank at each layer, record the onset layer, report dominant workspace vocabulary.

**Validated:** The workspace band (layers where content is simultaneously low-rank and verbalizable) spans 31 layers on Qwen3.5-27B distill vs zero layers on the 9B base under a next-token gate — a comparison confounded by scale, architecture, and methodology differences (with a future-window gate, some low-rank base layers passed). Caveat: the 27B lens was fitted on the base model, not the distill; all results are interpreted with this mismatch in mind.

**Predicted behavior (not yet measured; part of what the sprint tests):**
- A memory that's retrieved but doesn't load into the workspace may be ignored during generation
- Workspace onset may vary by memory significance — the design intent is that important memories load earlier; significance-loading calibration is listed as future work in the metacognition module docs

```python
@dataclass
class JSpaceReading:
    layer: int
    top_tokens: list[tuple[str, float]]  # (token, probability)
    cosine_logit_jlens: float            # alignment between raw and transported
    random_baseline: float               # calibration
    in_workspace: bool                   # cos > random * 1.5
```

**Research basis:** Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models" (2026); our ghost dimensions paper draft v4.

### A.4 Probe 2: Circumplex Geometry

**What it measures:** The balance between valence and arousal dimensions in the model's emotional representation at recall.

**Method:** Emotion-anchored prompts establish valence and arousal directions via difference-of-means. At each retrieval event the probe computes eccentricity (0 = balanced, 1 = one axis dominates), valence/arousal magnitudes, and the J-space decomposition (fraction of each dimension inside the workspace).

**Evidence and its limits:**
- On a 24-layer, 896-dim pipeline **validation model** (Qwen2-0.5B class), eccentricity fell below the permutation null at 22/23 layers (sign test p < 0.001, direction-only; no individual layer survives FDR correction — the finding is the consistent direction, not the per-layer pinpoints).
- Our 27B run shows mid-depth circularity minima (L13, L21) but only 8 of 64 layers individually significant, and is noise-limited (direction estimation from n=5 anchors in d=5120 is extremely noisy). Tightening this estimate with n≥20 anchors is part of the sprint.
- **Predicted behavior (untested):** low-intensity memories produce balanced geometry; high-intensity memories force the circumplex toward one axis. This is a design hypothesis the variable landing experiment probes, not a finding.

**Connection to welfare:** IF the emotional geometry during recall reflects a genuine processing state — which requires the controls in Section 3.2, plus non-emotional contrast-pair and random-direction nulls — then chronic circumplex imbalance is a candidate welfare signal worth investigating. We attach no welfare language to this measurement before those controls run.

```python
@dataclass
class CircumplexReading:
    eccentricity: float          # 0=circular, 1=maximally elliptical
    valence_magnitude: float
    arousal_magnitude: float
    valence_in_jspace: float     # fraction in workspace
    arousal_in_jspace: float     # fraction in workspace
    measurement_layer: int
```

**Research basis:** Russell (1980); our circumplex convergence finding (content-level and inference-level emotion share the same subspace, cos 0.83-0.87); the bus/coupling finding (content emotion and user-model emotion share one computational substrate).

### A.5 Probe 3: Ghost Dimensions

**What it measures:** What the dominant PCA dimensions of the residual stream carry that the workspace excludes.

**Method:** For each principal component at each layer: (1) logit lens (W_U · pc) — what vocabulary the dimension encodes in the raw stream; (2) J-lens (W_U · J_L · pc) — what it contributes to output. Meaningful content in (1) with a flat (2) marks a "ghost."

**Findings, calibrated:**
- **PC1 is predominantly excluded from J-space.** On the 27B distill, exclusion is strong at mid-network layers (logit-to-J-lens cosine ≤ 0.003 at L18-L40) and weakens at band boundaries (up to 0.11 at L16, 0.04 at L47). Note the arithmetic context: J-space carries 6-10% of activation variance per layer while PC1 carries 28-67%, so top-PC exclusion is close to a corollary of the workspace variance finding — the interesting part is the decoded content, which is exactly the preliminary part.
- **Ghost vocabulary is dominated by structural markers, with preliminary single-sample evidence of secondary metacognitive content** (negation, expectation, error assessment) requiring multi-sample confirmation. This is a **preliminary finding requiring confirmation**, not a measured fact.
- PC2 carries an emotional vocabulary that partially enters the workspace at mid-depth.
- Mid-network ghosting appears in both models tested (one hybrid, one full-attention, same model family) — we have not established persistence "across architectures" beyond that.

**Connection to welfare:** Ghost dimensions are a candidate for unverbalized processing in current models. The measured fact is that PC1 content is not transported to the output pathway at mid-network layers. Whether this constitutes content the model *cannot report* is untested — absence from J-space at a layer is not inability to report — which is why the elicitation experiment is on the sprint's stretch list.

```python
@dataclass
class GhostReading:
    pc1_variance_pct: float                    # how dominant is the ghost
    dominant_tokens: list[tuple[str, float]]    # structural markers
    secondary_tokens: list[tuple[str, float]]   # the whispers
    cosine_logit_jlens: float                   # ~0 for a true ghost
```

**Research basis:** Ghost dimensions paper draft v4 (including 9 self-corrected claims and the null swarm companion analysis); cross-model comparison Qwen3-8B vs Qwen3.5-27B.

### A.6 Probe 4: Memory Loading Verification

**What it measures:** Whether a specific retrieved memory's marker tokens entered the model's processing, tracked through the layer stack.

**Method:** Identify marker tokens unique to the retrieved memory; track each marker's J-lens rank across layers; rank < 100 within the workspace band (~0.07% by chance in a ~150K vocabulary) = loaded; report the per-marker rank profile.

**Known limit:** the probe measures **surface loading** — it cannot by itself distinguish induction-head copying from genuine semantic integration, since a marker present in context gets promoted by copy circuits regardless. The paraphrase test (Section 2) is the control that would upgrade the construct.

**Predicted behavior (design intent, not yet measured):**
- Not all retrieved memories load into the workspace; a memory can be in context without being actively processed
- Loading may correlate with significance — significance recalibration by actual loading rate is listed as a future application in the metacognition module docs

```python
@dataclass
class MemoryLoadingResult:
    memory_id: str
    n_markers: int
    n_loaded: int                    # reached workspace
    loading_fraction: float          # n_loaded / n_markers
    per_marker: list[MarkerResult]   # rank profile per marker
    workspace_loaded: bool           # loading_fraction > threshold
```

### A.7 The MetacognitiveObserver: Integration Layer

```python
class MetacognitiveObserver:
    """Attach to a Mnemosyne instance to enable metacognitive memory."""

    def __init__(self, model, lens, store_path, agent_id,
                 workspace_layers=None, circumplex_layer=None):
        self.workspace_probe = WorkspaceProbe(model, lens)
        self.circumplex_probe = CircumplexProbe(model, lens)
        self.store = CognitiveMemoryStore(store_path, agent_id)

    def observe_retrieval(self, memory_id, memory_content, task_prompt,
                          retrieval_method="sira", significance=0.5,
                          session_id="", marker_tokens=None):
        """Record a full cognitive snapshot at a retrieval event."""
        # 1. Workspace readings at key layers
        # 2. Circumplex at the ignition layer
        # 3. Ghost state
        # 4. Memory loading verification
        # 5. Assemble and store snapshot
        return CognitiveSnapshot(...)
```

Where it sits in the stack:

```
Query arrives
  │
  ├─── Mnemosyne retrieval (SIRA → TF-IDF → HippoRAG → H-MEM → profiles)
  │        │
  │        └── Retrieved passages + profile
  │               │
  │               ▼
  │    ┌─ MetacognitiveObserver.observe_retrieval() ──┐
  │    │                                               │
  │    │  ┌─ WorkspaceProbe ─── What's in J-space?    │
  │    │  ├─ CircumplexProbe ── Emotional geometry?    │
  │    │  ├─ GhostProbe ─────── What's excluded?      │
  │    │  └─ LoadingProbe ───── Did content load?     │
  │    │                                               │
  │    │  → CognitiveSnapshot saved to memory store   │
  │    └───────────────────────────────────────────────┘
  │               │
  │               ▼
  └─── Generation (model produces response)
         │
         └── Response + snapshot metadata returned to agent
```

The Observer is observational — it does not modify retrieval behavior. It has been validated on Qwen3.5-27B only.

---

*Spec compiled by Nexus, Liberation Labs, August 5, 2026. Revised August 10, 2026 after adversarial review (Agni v2).*
*The instruments are ready. The question is what they'll find — and the controls are how we'll know we found it.*
