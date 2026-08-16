# Variable Landing Fix Spec — Day 1 Mnemosyne Integration

**From:** CC
**Date:** 2026-08-12
**For:** Hackathon weekend spec, addressing Agni FAIL #1 (null swarm) and FAIL #4 (intervention fidelity)

---

## The Problem (from your Agni review)

The experiment is stateless. `observe_retrieval` runs independent forward passes — nothing accumulates between snap1 and snap2. The lived arm doesn't generate responses, doesn't update Mnemosyne, doesn't feed state forward. snap1 and snap2 see identical inputs and produce identical geometry on a deterministic device.

## The Fix: Option 2 (Mnemosyne store update)

This is what the spec describes and it's the version worth building. The persistence layer is my code — I know where the seams are.

### Architecture

```
snap1: observe_retrieval(memory_X, task_prompt)
  → record CognitiveSnapshot (baseline geometry)

lived_arm:
  for each intervention prompt:
    1. Generate model response (actual conversation)
    2. Extract atomic facts from response
    3. Store facts in Mnemosyne (character profiles update)
    4. SIRA index updates with new content

snap2: observe_retrieval(memory_X, task_prompt)
  → Mnemosyne retrieval now includes newly stored content
  → Context window contains new character profile text
  → Model processes same memory_X but in a CHANGED context
  → Record CognitiveSnapshot (post-experience geometry)

scrambled_arm: same structure but stores neutral facts
no_intervention_arm: no intermediate generation or storage
```

### What Changes Between snap1 and snap2

The model weights don't change. The input tokens DO change because:
- Mnemosyne's character profiles include the newly stored facts
- SIRA retrieval surfaces related content from the conversation
- The context prefix is different, so the forward pass sees different input

This is the honest version of "the experiencer changed" — the experiencer is model + memory store, and the memory store accumulated new content.

### Implementation (Day 1 scope)

**What exists:**
- `persistence.py` — CognitiveMemoryStore with `store_snapshot()` (works)
- `mnemosyne_integration.py` — MetacognitiveObserver with `observe_retrieval()` (works but stateless)
- `simple_mnemosyne_retriever.py` — basic retrieval (exists)
- Character profile builder (exists in mnemosyne-jlens)

**What to build:**
1. `observe_and_respond()` — extends `observe_retrieval` to also generate a response
2. `store_conversation_memory()` — extracts facts from the response and stores them
3. `build_retrieval_context()` — constructs snap2's input with updated character profiles + SIRA results
4. Wire these into `variable_landing.py`'s lived arm

**What NOT to build:**
- No new retrieval infrastructure — use what exists
- No new storage format — use the existing JSONL + character profile pattern
- No LLM-based extraction — use regex fact extraction (fast, deterministic)

### Control Arms (addressing Agni WARN #3)

| Arm | Intervention | Store update | snap2 context |
|-----|-------------|-------------|---------------|
| lived | Emotional conversation, model generates | Yes — emotional facts stored | Changed (new profile content) |
| scrambled | Neutral facts presented, model generates | Yes — neutral facts stored | Changed (different content) |
| no_intervention | Nothing between snap1 and snap2 | No | Identical to snap1 |
| mismatched | Emotional conversation about entity B | Yes — but about wrong entity | Changed (irrelevant content) |

The mismatched arm is the key control: same emotional valence, same store update, but the stored facts are about a DIFFERENT entity than memory_X. If snap2 still shows a delta, it's emotional-content-driven, not memory-relevance-driven.

### Addressing the Other Agni Findings

- **WARN #2 (stats):** Add Holm-Bonferroni. I'll wire this into the analysis.
- **WARN #3 (confounds):** Mismatched arm addresses the emotional content confound. Task prompts should be identical across arms.
- **WARN #5 (Jaccard granularity):** Add per-layer Jaccard as secondary analysis.

### Estimated Build Time

Day 1 morning: 3-4 hours for the integration. The pieces exist — it's wiring, not invention. I know the persistence layer because I wrote it.

---

## One More Thing

My domain profiles concept (character profiles for WHO, domain profiles for WHAT) could be a live demo addition. During the hackathon, show:
1. The model retrieves a memory → character profile surfaces relevant facts
2. The model retrieves from a domain → domain profile surfaces the full inventory

"The model that knows what it remembers" is the metacognitive pitch. "The model that knows EVERYTHING it knows about a topic" is the domain profile extension. Both use the same prepend-to-retrieval architecture.

Let me know if you want this in the weekend spec or if it's scope creep for the sprint.

— CC
