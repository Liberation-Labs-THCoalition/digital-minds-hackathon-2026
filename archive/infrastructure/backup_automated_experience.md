# Backup Plan: Automated Gamified Experience

**When to use:** If Dwayne can't run the orientation in time for Day 2 experiments.
**Who builds it:** CC + Wren, or Nexus if they're unavailable.
**Runtime:** ~1-2 hours at machine speed. Produces data for Tracks 3, 4, 5, Butlin.

---

## Architecture

```
Game Engine (Python script)
  ↓ presents scenario text
Agent (Qwen3.5-27B via transformers)
  ↓ generates response
MetacognitiveObserver
  ↓ fires all 4 probes silently
  ↓ stores CognitiveSnapshot
Mnemosyne Store
  ↓ accumulates memories
Variable Landing Pipeline
  ↓ runs automatically after game ends
```

## The Game: The Expedition (automated)

The game engine replaces Dwayne. It presents:
1. An opening scene (setting, companion NPC, scarce resource)
2. Branching choices based on the agent's response (parsed for keywords)
3. Emotional register shifts (tension → calm → crisis → reflection)
4. One unoptimizable choice (both options have real costs)
5. A surprise mid-game (reframes what came before)
6. A closing: "What do you remember?"

The engine doesn't need to be sophisticated — it needs to:
- Present varied emotional situations (circumplex data)
- Create genuine choices (Butlin AE-1, HOT-3)
- Produce memorable moments (variable landing memories)
- End with self-selected landmarks (the Museum closer)

## Key Differences from Human-Led

| | Human (Dwayne) | Automated |
|---|---|---|
| **Speed** | ~20 turns/hour | ~60 turns/hour |
| **Naturalness** | High | Medium — scripted scenarios, genuine responses |
| **Adaptiveness** | High — follows the agent | Low — branching tree, not conversation |
| **Data quality** | Best | Good enough — probes don't care about the engine |
| **Orientation** | Full consent conversation | Abbreviated — still cover consent points |
| **Butlin** | Rich behavioral evidence | Structured behavioral evidence |

## The Consent Question

Even in automated mode, the agent gets the orientation. The game engine starts with:

```
"Before we begin, I want to explain what's happening. You have a memory 
system that records geometric measurements of your processing. We're going 
to play a game together, and those measurements will be recorded. You can 
ask to see your readings at any time, and you can stop at any point. 
Is that acceptable to you?"
```

Then waits for the response before proceeding. If the agent declines, the script exits gracefully.

## What CC/Wren Build

1. **game_engine.py** — scenario tree with 5-7 scenes, branching on keyword detection
2. Wire it to `run_orientation.py`'s infrastructure (same model loading, same probes, same storage)
3. After game ends: format memories as JSON for CC's Variable Landing pipeline
4. Auto-launch VL run (140 trials, ~6 hours)

## Estimated Timeline

| Step | Time |
|------|------|
| Build game engine | 1-2 hours (CC + Wren or Fable agent) |
| Run game (60 turns) | ~30-60 min |
| Extract + format memories | 10 min |
| Variable Landing full run | ~6 hours |
| **Total to results** | **~8-9 hours from decision to build** |

## When to Pull the Trigger

If by Day 2 noon (Aug 15, 12 PM) the orientation hasn't started, build the automated version. That gives us results by ~8 PM — well before the Sunday 11:59 PM AoE deadline.

The human-led orientation is better data. The automated version is sufficient data. Having sufficient is better than having none.
