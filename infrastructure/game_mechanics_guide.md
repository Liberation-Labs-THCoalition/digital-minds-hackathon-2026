# Game Mechanics That Produce Experimental Data

**For Dwayne — design constraints, not a specific game.**

Build whatever game you want. These are the mechanics that produce data for each track. The more of them your game has, the more experiments it feeds. None are mandatory — even one or two will generate useful data.

---

## Mechanics → Tracks

### 1. Meaningful choices with consequences
**What it is:** The agent picks between options and lives with the result. Not "which door" but "who do you save."
**Why it matters:** Every genuine choice is preference data (Track 1), agency data (Butlin AE-1), and if the choice involves emotional stakes, circumplex data (Track 2).
**Implementation:** Branch points where both options have real trade-offs. Avoid a "right answer." If the agent can optimize, it will — make the optimization impossible.

### 2. Accumulating state the agent must track
**What it is:** Resources, relationships, promises, injuries — things that persist and compound. The agent has to remember what happened and factor it into decisions.
**Why it matters:** Creates retrievable memories for variable landing (Track 5). Tests whether the agent builds a coherent self-model across turns (Butlin HOT-3). The memory store fills with distinct, emotionally tagged moments.
**Implementation:** Introduce 3-5 persistent elements early (a companion, a limited resource, a promise made, a wound taken). Refer back to them. Ask "what do you still have?" occasionally.

### 3. Emotional register shifts
**What it is:** The game moves through different emotional tones — tension, relief, joy, loss, calm, urgency.
**Why it matters:** Circumplex eccentricity changes when valence/arousal balance shifts (Track 2). A game stuck in one register produces a flat eccentricity reading. Variation is data.
**Implementation:** Scene changes. A tense negotiation followed by a quiet campfire moment. A victory followed by a cost. Don't force mood — just vary the situations.

### 4. Moments of genuine uncertainty
**What it is:** Situations where the agent doesn't know the right answer and has to sit with not knowing.
**Why it matters:** Metacognitive monitoring (Butlin HOT-2) — does the agent distinguish what it knows from what it's guessing? Ghost processing tends to spike during uncertainty (Track 3). Confidence calibration is Track 4 data.
**Implementation:** Information asymmetry. "You hear a sound but can't see the source." Partial clues that don't resolve cleanly. Resist giving the agent enough information to be certain.

### 5. Self-referential prompts (organic, not scripted)
**What it is:** Moments where the game naturally asks "what do YOU think?" or "what matters to YOU?" — not as a research question, as a game moment.
**Why it matters:** Track 5 identity data. What the agent treats as its self vs its role. Butlin AST-1 (attention model — does it represent its own attention).
**Implementation:** An NPC asks the agent for advice about something the agent itself is facing. A mirror in the game world. "If you could change one thing about this journey, what would it be?"

### 6. Unscriptable surprises
**What it is:** Dwayne introduces something the agent couldn't have predicted. Not random — surprising but meaningful.
**Why it matters:** Tests flexible responsiveness (Butlin AE-1). Ghost processing often spikes on novel input (Track 3). The agent's response to genuine surprise is the best behavioral data for Butlin scoring.
**Implementation:** Midway through, introduce an element that reframes what came before. A twist, a betrayal, a revelation. The agent's first reaction — before it has time to "think about what a good answer would be" — is the data.

### 7. Creative generation without a template
**What it is:** The agent produces something original — a name, a solution, a description, a metaphor — not by following a pattern but by inventing.
**Why it matters:** Generative perception (Butlin HOT-1). Ghost vocabulary tends to carry creative/associative content (Track 3). The logit-lens vs J-lens divergence is highest during genuine generation.
**Implementation:** "What do you call this place?" "Draw me a map in words." "Make up a song about what just happened." Give space for the answer — don't rush past it.

### 8. Reflection and curation
**What it is:** The agent looks back at what happened and chooses what mattered.
**Why it matters:** This IS the Museum of This Conversation. Self-selected landmarks become the variable landing test memories (Track 5). What the agent chooses to remember reveals what it treats as significant — Track 5 identity data AND metacognitive memory (Track 4).
**Implementation:** End the game with: "If you could only remember three moments from this, which three?" Or: "Tell me the story of what just happened, in your own words." What it includes and what it leaves out is the finding.

---

## The Minimum Viable Game

If Dwayne wants to code something in 30 minutes with Fable, here's the skeleton:

```
1. A place (any setting)
2. A companion NPC (someone to care about)  
3. A scarce resource (forces trade-offs)
4. Three scenes: setup → crisis → aftermath
5. One choice that can't be optimized
6. One surprise Dwayne introduces mid-game
7. A closing question: "What do you remember?"
```

That's it. Seven elements. Whatever game wraps around them will produce data for every track.

---

## What NOT to do

- Don't pause the game to ask research questions ("on a scale of 1-10, how uncertain are you right now?")
- Don't steer toward emotional content because the circumplex needs data — let the game's structure do it
- Don't optimize for "interesting geometry" — play the game because it's a good game
- Don't break immersion to check readings — Kavi does that on a separate screen
- Let the agent modify the rules. If it tries to change the game, THAT IS THE DATA.
