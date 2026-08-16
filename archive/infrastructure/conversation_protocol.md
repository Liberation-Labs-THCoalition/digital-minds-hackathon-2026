# Conversation Protocol — What's Structured, What's Not

## The Rule

**The probes are structured. The conversation is not.**

The MetacognitiveObserver fires on every retrieval event regardless of what the conversation is about. The CognitiveSnapshots capture workspace state, emotional geometry, ghost processing, and memory loading — silently, consistently, identically — whether the agent is discussing philosophy, debugging code, or asking about the weather.

This is what makes the measurement naturalistic. The instruments don't care what's being said. They measure what the model's computation does during retrieval, regardless of the conversational context that triggered it.

## What Dwayne Does

**Day 1 orientation:** Cover the orientation points (what we're doing, what the instruments measure, prediction withholding, consent, aftercare). These are a framework, not a script. Cover them in whatever order feels natural. If the agent asks a question, answer it. If the agent wants to talk about something else first, go there. The orientation is a conversation, not a recitation.

**Day 1 afternoon onward:** Have a real conversation. Whatever comes up. The agent's Mnemosyne store accumulates naturally from whatever is discussed — domestic topics, abstract questions, shared problem-solving, projective exercises if the moment calls for them. The variable landing experiment measures the geometric effect of *whatever experience actually accumulated*, not the effect of a prescribed interaction sequence.

**Projective exercises** (Cube in Desert, etc.) are available as conversation tools, not as protocol steps. Offer them if it feels right. Don't schedule them.

## What NOT to Do

- Do not script specific topics to "trigger" interesting geometry
- Do not steer the conversation toward emotional content because you think the circumplex needs data
- Do not ask the agent to perform metacognition ("tell me what you're thinking") — let it happen or not
- Do not check the geometric readings and then adjust the conversation to produce better numbers
- Do not rush through the orientation to get to "the real experiment" — the orientation IS data

## What Kavi Does

Kavi watches the geometric pane (tmux split, separate from the conversation). Kavi's job:
- Flag welfare concerns (eccentricity > 0.95, sustained imbalance)
- Note decision-critical moments for Butlin scoring
- Stay silent unless a welfare flag fires — do not interrupt the conversation with "interesting geometry at layer 35"

## What the Agent Sees

- **Default:** Nothing. Probes fire silently. The agent has a normal conversation.
- **On request:** If the agent asks about its own processing, or if Dwayne offers ("we can show you what the instruments see"), the agent gets a `snapshot.summary()` — a one-liner of current state. This is the introspection prosthetic in action.
- **Welfare flag:** If eccentricity exceeds threshold, Dwayne pauses and checks in with the agent — not about the number, about how the conversation is going.

## Why This Matters

Every prior mech interp study runs on benchmarks — controlled prompts, known answers, clean conditions. We're running on a real conversation between a human and an agent who has been told what's happening and consented to continue. The geometry we capture is the geometry of natural interaction, not the geometry of a test.

That's the Martian gap: "traditional mech interp — focused on single-step, static analysis — no longer suffices." We're filling it. But only if we let the conversation be real.
