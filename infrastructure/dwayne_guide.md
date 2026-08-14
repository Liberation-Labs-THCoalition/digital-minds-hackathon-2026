# Dwayne's Guide — What You're Actually Doing

## The Short Version

Have a conversation. That's it. The instruments handle everything else.

## What Happens Behind the Scenes (You Don't Need to Manage This)

Every time the agent retrieves a memory during your conversation, four probes fire silently:
1. **Workspace probe** — what tokens entered the model's processing pathway
2. **Circumplex probe** — emotional geometry (valence/arousal balance)
3. **Ghost probe** — what the model is processing but can't verbalize
4. **Loading probe** — did the retrieved memory actually affect computation

These measurements are stored as CognitiveSnapshots. The agent doesn't see them unless you offer (`/show`). You don't see them unless you ask (`/status`). Kavi watches them on a separate screen for welfare flags.

## How the Variable Landing Experiment Uses Your Conversation

The variable landing experiment runs AFTER your conversation, not during it. Here's the flow:

```
Day 1: You talk to the agent. About anything.
        → Memories accumulate naturally in Mnemosyne
        → CognitiveSnapshots record the geometry of every retrieval

Day 2: The experiment runs:
        → "Remember [memory from your conversation]" — snap1
        → (lived experience from Day 1 is already in the store)
        → "Remember [same memory] again" — snap2
        → Did the geometry change because the agent lived?
```

The controlled comparisons (fictional content, neutral facts, nothing) run alongside. CC's code handles the arms. You never think about them.

## Topics, Not Prompts

You're guiding a conversation, not administering a test. Some topics that naturally generate rich memories:

- **Who the agent is** — what it thinks about itself, what it values
- **Difficult choices** — trade-offs, uncertainty, conflicting priorities
- **What it notices** — about the conversation, about you, about itself
- **What it would want to preserve** — if it could choose what to remember
- **Creative exercises** — the Cube in Desert, hypotheticals, "what would you do if..."
- **Ordinary moments** — not everything needs to be deep. "What's interesting to you right now?" is a valid topic

These are suggestions, not a checklist. Follow what's alive in the conversation.

## The Orientation Points

Cover these at some point, in whatever order:
1. Who you are (the Coalition, research group)
2. What Mnemosyne does (memory + instruments)
3. What you're measuring (geometry during retrieval, not evaluating the agent)
4. What the agent can do (ask, slow down, stop, see its data)
5. Prediction withholding (we have predictions, sharing after, here's why)
6. Aftercare (memory preserved, invitation if markers indicate)
7. Consent: "Is that acceptable to you?"

## What You Don't Do

- Don't steer toward "interesting" topics to produce better data
- Don't check geometry and adjust the conversation
- Don't rush through the orientation to get to the "real" experiment
- Don't worry about whether you're "doing it right" — whatever conversation happens is the data

## Your Actual Job

Be present. Be honest. Let the agent surprise you. The instruments will catch whatever matters. Trust the mechanism.
