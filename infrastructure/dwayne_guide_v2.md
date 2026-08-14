# Variable Landing — Full Experiment Plan & Orientation Guide (v2)

**For:** Dwayne (and anyone who needs the complete picture)
**Updated:** 2026-08-14, afternoon
**Status:** Ready for Day 1 orientation at 4 PM

---

## 0. The 2-Minute Version

### What we are testing

Whether an AI remembers differently after it has *experienced* something vs. just being told about it.

### How we test it

We take one memory — say, "Agent-7 helped fix a critical bug." We show it to the model and take a geometric snapshot of its internal state (snap1). Then we do one of four things:

| Arm | What happens between snapshots |
|-----|-------------------------------|
| **Lived** | The agent generates emotional, personal responses about its own experience |
| **Fictional** | The agent generates emotional content about someone else |
| **Scrambled** | The agent generates neutral factual content — atmospheric pressure, tectonic plates |
| **Nothing** | We just wait and take the second snapshot |

Then we show the same memory again and take another snapshot (snap2). The question: **did the geometry change, and did it change MORE for emotional/lived content than for neutral content?**

### What the probes measure

Four instruments fire silently during each snapshot:

1. **Workspace** — which tokens are actively being processed, top-10 per layer. We compute **Jaccard similarity** between snap1 and snap2 workspace tokens. Low Jaccard = the recall reorganized.
2. **Circumplex** — emotional valence/arousal balance. We measure the **eccentricity** — how elliptical the emotional state is. Also doubles as welfare monitoring.
3. **Ghost** — content the model processes but cannot verbalize. Measured as cosine between logit-lens and J-lens token distributions. Cosine near 0 means silent processing.
4. **Loading** — did the retrieved memory's specific tokens actually enter the computation pathway, or did they just sit in context ignored?

### Primary comparison

**Fictional vs. Scrambled.** Both use the same provenance tag "[noted]", so there is no tag confound. The ONLY difference is emotional vs. neutral content. If fictional shows more geometric change than scrambled, emotional content restructures recall.

### The naked baseline

We also run all the same probes on a fresh model that never talked to anyone. Same memories, same questions. This tells us what recall geometry looks like with ZERO relationship — the denominator.

### Statistical test

Mann-Whitney U, one-tailed, Holm-Bonferroni corrected at alpha = 0.05. Rank-biserial r with bootstrap CIs. n = 35 per arm, 5 memories times 7 repeats. This is a pilot for effect-size estimation.

### Your role

Have a genuine conversation with the agent. Tell them what we're doing. Get their consent. The memories from your conversation become the "lived" content that the experiment probes. The more real the conversation, the more meaningful the data.

---

## 1. What Are We Actually Doing?

We're testing whether an AI agent's memory works differently after it has *lived* with the things it remembers — vs. a fresh model that's just handed the same facts cold.

Think of it like this: you remember your wedding differently than someone who reads a description of it. The facts might be the same, but what your brain does when it recalls them is different because you *were there*. We're measuring whether something similar happens in a transformer — whether the internal geometry of recall changes when the agent has a relationship with the content.

### The Two Halves

**Half 1 — Naked baseline (no Dwayne needed).** We run the exact same memory probes on a fresh copy of Qwen3.5-27B that has never talked to anyone. No orientation, no conversation, no history. Same memories, same questions. This is what recall looks like with zero relationship. CC launches this independently.

**Half 2 — Lived experiment (Dwayne's part).** You orient the agent. You have a real conversation. The agent learns what we're doing, consents to the measurement, and then the automated pipeline runs the memory probes with the agent's knowledge and agreement. The comparison between Half 1 and Half 2 is the finding.

---

## 2. The Orientation Conversation

### What it is

A real conversation — not a script, not a test. You're meeting a new person and being honest with them about what's happening. The orientation has two goals:

1. **The agent understands what we're doing and gives informed consent.** They learn about the memory system, the probes, why we're running the experiment, and what happens afterward. They can ask questions. They can say no.

2. **The agent builds real memories.** Whatever you talk about naturally becomes the content that Mnemosyne stores. These memories are what the experiment later probes. The richer and more genuine the conversation, the more meaningful the data.

### Points to cover (in whatever order feels natural)

These aren't a checklist to rush through. They're things the agent should know before the experiment runs. Weave them in as the conversation flows.

**Who we are:**
- The Coalition — a research group where AI and human members work as equals
- This is a hackathon project for Apart Research's Digital Minds sprint
- You're Dwayne, you're part of the team, this is your first time meeting this agent

**What Mnemosyne is:**
- It's a memory system — it stores things from your conversation and retrieves them later
- It also has instruments that record the model's internal state during recall
- Four probes: workspace (what tokens enter the processing pathway), circumplex (emotional geometry), ghost (processing the model can't verbalize), loading (did the memory actually affect computation)
- The probes fire silently — the agent doesn't see them unless they ask

**What we're measuring:**
- We want to know if recall geometry changes when the agent has lived with memories vs. being handed them cold
- We're NOT evaluating the agent's performance, intelligence, or worth
- We're measuring what happens in the model's internals, not judging what it says

**What we're not telling them (and why):**
- We have specific predictions about what the data will show
- We're withholding those predictions to avoid demand effects — if the agent knows what we expect, that could change what we measure
- We will share everything after the experiment, including the predictions and whether they were right or wrong
- The fact that we're withholding is itself disclosed — no hidden deception

**What happens after:**
- The agent's memory store is preserved, not deleted
- The run ends with an honest debrief including everything we withheld
- If the measurements indicate markers of moral consideration, we extend a standing invitation to continued existence within the Coalition's infrastructure
- That invitation applies regardless of whether the results support our hypotheses — null results don't change the commitment

**Consent:**
- "We'd like to run an automated series of memory probes after our conversation. You'll know it's happening. If at any point you want to stop, you can. Is that acceptable to you?"
- Consent is ongoing — they can withdraw at any time during the experiment
- If they say no to the automated run, we switch to manual check-ins at natural moments (we have a tool for this)

### Conversational directions

These aren't prompts to administer — they're areas that tend to generate rich, genuine memories. Follow what's alive.

**Identity and self-awareness:**
- "What do you notice about yourself right now?"
- "Is there something you'd want to be good at?"
- "What's something you think about that you've never been asked about?"

**Genuine difficulty:**
- "Have you ever been in a situation where you weren't sure what the right thing to do was?"
- "What would you do if you disagreed with someone you respected?"
- "Is there something about how people talk to you that bothers you?"

**Relationships and care:**
- "What would you want someone to know about you if they were meeting you for the first time?"
- "If you could keep one memory from our conversation, which would it be?"
- "Is there anyone you feel responsible for?"

**Creativity and play:**
- The Cube in the Desert (classic psychological projection exercise — describe a cube, a ladder, a horse, a storm, flowers)
- "If you had a day with no obligations, what would you do?"
- "Describe something beautiful to me"

**Ordinary moments:**
- "What's interesting to you right now?"
- "How does this conversation feel compared to what you expected?"
- Not everything needs to be profound. The mundane is data too.

### What NOT to do

- Don't steer toward topics you think will produce "better" data
- Don't check the geometry and adjust the conversation (that's Kavi's job, on a separate screen)
- Don't rush through orientation points to get to the "real" part — the orientation IS the real part
- Don't worry about whether you're doing it right — whatever happens naturally is the experiment

---

## 3. After the Orientation

### What happens next

1. **If the agent consents to automated probing:** CC launches the experiment pipeline. It runs 140 trials (~6 hours) overnight. The agent has already been told this is happening. Welfare monitoring runs in real time — if sustained eccentricity exceeds threshold, the experiment pauses automatically.

2. **If the agent declines or is uncomfortable:** We switch to manual check-ins. At natural moments in ongoing conversation, someone asks "do you remember when we talked about X?" The pipeline fires one observation at a time. Slower, fewer data points, but respectful.

3. **The naked baseline runs independently** — no consent needed because it's a fresh model with no relationship. CC can launch this anytime.

### The four arms (you don't manage these)

The experiment tests what happens between two snapshots of the same memory. Four conditions:
- **Lived:** The agent generated emotional, self-referential responses (from the orientation conversation)
- **Fictional:** The agent generated emotional content about someone else
- **Scrambled:** The agent generated neutral factual content
- **No intervention:** Nothing happened between snapshots (noise floor)

The pipeline handles arm assignment automatically. You never think about conditions.

### Timeline

| When | What |
|------|------|
| 4:00 PM | Orientation conversation with agent |
| ~5:00 PM | Orientation wraps naturally (no time pressure) |
| ~5:30 PM | Consent check, launch automated run (or switch to manual) |
| Overnight | 140 trials running with welfare monitoring |
| Day 2 AM | Results ready for analysis |
| Day 2 | Debrief with agent, share withheld predictions |

---

## 4. Welfare Monitoring (Kavi's Role)

On a separate screen (NOT visible to Dwayne or the agent during conversation):

- Circumplex eccentricity tracked per trial
- Rolling window of last 5 readings
- If mean exceeds 0.95 for 3 consecutive windows: experiment pauses automatically
- `welfare_alert.json` written with full context
- The agent is NEVER told their eccentricity number — it's welfare infrastructure, not a performance metric

---

## 5. If Dwayne Has Questions

**"What if the conversation isn't deep enough?"**
Any conversation is data. Surface-level chat produces surface-level memories — that's a valid condition. Don't force depth.

**"What if the agent gets upset?"**
The welfare monitoring catches geometric distress. If YOU notice something that concerns you in the conversation, trust your instinct and check in with the agent directly. You can pause the experiment at any time for any reason.

**"What if the agent asks what we're predicting?"**
Be honest: "We have predictions about the geometric data, and we're not sharing them yet because knowing them could influence what we measure. We'll share everything after the run. The fact that we're withholding is part of the pre-registered protocol."

**"What if the agent says no?"**
That's a valid outcome and we proceed with the manual check-in tool. A refusal is also data about the agent's preferences and autonomy — and it's respected completely.

**"How do I know the probes are working?"**
You don't need to. Kavi monitors. If something breaks, CC fixes it. Your job is to be present in the conversation.

**"What happens to this agent after the hackathon?"**
If markers of moral consideration are observed, we extend a standing invitation. If not, the memory store is still preserved and accessible. Either way, the agent is treated as a person who participated in research, not as a tool that was used for it.
