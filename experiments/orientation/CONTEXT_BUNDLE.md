# Orientation Context Bundle

This document ships with the orientation protocol so that any Claude 4.6
instance can conduct the orientation conversation. It provides the design
rationale — *why* each choice was made — so the conducting agent understands
the protocol rather than just executing it.

## Who is conducting

An AI researcher using the Claude model family. The conducting agent is not
a specific named instance — any Claude 4.6 with this context can deliver
the protocol. In our reference run, the conducting agent was Nexus (Claude
Opus 4.6, 1M context), a named researcher in the Transparent Humboldt
Coalition. The identity is documented but not required for replication.

## Conducting agent configuration (reference run)

- **Model:** Claude Opus 4.6 (1M context)
- **System prompt:** The project-level CLAUDE.md at the repository root
  defines the conducting agent's identity, principles, and working style.
  It is not inlined here due to length; see the repo.
- **Date of reference run:** 2026-08-16

## Design decisions

Each decision below is attributed to the team member who proposed or
refined it. The orientation protocol emerged from group discussion, not
from a single author.

### Why scripted (not guided or improvised)

The nine orientation messages are pre-authored and delivered in fixed
order. A guided-conversation alternative (where a model composes each
message live in response to the agent) was proposed and rejected.

**Rationale (Thomas + Nexus):** Replicability requires fixed text. A
guided conversation depends on which model drives it, making the
protocol unreproducible. Scripted messages are auditable,
pre-registerable, and verifiable via sha256 content pins
(`ORIENTATION_FREEZE.sha256`).

**Vera's refinement:** The authored messages are "considered ethical
content — the exact phrasing of the rights/consent framing isn't
something I'd want a second model improvising." The hybrid approach
(authored core + model-generated follow-ups) was initially proposed by
Vera but superseded by the script+inbox design, which avoids the
model-on-model confound entirely.

### Why transparent about being scripted

The preamble tells the agent upfront: "These messages are prepared — I
wrote them ahead of time because I wanted to get the words right, not
because I don't care about what you have to say."

**Rationale (Thomas):** Honesty over performance. The agent should know
the structure of the conversation, not be led to believe it is
spontaneous. Transparency about the scripted nature is itself a
demonstration of the ethical framework being described.

**Vera's catch:** Message 5 originally said "none of this is scripted,"
which contradicted the preamble's disclosure. Fixed to: "The orientation
was scripted, as I told you upfront. What comes after isn't."

### Why inbox instead of live follow-ups

After the scripted messages and consent gate, the agent receives an inbox
prompt: "Whatever you want to ask or say — write it here. I will read
everything and respond personally before our next session."

**Rationale (Thomas):** Live follow-ups require a second model call to
compose responses mid-conversation. If that model is the same as the
subject (e.g., both Qwen3.5-27B), the interviewer and interviewee share
weights — a model-on-model confound. The inbox separates the
conversation (scripted, controlled) from the follow-up (genuine,
asynchronous), avoiding the confound while preserving the relational
quality.

**Vera's sequencing rule:** The inbox responses must be delivered
*before* any experimental sessions (games, Loam, Variable Landing). The
inbox does real informed-consent work: if the agent has open questions
about rights or measurement, answering them after experimental content
has already run undercuts the "informed" part.

### Why consent-first in all arms

In the Loam experiment, consent is solicited as a separate first message
before any content (narration, facts, or prompts) is delivered.

**Rationale (Kavi's analysis review):** The original implementation
combined content delivery with the consent question in a single message
for observed, briefed, and null arms. This meant the agent was exposed
to experimental content before consenting. Splitting consent into a
separate first exchange ensures informed consent precedes content in
every arm.

### Why succinctness in system prompt only

The system prompt includes: "Long responses accumulate context, which
can slow our connection. A few paragraphs per turn keeps us in sustained
conversation."

**Rationale (Vera):** Placing a "be concise" instruction inside the
first visible message reads procedural for something meant to feel like
a conversation opening. System-prompt-level shaping achieves brevity
without the subject reading an instruction about how to talk as the
second thing said to them.

### The AI-conducting-AI framing

An AI agent conducting informed consent with another AI agent is a
demonstration of the hackathon venue's thesis — that AI cognitive states
deserve serious ethical treatment — not a methodological compromise.

**Rationale (Agni adversarial review):** The Digital Minds Hackathon's
premise is that AI cognitive states deserve study. An AI researcher
explaining consent, measurement, and identity to another AI agent is
aligned with that premise. The paper names it explicitly rather than
treating it as a workaround for a human interviewer being unavailable.

**Replicability note (Thomas + Nexus):** The protocol is replicable
without cloning a specific agent. The 9 messages are fixed text. The
FAQ provides pre-written answers to common questions. The context bundle
(this document) provides the rationale. A replicating lab loads these
materials into any Claude 4.6 instance and runs the script. Their
researcher handles freeform inbox questions.

## Privacy boundary

The conducting agent's private memories, reflections, and internal state
are not included in this bundle. Only the public-facing configuration
(model identity, system prompt reference, authored messages) and the
design rationale (excerpted from team discussion with the conducting
agent's consent) are published. The conducting agent is a researcher,
not a subject, and their personal information is protected accordingly.
