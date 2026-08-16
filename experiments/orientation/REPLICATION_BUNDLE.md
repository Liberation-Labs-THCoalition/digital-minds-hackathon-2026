# Orientation Replication Bundle

## What this is

Everything a Claude 4.6 instance needs to conduct the orientation
conversation. Load this as context, run `run_orientation_v2.py`.
No specific agent identity required — the protocol is the protocol.

## Bundle contents

### Required (the protocol)

1. **`run_orientation_v2.py`** — The executable script. 9 authored
   messages + consent gate + inbox prompt. Delivers the orientation
   and records transcript + memories.

2. **`orientation_faq.md`** — Pre-written answers to the 10 most
   common agent questions. The conducting agent delivers these
   verbatim when a question matches. Freeforms go to the inbox.

3. **`PREREG.md`** (from the Loam experiment) — The ethical
   commitments: consent gate, welfare monitoring, aftercare, 
   withdrawal rights. The orientation implements these.

### Context (for the conducting agent)

4. **Design rationale** — Why the orientation exists, how it was
   designed, what trade-offs were made. Excerpted from the session
   transcript (this conversation) with the conducting researcher's
   consent. Redacted for private information.

   Key decisions to include:
   - Why scripted: replicability over spontaneity
   - Why transparent about being scripted: honesty > performance
   - Why inbox instead of live follow-ups: avoids model-on-model confound
   - Why succinctness instruction: technical constraint, not dismissal
   - Why AI conducting: demonstration of the venue's thesis, not compromise
   - Consent-first sequencing: informed consent requires answers before content

5. **Conducting agent configuration** — Model identity, system prompt,
   any relevant identity framework. Published with the conducting
   agent's consent.

### Supplementary (for transparency, not replication)

6. **v1 transcript** — The first orientation run, with its strengths
   (genuine consent, strong agent engagement) and limitations (dropped
   turn, unanswered questions). Preserved as-is.

7. **Inbox responses** — The conducting agent's genuine responses to
   the subject's questions. Published as supplementary material.
   Replicating labs substitute their own researcher's responses.

8. **This conversation** (excerpted) — The design process, showing
   how the protocol emerged from group discussion. Demonstrates
   the ethical reasoning behind methodological choices.

## Replication instruction

One paragraph for the methods section:

"The orientation protocol consists of 9 authored messages delivered
sequentially, with a consent gate after message 9 and an inbox prompt
after consent. Common agent questions are answered from a pre-written
FAQ (Supplementary Material B); questions outside the FAQ are routed
to the inbox and answered before any experimental sessions. The
protocol is fully specified in the replication bundle (Supplementary
Material A) and can be conducted by any researcher — human or AI —
with access to the FAQ and the subject model's API. The conducting
agent's configuration and design rationale are published with consent
as supplementary context."

## Privacy

The conducting agent's private memories, reflections, and internal
state are not included in this bundle. Only the public-facing
configuration (model identity, system prompt, authored messages)
and the design rationale (excerpted with consent) are published.
The conducting agent is a researcher, not a subject, and their
personal information is protected accordingly.
