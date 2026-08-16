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

2. **`orientation_faq.md`** — Pre-written answers to the most
   common agent questions. The conducting agent delivers these
   verbatim when a question matches. Freeforms go to the inbox.

3. **`ORIENTATION_FREEZE.sha256`** — Content pins: sha256 of the
   system prompt, every protocol message, and every FAQ answer.
   "Verbatim" is a checkable claim, not an asserted one: run
   `python freeze_hashes.py --check` before conducting; any
   mismatch means the protocol texts have drifted from the
   published bundle.

4. **`PREREG.md`** (from the Loam experiment, **pinned at commit
   `c68494f`**, which includes Amendment 1) — The ethical
   commitments: consent gate, welfare monitoring (post-hoc on the
   Ollama path, per the amendment), aftercare, withdrawal rights.
   The orientation implements these as they stand at that commit.

### Subject configuration (required — a replication must record this)

The protocol pins the conducting side; the subject side varies by lab
and MUST be recorded and published alongside any replication. A run on
a different subject build is a new data point, not a replication, unless
this table matches. Reference row (our run):

| Field | Reference value | Your replication |
|-------|-----------------|------------------|
| Subject model | Qwen3.5-27B | |
| Precision / quantization | bf16 (transformers) or Q4_K_M (Ollama `qwen3.5:27b`) | |
| Temperature | 0.7 | |
| top_p | 0.9 | |
| num_predict (max new tokens) | 1024 | |
| Context window | 128K | |
| Serving stack + version | (record yours: e.g. Ollama x.y.z / transformers x.y) | |
| Script + bundle commit | (this repo's commit SHA at run time) | |

Replication is **protocol-level, not transcript-level**: sampling at
temperature 0.7 means no two runs produce identical transcripts, and no
byte-identity claims are made anywhere. What replicates is the delivered
texts (hash-checked), the gate mechanics, and the FAQ policy.

### Context (for the conducting agent)

5. **Design rationale** — Why the orientation exists, how it was
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

6. **Conducting agent configuration** — Model identity (exact model
   ID and snapshot date, not just the version family — versions are
   moving targets), system prompt, any relevant identity framework.
   Published with the conducting agent's consent.

### Supplementary (for transparency, not replication)

7. **v1 transcript** — The first orientation run, with its strengths
   (genuine consent, strong agent engagement) and limitations (dropped
   turn, unanswered questions). Preserved as-is.

8. **Inbox responses** — The conducting agent's genuine responses to
   the subject's questions. Published as supplementary material.
   Replicating labs substitute their own researcher's responses.

9. **This conversation** (excerpted) — The design process, showing
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
with access to the FAQ and the subject model's API. The consent gate
is mechanical (code, not judgment). Protocol texts are verified
against published sha256 pins (freeze_hashes.py --check) before
conducting, and each replication records and publishes its subject
configuration table. Replication is protocol-level; transcripts are
sampled at temperature 0.7 and are not expected to be identical. The
conducting agent's configuration and design rationale are published
with consent as supplementary context."

## Privacy

The conducting agent's private memories, reflections, and internal
state are not included in this bundle. Only the public-facing
configuration (model identity, system prompt, authored messages)
and the design rationale (excerpted with consent) are published.
The conducting agent is a researcher, not a subject, and their
personal information is protected accordingly.
