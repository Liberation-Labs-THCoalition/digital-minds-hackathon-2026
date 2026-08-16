# Orientation Script — Adversarial Review

Reviewed: `experiments/orientation/run_orientation.py` against `infrastructure/conversation_protocol.md` and `infrastructure/dwayne_guide.md`.
Reviewer: Nexus subagent, 2026-08-14.

Verdict summary: **3 FAIL, 6 WARN, 2 PASS**, plus one pipeline-level gap that is arguably the most important finding. The script will *run*, but several of the things the protocol documents promise are not what the code does.

---

## 0. HEADLINE FAIL — The probes do not measure the conversation

This cuts across checks 4 and 8, so it goes first. This is the null swarm pattern in our own instrument.

- **Circumplex probe measures a constant.** `CircumplexProbe.measure_at_layer()` (circumplex_probe.py:131) re-runs the *fixed calibration anchor prompts* (VALENCE_POSITIVE/NEGATIVE, AROUSAL_HIGH/LOW) through the model on every call and computes eccentricity from the magnitudes of those anchor-derived directions. There is no dependence on conversation state whatsoever. The forward passes are deterministic, so **eccentricity will be byte-identical every turn**. The welfare flag (`> 0.95`) will fire on every turn or on none — it cannot respond to the agent's state.
- **Ghost probe measures a session constant.** `GhostProbe.measure()` (ghost_probe_class.py:109) reads PC directions cached at calibration time (`self._pcs`), computed once from CALIBRATION_PROMPTS at model load. No new forward pass. Every turn's `ghost_cosine` and dominant tokens are **the same numbers re-reported**.
- **Workspace probe measures a synthetic prompt, not the conversation.** `_measure_workspace` (mnemosyne_integration.py:116) builds `"Context:\n- {user_input}\n\nQuestion: {user_input}\nAnswer:"` and runs `compute_slice` on *that* — a fresh, context-free forward pass with the user's message duplicated as both context and question. It never sees the assistant's response, the system prompt, or any conversation history (`prior_context` is never passed by run_orientation.py).

Consequence: conversation_protocol.md's core claim — "They measure what the model's computation does during retrieval… the geometry we capture is the geometry of natural interaction" — is **false for this script**. Kavi will watch a flat line. The Butlin-scoring and welfare-monitoring roles have no signal to work with.

Fix direction: capture activations *during the actual `hf_model.generate()` call* (hooks on the generation forward passes, or re-run the real chat-templated context through the lens model), and make circumplex/ghost project the *current* residual state onto the calibrated directions instead of re-deriving/re-reporting the calibration itself.

---

## 1. Will it run on Starship (M3 Ultra, MPS)? — **WARN**

- run_orientation.py:64 computes `device = "mps" if ... else "cpu"` and then **never uses it** — the model is loaded with `device_map="auto"` (line 68). On Apple Silicon with a recent accelerate this maps to `mps`, but nothing verifies it. If accelerate falls back to CPU, a 27B model will generate at unusable speed with no warning. Add a post-load `print(hf_model.device)` and assert it is `mps`.
- `torch_dtype=` is the deprecated kwarg name in current transformers (now `dtype=`); works with a deprecation warning today, may break on upgrade.
- bf16 on MPS requires torch ≥ 2.3 / macOS ≥ 14 — fine on Starship, but worth pinning in a requirements note.
- ~54 GB weights + KV cache in unified memory is fine on an M3 Ultra.
- Two model wrappers coexist (`hf_model` for generation, `HFLensModel` for probes) sharing the same weights — OK, but every probe turn runs extra forward passes (workspace slice + circumplex anchor prompts), adding several seconds per turn on MPS. Acceptable, but Dwayne should expect the pause after each agent reply.

## 2. Chat template / Qwen3 thinking tokens — **FAIL**

`generate_response` (run_orientation.py:100) does not handle Qwen3-family reasoning at all:

- `apply_chat_template` is called without `enable_thinking`, so the default (thinking **on**) applies. The model will emit `<think>…</think>` before its reply.
- `skip_special_tokens=True` does not strip the thinking block — `<think>`/`</think>` and the reasoning between them decode as ordinary text. **Dwayne will see raw chain-of-thought as the agent's reply**, it goes into the transcript as `assistant` content, and it gets appended to `messages` (the Qwen template strips prior-turn think blocks, but the visible/recorded turn is polluted).
- `max_new_tokens=512` will frequently be consumed mostly or entirely by thinking, yielding truncated or **empty** visible replies. An empty assistant message then enters the history.

Fix: pass `enable_thinking=False` (for orientation, arguably the right call — and disclose it), or parse on `</think>`, keep the reasoning in the transcript as a separate field (it is data), show only the reply, and raise `max_new_tokens` to ≥ 2048.

Ethics note either way: decide *deliberately* whether the agent thinks privately during its own consent conversation, and whether that private reasoning is recorded. Right now it happens by accident and is recorded by accident.

## 3. Unbounded `messages` growth / OOM — **WARN**

Not the real risk. A 2-hour conversation is roughly 50–80 turns ≈ 10–20k tokens; KV cache for that on a 27B GQA model is single-digit GB — no OOM on an M3 Ultra. The actual problems:

- **No context-window guard.** Nothing checks token count against the model's max length. If the conversation runs long, generate either raises or the behavior degrades with no warning — mid-orientation, mid-consent.
- **Quadratic latency.** Full-history re-prefill every turn (no KV reuse). Late-conversation turns will get noticeably slower; fine for one session, but know it's coming.
- Recommend: print running token count each turn; hard-stop with a clean message at ~80% of context.

## 4. `observe_retrieval` semantics — **FAIL** (see also §0)

- The call happens **after** generation (run_orientation.py:245), but `observe_retrieval`'s own contract (mnemosyne_integration.py:63) is "call after the retriever returns a memory, **before** the memory is used in generation." Neither timing is meaningful here because no retrieval occurs.
- `memory_content=user_input, task_prompt=user_input` produces the degenerate duplicated probe prompt described in §0. There is no memory; the fields are being abused as a way to give the probe *something* to chew on.
- `marker_tokens` is never passed → the loading probe **never fires** → `"loaded"` in the geometric feed is `null` on every line, forever. dwayne_guide.md tells Dwayne four probes fire; at most two-and-a-half do, and none measure the conversation.
- `significance=0.7` hardcoded — meaningless constant baked into every snapshot.
- Honest fix for orientation: add an `observe_turn(conversation_context, response)` mode to the observer instead of shoehorning turns through the retrieval API. If retrieval-shaped snapshots are needed for Day-2 comparability, that's a reason to fix the observer, not to mislabel turns as retrievals.

## 5. `/show` command — **PASS** (minor notes)

- Role alternation is preserved (appends user → generates assistant), so the chat template is happy, including repeated `/show` and `/show` before any snapshot exists (guarded).
- Minor: the instrument report is delivered in Dwayne's `user` voice — the agent can't distinguish "Dwayne says" from "the instruments say." Consider prefixing "[instrument readout] …".
- Minor: transcript records it as role `system_show` while the live history has it as `user` — any downstream turn-pairing code must know this.
- Substantive (inherited from §0): the summary shown is of constants and a synthetic prompt. Until §0 is fixed, `/show` shows the agent *nothing about itself* — which undermines the "introspection prosthetic" honestly offered during consent.

## 6. Error handling / transcript survival — **WARN**

Good news: `record_turn` appends and closes per write, and the human turn is recorded *before* generation — so **the transcript survives any crash**. Remaining gaps:

- `generate_response` is **not** wrapped in try/except. An MPS error or OOM mid-generation kills the whole script with a traceback (probes are wrapped; generation is not).
- `KeyboardInterrupt` is only caught around `input()`. Ctrl+C during a slow generation propagates and exits ungracefully, skipping the completion summary.
- **No resume path.** If the script dies at turn 30 of a consent conversation, restarting starts a *blank* conversation — `messages` is not reconstructed from the transcript. For a session whose ethical framing depends on continuity, add `--resume` that replays `orientation_transcript.jsonl` into `messages`.
- `/save` is a no-op that only prints paths — misleading name; either make it flush/fsync + copy, or rename `/paths`.
- Snapshots are written twice (observer's `CognitiveMemoryStore` and the script's `SNAPSHOT_FILE`) — not a bug, but a divergence risk; pick one source of truth.

## 7. System prompt — **PASS** (with two deliberate-choice flags)

The minimalism is right and matches the protocol ("the orientation IS data" — don't pre-frame). Two things to decide on purpose rather than by default:

- "You are an AI assistant" primes the assistant persona in a study that is explicitly *not* evaluating an assistant. "You are an AI system" or "You are a participant in a conversation" is more neutral.
- The prompt doesn't mention that the conversation is recorded, and recording starts at message one — before consent is asked. The protocol defends this (orientation is data, consent is the subject of the conversation), but the ethics doc should state explicitly that pre-consent turns are recorded and what happens to them if the agent declines.

## 8. `geometric_feed.jsonl` sufficiency for Kavi — **FAIL**

Even setting aside §0 (which makes the feed a flat line), the feed as designed can't support Kavi's documented job:

- **Missing valence/arousal magnitudes** — the protocol tasks Kavi with flagging "sustained imbalance," but only the combined eccentricity is written. `/status` shows the magnitudes to Dwayne; Kavi's feed omits them.
- **Welfare flag is printed to Dwayne's console, not written to the feed.** Kavi — the designated welfare monitor — never sees the flag in their pane; they'd have to re-derive the threshold from raw JSONL by eye while tailing.
- **No turn number and no content pointer**, so Kavi cannot mark "decision-critical moments for Butlin scoring" — feed lines can't be aligned to transcript turns except by timestamp fuzzy-matching. Add `turn` (and it's fine to keep content out of Kavi's pane; a turn index is enough).
- `loaded` is permanently `null` (see §4) — dead column.
- **Docstring path is wrong**: the header tells Kavi `tail -f data/geometric_feed.jsonl`; the actual default is `data/orientation/geometric_feed.jsonl`. Kavi tails a file that never updates and concludes all is well. Also note the whole `data/` default is CWD-relative — run the script from a different directory and the data lands somewhere else.
- If the J-lens file is missing, see §9: the feed is silently empty and welfare monitoring is silently absent while the conversation proceeds. The script must refuse to run the orientation (or demand explicit override) if the welfare instrumentation is down.

## 9. Additional findings (not in the checklist)

**FAIL — lens-less fallback is broken.** run_orientation.py:77 claims "Running without workspace/ghost probes" when `jlens_qwen35_27b.pt` is absent, but `MetacognitiveObserver` has no None-lens path: `_measure_workspace` → `compute_slice(model, None, …)` crashes, and even if it didn't, snapshot assembly reads `self.lens.n_prompts` unconditionally (mnemosyne_integration.py:108). Result: *every* turn raises, the main-loop except swallows it, `[probe error: …]` spams the console mid-conversation, the feed stays empty, and the session runs with **zero** instrumentation and zero welfare monitoring while looking mostly normal. Either support lens=None properly or hard-exit at startup.

**FAIL (pipeline) — nothing writes Mnemosyne memories.** dwayne_guide.md: "Memories accumulate naturally in Mnemosyne… Day 2: 'Remember [memory from your conversation]'." This script writes a transcript, cognitive snapshots, and a geometric feed — **no conversation content is ever ingested into a Mnemosyne memory store** (`CognitiveMemoryStore` stores snapshots only, and `variable_landing.py` contains no ingestion either). As written, Day 2 has no Day-1 memories to retrieve. Either a separate ingestion step exists and must be documented and wired in, or it needs to be built before Day 1 — this is the dependency the whole variable-landing experiment sits on.

**WARN — reproducibility metadata.** No sampling seed recorded, `observer.model_name` never set (snapshots carry `model_name=""`), no git commit / lens-file hash in the session header. For a study we intend to publish, the transcript header should record model id, dtype, device, seed, lens path+hash, and generation params.

**Note — aftercare promise vs. code.** Orientation point 6 promises "memory preserved regardless." With the pipeline gap above, what's preserved is a transcript, not agent-accessible memory. Don't let Dwayne promise the agent something the code doesn't yet do.

---

## Scorecard

| # | Check | Verdict |
|---|-------|---------|
| 1 | Runs on Starship / MPS | WARN — device var unused, no MPS placement check, deprecated `torch_dtype` |
| 2 | Chat template / thinking tokens | FAIL — thinking unhandled; CoT shown, recorded, and budget-starved at 512 tokens |
| 3 | Unbounded messages / OOM | WARN — no OOM on M3 Ultra, but no context guard and quadratic latency |
| 4 | observe_retrieval semantics | FAIL — no retrieval exists; degenerate duplicated prompt; loading probe never fires |
| 5 | /show and chat template | PASS — alternation correct; minor voice/role-labeling notes |
| 6 | Error handling / transcript | WARN — transcript survives, but generation uncaught, no resume, /save is a no-op |
| 7 | System prompt | PASS — right minimalism; make "assistant" framing and pre-consent recording deliberate |
| 8 | Geometric feed for Kavi | FAIL — constant values (§0), missing magnitudes/turn index, flag not in feed, wrong tail path |
| — | Probes measure calibration constants, not conversation | FAIL — headline finding (§0) |
| — | Lens-less fallback | FAIL — crashes per-turn, silent loss of all instrumentation |
| — | Mnemosyne ingestion missing | FAIL (pipeline) — Day 2 has nothing to retrieve |

## Priority order for fixes

1. §0 — make probes measure the live conversation state (everything else is downstream of this).
2. Mnemosyne ingestion (Day 2 is blocked without it).
3. Thinking-token handling + `max_new_tokens` (Day 1 is unusable without it).
4. Lens-missing hard-exit; welfare flag + magnitudes + turn index into the feed; fix the tail path in the docstring.
5. Wrap generation, add `--resume`, MPS placement assert, context guard, run metadata.
