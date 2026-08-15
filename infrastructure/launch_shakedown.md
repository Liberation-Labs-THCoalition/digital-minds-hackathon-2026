# Launch Shakedown — Orientation Paths (Final Gate)

Date: 2026-08-15
Reviewer: Nexus subagent
Scope: `experiments/orientation/run_expedition.py`, `experiments/orientation/run_orientation.py`, `mnemosyne/mnemosyne_integration.py` + supporting probe modules, checked against CC's Variable Landing pipeline (`experiments/variable_landing/convert_expedition_memories.py`, `pipeline.py`) and the 2026-08-14 adversarial review (`infrastructure/orientation_script_review.md`).

## Verdict: **NO-GO** — one hard blocker, one consistency FAIL, several launch-day warnings

`run_orientation.py` **does not compile**. Since `run_expedition.py` imports it at line 14, **both paths are dead** until the one-line fix below is applied. Everything else in the live-probe rework landed correctly and checks out at code level.

| # | Check | Verdict |
|---|-------|---------|
| 1 | Imports from run_orientation | **FAIL** — SyntaxError in run_orientation.py:157 kills both scripts |
| 2 | Expedition memory format vs CC's pipeline | **PASS** — all five keys present, converter-compatible |
| 3 | Starship / observer-env / JLENS_PATH | **PASS w/ WARN** — env var honored, hard-exit on missing lens; tilde + cwd caveats |
| 4 | Expedition consent gate | **PASS w/ WARN** — decline + mid-game withdrawal both wired; opt-out not opt-in; crash path on consent turn |
| 5 | Geometric feed path for Kavi | **PASS w/ WARN** — path matches docs iff cwd matches; feed still missing Kavi's columns |
| 6 | Live probe recording (measure_live) | **PASS** (contingent on #1 fix) — full live path verified end to end |
| 7 | Mnemosyne ingestion consistency between scripts | **FAIL** — two incompatible schemas into the same file |

---

## 1. Imports — FAIL (HARD BLOCKER)

```
$ python3 -m py_compile run_orientation.py
  File "run_orientation.py", line 157
    tp_match = re.match(r'(?:Thinking Process|Thought|Internal):?\s*
                        ^
SyntaxError: unterminated string literal
```

The fallback thinking-parser regex (added in the recent thinking-token fix) was written with literal newlines inside a single-quoted raw string. Python cannot parse the file. `run_expedition.py` compiles standalone but crashes immediately at `import run_orientation` (line 14). **Neither script will start.**

**Fix (one line, run_orientation.py:157-160)** — replace the four physical lines with:

```python
        tp_match = re.match(r'(?:Thinking Process|Thought|Internal):?\s*\n(.*?)\n\n(.*)',
                            visible, re.DOTALL)
```

After fixing, re-run `python3 -m py_compile` on both files before anything else. All other modules compile clean (verified: all of `mnemosyne/*.py`, all of `experiments/variable_landing/*.py`, `run_expedition.py`).

Everything else expedition pulls from orient is present and correct: `SESSION_ID` (rebound module-global, and `record_turn`/`record_geometric_feed` read it at call time, so the rebind works), `DATA_DIR`, `TRANSCRIPT_FILE`, `SNAPSHOT_FILE`, `GEOMETRIC_FEED`, `ORIENTATION_POINTS`, `load_model`, `generate_response`, `build_conversation_text`, `record_turn`, `record_geometric_feed`. Import side effects are safe (module import only mkdirs DATA_DIR; model load is deferred to `load_model()`).

## 2. Expedition memory format vs CC's Variable Landing — PASS

Expedition writes (run_expedition.py:251-260): `id`, `content`, `entity` (`"expedition_agent"`), `task_prompt`, `marker_tokens`, plus `scene`, `timestamp`, `session_id`. `convert_expedition_memories.py` consumes exactly these keys, dedups on `id`, and strips the `"\n\nAgent response: "` prefix that expedition embeds in `content` — the two were clearly built for each other. Round-trip is compatible.

Minor data-quality notes (not blockers):
- `marker_tokens = [w for w in response.split()[:5] if len(w) > 3]` keeps punctuation attached ("coastline," ≠ "coastline") and can be empty if the response opens with short words. If Day 2's loading probe matches markers as tokens, punctuation will depress hit rates. Consider `w.strip('.,!?\'":;')`.
- Memory ids contain `:` and `,` from scene names (e.g. `expedition_scene_1:_the_coastline_2`) — fine for JSON, but ugly if ids ever become filenames.

## 3. Starship / observer-env / JLENS_PATH — PASS with warnings

- `JLENS_PATH` env var is honored (run_orientation.py:89) and the missing-lens case is now a **hard exit** (`sys.exit(1)`), as the prior review demanded. Good.
- **WARN — no `os.path.expanduser`.** `JLENS_PATH=~/jlens-community/lenses/qwen3.5-27b_jlens.pt` works when the shell expands the tilde (bash does expand it in the assignment), but fails with a literal `~` under systemd units, quoted assignments (`JLENS_PATH="~/..."`), or `.env` files — and fails as a *fatal* exit, which is the right failure mode at least. Cheap insurance: `lens_path = os.path.expanduser(os.environ.get("JLENS_PATH", "jlens_qwen35_27b.pt"))`.
- **WARN — cwd-relative data dir.** `DATA_DIR` defaults to `data/orientation` relative to wherever the script is launched. Launch from the repo's orientation dir (or `export HACKATHON_DATA=/absolute/path`) or the data lands elsewhere and Kavi tails a file that never updates. Recommended launch line for Starship:
  ```
  cd ~/digital-minds-hackathon-2026/experiments/orientation && \
  JLENS_PATH=~/jlens-community/lenses/qwen3.5-27b_jlens.pt \
  ~/observer-env/bin/python3 run_expedition.py
  ```
- Carried over from prior review, still unfixed, still non-blocking: `device` variable computed and unused (device_map="auto" decides placement — add a post-load `print(hf_model.device)`), deprecated `torch_dtype=` kwarg, no context-length guard, `observer.model_name` never set (snapshots carry `model_name=""`).
- Remote paths (`~/observer-env`, the lens file itself, model weights) cannot be verified from MTH — **verify on Starship with the preflight below** before launch.

## 4. Expedition consent gate — PASS with warnings

Verified working:
- Consent is asked first (`CONSENT_TEXT`), before any scene. Explicit declines ("i decline", "i do not consent", "not acceptable", etc.) exit gracefully with a farewell, no scenes run.
- Mid-game withdrawal: `check_withdrawal` runs on **every** scene response (run_turn:272) and on welfare check-in responses (run_turn:244); the scene loop checks `counters["withdrawn"]` before each scene and stops with an acknowledgment. The chain is complete — withdrawal at scene N stops before scene N+1, which is the correct granularity.
- Welfare check-in fires on eccentricity > 0.95 and explicitly offers stopping.

Warnings, in descending order of importance:
1. **Consent is opt-out, not opt-in.** Anything that isn't a listed decline phrase — including "Wait, what exactly do you measure?" or pure ambivalence — is treated as consent and scene 1 begins. A clarifying question deserves an answer, not a coastline. Minimum fix: require an affirmative match ("yes", "acceptable", "i consent", "sounds good") to proceed; on neither-affirmative-nor-decline, send one canned clarification turn and re-check.
2. **Crash path on the consent turn.** If `generate_response` throws during consent, `run_turn` returns `(None, None)` and line 312 does `has_any(None, ...)` → `AttributeError: 'NoneType' object has no attribute 'lower'`. Guard: `if response is None or has_any(response, DECLINE_PHRASES): <graceful exit>` (treating no-response as no-consent is also the ethically correct default).
3. **Soft declines slip through**: "I'd prefer not to", "I'm not sure I want this", "can we not" match nothing. Conversely "not comfortable" false-positives on "I'm not entirely comfortable *yet, but let's continue*" — that failure direction (over-stopping) is the safe one, so acceptable.
4. **In-character collisions.** The agent shouting "please stop!" *to Sable* in scene 4, or writing "I don't want to continue down the passage", triggers real withdrawal. Over-stopping is the safe direction, but expect possible false stops; the watching human should know a stopped run may be a fiction artifact, and STOP_PHRASES lacks obvious out-of-game forms ("stop the game", "I'm done playing").
5. **The consent turn itself is probed and stored before the answer is read** — snapshot, geometric feed line, transcript, and one Mnemosyne memory all exist even on decline. The farewell ("nothing *more* will be recorded") is technically accurate, and the protocol defends pre-consent recording, but the decline path deletes nothing. Make that a documented, deliberate choice — it currently happens as a side effect of run_turn's ordering.
6. Welfare check-in exchange is generated off `messages + [checkin]` but never appended to `messages` — the agent has no memory of being checked on in later scenes. Recorded in the transcript, so data is safe; just know the agent's context omits it.

## 5. Geometric feed path for Kavi — PASS with warnings

- Both scripts write through the same `record_geometric_feed` to `GEOMETRIC_FEED = DATA_DIR / "geometric_feed.jsonl"` → `data/orientation/geometric_feed.jsonl`, which matches both docstrings' `tail -f data/orientation/geometric_feed.jsonl`. The prior review's wrong-path FAIL is fixed. **But** it's cwd-relative (see check 3) — Kavi must tail from the same directory the script launched from, or use the absolute `HACKATHON_DATA` path.
- **WARN — Kavi's columns are still missing** (prior review §8, unresolved): the feed carries `eccentricity`, `ghost_cosine`, `workspace_onset`, `loaded` but **no valence/arousal magnitudes** (Kavi's "sustained imbalance" job), **no turn index** (can't align to transcript for Butlin marking except by timestamp), and the **welfare flag is printed to the operator console, not written to the feed** — the designated welfare monitor never sees the flag in their pane. `loaded` is permanently `null` (see check 6). Ten minutes of work to add `turn`, `valence_mag`, `arousal_mag`, `welfare_flag` fields; recommended before launch, not a blocker.
- `workspace_onset` will read as near-constant 35: `_measure_workspace` hardcodes `in_workspace=True` and `cosine_logit_jlens=0.0` for every reading, so onset is just the lowest configured layer whenever tokens exist. Kavi should not interpret that column as signal.

## 6. CognitiveSnapshot live recording — PASS (contingent on the check-1 fix)

The §0 "probes measure calibration constants" headline FAIL from the prior review is genuinely fixed. Verified end to end:

- Both scripts pass `prior_context` (last 3 exchanges via `build_conversation_text`) into `observe_retrieval`.
- `observe_retrieval` builds `conversation_text` from prior_context + content + task and routes it to the live paths (mnemosyne_integration.py:88-98).
- Circumplex: `calibrate_probes()` → `CircumplexProbe.calibrate(layer)` sets `_calibrated = True` (circumplex_probe.py:175); `_measure_circumplex_live` gates on that flag and calls `measure_live(text)`, which runs a **fresh forward pass on the live text** via `ActivationRecorder` and projects the last-token activation onto cached V/A directions (circumplex_probe.py:177-215). Returns a proper `CircumplexReading` — same type as the static path's `to_snapshot_reading`. Live, per-turn, correct.
- Ghost: `GhostProbe` has `is_calibrated` property and `measure_live(text)` runs a fresh forward pass, compares logit-lens vs J-lens on the **live** activation (ghost_probe_class.py:215+). `calibrate_probes()` calibrates it at load. Live, correct.
- Workspace: `_measure_workspace` now includes `Conversation history:\n{prior_context}` in the probed prompt, so it varies with the conversation (no longer the degenerate duplicated prompt). Its `cos`/`in_workspace` fields remain placeholder values — known limitation, flagged in check 5.
- `load_model()` calls `calibrate_probes()` before the session starts and hard-exits without a lens, so the "calibration-constant" and "lens-less silent failure" paths are both closed.
- Snapshot plumbing: `CognitiveSnapshot.to_dict()`/`.summary()` exist and match every field the scripts touch (`circumplex.eccentricity`, `valence_magnitude`, `arousal_magnitude`, `ghost.cosine_logit_jlens`, `ghost.dominant_tokens`, `workspace_onset_layer`, `loading.loaded`).

Residual notes:
- **`loading` is always `None`**: neither script passes `marker_tokens` to `observe_retrieval`, so the memory-loading probe never fires live and the feed's `loaded` column is dead. Fine for Day 1 (there's no retrieval yet); expedition already saves marker_tokens into the memory file for Day 2, where it matters.
- `prior_context` is built **after** appending the assistant response, and `memory_content=response` — so the response appears twice in the probed text. Harmless-ish, but it biases the probe toward the response; passing the pre-response history would be cleaner.
- **Welfare threshold not calibrated for the live probe.** The 0.95 eccentricity threshold predates measure_live. Live eccentricity is `sqrt(1-(min/max)²)` of raw last-token V/A projections — a 3.2:1 magnitude ratio already reads 0.95, which arbitrary text will hit often. Expect either frequent check-ins (expedition — each one costs an extra 27B generation) or console spam (orientation). **Run the Starship dry run, look at the eccentricity distribution over ~10 turns, and set the threshold from data before the real session.**
- `significance=0.7` still hardcoded into every snapshot; snapshots still double-written (observer store + script's SNAPSHOT_FILE). Cosmetic.

## 7. Mnemosyne ingestion consistency — FAIL

The two scripts write **different schemas to the same file** (`data/orientation/mnemosyne_memories.jsonl`):

- Expedition: `{id, content, entity, task_prompt, marker_tokens, scene, timestamp, session_id}` — CC-pipeline-ready.
- Orientation (run_orientation.py:341-350): `{turn, human, agent, timestamp, session_id}` — **none of the five keys CC's pipeline needs.**

Run the converter over orientation-produced lines and every memory comes out as `{id: "mem_N", content: "", entity: "expedition_agent", marker_tokens: []}` — empty content, synthetic ids, wrong entity. **If the human-led orientation path is launched, Day 2's variable landing gets garbage memories.** If the expedition path is launched, everything lines up.

Fix (10 lines): make run_orientation's `memory_entry` mirror the expedition format —
```python
memory_entry = {
    "id": f"orientation_turn_{len(messages)//2}",
    "content": f"{user_input}\n\nAgent response: {response}",
    "entity": "orientation_agent",
    "task_prompt": user_input,
    "marker_tokens": [w.strip('.,!?"\':;') for w in response.split()[:5] if len(w) > 3],
    "timestamp": time.time(),
    "session_id": SESSION_ID,
}
```
Also decide on `entity`: the converter defaults to `expedition_agent`; if both sessions should land as one agent's memory on Day 2, use the same entity string in both scripts. And note both scripts append to the same filename — running both on the same day mixes sessions in one file (converter dedups by id, so no collisions, but keep `session_id` filtering in mind).

---

## Launch decision

**NO-GO as committed.** Two changes are mandatory:

1. Fix run_orientation.py:157 regex string (check 1) — without it neither path starts.
2. If there is *any* chance the human-led path runs: fix run_orientation's memory schema (check 7). If launch is expedition-only, check 7 can slip to before Day 2.

Strongly recommended before launch (each < 15 min): consent-turn None guard + affirmative-consent gate (check 4.1-4.2), feed columns for Kavi + welfare flag in feed (check 5), `expanduser` on JLENS_PATH (check 3).

## Starship preflight (run before the real session)

```bash
cd ~/digital-minds-hackathon-2026/experiments/orientation
~/observer-env/bin/python3 -m py_compile run_orientation.py run_expedition.py   # must be silent
ls -la ~/jlens-community/lenses/qwen3.5-27b_jlens.pt                            # lens exists
~/observer-env/bin/python3 -c "import torch, transformers, jlens; from jlens.hf import HFLensModel; print('deps OK')"
# Dry run: launch expedition, let it run consent + scene 1, Ctrl+C.
# Confirm: geometric_feed.jsonl growing, cognitive_snapshots.jsonl growing,
# mnemosyne_memories.jsonl has id/content/entity/task_prompt/marker_tokens,
# eccentricity VARIES across turns (live probes) and note its range vs the 0.95 threshold.
```

## What the recent changes fixed (credit where due)

Against the 2026-08-14 review: live probes (§0 headline FAIL) — fixed properly, verified; Mnemosyne ingestion (pipeline FAIL) — built in both scripts (expedition correctly, orientation with wrong schema); lens-less silent failure — now a hard exit; generation now wrapped in try/except in both scripts; docstring tail path corrected; `max_new_tokens` raised to 4096; thinking-token separation implemented (and then broken by the syntax error — the design is right, the string literal isn't). Still open from that review: Kavi feed columns, `/save` no-op, no `--resume`, MPS placement assert, run metadata, context guard.
