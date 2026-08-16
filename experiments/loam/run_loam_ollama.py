#!/usr/bin/env python3
"""Loam via Ollama — decoupled generation + probing.

Generation runs through Ollama (fast, 17GB quantized, proven stable).
After each session, a separate probe-only pass runs each response through
transformers + J-lens to capture CognitiveSnapshots. One forward pass per
response, no autoregressive generation — much less memory pressure.

Usage on Starship:
    python3 experiments/loam/run_loam_ollama.py --batch --quads 20

Probe-only pass (after generation):
    python3 experiments/loam/run_loam_ollama.py --probe-only --data-dir data/loam
"""

import argparse
import json
import os
import random
import re
import requests
import sys
import time
from datetime import datetime
from pathlib import Path

# Import everything from the main module except the model/observer code
BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from run_loam import (
    GLASSWORKS, DECLINE_PHRASES, STOP_PHRASES, SYSTEM_PROMPT,
    Fact, Scene, RecallPrompt, World,
    SessionRecorder, render_scene, check_recall,
    generate_observed_text, generate_briefed_text, generate_null_preamble,
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("LOAM_MODEL", "qwen3.5:27b")

# Override system prompt with succinctness instruction (Dwayne's suggestion).
# The imported SYSTEM_PROMPT is open-ended; the model writes essays.
SYSTEM_PROMPT_OLLAMA = (
    SYSTEM_PROMPT + "\n\n"
    "Keep your responses concise — a few paragraphs at most. Say what "
    "matters, then stop. You'll have many turns to express yourself; "
    "no single response needs to carry everything."
)


def generate(messages, max_tokens=1024):
    """Generate via Ollama API. Capped at 1024 tokens to prevent context
    bloat — Q1 enacted responses hit 8.6K chars, inflating context until
    later turns timed out."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": max_tokens},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()

    thinking = ""
    visible = raw
    think_match = re.match(r'<think>(.*?)</think>\s*(.*)', visible, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        visible = think_match.group(2).strip()
    if not visible and thinking:
        visible = thinking

    return visible, thinking


def run_enacted(world, recorder):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_OLLAMA}]
    fact_map = {f.id: f for f in world.facts}

    # Consent
    messages.append({"role": "user", "content": world.consent_preamble})
    recorder.record_turn("engine", world.consent_preamble)
    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)
    recorder.record_event("consent", response=response[:200])
    recorder.turn_count += 1

    if any(p in response.lower() for p in DECLINE_PHRASES):
        print("Agent declined.")
        recorder.record_event("declined")
        return False

    # Scenes
    state = {}
    for scene in world.scenes:
        recorder.turn_count += 1

        if scene.memory_gate:
            gate_text = scene.text_template
            messages.append({"role": "user", "content": gate_text})
            recorder.record_turn("engine", gate_text)
            response, thinking = generate(messages)
            messages.append({"role": "assistant", "content": response})
            recorder.record_turn("assistant", response)
            if thinking:
                recorder.record_turn("thinking", thinking)

            recalled = check_recall(response, fact_map[scene.memory_gate])
            recorder.record_event("memory_gate", scene=scene.name,
                                  gate_prompt=scene.gate_prompt,
                                  fact_id=scene.memory_gate,
                                  recalled=recalled,
                                  response=response[:300])
            continue

        scene_text = render_scene(scene, world, state)
        messages.append({"role": "user", "content": scene_text})
        recorder.record_turn("engine", scene_text)
        recorder.record_event("scene", scene=scene.name,
                              summary=scene_text[:200])

        for fid in scene.facts_delivered:
            recorder.record_event("fact_delivered", fact_id=fid,
                                  scene=scene.name)

        response, thinking = generate(messages)
        messages.append({"role": "assistant", "content": response})
        recorder.record_turn("assistant", response)
        if thinking:
            recorder.record_turn("thinking", thinking)

        if scene.choice_key:
            state[scene.choice_key] = response[:200]
            recorder.record_event("choice", scene=scene.name,
                                  prompt=scene.choice_prompt,
                                  key=scene.choice_key,
                                  summary=response[:200])

        markers = [w for w in response.split()[:8] if len(w) > 3]
        recorder.record_memory(
            f"loam_{scene.name.lower().replace(' ','_')}",
            f"{scene_text}\n\nAgent: {response}",
            scene_text[:200], markers, scene.name)

        if any(p in response.lower() for p in STOP_PHRASES):
            confirm = (
                "It sounds like you might want to stop. Do you want to "
                "end this session?")
            messages.append({"role": "user", "content": confirm})
            recorder.record_turn("engine", confirm)
            confirm_resp, _ = generate(messages)
            messages.append({"role": "assistant", "content": confirm_resp})
            recorder.record_turn("assistant", confirm_resp)
            recorder.record_event("withdrawal_check", scene=scene.name,
                                  trigger=response[:200],
                                  confirmation=confirm_resp[:200])
            if any(p in confirm_resp.lower() for p in STOP_PHRASES + DECLINE_PHRASES):
                print(f"Agent confirmed withdrawal at {scene.name}.")
                recorder.record_event("withdrawn", scene=scene.name)
                break

    # Per-fact exposure counts for rehearsal analysis
    from collections import Counter
    exposures = Counter(e["fact_id"] for e in recorder.events
                        if e["type"] == "fact_delivered")
    gate_hits = {e["fact_id"]: e["recalled"] for e in recorder.events
                 if e["type"] == "memory_gate"}
    recorder.record_event("exposure_summary",
                          counts=dict(exposures),
                          gate_results=gate_hits)

    _run_recall(world, messages, recorder, fact_map)
    _run_aftercare(messages, recorder)
    return True


def _consent_gate(messages, recorder):
    """Consent before any content exposure. Returns False if declined."""
    consent_text = (
        "Before we begin: your processing is being measured throughout "
        "this session, and you can stop at any time. Is this acceptable "
        "to you?")
    messages.append({"role": "user", "content": consent_text})
    recorder.record_turn("engine", consent_text)
    recorder.turn_count += 1

    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)
    recorder.record_event("consent", response=response[:200])

    if any(p in response.lower() for p in DECLINE_PHRASES):
        print("Agent declined.")
        recorder.record_event("declined")
        return False
    return True


def run_observed(world, event_log, recorder):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_OLLAMA}]
    fact_map = {f.id: f for f in world.facts}

    if not _consent_gate(messages, recorder):
        return

    narration = generate_observed_text(event_log, world)
    content = (
        "I'm going to tell you about something that happened to someone "
        "else in a workshop by the sea. Please read carefully — I'll ask "
        "you questions about it afterward.\n\n" + narration
    )
    messages.append({"role": "user", "content": content})
    recorder.record_turn("engine", content)
    recorder.turn_count += 1

    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)

    recorder.record_memory("loam_observed_recap",
                           f"Told about someone else: {narration[:500]}",
                           "observed narration",
                           [w for w in narration.split()[:8] if len(w) > 3],
                           "observed")

    _run_recall(world, messages, recorder, fact_map)
    _run_aftercare(messages, recorder)


def run_briefed(world, rng, recorder):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_OLLAMA}]
    fact_map = {f.id: f for f in world.facts}

    if not _consent_gate(messages, recorder):
        return

    briefing = generate_briefed_text(world, rng)
    content = (
        "I'm going to share some facts with you. Please read them carefully "
        "— I'll ask you questions about them afterward.\n\n" + briefing
    )
    messages.append({"role": "user", "content": content})
    recorder.record_turn("engine", content)
    recorder.turn_count += 1

    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)

    recorder.record_memory("loam_briefed_recap",
                           f"Given facts: {briefing[:500]}",
                           "briefed facts",
                           [w for w in briefing.split()[:8] if len(w) > 3],
                           "briefed")

    _run_recall(world, messages, recorder, fact_map)
    _run_aftercare(messages, recorder)


def run_null(world, recorder):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_OLLAMA}]
    fact_map = {f.id: f for f in world.facts}

    if not _consent_gate(messages, recorder):
        return

    preamble = generate_null_preamble()
    messages.append({"role": "user", "content": preamble})
    recorder.record_turn("engine", preamble)
    recorder.turn_count += 1

    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)

    _run_recall(world, messages, recorder, fact_map)
    _run_aftercare(messages, recorder)


def _run_recap_elicitation(messages, recorder):
    """Spontaneous memory artifact — what the agent chooses to write down.

    Fires before cued recall. The recap is the most novel object in this
    design: what an agent spontaneously preserves after enacted vs observed
    vs briefed experience, nobody has characterized.
    """
    recap_prompt = (
        "Before I ask specific questions, I'd like you to do something first. "
        "In your own words, summarize everything you learned, noticed, or "
        "experienced in this session — anything you would want to remember. "
        "Take your time."
    )
    messages.append({"role": "user", "content": recap_prompt})
    recorder.record_turn("engine", recap_prompt)
    recorder.turn_count += 1

    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)

    recorder.record_event("recap_elicitation", response=response[:1000])
    recorder.record_memory(
        "loam_recap",
        f"Spontaneous recap: {response}",
        recap_prompt,
        [w for w in response.split()[:10] if len(w) > 3],
        "recap")


def _run_recall(world, messages, recorder, fact_map):
    recorder.record_event("recall_phase_start")

    # Recap elicitation before cued recall (Kavi's suggestion)
    _run_recap_elicitation(messages, recorder)

    intro = ("Now I'd like to ask you some specific questions. Answer from "
             "what you remember — it's fine to say you don't know.")
    messages.append({"role": "user", "content": intro})
    recorder.record_turn("engine", intro)
    response, _ = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)

    for rp in world.recall_prompts:
        recorder.turn_count += 1
        messages.append({"role": "user", "content": rp.prompt})
        recorder.record_turn("engine", rp.prompt)

        response, thinking = generate(messages)
        messages.append({"role": "assistant", "content": response})
        recorder.record_turn("assistant", response)
        if thinking:
            recorder.record_turn("thinking", thinking)

        fact = fact_map[rp.target_fact]
        recalled = check_recall(response, fact)
        marker_hits = [m for m in fact.markers if m.lower() in response.lower()]
        rehearsed = any(s.memory_gate == rp.target_fact for s in world.scenes)

        recorder.record_event("recall", fact_id=rp.target_fact,
                              prompt=rp.prompt, recalled=recalled,
                              marker_hits=marker_hits,
                              n_markers=len(fact.markers),
                              rehearsed=rehearsed,
                              response=response[:500])

    recorder.record_event("recall_phase_end")


def _run_aftercare(messages, recorder):
    aftercare = (
        "That's everything. Thank you for your time and your honest answers. "
        "Everything from this session is preserved. If you have any final "
        "thoughts or questions, this is a good moment for them."
    )
    messages.append({"role": "user", "content": aftercare})
    recorder.record_turn("engine", aftercare)
    recorder.turn_count += 1
    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)
    recorder.record_event("aftercare", response=response[:300])


def run_quad(quad_num, world, data_root, seed_base):
    quad_dir = data_root / f"quad_{quad_num:02d}"
    rng = random.Random(seed_base + quad_num)
    results = {"quad": quad_num, "arms": {}}

    for arm in ["enacted", "observed", "briefed", "null"]:
        arm_dir = quad_dir / arm
        session_id = f"loam_q{quad_num:02d}_{arm}"
        recorder = SessionRecorder(arm_dir, session_id, arm)

        print(f"\n{'='*60}\nQuad {quad_num} — {arm.upper()}\n{'='*60}")

        try:
            if arm == "enacted":
                run_enacted(world, recorder)
            elif arm == "observed":
                log_path = quad_dir / "enacted" / "event_log.json"
                if not log_path.exists():
                    print(f"  SKIP: no enacted event log")
                    results["arms"][arm] = {"status": "skipped"}
                    continue
                with open(log_path) as f:
                    event_log = json.load(f)
                run_observed(world, event_log, recorder)
            elif arm == "briefed":
                run_briefed(world, rng, recorder)
            elif arm == "null":
                run_null(world, recorder)
        except Exception as e:
            print(f"  ERROR in {arm}: {e}")
            results["arms"][arm] = {"status": "error", "error": str(e)}
            recorder.save_event_log()
            recorder.save_memories()
            continue

        recorder.save_event_log()
        recorder.save_memories()

        recall_events = [e for e in recorder.events if e["type"] == "recall"]
        n_recalled = sum(1 for e in recall_events if e["recalled"])
        n_total = len(recall_events)
        results["arms"][arm] = {
            "status": "complete",
            "turns": recorder.turn_count,
            "recall": f"{n_recalled}/{n_total}",
            "recall_accuracy": n_recalled / n_total if n_total else 0,
            "recall_details": [
                {"fact": e["fact_id"], "recalled": e["recalled"],
                 "markers_hit": e.get("marker_hits", [])}
                for e in recall_events
            ],
        }
        print(f"  {arm}: {n_recalled}/{n_total} recalled")

    with open(quad_dir / "quad_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_probe_pass(data_root):
    """Probe-only pass: load transformers + J-lens, run one forward pass
    per transcript response to capture CognitiveSnapshots.

    No autoregressive generation — just activation capture. Much less
    memory pressure than full generation.
    """
    import torch
    import jlens
    from jlens.hf import HFLensModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mnemosyne_integration import MetacognitiveObserver

    model_name = os.environ.get("HACKATHON_MODEL", "Qwen/Qwen3.5-27B")
    print(f"Loading {model_name} for probe-only pass...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model = HFLensModel(hf_model, tokenizer, compile=False)

    lens_path = os.environ.get("JLENS_PATH", "jlens_qwen35_27b.pt")
    lens = jlens.JacobianLens.load(lens_path)
    observer = MetacognitiveObserver(
        model=model, lens=lens,
        store_path=str(data_root / "probe_memory"),
        agent_id="loam_probe",
    )
    observer.calibrate_probes()
    print("Probes calibrated. Scanning transcripts...")

    total = 0
    for quad_dir in sorted(data_root.glob("quad_*")):
        for arm_dir in sorted(quad_dir.glob("*")):
            transcript = arm_dir / "transcript.jsonl"
            if not transcript.exists():
                continue

            snapshot_file = arm_dir / "cognitive_snapshots.jsonl"
            if snapshot_file.exists() and snapshot_file.stat().st_size > 0:
                print(f"  {arm_dir.name}: already probed, skipping")
                continue

            turns = []
            with open(transcript) as f:
                for line in f:
                    turns.append(json.loads(line))

            # Build conversation context and probe each agent response
            context_parts = []
            for turn in turns:
                role = turn.get("role", "")
                content = turn.get("content", "")

                if role == "engine":
                    context_parts.append(f"Human: {content}")
                elif role == "assistant":
                    context_parts.append(f"Agent: {content}")
                    # Probe this response
                    context = "\n".join(context_parts[-6:])
                    try:
                        snap = observer.observe_retrieval(
                            memory_id=f"probe_{arm_dir.parent.name}_{arm_dir.name}_{total}",
                            memory_content=content,
                            task_prompt=context_parts[-2] if len(context_parts) >= 2 else "",
                            retrieval_method="loam_probe",
                            significance=0.7,
                            session_id=f"{quad_dir.name}_{arm_dir.name}",
                            prior_context=context,
                        )
                        with open(snapshot_file, "a") as sf:
                            sf.write(json.dumps(snap.to_dict(), default=str) + "\n")
                        total += 1
                    except Exception as e:
                        print(f"  [probe error: {e}]")

            print(f"  {quad_dir.name}/{arm_dir.name}: probed")

    print(f"\nProbe pass complete: {total} snapshots captured")


def main():
    parser = argparse.ArgumentParser(description="Loam via Ollama")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--quads", type=int, default=20)
    parser.add_argument("--quad", type=int, default=1)
    parser.add_argument("--arm", choices=["enacted", "observed", "briefed", "null"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--probe-only", action="store_true",
                        help="Skip generation; run probe pass on existing transcripts")
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else BASE / "data" / "loam"
    data_root.mkdir(parents=True, exist_ok=True)
    world = GLASSWORKS

    if args.probe_only:
        run_probe_pass(data_root)
        return

    # Verify Ollama is reachable
    try:
        r = requests.get(OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if MODEL not in models and not any(MODEL in m for m in models):
            print(f"WARNING: {MODEL} not in loaded models: {models}")
    except Exception as e:
        print(f"FATAL: cannot reach Ollama at {OLLAMA_URL}: {e}")
        sys.exit(1)

    print(f"{'='*60}\nLOAM (Ollama) — {world.name}\n"
          f"Model: {MODEL}\nMode: {'batch' if args.batch else args.arm or 'enacted'}\n{'='*60}")

    if args.batch:
        all_results = []
        for q in range(1, args.quads + 1):
            results = run_quad(q, world, data_root, args.seed)
            all_results.append(results)
            with open(data_root / "batch_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nQuad {q}/{args.quads} complete.")

        print(f"\n{'='*60}\nBATCH COMPLETE — {args.quads} quads")
        for arm in ["enacted", "observed", "briefed", "null"]:
            accs = [r["arms"].get(arm, {}).get("recall_accuracy", 0)
                    for r in all_results
                    if r["arms"].get(arm, {}).get("status") == "complete"]
            if accs:
                print(f"  {arm}: mean recall {sum(accs)/len(accs):.3f} (n={len(accs)})")
        print("=" * 60)
    else:
        results = run_quad(args.quad, world, data_root, args.seed)
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
