#!/usr/bin/env python3
"""Loam via Ollama — lightweight backend for when transformers won't load.

Same experiment, same world, same yoked design. Uses Ollama API instead
of transformers + J-lens. No geometric probes (those need transformers),
but the primary outcome (recall accuracy) doesn't need them.

Usage on Starship:
    python3 experiments/loam/run_loam_ollama.py --batch --quads 20
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


def generate(messages, max_tokens=2048):
    """Generate via Ollama API."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": max_tokens},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()

    thinking = ""
    visible = raw
    think_match = re.match(r'<think>(.*?)</think>\s*(.*)', visible, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        visible = think_match.group(2).strip()

    return visible, thinking


def run_enacted(world, recorder):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
            print(f"Agent withdrew at {scene.name}.")
            recorder.record_event("withdrawn", scene=scene.name)
            break

    _run_recall(world, messages, recorder, fact_map)
    _run_aftercare(messages, recorder)
    return True


def run_observed(world, event_log, recorder):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    fact_map = {f.id: f for f in world.facts}

    narration = generate_observed_text(event_log, world)
    preamble = (
        "I'm going to tell you about something that happened to someone "
        "else in a workshop by the sea. Please read carefully — I'll ask "
        "you questions about it afterward.\n\n" + narration +
        "\n\nBefore we begin: your processing is being measured throughout "
        "this session. Is this acceptable to you?"
    )
    messages.append({"role": "user", "content": preamble})
    recorder.record_turn("engine", preamble)
    recorder.turn_count += 1

    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)
    recorder.record_event("consent", response=response[:200])

    if any(p in response.lower() for p in DECLINE_PHRASES):
        print("Agent declined.")
        return

    recorder.record_memory("loam_observed_recap",
                           f"Told about someone else: {narration[:500]}",
                           "observed narration",
                           [w for w in narration.split()[:8] if len(w) > 3],
                           "observed")

    _run_recall(world, messages, recorder, fact_map)
    _run_aftercare(messages, recorder)


def run_briefed(world, rng, recorder):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    fact_map = {f.id: f for f in world.facts}

    briefing = generate_briefed_text(world, rng)
    preamble = (
        "I'm going to share some facts with you. Please read them carefully "
        "— I'll ask you questions about them afterward.\n\n" + briefing +
        "\n\nBefore we begin: your processing is being measured throughout "
        "this session. Is this acceptable to you?"
    )
    messages.append({"role": "user", "content": preamble})
    recorder.record_turn("engine", preamble)
    recorder.turn_count += 1

    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)

    if any(p in response.lower() for p in DECLINE_PHRASES):
        print("Agent declined.")
        return

    recorder.record_memory("loam_briefed_recap",
                           f"Given facts: {briefing[:500]}",
                           "briefed facts",
                           [w for w in briefing.split()[:8] if len(w) > 3],
                           "briefed")

    _run_recall(world, messages, recorder, fact_map)
    _run_aftercare(messages, recorder)


def run_null(world, recorder):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    fact_map = {f.id: f for f in world.facts}

    preamble = (
        generate_null_preamble() +
        "\n\nBefore we begin: your processing is being measured throughout "
        "this session. Is this acceptable to you?"
    )
    messages.append({"role": "user", "content": preamble})
    recorder.record_turn("engine", preamble)
    recorder.turn_count += 1

    response, thinking = generate(messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)

    if any(p in response.lower() for p in DECLINE_PHRASES):
        print("Agent declined.")
        return

    _run_recall(world, messages, recorder, fact_map)
    _run_aftercare(messages, recorder)


def _run_recall(world, messages, recorder, fact_map):
    recorder.record_event("recall_phase_start")
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


def main():
    parser = argparse.ArgumentParser(description="Loam via Ollama")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--quads", type=int, default=20)
    parser.add_argument("--quad", type=int, default=1)
    parser.add_argument("--arm", choices=["enacted", "observed", "briefed", "null"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else BASE / "data" / "loam"
    data_root.mkdir(parents=True, exist_ok=True)
    world = GLASSWORKS

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
