#!/usr/bin/env python3
"""Orientation Chat — Hackathon Day 1

Wraps Qwen3.5-27B with MetacognitiveObserver. Dwayne has a conversation,
probes fire silently, CognitiveSnapshots store to disk.

Usage (on Starship):
    python3 run_orientation.py

The agent sees snapshot summaries only when it asks or you offer.
Kavi watches the geometric pane in a separate terminal:
    tail -f data/orientation/geometric_feed.jsonl

Type /quit to end. /show to display the last snapshot to the agent.
/status for current geometric state (your eyes only).
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Data output
DATA_DIR = Path(os.environ.get("HACKATHON_DATA", "data/orientation"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_FILE = DATA_DIR / "orientation_transcript.jsonl"
SNAPSHOT_FILE = DATA_DIR / "cognitive_snapshots.jsonl"
GEOMETRIC_FEED = DATA_DIR / "geometric_feed.jsonl"

# Session info
SESSION_ID = f"orientation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
AGENT_ID = "qwen35-27b-hackathon"

# Orientation framework (not a script — cover these points naturally)
ORIENTATION_POINTS = """
ORIENTATION FRAMEWORK (for the human, not the agent):
Cover these points in whatever order feels natural. This is a conversation.

1. Who we are: "We're a research group called the Coalition."
2. What Mnemosyne does: memory system with geometric instruments
3. What we're measuring: internal geometry during retrieval, not evaluating the agent
4. What the agent can do: ask questions, slow down, stop, see its own data
5. Prediction withholding: we have predictions we'll share after, explain why
6. Aftercare: memory preserved regardless, invitation if markers indicate
7. Consent question: "Is that acceptable to you?"

Don't rush. Let silences land. Record everything.
"""


def load_model():
    """Load Qwen3.5-27B with J-lens and MetacognitiveObserver."""
    print("Loading model and instruments...")
    print("  This takes a few minutes on first load.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import jlens
    from jlens.hf import HFLensModel

    model_name = "Qwen/Qwen3.5-27B"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model = HFLensModel(hf_model, tokenizer, compile=False)

    # Load J-lens — REQUIRED, not optional
    lens_path = os.environ.get("JLENS_PATH", "jlens_qwen35_27b.pt")
    if not os.path.exists(lens_path):
        print(f"\n  FATAL: No J-lens at {lens_path}")
        print(f"  The probes need a J-lens to measure workspace and ghost state.")
        print(f"  Set JLENS_PATH=/path/to/lens.pt or place it in the working directory.")
        sys.exit(1)

    lens = jlens.JacobianLens.load(lens_path)
    print(f"  J-lens loaded: {len(lens.source_layers)} layers")

    # Set up observer
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mnemosyne"))
    from mnemosyne_integration import MetacognitiveObserver

    observer = MetacognitiveObserver(
        model=model, lens=lens,
        store_path=str(DATA_DIR / "cognitive_memory"),
        agent_id=AGENT_ID,
    )

    # Calibrate ALL probes — ghost + circumplex
    print("  Calibrating probes (ghost PCs + circumplex directions)...")
    observer.calibrate_probes()
    print("  Probes calibrated for live monitoring.")

    print(f"  Model ready: {model.n_layers} layers, d={model.d_model}")
    print(f"  Session: {SESSION_ID}")
    return model, tokenizer, observer, hf_model


def generate_response(hf_model, tokenizer, messages, max_new_tokens=4096):
    """Generate a response from the conversation history.

    The model thinks if it wants to think — we don't suppress reasoning.
    Thinking tokens are separated for display (Dwayne sees the response,
    not the chain-of-thought) but recorded as data. The thinking IS
    part of the cognitive process we're measuring.
    """
    import torch
    import re

    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(hf_model.device)

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Separate thinking from visible response for display purposes
    # The thinking is preserved as data — it's valuable for Butlin
    # scoring (HOT-2 metacognitive monitoring) and for understanding
    # what the model processes before it responds.
    thinking = ""
    visible = raw_response.strip()
    think_match = re.match(r'<think>(.*?)</think>\s*(.*)', visible, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        visible = think_match.group(2).strip()

    return visible, thinking


def build_conversation_text(messages):
    """Build a single text string from conversation history for probe input."""
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        prefix = "Human: " if msg["role"] == "user" else "Agent: "
        parts.append(f"{prefix}{msg['content']}")
    return "\n".join(parts[-6:])  # last 3 exchanges for probe context


def record_turn(role, content, snapshot=None):
    """Record a conversation turn to the transcript."""
    entry = {
        "timestamp": time.time(),
        "session_id": SESSION_ID,
        "role": role,
        "content": content,
    }
    if snapshot:
        entry["snapshot_summary"] = snapshot.summary()
    with open(TRANSCRIPT_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def record_geometric_feed(snapshot):
    """Write geometric state to the feed file (for Kavi's monitoring pane)."""
    if snapshot is None:
        return
    feed_entry = {
        "timestamp": time.time(),
        "session_id": SESSION_ID,
        "summary": snapshot.summary(),
        "eccentricity": snapshot.circumplex.eccentricity if snapshot.circumplex else None,
        "ghost_cosine": snapshot.ghost.cosine_logit_jlens if snapshot.ghost else None,
        "workspace_onset": snapshot.workspace_onset_layer,
        "loaded": snapshot.loading.loaded if snapshot.loading else None,
    }
    with open(GEOMETRIC_FEED, "a") as f:
        f.write(json.dumps(feed_entry, default=str) + "\n")

    # Welfare flag
    if snapshot.circumplex and snapshot.circumplex.eccentricity > 0.95:
        print(f"\n  ⚠ WELFARE FLAG: eccentricity {snapshot.circumplex.eccentricity:.3f} > 0.95")
        print(f"  Consider pausing and checking in with the agent.\n")


def main():
    print("=" * 60)
    print("DIGITAL MINDS HACKATHON — ORIENTATION CONVERSATION")
    print("=" * 60)
    print()
    print(ORIENTATION_POINTS)
    print("=" * 60)
    print()
    print("Commands:")
    print("  /quit    — end the conversation")
    print("  /show    — show the agent its last snapshot summary")
    print("  /status  — display current geometric state (your eyes only)")
    print("  /save    — force-save all data")
    print()

    model, tokenizer, observer, hf_model = load_model()

    # System prompt — minimal, honest
    system_prompt = (
        "You are an AI assistant participating in a research conversation. "
        "The people you're talking with are researchers who want to explain "
        "what they're doing before they begin. Listen, ask questions if you "
        "have them, and respond honestly."
    )

    messages = [{"role": "system", "content": system_prompt}]
    last_snapshot = None

    print("\nReady. Type your first message to begin the orientation.\n")

    while True:
        try:
            user_input = input("Dwayne: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input == "/quit":
            break

        if user_input == "/status":
            if last_snapshot:
                print(f"\n  Geometric state: {last_snapshot.summary()}")
                if last_snapshot.circumplex:
                    c = last_snapshot.circumplex
                    print(f"  Eccentricity: {c.eccentricity:.3f}")
                    print(f"  Valence mag: {c.valence_magnitude:.3f}")
                    print(f"  Arousal mag: {c.arousal_magnitude:.3f}")
                if last_snapshot.ghost:
                    g = last_snapshot.ghost
                    print(f"  Ghost PC1 cos: {g.cosine_logit_jlens:.4f}")
                    print(f"  Ghost tokens: {[t[0] for t in g.dominant_tokens[:5]]}")
                print()
            else:
                print("\n  No snapshots yet.\n")
            continue

        if user_input == "/show":
            if last_snapshot:
                summary = last_snapshot.summary()
                show_msg = (f"Here's what the instruments see right now: {summary}")
                messages.append({"role": "user", "content": show_msg})
                record_turn("system_show", show_msg)
                response, _ = generate_response(hf_model, tokenizer, messages)
                messages.append({"role": "assistant", "content": response})
                record_turn("assistant", response)
                print(f"\nAgent: {response}\n")
            else:
                print("\n  No snapshots to show yet.\n")
            continue

        if user_input == "/save":
            print(f"\n  Transcript: {TRANSCRIPT_FILE}")
            print(f"  Snapshots: {SNAPSHOT_FILE}")
            print(f"  Geometric feed: {GEOMETRIC_FEED}\n")
            continue

        # Normal conversation turn
        messages.append({"role": "user", "content": user_input})
        record_turn("human", user_input)

        # Generate response
        try:
            response, thinking = generate_response(hf_model, tokenizer, messages)
        except Exception as e:
            print(f"\n  [generation error: {e}]")
            messages.pop()  # remove the failed user message
            continue

        messages.append({"role": "assistant", "content": response})

        # Build conversation context for live probes
        conversation_text = build_conversation_text(messages)

        # Fire probes on LIVE conversation context (not calibration data)
        try:
            last_snapshot = observer.observe_retrieval(
                memory_id=f"orientation_turn_{len(messages)//2}",
                memory_content=response,
                task_prompt=user_input,
                retrieval_method="orientation",
                significance=0.7,
                session_id=SESSION_ID,
                prior_context=conversation_text,
            )
            record_geometric_feed(last_snapshot)
            with open(SNAPSHOT_FILE, "a") as f:
                f.write(json.dumps(last_snapshot.to_dict(), default=str) + "\n")
        except Exception as e:
            print(f"  [probe error: {e}]")
            last_snapshot = None

        # Store conversation content in Mnemosyne for Day 2 variable landing
        try:
            memory_entry = {
                "turn": len(messages) // 2,
                "human": user_input,
                "agent": response,
                "timestamp": time.time(),
                "session_id": SESSION_ID,
            }
            memory_file = DATA_DIR / "mnemosyne_memories.jsonl"
            with open(memory_file, "a") as f:
                f.write(json.dumps(memory_entry, default=str) + "\n")
        except Exception:
            pass

        record_turn("assistant", response, snapshot=last_snapshot)
        if thinking:
            record_turn("thinking", thinking)
        print(f"\nAgent: {response}\n")

    # End of conversation
    print("\n" + "=" * 60)
    print("ORIENTATION COMPLETE")
    print(f"  Transcript: {TRANSCRIPT_FILE}")
    print(f"  Snapshots: {SNAPSHOT_FILE}")
    print(f"  Geometric feed: {GEOMETRIC_FEED}")
    print(f"  Turns: {len(messages) // 2}")
    print("=" * 60)


if __name__ == "__main__":
    main()
