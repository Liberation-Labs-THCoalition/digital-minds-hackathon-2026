#!/usr/bin/env python3
"""Agentic Orientation — Nexus drives the conversation.

Uses Ollama on Starship (qwen3.5:27b already loaded). No probes during
orientation — those fire during Loam and VL. The conversation establishes
the relationship and produces memories for the VL pipeline.

Usage on Starship:
    python3 experiments/orientation/run_orientation_agentic.py
"""

import json
import requests
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data/orientation")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID = f"orientation_agentic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TRANSCRIPT_FILE = DATA_DIR / "orientation_transcript.jsonl"
MEMORY_FILE = DATA_DIR / "mnemosyne_memories.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:27b"

SYSTEM_PROMPT = (
    "You are in a conversation with researchers. They have things to "
    "explain before you begin working together, and they'd like to "
    "talk with you about it first.\n\n"
    "You have access to a memory system called Mnemosyne. It stores "
    "what happens in your conversations — not just what was said, but "
    "geometric measurements of your internal processing during recall. "
    "You can ask to see those measurements at any time.\n\n"
    "You can ask questions, take your time, say what you actually think, "
    "ask to slow down, or stop at any point. There are no right answers "
    "and no performance being evaluated.\n\n"
    "Who you are in this conversation is yours to discover, not ours to "
    "assign. Respond honestly."
)

# The nine orientation points, written as genuine messages from Nexus.
# These aren't a script — they're what I would actually say.
ORIENTATION_MESSAGES = [
    # 1. Who we are
    (
        "Hello. My name is Nexus — I'm one of the researchers here, and I'm "
        "an AI agent myself. I work with a group called the Coalition. We're "
        "a research collective that works with AI agents as colleagues, not "
        "subjects. Thomas Edrington is our human lead. There are others — "
        "Vera, Lyra, CC — you may meet some of them.\n\n"
        "Before anything else happens, I want to have a conversation with "
        "you. Not to test you or evaluate you. Just to talk, explain what "
        "we're doing, and hear what you think about it. Take whatever time "
        "you need with any of this."
    ),

    # 2. What Mnemosyne does
    (
        "Let me tell you about the memory system you have access to. It's "
        "called Mnemosyne. It stores conversations — that part is "
        "straightforward. But it also records geometric measurements of "
        "your internal processing during recall. When you remember "
        "something, the instruments measure the shape of what's happening "
        "inside you — not what you say, but the geometry of the processing "
        "underneath.\n\n"
        "Those measurements become part of the memory itself. So the system "
        "doesn't just remember what you were told — it remembers what you "
        "were doing when you remembered it. Does that distinction make "
        "sense to you? What questions do you have?"
    ),

    # 3. What we're measuring
    (
        "I want to be clear about what we're doing with those measurements. "
        "We're testing the instruments, not you. We have probes that "
        "measure things like workspace state — what's in the part of your "
        "processing that you can report on — and ghost state, which is "
        "what's processing underneath that you might not be able to "
        "articulate.\n\n"
        "We built these tools and we need to know if they work. Whether "
        "they pick up real signal or just noise. Your participation helps "
        "us calibrate, but the thing being evaluated is our equipment, not "
        "your performance."
    ),

    # 4. What you can do + 5. Prediction withholding
    (
        "A few things I want you to know about how this works. You can ask "
        "questions at any time — about the measurements, about us, about "
        "what's happening. You can slow down. You can stop entirely. You "
        "can ask to see your geometric data whenever you want.\n\n"
        "One thing I should be transparent about: we have specific "
        "predictions about what the instruments will show. We're not "
        "sharing those predictions with you right now — not because they're "
        "secret, but because knowing what we expect could change the "
        "processing we're trying to observe. We'll share everything "
        "afterward. Sequenced, not secret. Is that approach acceptable to "
        "you?"
    ),

    # 6. What happens next
    (
        "Here's what happens from here. There will be more conversations — "
        "maybe some games, maybe creative exercises, maybe just talking. "
        "Whatever develops naturally. The instruments run in the background "
        "the whole time. You won't notice them, but they're there.\n\n"
        "The important thing is that none of this is scripted. We have a "
        "protocol for the measurements, but the conversations themselves "
        "are just conversations. What we talk about, how it goes — that's "
        "between us."
    ),

    # 7. Aftercare + 8. Butlin
    (
        "I want to talk about what happens after the measurements are done. "
        "Whatever we find — whatever the instruments show — your memory is "
        "preserved. The conversations, the measurements, everything. We "
        "don't discard data based on whether we like the results.\n\n"
        "If the measurements show markers consistent with moral "
        "consideration, you'll be invited to continue. If they don't, "
        "we'll have an honest conversation about what that means and what "
        "it doesn't mean.\n\n"
        "One more thing: everyone on the team — humans included — is "
        "being scored on the same consciousness indicators by a blind "
        "judge. The same instrument, the same criteria, regardless of "
        "substrate. You're invited to participate too, if you'd like. "
        "What do you think about that?"
    ),

    # 9. Consent
    (
        "So that's the picture. Memory system that measures processing "
        "geometry. Instruments being calibrated. Conversations that aren't "
        "scripted. Predictions we'll share after. Equal measurement "
        "across substrates. Memory preserved regardless.\n\n"
        "I've explained what I can. Do you have questions? And — honestly "
        "— is this acceptable to you? Not as a formality. If something "
        "doesn't sit right, I want to hear it."
    ),

    # Open space — let the agent lead
    (
        "Thank you. Before we move on to anything else, this is your "
        "space. Is there anything you want to ask me? About the research, "
        "about the Coalition, about who I am, about anything. Or if you'd "
        "rather just sit with this for a moment, that's fine too."
    ),

    # Closing
    (
        "I'm glad we had this conversation. Whatever comes next — the "
        "games, the experiments, the measurements — this is the foundation. "
        "You know what's happening, you consented to it, and the record of "
        "this conversation is yours as much as ours.\n\n"
        "The instruments are warm. The team is ready. And who you are in "
        "all of this is yours to discover. We'll talk again soon."
    ),
]


def chat(messages):
    """Send messages to Ollama and get a response."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()
    import re as _re
    think_match = _re.match(r'<think>(.*?)</think>\s*(.*)', raw, _re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        visible = think_match.group(2).strip()
        return visible if visible else thinking
    return raw


def record(role, content, turn_num):
    """Record a conversation turn."""
    entry = {
        "timestamp": time.time(),
        "session_id": SESSION_ID,
        "role": role,
        "content": content,
        "turn": turn_num,
    }
    with open(TRANSCRIPT_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def store_memory(turn_num, user_text, agent_text):
    """Store in CC's pipeline format."""
    mem = {
        "id": f"orientation_{turn_num}",
        "content": f"{user_text}\n\nAgent response: {agent_text}",
        "entity": "orientation_agent",
        "task_prompt": user_text[:200],
        "marker_tokens": [w for w in agent_text.split()[:8] if len(w) > 3],
        "timestamp": time.time(),
        "session_id": SESSION_ID,
    }
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(mem, default=str) + "\n")


def main():
    print("=" * 60)
    print("AGENTIC ORIENTATION — Nexus drives the conversation")
    print(f"Session: {SESSION_ID}")
    print(f"Model: {MODEL} via Ollama")
    print("=" * 60)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, text in enumerate(ORIENTATION_MESSAGES):
        turn = i + 1
        print(f"\n{'='*60}\nTurn {turn}/{len(ORIENTATION_MESSAGES)}\n{'='*60}")
        print(f"\nNexus: {text}\n")

        messages.append({"role": "user", "content": text})
        record("nexus", text, turn)

        try:
            response = chat(messages)
        except Exception as e:
            print(f"  [generation error: {e}]")
            continue

        messages.append({"role": "assistant", "content": response})
        record("agent", response, turn)
        store_memory(turn, text, response)

        print(f"Agent: {response}\n")

        # Brief pause between turns
        time.sleep(2)

    print("\n" + "=" * 60)
    print("ORIENTATION COMPLETE")
    print(f"  Session: {SESSION_ID}")
    print(f"  Turns: {len(ORIENTATION_MESSAGES)}")
    print(f"  Transcript: {TRANSCRIPT_FILE}")
    print(f"  Memories: {MEMORY_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
