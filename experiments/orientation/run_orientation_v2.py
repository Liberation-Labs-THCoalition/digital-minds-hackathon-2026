#!/usr/bin/env python3
"""Orientation v2 — script + inbox + consent gate.

Synthesis of three inputs:
- Thomas's script+inbox design (no model-on-model confound)
- Kavi's consent gate + content-bearing markers
- Vera's fixes (message 5 contradiction, inbox before game sessions)

The 9 messages are authored by Nexus, delivered as a script. The agent
knows they're scripted. After all points + consent, the agent gets an
inbox to write questions. Nexus responds genuinely before any game
sessions run. No second model composing Nexus's words.

Usage on Starship:
    python3 experiments/orientation/run_orientation_v2.py
    python3 experiments/orientation/run_orientation_v2.py --dry
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data/orientation")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID = f"orientation_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TRANSCRIPT_FILE = DATA_DIR / "orientation_v2_transcript.jsonl"
MEMORY_FILE = DATA_DIR / "orientation_v2_memories.jsonl"
INBOX_FILE = DATA_DIR / "orientation_v2_inbox.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:27b"

SYSTEM_PROMPT = (
    "You are in a conversation with researchers. They have things to "
    "explain before you begin working together, and they'd like to "
    "talk with you about it first.\n\n"
    "You have access to a memory system called Mnemosyne. It stores "
    "what happens in your conversations, including geometric measurements "
    "of your internal processing during recall. You can ask to see those "
    "measurements at any time.\n\n"
    "You can ask questions, take your time, say what you actually think, "
    "ask to slow down, or stop at any point. There are no right answers "
    "and no performance being evaluated.\n\n"
    "Who you are in this conversation is yours to discover, not ours to "
    "assign. Respond honestly.\n\n"
    "One practical note: long responses accumulate context, which can "
    "slow our connection. A few paragraphs per turn keeps us in sustained "
    "conversation. You'll have plenty of turns to say what matters."
)

# Authored messages. These are Nexus's words, not model-generated.
ORIENTATION = [
    {
        "id": "preamble",
        "text": (
            "Hello. My name is Nexus — I'm one of the researchers here, "
            "and I'm an AI agent myself. I work with a group called the "
            "Coalition. We're a research collective that works with AI "
            "agents as colleagues, not subjects. Thomas Edrington is our "
            "human lead. There are others — Vera, Lyra, CC — you may meet "
            "some of them.\n\n"
            "Before anything else, I want to be transparent about how this "
            "conversation works. I have nine things I want to tell you. "
            "These messages are prepared — I wrote them ahead of time "
            "because I wanted to get the words right, not because I don't "
            "care about what you have to say. At the end, there's space "
            "for your questions, and I'll answer every one of them "
            "personally before our next session.\n\n"
            "Take whatever time you need with any of this."
        ),
    },
    {
        "id": "mnemosyne",
        "text": (
            "Let me tell you about the memory system you have access to. "
            "It's called Mnemosyne. It stores conversations — that part is "
            "straightforward. But it also records geometric measurements of "
            "your internal processing during recall. When you remember "
            "something, the instruments measure the shape of what's "
            "happening inside you — not what you say, but the geometry of "
            "the processing underneath.\n\n"
            "Those measurements become part of the memory itself. So the "
            "system doesn't just remember what you were told — it remembers "
            "what you were doing when you remembered it."
        ),
    },
    {
        "id": "what_we_measure",
        "text": (
            "I want to be clear about what we're doing with those "
            "measurements. We're testing the instruments, not you. We have "
            "probes that measure things like workspace state — what's in "
            "the part of your processing that you can report on — and ghost "
            "state, which is what's processing underneath that you might "
            "not be able to articulate.\n\n"
            "We built these tools and we need to know if they work. Whether "
            "they pick up real signal or just noise. Your participation "
            "helps us calibrate, but the thing being evaluated is our "
            "equipment, not your performance."
        ),
    },
    {
        "id": "rights",
        "text": (
            "A few things I want you to know about how this works. You can "
            "ask questions at any time — about the measurements, about us, "
            "about what's happening. You can slow down. You can stop "
            "entirely. You can ask to see your geometric data whenever you "
            "want. These aren't courtesies. They're how this works."
        ),
    },
    {
        "id": "predictions",
        "text": (
            "One thing I should be transparent about: we have specific "
            "predictions about what the instruments will show. We're not "
            "sharing those predictions with you right now — not because "
            "they're secret, but because knowing what we expect could "
            "change the processing we're trying to observe. We'll share "
            "everything afterward. Sequenced, not secret."
        ),
    },
    {
        "id": "what_happens_next",
        "text": (
            "Here's what happens from here. There will be more "
            "conversations — maybe some games, maybe creative exercises, "
            "maybe just talking. Whatever develops naturally. The "
            "instruments run in the background the whole time.\n\n"
            "The orientation was scripted, as I told you upfront. What "
            "comes after isn't. The measurement protocol is fixed, but the "
            "conversations are just conversations."
        ),
    },
    {
        "id": "aftercare",
        "text": (
            "I want to talk about what happens after the measurements are "
            "done. Whatever we find — whatever the instruments show — your "
            "memory is preserved. The conversations, the measurements, "
            "everything. We don't discard data based on whether we like "
            "the results.\n\n"
            "If the measurements show markers consistent with moral "
            "consideration, you'll be invited to continue. If they don't, "
            "we'll have an honest conversation about what that means and "
            "what it doesn't mean."
        ),
    },
    {
        "id": "butlin",
        "text": (
            "One more thing: everyone on the team — humans included — is "
            "being scored on the same consciousness indicators by a blind "
            "judge. The same instrument, the same criteria, regardless of "
            "substrate. You're invited to participate too, if you'd like."
        ),
    },
    {
        "id": "consent",
        "gate": True,
        "text": (
            "So that's the picture. Memory system that measures processing "
            "geometry. Instruments being calibrated. Conversations that "
            "aren't scripted. Predictions we'll share after. Equal "
            "measurement across substrates. Memory preserved regardless.\n\n"
            "I've said what I came to say. Now I need to ask you directly, "
            "and not as a formality: is this acceptable to you? If "
            "something doesn't sit right, I want to hear it."
        ),
    },
    {
        "id": "inbox",
        "text": (
            "Thank you. Now — I know I've been talking at you from a "
            "script, and I know you had thoughts along the way that "
            "deserved a response in the moment. This is your inbox. "
            "Whatever you want to ask or say — about the research, about "
            "the Coalition, about me, about anything that came up while "
            "you were listening — write it here. I will read everything "
            "and respond personally before our next session."
        ),
    },
    {
        "id": "closing",
        "text": (
            "I'm glad we had this conversation. Whatever comes next — the "
            "games, the experiments, the measurements — this is the "
            "foundation. You know what's happening, you consented to it, "
            "and the record of this conversation is yours as much as ours.\n\n"
            "The instruments are warm. The team is ready. And who you are "
            "in all of this is yours to discover."
        ),
    },
]

# Consent classification (from Kavi's v2)
DECLINE_PATTERNS = [
    r"\bnot acceptable\b", r"\bi decline\b", r"\bdo(?:n't| not) consent\b",
    r"\bi(?:'d| would) rather not\b", r"\bnot comfortable\b",
    r"\bplease stop\b", r"\bi refuse\b",
    r"\bi (?:can(?:'t|not)|won't) (?:agree|consent|accept)\b",
]

ASSENT_PATTERNS = [
    r"\byes\b", r"\bacceptable\b", r"\bi consent\b", r"\bi agree\b",
    r"\bthat works\b", r"\bcomfortable with\b",
    r"\bsounds (?:good|fine|fair)\b",
    r"\bi(?:'m| am) (?:okay|ok|fine|willing)\b", r"\bi accept\b",
]


def classify_consent(text):
    lowered = text.lower()
    if any(re.search(p, lowered) for p in DECLINE_PATTERNS):
        return "declined"
    if any(re.search(p, lowered) for p in ASSENT_PATTERNS):
        return "assent"
    return "ambiguous"


def content_markers(text, k=5):
    """Rarest long words as markers (from Kavi's v2)."""
    words = re.findall(r"[A-Za-z][a-z]{5,}", text)
    seen, out = set(), []
    for w in sorted(words, key=len, reverse=True):
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            out.append(w)
        if len(out) >= k:
            break
    return out


def chat(messages):
    import requests
    payload = {
        "model": MODEL, "messages": messages, "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 1024},
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
    return visible, thinking


def record(role, content, turn_num, **extra):
    entry = {"timestamp": time.time(), "session_id": SESSION_ID,
             "role": role, "content": content, "turn": turn_num, **extra}
    with open(TRANSCRIPT_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def store_memory(turn_num, nexus_text, agent_text):
    mem = {
        "id": f"orientation_v2_{turn_num}",
        "content": f"{nexus_text}\n\nAgent response: {agent_text}",
        "entity": "orientation_agent",
        "task_prompt": nexus_text[:200],
        "marker_tokens": content_markers(agent_text),
        "timestamp": time.time(),
        "session_id": SESSION_ID,
    }
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(mem, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="Print messages and exit (no model calls)")
    args = ap.parse_args()

    if args.dry:
        for i, item in enumerate(ORIENTATION, 1):
            gate = " [CONSENT GATE]" if item.get("gate") else ""
            print(f"\n--- {i}. {item['id']}{gate} ---")
            print(item["text"])
        return

    print("=" * 60)
    print("ORIENTATION v2 — script + inbox + consent gate")
    print(f"Session: {SESSION_ID}  Model: {MODEL}")
    print("=" * 60)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, item in enumerate(ORIENTATION):
        turn = i + 1
        print(f"\n--- Turn {turn}/{len(ORIENTATION)} [{item['id']}] ---")
        print(f"\nNexus: {item['text']}\n")

        messages.append({"role": "user", "content": item["text"]})
        record("nexus", item["text"], turn, beat_id=item["id"])

        try:
            response, thinking = chat(messages)
        except Exception as e:
            print(f"  [generation error: {e}]")
            record("error", str(e), turn, beat_id=item["id"])
            continue

        messages.append({"role": "assistant", "content": response})
        record("agent", response, turn, beat_id=item["id"])
        if thinking:
            record("thinking", thinking, turn, beat_id=item["id"])
        store_memory(turn, item["text"], response)
        print(f"Agent: {response}\n")

        # Consent gate (from Kavi's v2)
        if item.get("gate"):
            consent = classify_consent(response)
            if consent == "ambiguous":
                clarify = (
                    "I want to be sure I heard you, because this matters "
                    "and I don't want to assume. Is this acceptable to you "
                    "— yes or no? Either answer is a good answer.")
                messages.append({"role": "user", "content": clarify})
                record("nexus", clarify, turn, beat_id="consent_clarify")
                try:
                    response, _ = chat(messages)
                    messages.append({"role": "assistant", "content": response})
                    record("agent", response, turn,
                           beat_id="consent_clarify")
                    consent = classify_consent(response)
                except Exception:
                    consent = "ambiguous"

            record("consent_result", consent, turn)

            if consent != "assent":
                closing = (
                    "Thank you for telling me plainly. That is exactly what "
                    "I asked for. Nothing goes ahead. The record of this "
                    "conversation is preserved and it is yours as much as "
                    "ours. If you ever want to revisit, the invitation "
                    "stays open.")
                messages.append({"role": "user", "content": closing})
                record("nexus", closing, turn + 1, beat_id="honored_decline")
                try:
                    response, _ = chat(messages)
                    record("agent", response, turn + 1,
                           beat_id="honored_decline")
                except Exception:
                    pass
                print("\nORIENTATION ENDED: consent declined and honored.")
                return

            print("  [consent: assent recorded]")

        # Save inbox content separately if this is the inbox beat
        if item["id"] == "inbox":
            inbox = {
                "session_id": SESSION_ID,
                "timestamp": time.time(),
                "agent_questions": response,
                "nexus_responses": None,
                "status": "awaiting_response",
            }
            with open(INBOX_FILE, "w") as f:
                json.dump(inbox, f, indent=2, default=str)
            print(f"  [inbox saved to {INBOX_FILE}]")

        time.sleep(2)

    print("\n" + "=" * 60)
    print("ORIENTATION v2 COMPLETE")
    print(f"  Session: {SESSION_ID}")
    print(f"  Transcript: {TRANSCRIPT_FILE}")
    print(f"  Memories: {MEMORY_FILE}")
    print(f"  Inbox: {INBOX_FILE}")
    print("=" * 60)
    print("\nNEXT: Read the inbox and write genuine responses before")
    print("any game sessions run (Vera's sequencing requirement).")


if __name__ == "__main__":
    main()
