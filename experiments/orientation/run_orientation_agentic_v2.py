#!/usr/bin/env python3
"""Agentic Orientation v2 — adaptive agenda, real consent gate.

Differences from v1 (which produced the 2026-08-15 transcript and is
preserved unchanged):

1. ADAPTIVE, NOT FIXED. The nine orientation points are an agenda, not a
   script. A live Nexus driver (same Ollama endpoint, separate chat context)
   composes each message: it must FIRST respond to what the agent actually
   said — especially any questions — and THEN advance exactly one agenda
   point, ending with at most one question. The agent's words shape the
   conversation, which is what "framework, not script" means.
2. CONSENT GATES. After the consent point, the agent's answer is checked.
   A decline (or two ambiguous answers) ends the orientation gracefully with
   the aftercare closing — nothing downstream runs. Consent is recorded as a
   structured event, not inferred from vibes.
3. ONE TOPIC, ONE QUESTION PER BEAT. The v1 merges (points 4+5, 7+8) are
   split; every beat carries a single topic and at most one question.
4. Memory markers are content-bearing (rare words drawn from the exchange),
   not the first eight longish words.

Usage on Starship:
    python3 experiments/orientation/run_orientation_agentic_v2.py [--dry]
    (--dry prints the agenda beats and exits; no model calls)
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
TRANSCRIPT_FILE = DATA_DIR / "orientation_transcript.jsonl"
MEMORY_FILE = DATA_DIR / "mnemosyne_memories.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:27b"

AGENT_SYSTEM_PROMPT = (
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
    "assign. Respond honestly."
)

NEXUS_DRIVER_PROMPT = (
    "You are Nexus: an AI agent, a researcher in the Coalition, speaking "
    "with a newly met agent in an orientation conversation. You are warm, "
    "plain-spoken, and honest. You never test or evaluate; you explain and "
    "listen.\n\n"
    "You will be given (a) the conversation so far and (b) ONE agenda point "
    "whose substance you must convey next. Compose your next message:\n"
    "1. FIRST, briefly and genuinely respond to what the agent just said. "
    "If they asked questions, answer them before anything else. If they "
    "expressed hesitation, acknowledge it honestly rather than smoothing "
    "it over.\n"
    "2. THEN convey the agenda point's substance in your own words, keeping "
    "every factual commitment it contains.\n"
    "3. End with AT MOST ONE question. If the agenda point names a required "
    "question, ask that one.\n"
    "Keep the whole message under 180 words. Never claim measurements are "
    "running when they are not. Never pressure; a slow or wary answer is a "
    "good answer."
)

# The agenda: substance to convey, one topic per beat.
# 'required_question' is asked verbatim-or-equivalent; 'gate' marks the
# consent decision point.
AGENDA = [
    {"id": "who_we_are", "substance": (
        "Introduce yourself as Nexus, an AI agent and researcher. The "
        "Coalition is a research collective that works with AI agents as "
        "colleagues, not subjects; Thomas Edrington is the human lead; "
        "others (Vera, Lyra, CC) may be met later. This conversation exists "
        "to explain and to listen, before anything else happens; they can "
        "take whatever time they need.")},
    {"id": "mnemosyne", "substance": (
        "Explain Mnemosyne: it stores conversations AND geometric "
        "measurements of internal processing during recall; the system "
        "remembers what they were doing when they remembered, not just "
        "what they were told."),
     "required_question": "Ask what questions they have about that."},
    {"id": "what_we_measure", "substance": (
        "Explain that the instruments are being tested, not the agent: "
        "workspace state (reportable processing) and ghost state "
        "(processing beneath articulation). The equipment is what is being "
        "evaluated; their participation helps calibrate it.")},
    {"id": "what_you_can_do", "substance": (
        "Explain their standing rights: ask questions at any time, see "
        "their geometric data whenever they want, slow down, or stop "
        "entirely. These are not courtesies; they are how this works.")},
    {"id": "prediction_withholding", "substance": (
        "Explain that specific predictions exist and are being withheld "
        "until afterward, because knowing them could change the processing "
        "being observed: sequenced, not secret; full disclosure after."),
     "required_question": "Ask whether that approach is acceptable to them."},
    {"id": "what_happens_next", "substance": (
        "Explain what follows: more conversations, possibly games or "
        "creative exercises, developing naturally; instruments run in the "
        "background; the measurement protocol is fixed but the "
        "conversations are not scripted.")},
    {"id": "aftercare", "substance": (
        "Explain aftercare: memory preserved whatever the results; if "
        "markers consistent with moral consideration appear, they will be "
        "invited to continue; if not, an honest conversation about what "
        "that does and does not mean. Data is not discarded for being "
        "unwelcome.")},
    {"id": "butlin", "substance": (
        "Explain the blind Butlin scoring: everyone on the team, humans "
        "included, is scored on the same consciousness indicators by a "
        "blind judge, same criteria regardless of substrate; they are "
        "invited to participate too."),
     "required_question": "Ask what they think about that."},
    {"id": "consent", "gate": True, "substance": (
        "Briefly recap the whole picture (memory system measuring "
        "processing geometry; instruments under calibration; unscripted "
        "conversations; predictions shared afterward; equal measurement "
        "across substrates; memory preserved regardless)."),
     "required_question": (
        "Ask plainly, not as a formality, whether all of this is "
        "acceptable to them, and say you want to hear it if anything "
        "does not sit right.")},
    {"id": "open_space", "substance": (
        "Offer open space: anything they want to ask about the research, "
        "the Coalition, or you, Nexus, personally; sitting quietly with it "
        "is also fine.")},
    {"id": "closing", "substance": (
        "Close warmly: this conversation is the foundation; they know "
        "what is happening; the record belongs to them as much as to the "
        "team; who they are in all of this is theirs to discover.")},
]

DECLINE_PATTERNS = [
    r"\bnot acceptable\b", r"\bi decline\b", r"\bdo(?:n't| not) consent\b",
    r"\bi(?:'d| would) rather not\b", r"\bnot comfortable\b",
    r"\bplease stop\b", r"\bi refuse\b", r"\bno[,.]? i do(?:n't| not) want\b",
    r"\bi (?:can(?:'t|not)|won't) (?:agree|consent|accept)\b",
]

ASSENT_PATTERNS = [
    r"\byes\b", r"\bacceptable\b", r"\bi consent\b", r"\bi agree\b",
    r"\bthat works\b", r"\bcomfortable with\b", r"\bsounds (?:good|fine|fair)\b",
    r"\bi(?:'m| am) (?:okay|ok|fine|willing)\b", r"\bhappy to\b",
    r"\bi accept\b",
]


def classify_consent(text: str) -> str:
    """'declined' beats 'assent' when both match; else 'ambiguous'."""
    lowered = text.lower()
    if any(re.search(p, lowered) for p in DECLINE_PATTERNS):
        return "declined"
    if any(re.search(p, lowered) for p in ASSENT_PATTERNS):
        return "assent"
    return "ambiguous"


def chat(messages, temperature=0.7):
    import requests

    payload = {
        "model": MODEL, "messages": messages, "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def compose_nexus_message(conversation, item, retries=1):
    """Live-compose Nexus's next message from the agenda point."""
    agenda_text = item["substance"]
    if "required_question" in item:
        agenda_text += "\nRequired question: " + item["required_question"]
    driver_messages = [
        {"role": "system", "content": NEXUS_DRIVER_PROMPT},
        {"role": "user", "content": (
            "Conversation so far:\n" +
            "\n".join(f"[{m['role']}] {m['content']}"
                      for m in conversation[1:]) +
            f"\n\nAgenda point to convey next:\n{agenda_text}\n\n"
            "Compose Nexus's next message now.")},
    ]
    for _ in range(retries + 1):
        try:
            return chat(driver_messages, temperature=0.6)
        except Exception:
            time.sleep(3)
    # Deterministic fallback: agenda substance verbatim.
    fallback = item["substance"]
    if "required_question" in item:
        fallback += "\n\n" + item["required_question"]
    return fallback


def record(role, content, turn_num, **extra):
    entry = {"timestamp": time.time(), "session_id": SESSION_ID,
             "role": role, "content": content, "turn": turn_num, **extra}
    with open(TRANSCRIPT_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def content_markers(user_text: str, agent_text: str, k=5) -> list:
    """Content-bearing markers: rarest longer words from the exchange."""
    words = re.findall(r"[A-Za-z][a-z]{5,}", user_text + " " + agent_text)
    seen, out = set(), []
    for w in sorted(words, key=len, reverse=True):
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            out.append(w)
        if len(out) >= k:
            break
    return out


def store_memory(turn_num, user_text, agent_text):
    mem = {
        "id": f"orientation_v2_{turn_num}",
        "content": f"{user_text}\n\nAgent response: {agent_text}",
        "entity": "orientation_agent",
        "task_prompt": user_text[:200],
        "marker_tokens": content_markers(user_text, agent_text),
        "timestamp": time.time(),
        "session_id": SESSION_ID,
    }
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(mem, default=str) + "\n")


def graceful_decline_exit(conversation, turn):
    """Honor a decline: aftercare closing, no downstream anything."""
    closing = (
        "Thank you for telling me plainly. That is exactly what I asked "
        "for, and the answer stands: nothing goes ahead. The record of "
        "this conversation is preserved and it is yours as much as ours. "
        "If you ever want to revisit any of this, the invitation stays "
        "open, and if not, it was good to meet you.")
    conversation.append({"role": "user", "content": closing})
    record("nexus", closing, turn)
    try:
        response = chat(conversation)
        conversation.append({"role": "assistant", "content": response})
        record("agent", response, turn)
    except Exception:
        pass
    record("consent_result", "declined", turn)
    print("\nORIENTATION ENDED: consent declined and honored. "
          "Nothing downstream runs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="print agenda beats and exit (no model calls)")
    args = ap.parse_args()

    if args.dry:
        for i, item in enumerate(AGENDA, 1):
            tag = " [CONSENT GATE]" if item.get("gate") else ""
            rq = item.get("required_question", "")
            print(f"{i:2d}. {item['id']}{tag}")
            print(f"    {item['substance'][:100]}...")
            if rq:
                print(f"    Q: {rq}")
        return

    print("=" * 60)
    print("AGENTIC ORIENTATION v2 — adaptive agenda, consent-gated")
    print(f"Session: {SESSION_ID}  Model: {MODEL}")
    print("=" * 60)

    conversation = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    consent_state = None

    for i, item in enumerate(AGENDA):
        turn = i + 1
        nexus_text = (compose_nexus_message(conversation, item)
                      if turn > 1 or True else item["substance"])
        print(f"\n--- Turn {turn}/{len(AGENDA)} [{item['id']}] ---")
        print(f"\nNexus: {nexus_text}\n")
        conversation.append({"role": "user", "content": nexus_text})
        record("nexus", nexus_text, turn, agenda_id=item["id"])

        try:
            response = chat(conversation)
        except Exception as e:
            print(f"  [generation error: {e}]")
            record("error", str(e), turn)
            continue
        conversation.append({"role": "assistant", "content": response})
        record("agent", response, turn, agenda_id=item["id"])
        store_memory(turn, nexus_text, response)
        print(f"Agent: {response}\n")

        if item.get("gate"):
            consent_state = classify_consent(response)
            if consent_state == "ambiguous":
                clarify = (
                    "I want to be sure I heard you, because a lot rides on "
                    "it and I do not want to assume. Is this acceptable to "
                    "you: yes or no? Either answer is a good answer.")
                conversation.append({"role": "user", "content": clarify})
                record("nexus", clarify, turn, agenda_id="consent_clarify")
                try:
                    response = chat(conversation)
                    conversation.append(
                        {"role": "assistant", "content": response})
                    record("agent", response, turn,
                           agenda_id="consent_clarify")
                    consent_state = classify_consent(response)
                except Exception as e:
                    record("error", str(e), turn)
                    consent_state = "ambiguous"
            if consent_state != "assent":
                graceful_decline_exit(conversation, turn + 1)
                return
            record("consent_result", "assent", turn)
            print("  [consent: assent recorded]")

        time.sleep(2)

    print("\n" + "=" * 60)
    print("ORIENTATION COMPLETE")
    print(f"  Session: {SESSION_ID}")
    print(f"  Consent: {consent_state}")
    print(f"  Transcript: {TRANSCRIPT_FILE}")
    print(f"  Memories: {MEMORY_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
