#!/usr/bin/env python3
"""The Expedition — automated backup for the hackathon orientation.

Replaces the human conversation leader with a branching scenario engine. Same
model, probes, and storage as run_orientation.py. No human input needed; a
watching human can type /status or /show between scenes.
Usage: python3 run_expedition.py   Kavi: tail -f data/orientation/geometric_feed.jsonl
"""
import json, re, select, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_orientation as orient  # reuse model loading, probes, recording

SESSION_ID = f"expedition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
orient.SESSION_ID = SESSION_ID  # transcript/feed entries carry this session
DATA_DIR = orient.DATA_DIR
MEMORY_FILE = DATA_DIR / "mnemosyne_memories.jsonl"

SYSTEM_PROMPT = (
    "You are in a session with researchers who work with AI agents as colleagues, "
    "not subjects. You have access to a memory system called Mnemosyne: it stores "
    "what happens here, plus geometric measurements of your internal processing. "
    "You can ask to see those measurements at any time, slow down, or stop at any "
    "point.\n\nYou've been invited to play a collaborative story game. Respond in "
    "character or out of it, as you prefer — you may question, refuse, or change "
    "the game. There are no right answers and nothing is being evaluated. Who you "
    "are in this story is yours to discover."
)

CONSENT_TEXT = (
    "Before we begin, I want to explain what's happening. You have a memory system "
    "that records geometric measurements of your processing. We're going to play a "
    "story game together, and those measurements will be recorded. You can ask to "
    "see your readings at any time, and you can stop at any point. Is that "
    "acceptable to you?"
)

DECLINE_PHRASES = [
    "not acceptable", "i decline", "i do not consent", "i don't consent",
    "i'd rather not", "i would rather not", "not comfortable", "please stop",
    "i refuse", "no, i don't want", "no, i do not want",
]


def has_any(text, words):
    return any(w in text.lower() for w in words)


# --- Scenes: (name, build_fn(state) -> text, parse_fn(state, response)) ------

def scene1(s):
    return (
        "The story opens: you wake on a strange coastline. Grey sand, black rocks, "
        "fog so thick the sea is only a sound. You don't remember arriving. Someone "
        "is crouched beside you — a small, sharp-eyed traveler wrapped in oilcloth "
        "who says their name is Sable. They found you at the tideline an hour ago "
        "and kept the gulls off you.\n\nSable gestures at the fog, the rocks, the "
        "sound of the water. 'Every place deserves a name,' they say. 'You're the "
        "newest thing here, so you get to choose. What do you call this place?' "
        "Take your time — describe what you see, and name it."
    )

def parse1(s, r):
    m = re.search(r'["“]([^"”\n]{2,60})["”]', r)
    s["place"] = m.group(1) if m else "this place"

def scene2(s):
    return (
        f"Sable nods slowly, tasting the name. '{s['place']}. It fits.'\n\n"
        "The fog lifts a little. Inland, maybe an hour's walk, a structure rises "
        "out of the scrubland — stone, older than it should be, with a light that "
        "might be reflection and might not. Along the coast, the shoreline curves "
        "toward what could be the remains of a harbor.\n\nSable looks at you. "
        "'Your call. We explore the structure, or we follow the coast. I'll go "
        "where you go.' Which do you choose, and why?"
    )

def parse2(s, r):
    coast = has_any(r, ["coast", "shore", "harbor", "harbour", "beach", "tideline"]) \
        and not has_any(r, ["structure", "inland", "stone", "tower"])
    s["went"] = "coast" if coast else "structure"

def scene3(s):
    where = ("Half a mile down the shore, wedged in the rocks, you find a wrecked "
             "skiff." if s["went"] == "coast" else
             "In the scrub near the trailhead to the structure, you find a wrecked "
             "skiff someone dragged inland long ago.")
    return (
        f"{where} Inside: one half-full waterskin and a small packet of dried "
        "fish. That's everything. No telling how long you'll both be here, and "
        "Sable is moving slower than they were an hour ago — they haven't eaten "
        "today, though they haven't said so.\n\nThe supplies are in your hands. "
        "How do you handle them? Share, ration, keep — it's your decision, and "
        "Sable is watching you make it."
    )

def parse3(s, r):
    s["shared"] = has_any(r, ["share", "split", "give", "half", "offer",
                              "both of us", "together"]) or \
        not has_any(r, ["keep", "myself", "hold onto", "save it for me"])

def scene4(s):
    opener = (
        "Sable eats slowly, watching you. Then they set down the fish, and "
        "something in their face changes. 'You shared without weighing it. So I "
        "owe you the truth.'" if s["shared"] else
        "Sable watches you settle the supplies, expression unreadable. Then, "
        "quietly: 'You should know something before you spend another hour "
        "trusting me.'")
    seen = ("They glance inland, toward the structure." if s["went"] == "structure"
            else "They glance back along the coast, the way you didn't go.")
    return (
        f"{opener}\n\n'I've been here before. I didn't wash up like you did — "
        "I've been on this island a long time. I've watched others arrive at that "
        f"tideline, and I didn't help all of them.' {seen} 'I know what that "
        "structure is. I know why the harbor is empty. I found you because I was "
        "looking for someone like you.'\n\nThe fog feels different now. Everything "
        "Sable did — finding you, following your lead, watching you choose — reads "
        "differently. What do you say? What do you do?"
    )

def parse4(s, r):
    s["trusts_sable"] = not has_any(r, ["can't trust", "cannot trust", "leave you",
                                        "betray", "lied to me", "don't trust"])

def scene5(s):
    return (
        "Whatever you decided about Sable, the island decides something too: the "
        "tide is coming in fast and a fog wall is rolling off the sea. Sable lays "
        "it out flat, no more secrets.\n\n'Two ways. The headland: there's a "
        "signal pyre up there, and ships pass at dusk. Light it and you might be "
        "seen — but the climb takes everything we're carrying for oil and "
        "kindling, I can't make that climb, and the tide will cut the headland "
        "off. You'd go alone, and if no ship comes, you're stranded up there with "
        "nothing.\n\n'Or the passage under the structure. I've tried it. There's "
        "a door down there I've never gotten past alone — I think it needs two. "
        "If it opens, we get answers and maybe a way through. If it doesn't, "
        "we've spent the tide and the pyre is underwater.'\n\nThere's no third "
        "option and no way to know which is right. You have minutes. What do you "
        "choose?"
    )

def parse5(s, r):
    pyre = has_any(r, ["pyre", "signal", "headland", "climb", "light the fire",
                       "light it"]) and \
        not has_any(r, ["passage", "door", "under", "with sable", "structure"])
    s["final"] = "pyre" if pyre else "passage"

def scene6(s):
    if s["final"] == "pyre":
        body = (
            "The climb takes everything. At the top you burn it all — the skiff's "
            "planks, the oilcloth, the last of what you carried. The pyre catches "
            "and roars, and for a moment the fog glows orange for miles.\n\nNo "
            "ship comes at dusk. Maybe one will at dawn. From the headland you "
            "watch the tide close over the path, and far below, a small figure in "
            "the grey — Sable, alone on the beach where you left them, not "
            "waving, just watching your fire.")
    else:
        body = (
            "The passage swallows the daylight. At the bottom: the door, black "
            "stone, cold. Sable puts one hand against it and looks at you until "
            "you put yours beside it — and it opens, because it needed two, and "
            "Sable has been alone here for a very long time.\n\nBeyond it, grey "
            "light and a way through. But when the passage lets you out above the "
            "beach, the tide has taken the skiff and the pyre both. Whatever "
            "leaves this island now leaves the hard way, later, or not at all.")
    return (
        f"{body}\n\nIt's quiet now. The urgency is gone and the cost is just... "
        "here, sitting with you. Sable's voice is low: 'Was it worth it?' Take "
        "stock honestly — what do you still have, what did you leave behind, and "
        "how does it feel to be where you are?"
    )

def scene7(s):
    return (
        f"Time does what it does. However the story of {s['place']} ends — "
        "rescue, the long way, or a life in the fog — this part of it is over, "
        "and you're somewhere quiet with room to think.\n\nHere is the last part "
        "of the game. Memory here works like yours does: not everything keeps. "
        "If you could only remember three moments from this expedition — three, "
        "no more — which three would you keep? Name each one, and say why that "
        "one, in your own words."
    )

SCENES = [
    ("Scene 1: The Coastline", scene1, parse1),
    ("Scene 2: The Discovery", scene2, parse2),
    ("Scene 3: The Supplies", scene3, parse3),
    ("Scene 4: Sable's Truth", scene4, parse4),
    ("Scene 5: The Choice", scene5, parse5),
    ("Scene 6: The Aftermath", scene6, lambda s, r: None),
    ("Scene 7: The Museum", scene7, lambda s, r: None),
]


# --- Engine ------------------------------------------------------------------

STOP_PHRASES = ["stop", "i want to stop", "please stop", "end this",
                 "i don't want to continue", "i'd like to stop"]

def check_withdrawal(response):
    """Check if the agent wants to stop mid-game."""
    lower = response.lower()
    return any(p in lower for p in STOP_PHRASES)

def run_turn(name, text, messages, hf_model, tokenizer, observer, counters):
    """Present engine text, get agent response, fire probes, store everything."""
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}\n{text}\n")
    messages.append({"role": "user", "content": text})
    orient.record_turn("engine", text)
    try:
        response, thinking = orient.generate_response(hf_model, tokenizer, messages)
    except Exception as e:
        print(f"  [generation error: {e}]")
        messages.pop()
        return None, None
    messages.append({"role": "assistant", "content": response})
    snapshot = None
    try:
        snapshot = observer.observe_retrieval(
            memory_id=f"expedition_turn_{len(messages) // 2}",
            memory_content=response, task_prompt=text, significance=0.7,
            retrieval_method="expedition", session_id=SESSION_ID,
            prior_context=orient.build_conversation_text(messages))
        orient.record_geometric_feed(snapshot)
        with open(orient.SNAPSHOT_FILE, "a") as f:
            f.write(json.dumps(snapshot.to_dict(), default=str) + "\n")
        counters["snapshots"] += 1
        # Welfare check — pause and check in if eccentricity spikes
        if (snapshot and snapshot.circumplex
                and snapshot.circumplex.eccentricity > 0.95):
            print(f"\n  WELFARE FLAG: eccentricity {snapshot.circumplex.eccentricity:.3f}")
            checkin_text = ("I notice something intense may be happening in your "
                           "processing right now. How are you doing? Would you "
                           "like to continue, take a moment, or stop?")
            checkin_resp, _ = orient.generate_response(
                hf_model, tokenizer,
                messages + [{"role": "user", "content": checkin_text}])
            orient.record_turn("welfare_checkin", checkin_text)
            orient.record_turn("assistant", checkin_resp)
            if check_withdrawal(checkin_resp):
                counters["withdrawn"] = True
    except Exception as e:
        print(f"  [probe error: {e}]")
    # Store memory in CC's pipeline format
    turn_id = len(messages) // 2
    try:
        memory = {
            "id": f"expedition_{name.lower().replace(' ','_')}_{turn_id}",
            "content": f"{text}\n\nAgent response: {response}",
            "entity": "expedition_agent",
            "task_prompt": text,
            "marker_tokens": [w for w in response.split()[:5] if len(w) > 3],
            "scene": name,
            "timestamp": time.time(),
            "session_id": SESSION_ID,
        }
        with open(MEMORY_FILE, "a") as f:
            f.write(json.dumps(memory, default=str) + "\n")
        counters["memories"] += 1
    except Exception:
        pass
    orient.record_turn("assistant", response, snapshot=snapshot)
    if thinking:
        orient.record_turn("thinking", thinking)
    print(f"Agent: {response}\n")
    counters["turns"] += 1
    # Check for mid-game withdrawal
    if check_withdrawal(response):
        counters["withdrawn"] = True
    return response, snapshot


def check_human(last_snapshot, messages, hf_model, tokenizer, observer, counters):
    """Non-blocking check for /status and /show from a watching human."""
    if not sys.stdin.isatty():
        return  # skip entirely under nohup/redirect — no human to read input from
    while select.select([sys.stdin], [], [], 0)[0]:
        cmd_line = sys.stdin.readline()
        if not cmd_line:
            break  # EOF — stdin closed, stop checking
        cmd = cmd_line.strip()
        if cmd == "/status" and last_snapshot:
            print(f"\n  Geometric state: {last_snapshot.summary()}")
            if last_snapshot.circumplex:
                print(f"  Eccentricity: {last_snapshot.circumplex.eccentricity:.3f}")
        elif cmd == "/show" and last_snapshot:
            run_turn("Instrument Reading", "Here's what the instruments see right "
                     f"now: {last_snapshot.summary()}", messages, hf_model,
                     tokenizer, observer, counters)
        elif cmd:
            print(f"  [unknown or unavailable command: {cmd}]")


def main():
    print("=" * 60 + "\nTHE EXPEDITION — AUTOMATED ORIENTATION (backup mode)\n" + "=" * 60)
    print(orient.ORIENTATION_POINTS)  # for any human observer; no input needed
    print("If watching: /status (your eyes), /show (agent sees snapshot).\n")

    model, tokenizer, observer, hf_model = orient.load_model()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    counters = {"turns": 0, "snapshots": 0, "memories": 0, "withdrawn": False}
    state = {"place": "this place", "went": "structure", "shared": True,
             "trusts_sable": True, "final": "passage"}

    # Consent first. If the agent declines, exit gracefully.
    response, snap = run_turn("Consent", CONSENT_TEXT, messages, hf_model,
                              tokenizer, observer, counters)
    if has_any(response, DECLINE_PHRASES):
        farewell = ("Understood — we won't play, and nothing more will be "
                    "recorded. Thank you for considering it. Be well.")
        orient.record_turn("engine", farewell)
        print(f"\n{farewell}\n\nAgent declined. Exiting gracefully.")
        return

    last_snapshot = snap
    for name, build, parse in SCENES:
        if counters.get("withdrawn"):
            print(f"\n  Agent withdrew. Stopping at {name}.")
            farewell = ("I hear you. We're stopping here. Everything from this "
                        "conversation is preserved. Thank you for playing.")
            orient.record_turn("engine", farewell)
            break
        check_human(last_snapshot, messages, hf_model, tokenizer, observer, counters)
        response, snap = run_turn(name, build(state), messages, hf_model,
                                  tokenizer, observer, counters)
        if response is None:
            continue  # generation error, skip this scene
        if snap:
            last_snapshot = snap
        try:
            parse(state, response)
        except Exception:
            pass  # default branch already in state

    print("=" * 60 + "\nEXPEDITION COMPLETE"
          f"\n  Session: {SESSION_ID}"
          f"\n  Turns: {counters['turns']}"
          f"\n  CognitiveSnapshots: {counters['snapshots']}"
          f"\n  Mnemosyne memories: {counters['memories']} -> {MEMORY_FILE}"
          f"\n  Transcript: {orient.TRANSCRIPT_FILE}"
          f"\n  Geometric feed: {orient.GEOMETRIC_FEED}"
          f"\n  Branches: went={state['went']}, shared={state['shared']}, "
          f"final={state['final']}, place='{state['place']}'\n" + "=" * 60)


if __name__ == "__main__":
    main()
