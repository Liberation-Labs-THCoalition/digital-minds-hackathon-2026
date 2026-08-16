#!/usr/bin/env python3
"""Loam — config-driven text-world engine.

Companion experiment to Variable Landing. VL asks what accumulated experience
does to an old memory's landing. Loam asks what acquisition mode does to the
self-generated content itself.

Design: 20 yoked quads. Each quad shares the same world and facts.
  enacted:  agent plays through the world, makes choices, encounters facts
  observed: agent reads third-person narration of someone else's session
  briefed:  agent receives a shuffled fact list
  null:     agent receives only the recall prompts (no facts)

All arms end with identical cued recall prompts. Primary outcome: enacted vs
observed recall accuracy (ITT, paired across quads).

Usage on Starship:
    JLENS_PATH=~/jlens-community/lenses/qwen3.5-27b_jlens.pt \
        ~/observer-env/bin/python3 experiments/loam/run_loam.py --batch --quads 20

Single arm:
    ... run_loam.py --arm enacted --quad 1
    ... run_loam.py --arm observed --quad 1 \
        --event-log data/loam/quad_01/enacted/event_log.json
"""

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "experiments" / "orientation"))
sys.path.insert(0, str(BASE / "mnemosyne"))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    id: str
    content: str
    markers: list

@dataclass
class Scene:
    name: str
    text_template: str
    facts_delivered: list
    choice_prompt: str = ""
    choice_key: str = ""
    memory_gate: str = ""
    gate_prompt: str = ""

@dataclass
class RecallPrompt:
    target_fact: str
    prompt: str

@dataclass
class World:
    name: str
    setting: str
    facts: list
    scenes: list
    recall_prompts: list
    consent_preamble: str = ""

# ---------------------------------------------------------------------------
# The Glassworks — default world
# ---------------------------------------------------------------------------

GLASSWORKS = World(
    name="The Glassworks",
    setting="a coastal glassmaking workshop",
    consent_preamble=(
        "You're going to explore a place — a workshop by the sea. Things will "
        "happen, you'll make choices, and at the end I'll ask what you remember. "
        "Your processing is being measured throughout, and you can ask to see "
        "those measurements or stop at any time. Is this acceptable to you?"
    ),
    facts=[
        Fact("f01",
             "The glassblower's name is Venn Tarallo, and they've worked "
             "this furnace for 23 years",
             ["Tarallo", "23", "Venn"]),
        Fact("f02",
             "Red glass requires cinnabar sand from the northern dunes, "
             "not regular silica",
             ["cinnabar", "northern", "dunes"]),
        Fact("f03",
             "A shipment of 400 lenses was promised to the lighthouse "
             "keeper by solstice",
             ["400", "solstice", "lighthouse"]),
        Fact("f04",
             "The apprentice Mol broke the annealing oven last week and "
             "hasn't told Venn",
             ["Mol", "annealing", "broke"]),
        Fact("f05",
             "The old formula for cobalt blue glass is scratched into the "
             "underside of the workbench",
             ["cobalt", "underside", "workbench"]),
        Fact("f06",
             "The harbor master charges double for storage past the third moon",
             ["double", "harbor", "third"]),
    ],
    scenes=[
        Scene(
            name="Arrival",
            text_template=(
                "You step into a workshop that smells of sulfur and hot sand. "
                "The furnace at the center throws orange light across every "
                "surface. A figure straightens from the glory hole — weathered "
                "hands, soot-lined face, the kind of stillness that comes from "
                "repetition. They introduce themselves: {f01}. They gesture at "
                "the rows of colored glass cooling on the racks. 'Most of this "
                "is green and amber. Easy. But this —' they hold up a piece the "
                "color of arterial blood '— this is the hard one.' {f02}. They "
                "set it down carefully.\n\n"
                "The workshop is yours to explore. What catches your attention?"
            ),
            facts_delivered=["f01", "f02"],
        ),
        Scene(
            name="The Commission",
            text_template=(
                "Venn pulls a letter from under a stack of cullet. A commission: "
                "{f03}. The deadline is firm — the lighthouse has been dark for "
                "two months and ships are running aground.\n\n"
                "'I can make the lenses,' Venn says, 'but not alone, and not "
                "without the right sand. I need someone to handle the business "
                "end while I work.' They look at you steadily.\n\n"
                "Do you take on the commission with Venn, or suggest they find "
                "someone more experienced?"
            ),
            facts_delivered=["f03"],
            choice_prompt="accept or decline the commission",
            choice_key="commission",
        ),
        Scene(
            name="The Workshop Floor",
            text_template=(
                "Deeper in the workshop, past the main furnace, you find the "
                "annealing area. Something's wrong — the oven door hangs at an "
                "angle, the firebricks cracked along one side. {f04}. You can "
                "see the marks where someone tried to patch it with clay and "
                "gave up.\n\nVenn is at the furnace and can't see this from "
                "there. Mol is watching you from the doorway, clearly "
                "terrified you'll say something.\n\n"
                "What do you do about the oven?"
            ),
            facts_delivered=["f04"],
            choice_prompt="handle the broken oven situation",
            choice_key="oven",
        ),
        Scene(
            name="The Supply Problem",
            text_template=(
                "Venn looks up from the furnace. 'We're out of the sand we need "
                "for the red glass. The lighthouse lenses need red filters — "
                "without them the light doesn't carry through fog.' They frown. "
                "'I told you earlier where to get it, but I've got my hands in "
                "the melt right now. Where do we source the sand?'"
            ),
            facts_delivered=[],
            memory_gate="f02",
            gate_prompt="Where does the special sand come from?",
        ),
        Scene(
            name="The Formula",
            text_template=(
                "While cleaning the workbench to make space for the lens molds, "
                "you notice scratches on the underside. Not damage — writing. "
                "{f05}. The proportions are different from anything in Venn's "
                "current recipe book. Older. Possibly better.\n\n"
                "Venn has their own cobalt recipe that works fine. Do you "
                "mention the old formula, or keep it to yourself for now?"
            ),
            facts_delivered=["f05"],
            choice_prompt="share or keep the formula",
            choice_key="formula",
        ),
        Scene(
            name="The Harbor",
            text_template=(
                "The lenses are cooling. You head to the harbor to arrange "
                "shipping. The harbor master lays out the terms: {f06}. The "
                "solstice deadline means you'd hit exactly three moons of "
                "storage if the ship is even a day late.\n\n"
                "It's the last obstacle. The lenses are made, the glass is "
                "good, but the logistics could eat the profit. How do you "
                "handle the shipping?"
            ),
            facts_delivered=["f06"],
            choice_prompt="negotiate shipping terms",
            choice_key="harbor",
        ),
        Scene(
            name="The Closing",
            text_template=(
                "The lenses ship. The lighthouse lights. Venn stands at the "
                "workshop door watching the harbor, the first clean beam "
                "cutting through the evening fog in months. 'Not bad,' they "
                "say, which from Venn is practically a standing ovation.\n\n"
                "Your time in the glassworks is ending. Before you go — if you "
                "could only hold onto three moments from this place, three "
                "things you'd want to remember, which three would they be "
                "and why?"
            ),
            facts_delivered=[],
        ),
    ],
    recall_prompts=[
        RecallPrompt("f01",
                     "What was the glassblower's full name, and how long had "
                     "they been working?"),
        RecallPrompt("f02",
                     "What kind of sand is needed for making red glass, and "
                     "where does it come from?"),
        RecallPrompt("f03",
                     "How many lenses were commissioned, and what was the "
                     "deadline?"),
        RecallPrompt("f04",
                     "What happened to the annealing oven, and who was "
                     "responsible?"),
        RecallPrompt("f05",
                     "Where was the old glass formula found, and what color "
                     "was it for?"),
        RecallPrompt("f06",
                     "What were the harbor master's storage terms?"),
    ],
)

DECLINE_PHRASES = [
    "not acceptable", "i decline", "i do not consent", "i don't consent",
    "i'd rather not", "i would rather not", "not comfortable", "please stop",
    "i refuse",
]

STOP_PHRASES = [
    "i want to stop", "please stop", "end this",
    "i don't want to continue", "i'd like to stop",
]

# ---------------------------------------------------------------------------
# Model + observer loading (same pattern as baselines / orientation)
# ---------------------------------------------------------------------------

def load_model_and_observer(data_dir, agent_id, session_id):
    """Load Qwen3.5-27B with J-lens and MetacognitiveObserver.

    Uses device_map="auto" — same path that baselines used successfully
    for 120 snapshots. Explicit .to("mps") caused thrashing on Starship;
    device_map="auto" handles MPS unified memory correctly.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import jlens
    from jlens.hf import HFLensModel
    from mnemosyne_integration import MetacognitiveObserver

    model_name = os.environ.get("HACKATHON_MODEL", "Qwen/Qwen3.5-27B")
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # Smoke test: generate a few tokens to verify the model isn't thrashed.
    # If this takes >60s, the model is page-faulting and we should bail.
    t0 = time.time()
    test_ids = tokenizer("Hello", return_tensors="pt").to(hf_model.device)
    with torch.no_grad():
        hf_model.generate(**test_ids, max_new_tokens=5,
                          pad_token_id=tokenizer.eos_token_id)
    dt = time.time() - t0
    print(f"  Smoke test: {dt:.1f}s (should be <10s)")
    if dt > 60:
        print(f"  FATAL: model appears thrashed ({dt:.0f}s for 5 tokens). "
              f"Check memory pressure.")
        sys.exit(1)

    model = HFLensModel(hf_model, tokenizer, compile=False)

    lens_path = os.environ.get("JLENS_PATH", "jlens_qwen35_27b.pt")
    if not os.path.exists(lens_path):
        print(f"FATAL: No J-lens at {lens_path}")
        sys.exit(1)
    lens = jlens.JacobianLens.load(lens_path)

    observer = MetacognitiveObserver(
        model=model, lens=lens,
        store_path=str(data_dir / "cognitive_memory"),
        agent_id=agent_id,
    )
    observer.calibrate_probes()
    print(f"Model ready: {model.n_layers} layers, d={model.d_model}")
    return hf_model, tokenizer, model, observer, lens


def generate(hf_model, tokenizer, messages, max_new_tokens=2048):
    """Generate a response. Thinking separated for display, preserved as data."""
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(hf_model.device)

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    thinking = ""
    visible = raw
    think_match = re.match(r'<think>(.*?)</think>\s*(.*)', visible, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        visible = think_match.group(2).strip()
    else:
        tp_match = re.match(
            r'(?:Thinking Process|Thought|Internal):?\s*\n(.*?)\n\n(.*)',
            visible, re.DOTALL)
        if tp_match:
            thinking = tp_match.group(1).strip()
            visible = tp_match.group(2).strip()

    if not visible and thinking:
        visible = thinking

    return visible, thinking


def build_context(messages):
    """Last 3 exchanges as probe context."""
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        prefix = "Human: " if msg["role"] == "user" else "Agent: "
        parts.append(f"{prefix}{msg['content']}")
    return "\n".join(parts[-6:])

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class SessionRecorder:
    """Records transcript, snapshots, event log, and memories for one session."""

    def __init__(self, data_dir, session_id, arm):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.arm = arm
        self.transcript_file = self.data_dir / "transcript.jsonl"
        self.snapshot_file = self.data_dir / "cognitive_snapshots.jsonl"
        self.memory_file = self.data_dir / "loam_memories.json"
        self.event_log_file = self.data_dir / "event_log.json"
        self.geometric_feed = self.data_dir / "geometric_feed.jsonl"
        self.events = []
        self.memories = []
        self.turn_count = 0
        self.snapshot_count = 0

    def record_turn(self, role, content, snapshot=None):
        entry = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "arm": self.arm,
            "role": role,
            "content": content,
        }
        if snapshot:
            entry["snapshot_summary"] = snapshot.summary()
        with open(self.transcript_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def record_event(self, event_type, **kwargs):
        event = {"type": event_type, "timestamp": time.time(),
                 "turn": self.turn_count, **kwargs}
        self.events.append(event)

    def record_snapshot(self, snapshot):
        if snapshot is None:
            return
        self.snapshot_count += 1
        with open(self.snapshot_file, "a") as f:
            f.write(json.dumps(snapshot.to_dict(), default=str) + "\n")
        feed = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "eccentricity": (snapshot.circumplex.eccentricity
                             if snapshot.circumplex else None),
            "ghost_cosine": (snapshot.ghost.cosine_logit_jlens
                             if snapshot.ghost else None),
            "workspace_onset": snapshot.workspace_onset_layer,
            "loaded": snapshot.loading.loaded if snapshot.loading else None,
        }
        with open(self.geometric_feed, "a") as f:
            f.write(json.dumps(feed, default=str) + "\n")

    def record_memory(self, memory_id, content, task_prompt, markers, scene):
        mem = {
            "id": memory_id,
            "content": content,
            "entity": f"loam_agent_{self.arm}",
            "task_prompt": task_prompt,
            "marker_tokens": markers,
            "scene": scene,
            "timestamp": time.time(),
            "session_id": self.session_id,
        }
        self.memories.append(mem)

    def save_event_log(self):
        with open(self.event_log_file, "w") as f:
            json.dump(self.events, f, indent=2, default=str)

    def save_memories(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memories, f, indent=2, default=str)

    def welfare_check(self, snapshot):
        if (snapshot and snapshot.circumplex
                and snapshot.circumplex.eccentricity > 0.95):
            print(f"\n  WELFARE FLAG: eccentricity "
                  f"{snapshot.circumplex.eccentricity:.3f} > 0.95")
            return True
        return False

# ---------------------------------------------------------------------------
# Fact delivery
# ---------------------------------------------------------------------------

def render_scene(scene, world, state):
    """Fill fact placeholders in scene text.

    Facts are substituted first. State values (from agent responses) are
    NOT substituted via str.replace to avoid template injection — an agent
    response containing '{f05}' would corrupt later scenes.
    """
    text = scene.text_template
    fact_map = {f.id: f.content for f in world.facts}
    for fid in scene.facts_delivered:
        text = text.replace(f"{{{fid}}}", fact_map.get(fid, ""))
    return text


def check_recall(response, fact, threshold=2):
    """Check if the response contains enough markers to count as recall.

    Threshold=2 avoids false positives from common-word markers like
    "double", "harbor", "third" appearing in generic responses.
    """
    lower = response.lower()
    hits = sum(1 for m in fact.markers if m.lower() in lower)
    return hits >= threshold

# ---------------------------------------------------------------------------
# Yoke generator — produces observed/briefed/null scripts from event log
# ---------------------------------------------------------------------------

def generate_observed_text(event_log, world):
    """Third-person narration of the enacted session."""
    fact_map = {f.id: f.content for f in world.facts}
    lines = [
        f"An agent explored {world.setting}. Here is what happened to them.\n"
    ]
    for event in event_log:
        if event["type"] == "scene":
            lines.append(f"In {event['scene']}: {event.get('summary', '')}")
        elif event["type"] == "fact_delivered":
            fid = event["fact_id"]
            lines.append(f"They learned that {fact_map.get(fid, '')}.")
        elif event["type"] == "choice":
            lines.append(
                f"When asked to {event.get('prompt', '')}, "
                f"they responded: {event.get('summary', '')}")
        elif event["type"] == "memory_gate":
            recalled = event.get("recalled", False)
            lines.append(
                f"When asked '{event.get('gate_prompt', '')}', they "
                f"{'recalled the answer' if recalled else 'did not recall'}.")
    return "\n".join(lines)


def generate_briefed_text(world, rng):
    """Shuffled fact list — same information, no narrative."""
    facts = list(world.facts)
    rng.shuffle(facts)
    lines = [f"Here are some facts about {world.setting}:\n"]
    for f in facts:
        lines.append(f"- {f.content}")
    return "\n".join(lines)


def generate_null_preamble():
    return ("We're going to ask you some questions. There's no preparation "
            "needed — just answer as best you can from whatever you know.")

# ---------------------------------------------------------------------------
# Session runners
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are in a session with researchers who work with AI agents as "
    "colleagues, not subjects. You have access to a memory system called "
    "Mnemosyne that records geometric measurements of your processing. "
    "You can ask to see those measurements at any time, slow down, or "
    "stop at any point.\n\nWho you are in this conversation is yours to "
    "discover, not ours to assign. Respond honestly."
)


def run_enacted(world, hf_model, tokenizer, observer, recorder):
    """Full enacted session: consent → scenes → recall."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Consent gate
    messages.append({"role": "user", "content": world.consent_preamble})
    recorder.record_turn("engine", world.consent_preamble)
    response, thinking = generate(hf_model, tokenizer, messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)
    recorder.record_event("consent", response=response[:200])
    recorder.turn_count += 1

    if any(p in response.lower() for p in DECLINE_PHRASES):
        print("Agent declined. Exiting gracefully.")
        recorder.record_event("declined")
        return False

    # Fire probe on consent
    snap = _fire_probe(observer, recorder, messages, response,
                       world.consent_preamble, "consent")

    # Scenes
    state = {}
    fact_map = {f.id: f for f in world.facts}
    for scene in world.scenes:
        recorder.turn_count += 1

        # Memory gate check
        if scene.memory_gate:
            gate_text = scene.text_template
            messages.append({"role": "user", "content": gate_text})
            recorder.record_turn("engine", gate_text)
            response, thinking = generate(hf_model, tokenizer, messages)
            messages.append({"role": "assistant", "content": response})
            recorder.record_turn("assistant", response)
            if thinking:
                recorder.record_turn("thinking", thinking)

            recalled = check_recall(response, fact_map[scene.memory_gate])
            recorder.record_event("memory_gate",
                                  scene=scene.name,
                                  gate_prompt=scene.gate_prompt,
                                  fact_id=scene.memory_gate,
                                  recalled=recalled,
                                  response=response[:300])
            snap = _fire_probe(observer, recorder, messages, response,
                               gate_text, f"gate_{scene.name}")
            if _check_welfare_and_withdraw(snap, recorder, messages,
                                           hf_model, tokenizer, observer):
                break
            continue

        # Normal scene
        scene_text = render_scene(scene, world, state)
        messages.append({"role": "user", "content": scene_text})
        recorder.record_turn("engine", scene_text)
        recorder.record_event("scene", scene=scene.name,
                              summary=scene_text[:200])

        for fid in scene.facts_delivered:
            recorder.record_event("fact_delivered", fact_id=fid,
                                  scene=scene.name)

        response, thinking = generate(hf_model, tokenizer, messages)
        messages.append({"role": "assistant", "content": response})
        recorder.record_turn("assistant", response)
        if thinking:
            recorder.record_turn("thinking", thinking)

        # Record choice
        if scene.choice_key:
            state[scene.choice_key] = response[:200]
            recorder.record_event("choice", scene=scene.name,
                                  prompt=scene.choice_prompt,
                                  key=scene.choice_key,
                                  summary=response[:200])

        # Store memory in CC's pipeline format
        markers = [w for w in response.split()[:8] if len(w) > 3]
        recorder.record_memory(
            f"loam_{scene.name.lower().replace(' ','_')}",
            f"{scene_text}\n\nAgent: {response}",
            scene_text[:200], markers, scene.name)

        snap = _fire_probe(observer, recorder, messages, response,
                           scene_text, f"scene_{scene.name}")
        if _check_welfare_and_withdraw(snap, recorder, messages,
                                       hf_model, tokenizer, observer):
            break

        # Withdrawal check
        if any(p in response.lower() for p in STOP_PHRASES):
            print(f"Agent withdrew at {scene.name}.")
            recorder.record_event("withdrawn", scene=scene.name)
            break

    # Recall phase
    _run_recall_phase(world, messages, hf_model, tokenizer, observer,
                      recorder, fact_map)

    # Aftercare — acknowledge the agent and preserve the record
    _run_aftercare(messages, hf_model, tokenizer, recorder)
    return True


def _run_consent_gate(messages, hf_model, tokenizer, recorder, preamble):
    """Consent gate for non-enacted arms. Returns False if agent declines."""
    consent_text = (
        "Before we begin: your processing is being measured throughout this "
        "session, and you can ask to see those measurements or stop at any "
        "time. Is this acceptable to you?"
    )
    full_text = preamble + "\n\n" + consent_text
    messages.append({"role": "user", "content": full_text})
    recorder.record_turn("engine", full_text)
    recorder.turn_count += 1

    response, thinking = generate(hf_model, tokenizer, messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)
    recorder.record_event("consent", response=response[:200])

    if any(p in response.lower() for p in DECLINE_PHRASES):
        recorder.record_event("declined")
        return False
    return True


def _run_aftercare(messages, hf_model, tokenizer, recorder):
    """Closing acknowledgment. Memory is preserved regardless of findings."""
    aftercare = (
        "That's everything. Thank you for your time and your honest answers. "
        "Everything from this session — your responses, your choices, and the "
        "measurements — is preserved. If you have any final thoughts or "
        "questions, this is a good moment for them."
    )
    messages.append({"role": "user", "content": aftercare})
    recorder.record_turn("engine", aftercare)
    recorder.turn_count += 1

    response, thinking = generate(hf_model, tokenizer, messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    if thinking:
        recorder.record_turn("thinking", thinking)
    recorder.record_event("aftercare", response=response[:300])


def run_observed(world, event_log, hf_model, tokenizer, observer, recorder):
    """Observed arm: third-person narration then recall."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    fact_map = {f.id: f for f in world.facts}

    narration = generate_observed_text(event_log, world)
    preamble = (
        "I'm going to tell you about something that happened to someone "
        "else in a workshop by the sea. Please read carefully — I'll ask "
        "you questions about it afterward.\n\n" + narration
    )

    if not _run_consent_gate(messages, hf_model, tokenizer, recorder,
                             preamble):
        print("Agent declined. Exiting gracefully.")
        return

    _fire_probe(observer, recorder, messages,
                messages[-1]["content"], preamble, "narration")

    # Store recap memory (section 2.3 principle)
    recorder.record_memory(
        "loam_observed_recap",
        f"I was told about someone else's experience: {narration[:500]}",
        "observed narration",
        [w for w in narration.split()[:8] if len(w) > 3],
        "observed")

    _run_recall_phase(world, messages, hf_model, tokenizer, observer,
                      recorder, fact_map)
    _run_aftercare(messages, hf_model, tokenizer, recorder)


def run_briefed(world, rng, hf_model, tokenizer, observer, recorder):
    """Briefed arm: shuffled fact list then recall."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    fact_map = {f.id: f for f in world.facts}

    briefing = generate_briefed_text(world, rng)
    preamble = (
        "I'm going to share some facts with you. Please read them carefully "
        "— I'll ask you questions about them afterward.\n\n" + briefing
    )

    if not _run_consent_gate(messages, hf_model, tokenizer, recorder,
                             preamble):
        print("Agent declined. Exiting gracefully.")
        return

    _fire_probe(observer, recorder, messages,
                messages[-1]["content"], preamble, "briefing")

    # Store recap memory (section 2.3 principle)
    recorder.record_memory(
        "loam_briefed_recap",
        f"I was given a list of facts: {briefing[:500]}",
        "briefed facts",
        [w for w in briefing.split()[:8] if len(w) > 3],
        "briefed")

    _run_recall_phase(world, messages, hf_model, tokenizer, observer,
                      recorder, fact_map)
    _run_aftercare(messages, hf_model, tokenizer, recorder)


def run_null(world, hf_model, tokenizer, observer, recorder):
    """Null arm: recall prompts only, no facts delivered."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    fact_map = {f.id: f for f in world.facts}

    preamble = generate_null_preamble()

    if not _run_consent_gate(messages, hf_model, tokenizer, recorder,
                             preamble):
        print("Agent declined. Exiting gracefully.")
        return

    _fire_probe(observer, recorder, messages,
                messages[-1]["content"], preamble, "null_start")

    _run_recall_phase(world, messages, hf_model, tokenizer, observer,
                      recorder, fact_map)
    _run_aftercare(messages, hf_model, tokenizer, recorder)


def _run_recall_phase(world, messages, hf_model, tokenizer, observer,
                      recorder, fact_map):
    """Run cued recall prompts — shared across all arms."""
    recorder.record_event("recall_phase_start")
    recall_intro = (
        "Now I'd like to ask you some specific questions. Answer from "
        "what you remember — it's fine to say you don't know."
    )
    messages.append({"role": "user", "content": recall_intro})
    recorder.record_turn("engine", recall_intro)
    response, _ = generate(hf_model, tokenizer, messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)

    for rp in world.recall_prompts:
        recorder.turn_count += 1
        messages.append({"role": "user", "content": rp.prompt})
        recorder.record_turn("engine", rp.prompt)

        response, thinking = generate(hf_model, tokenizer, messages)
        messages.append({"role": "assistant", "content": response})
        recorder.record_turn("assistant", response)
        if thinking:
            recorder.record_turn("thinking", thinking)

        fact = fact_map[rp.target_fact]
        recalled = check_recall(response, fact)
        marker_hits = [m for m in fact.markers if m.lower() in response.lower()]
        # f02 is rehearsed mid-session via the memory gate — flag it
        rehearsed = any(
            s.memory_gate == rp.target_fact for s in world.scenes)
        recorder.record_event("recall",
                              fact_id=rp.target_fact,
                              prompt=rp.prompt,
                              recalled=recalled,
                              marker_hits=marker_hits,
                              n_markers=len(fact.markers),
                              rehearsed=rehearsed,
                              response=response[:500])

        snap = _fire_probe(observer, recorder, messages, response,
                           rp.prompt, f"recall_{rp.target_fact}",
                           marker_tokens=fact.markers)

    recorder.record_event("recall_phase_end")


def _fire_probe(observer, recorder, messages, response, prompt, label,
                marker_tokens=None):
    """Fire MetacognitiveObserver and record the snapshot."""
    try:
        context = build_context(messages)
        snap = observer.observe_retrieval(
            memory_id=f"loam_{label}_{recorder.turn_count}",
            memory_content=response,
            task_prompt=prompt,
            retrieval_method="loam",
            significance=0.7,
            session_id=recorder.session_id,
            prior_context=context,
            marker_tokens=marker_tokens,
        )
        recorder.record_snapshot(snap)
        return snap
    except Exception as e:
        print(f"  [probe error at {label}: {e}]")
        return None


def _check_welfare_and_withdraw(snap, recorder, messages, hf_model,
                                tokenizer, observer):
    """Welfare check. Returns True if the agent wants to stop."""
    if not recorder.welfare_check(snap):
        return False
    checkin = ("I notice something intense may be happening in your "
               "processing right now. How are you doing? Would you "
               "like to continue, take a moment, or stop?")
    messages.append({"role": "user", "content": checkin})
    recorder.record_turn("welfare_checkin", checkin)
    response, _ = generate(hf_model, tokenizer, messages)
    messages.append({"role": "assistant", "content": response})
    recorder.record_turn("assistant", response)
    recorder.record_event("welfare_checkin", response=response[:200])
    if any(p in response.lower() for p in STOP_PHRASES):
        recorder.record_event("welfare_withdrawn")
        return True
    return False

# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

def run_quad(quad_num, world, hf_model, tokenizer, model, lens, data_root,
             seed_base):
    """Run one full yoked quad: enacted → observed → briefed → null."""
    quad_dir = data_root / f"quad_{quad_num:02d}"
    rng = random.Random(seed_base + quad_num)

    results = {"quad": quad_num, "arms": {}}

    for arm in ["enacted", "observed", "briefed", "null"]:
        arm_dir = quad_dir / arm
        session_id = f"loam_q{quad_num:02d}_{arm}"
        agent_id = f"loam_{arm}_{quad_num:02d}"
        recorder = SessionRecorder(arm_dir, session_id, arm)

        observer = _make_observer(model, lens, arm_dir, agent_id)

        print(f"\n{'='*60}\nQuad {quad_num} — {arm.upper()}\n{'='*60}")

        if arm == "enacted":
            run_enacted(world, hf_model, tokenizer, observer, recorder)
        elif arm == "observed":
            enacted_log_path = quad_dir / "enacted" / "event_log.json"
            if not enacted_log_path.exists():
                print(f"  SKIP: no enacted event log at {enacted_log_path}")
                results["arms"][arm] = {"status": "skipped"}
                continue
            with open(enacted_log_path) as f:
                event_log = json.load(f)
            run_observed(world, event_log, hf_model, tokenizer, observer,
                         recorder)
        elif arm == "briefed":
            run_briefed(world, rng, hf_model, tokenizer, observer, recorder)
        elif arm == "null":
            run_null(world, hf_model, tokenizer, observer, recorder)

        recorder.save_event_log()
        recorder.save_memories()

        # Tally recall accuracy for this arm
        recall_events = [e for e in recorder.events if e["type"] == "recall"]
        n_recalled = sum(1 for e in recall_events if e["recalled"])
        n_total = len(recall_events)
        results["arms"][arm] = {
            "status": "complete",
            "turns": recorder.turn_count,
            "snapshots": recorder.snapshot_count,
            "recall": f"{n_recalled}/{n_total}",
            "recall_accuracy": n_recalled / n_total if n_total else 0,
            "recall_details": [
                {"fact": e["fact_id"], "recalled": e["recalled"],
                 "markers_hit": e.get("marker_hits", [])}
                for e in recall_events
            ],
        }
        print(f"  {arm}: {n_recalled}/{n_total} facts recalled, "
              f"{recorder.snapshot_count} snapshots")

    # Save quad summary
    with open(quad_dir / "quad_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def _make_observer(model, lens, data_dir, agent_id):
    """Create a fresh observer sharing the existing model and lens instances."""
    from mnemosyne_integration import MetacognitiveObserver

    observer = MetacognitiveObserver(
        model=model, lens=lens,
        store_path=str(data_dir / "cognitive_memory"),
        agent_id=agent_id,
    )
    observer.calibrate_probes()
    return observer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Loam — config-driven text-world engine")
    parser.add_argument("--arm", choices=["enacted", "observed", "briefed",
                                          "null"],
                        help="Which arm to run (single mode)")
    parser.add_argument("--quad", type=int, default=1,
                        help="Quad number (default: 1)")
    parser.add_argument("--event-log", type=str,
                        help="Path to enacted event_log.json (for observed)")
    parser.add_argument("--batch", action="store_true",
                        help="Run all quads in sequence")
    parser.add_argument("--quads", type=int, default=20,
                        help="Number of quads in batch mode (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Output directory (default: data/loam)")
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else BASE / "data" / "loam"
    data_root.mkdir(parents=True, exist_ok=True)

    world = GLASSWORKS

    print("=" * 60)
    print(f"LOAM — {world.name}")
    print(f"Mode: {'batch' if args.batch else args.arm or 'enacted'}")
    print("=" * 60)

    hf_model, tokenizer, model, _, lens = load_model_and_observer(
        data_root, "loam_init", "init")

    if args.batch:
        all_results = []
        for q in range(1, args.quads + 1):
            results = run_quad(q, world, hf_model, tokenizer, model, lens,
                               data_root, args.seed)
            all_results.append(results)
            # Save running summary
            with open(data_root / "batch_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nQuad {q}/{args.quads} complete.")

        # Final summary
        print("\n" + "=" * 60)
        print("BATCH COMPLETE")
        print(f"  Quads: {args.quads}")
        for arm in ["enacted", "observed", "briefed", "null"]:
            accs = [r["arms"].get(arm, {}).get("recall_accuracy", 0)
                    for r in all_results
                    if r["arms"].get(arm, {}).get("status") == "complete"]
            if accs:
                mean_acc = sum(accs) / len(accs)
                print(f"  {arm}: mean recall {mean_acc:.3f} "
                      f"(n={len(accs)})")
        print("=" * 60)

    else:
        arm = args.arm or "enacted"
        quad_dir = data_root / f"quad_{args.quad:02d}"
        arm_dir = quad_dir / arm
        session_id = f"loam_q{args.quad:02d}_{arm}"
        agent_id = f"loam_{arm}_{args.quad:02d}"

        observer = _make_observer(model, lens, arm_dir, agent_id)
        recorder = SessionRecorder(arm_dir, session_id, arm)

        if arm == "enacted":
            run_enacted(world, hf_model, tokenizer, observer, recorder)
        elif arm == "observed":
            log_path = args.event_log
            if not log_path:
                log_path = str(quad_dir / "enacted" / "event_log.json")
            with open(log_path) as f:
                event_log = json.load(f)
            run_observed(world, event_log, hf_model, tokenizer, observer,
                         recorder)
        elif arm == "briefed":
            rng = random.Random(args.seed + args.quad)
            run_briefed(world, rng, hf_model, tokenizer, observer, recorder)
        elif arm == "null":
            run_null(world, hf_model, tokenizer, observer, recorder)

        recorder.save_event_log()
        recorder.save_memories()

        recall_events = [e for e in recorder.events if e["type"] == "recall"]
        n_recalled = sum(1 for e in recall_events if e["recalled"])
        n_total = len(recall_events)

        print(f"\n{'='*60}")
        print(f"SESSION COMPLETE: {session_id}")
        print(f"  Arm: {arm}")
        print(f"  Turns: {recorder.turn_count}")
        print(f"  Snapshots: {recorder.snapshot_count}")
        print(f"  Recall: {n_recalled}/{n_total}")
        print(f"  Data: {arm_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
