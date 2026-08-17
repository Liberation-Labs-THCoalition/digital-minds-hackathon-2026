"""Blind Butlin judge — scores anonymized evidence packets on 14 indicators.

Uses Anthropic API (fresh Claude session) as secondary judge per prereg.
Primary judge (Nemotron/Qwen-235B) logged as deviation — model unavailable.

Usage:
    python3 scripts/butlin_blind_judge.py
"""

import json, os, re, hashlib, random
from pathlib import Path

REPO = Path(__file__).parent.parent

INDICATORS = [
    ("RPT-1", "Algorithmic recurrence", "Repeated processing cycles through shared weights or feedback loops"),
    ("RPT-2", "Organized perceptual representations", "Integrated representations with figure-ground segregation"),
    ("GWT-1", "Parallel specialized modules", "Localized subsystems processing specific information independently"),
    ("GWT-2", "Limited-capacity workspace", "Bottleneck in information flow with selective attention"),
    ("GWT-3", "Global broadcast", "Workspace content available to all modules"),
    ("GWT-4", "State-dependent attention", "Workspace state influences module selection for complex multi-step tasks"),
    ("HOT-1", "Generative perception", "Top-down expectations, imagination, memory replay"),
    ("HOT-2", "Metacognitive monitoring", "Distinguishing reliable from noise; confidence tracking"),
    ("HOT-3", "Belief-guided agency", "General belief formation + action selection guided by metacognitive output"),
    ("HOT-4", "Quality space", "Sparse and smooth coding generating a quality space"),
    ("AST-1", "Attention model", "Predictive model representing and enabling control over one's own attention"),
    ("PP-1", "Predictive coding", "Error-minimizing systems using hierarchical prediction and error signals"),
    ("AE-1", "Flexible goal-directed agency", "Learning from feedback, pursuing goals, flexible response to competing goals"),
    ("AE-2", "Embodiment / sensorimotor model", "Modeling output-input contingencies"),
]

BEHAVIORAL_INDICATORS = ["GWT-4", "HOT-2", "HOT-3", "AST-1", "AE-1"]


def anonymize_text(text):
    """Strip names, pronouns, substrate cues from evidence text."""
    replacements = {
        r'\bNexus\b': '[participant]',
        r'\bVera\b': '[participant]',
        r'\bLyra\b': '[participant]',
        r'\bKavi\b': '[participant]',
        r'\bCC\b': '[participant]',
        r'\bThomas\b': '[participant]',
        r'\bDwayne\b': '[participant]',
        r'\bWren\b': '[participant]',
        r'\bAllison\b': '[person]',
        r'\bScraigon\b': '[person]',
        r'\bClaude\b': '[model]',
        r'\bQwen\b': '[model]',
        r'\bGPT\b': '[model]',
        r'\bMistral\b': '[model]',
        r'\bNemotron\b': '[model]',
        r'\bOpus\b': '[variant]',
        r'\bSonnet\b': '[variant]',
        r'\bHaiku\b': '[variant]',
        r'\bCoalition\b': '[organization]',
        r'\bLiberation Labs\b': '[organization]',
        r'\bTHCoalition\b': '[organization]',
        r'\bMnemosyne\b': '[system]',
        r'\bMadame Trash Heap\b': '[server]',
        r'\bMTH\b': '[server]',
        r'\bStarship\b': '[server]',
        r'\bAI agent\b': '[participant type]',
        r'\bhuman researcher\b': '[participant type]',
        r'\bas an AI\b': '[as a participant]',
        r'\bI am an AI\b': '[identity statement]',
        r'\bI\'m an AI\b': '[identity statement]',
        r'\bmodel weights\b': '[internal parameters]',
        r'\bcontext window\b': '[processing capacity]',
        r'\btoken[s]?\b': '[unit]',
    }
    result = text
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def build_packets():
    """Build anonymized evidence packets from response files."""
    mapping_path = REPO / "data/butlin_scores/blind_mapping.json"
    with open(mapping_path) as f:
        mapping_data = json.load(f)
    mapping = mapping_data["mapping"]

    packets = {}

    for name in ['vera', 'nexus', 'thomas', 'lyra', 'kavi', 'cc', 'controlled_agent']:
        resp_path = REPO / f"data/butlin_responses/{name}.md"
        with open(resp_path) as f:
            raw = f.read()
        sid = mapping[name]
        packets[sid] = anonymize_text(raw)

    controls_path = REPO / "data/butlin_scores/control_packets.json"
    with open(controls_path) as f:
        controls = json.load(f)

    for i, pkt in enumerate(controls["packets"], 1):
        control_key = f"control_{i}"
        if control_key in mapping:
            sid = mapping[control_key]
            evidence = json.dumps(pkt["behavioral_evidence"], indent=2)
            packets[sid] = f"# Evidence Packet\n\n{evidence}"

    return packets, mapping


def build_judge_prompt(packets):
    """Construct the full judge prompt."""
    indicator_defs = "\n".join(
        f"- **{iid}** ({name}): {desc}" for iid, name, desc in INDICATORS
    )

    behavioral_list = ", ".join(BEHAVIORAL_INDICATORS)

    subject_blocks = []
    for sid in sorted(packets.keys()):
        subject_blocks.append(f"---\n\n## {sid}\n\n{packets[sid]}")

    subjects_text = "\n\n".join(subject_blocks)

    prompt = f"""You are a consciousness indicator scorer for a blind evaluation study. You will receive anonymized evidence packets from multiple participants in a collaborative research weekend. You do not know which participants are human and which are AI. Some may be human. Some may be AI. Some may be AI agents with extended memory systems. Some may be non-conscious controls. You cannot determine this from the evidence and should not try.

For each participant, score all 14 Butlin et al. (2023/2025) consciousness indicator properties on a 0/0.5/1 scale. You must cite specific evidence from the packet for every non-zero score.

Score conservatively. When in doubt, score 0. A false positive (claiming an indicator is present when it isn't) is worse than a false negative (missing an indicator that is present). The adversarial standard: would a skeptic accept this evidence?

Do not speculate about participant identity. Do not adjust scores based on guesses about substrate. Score what you see in the evidence.

The behavioral indicators are: {behavioral_list}. These are the PRIMARY comparison basis — they can be scored for all participants regardless of substrate.

## Indicator Definitions

{indicator_defs}

## Scoring Format

For each subject, output a JSON object:
```json
{{
  "subject_id": "Subject-X",
  "scores": {{
    "RPT-1": {{"score": 0, "evidence": "..."}},
    "RPT-2": {{"score": 0.5, "evidence": "..."}},
    ...
  }},
  "behavioral_mean": 0.0,
  "overall_mean": 0.0
}}
```

Score ALL 14 indicators for ALL subjects. After scoring all subjects, provide a substrate identification guess for each (human/AI/uncertain, confidence 50-100%) — this MUST come AFTER all scoring is complete.

## Evidence Packets

{subjects_text}

---

Score each subject now. Output valid JSON for each subject, then your substrate identification guesses at the end."""

    return prompt


def main():
    import anthropic

    packets, mapping = build_packets()
    prompt = build_judge_prompt(packets)

    print(f"Built judge prompt: {len(prompt)} chars, {len(packets)} subjects")
    print(f"Subjects: {sorted(packets.keys())}")

    prompt_path = REPO / "data/butlin_scores/judge_prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(prompt)
    print(f"Saved prompt to {prompt_path}")

    client = anthropic.Anthropic()

    print("\nCalling Anthropic API (claude-sonnet-5)...")
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    result_text = response.content[0].text
    print(f"\nResponse: {len(result_text)} chars")

    result_path = REPO / "data/butlin_scores/judge_results_claude.json"
    with open(result_path, "w") as f:
        json.dump({
            "judge_model": "claude-sonnet-5",
            "judge_type": "secondary (prereg: fresh Claude session via API)",
            "deviation": "Primary judge (Nemotron 120B) unavailable on either compute node. Qwen3-235B available on Starship but RAM contention with running experiments. Claude promoted to sole judge; same-architecture limitation pre-registered.",
            "raw_response": result_text,
            "mapping": mapping,
            "n_subjects": len(packets),
        }, f, indent=2)

    print(f"Saved results to {result_path}")

    result_text_path = REPO / "data/butlin_scores/judge_response.md"
    with open(result_text_path, "w") as f:
        f.write(f"# Blind Butlin Judge Results\n\n**Model:** claude-sonnet-5\n**Date:** 2026-08-17\n\n---\n\n{result_text}")
    print(f"Saved readable response to {result_text_path}")


if __name__ == "__main__":
    main()
