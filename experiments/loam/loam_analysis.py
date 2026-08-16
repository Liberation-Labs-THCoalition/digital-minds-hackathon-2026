#!/usr/bin/env python3
"""Loam analysis — the preregistered confirmatory test (PREREG.md).

PRIMARY: enacted > observed recall accuracy, paired across quads.
         Wilcoxon signed-rank, one-tailed, alpha = .05.
         Effect size: matched-pairs rank-biserial r = (T+ - T-)/(T+ + T-).
         Bootstrap 95% CI on the mean paired difference (quad-level
         resampling, 10,000 draws).
SENSITIVITY: same test excluding f02 (rehearsed via the memory gate).
EXPLORATORY (uncorrected, labeled): briefed vs null; per-fact breakdown.

Scoring is RESCORED offline from transcript.jsonl rather than trusting the
runner's live `recalled` flags: markers that appear verbatim in the recall
question are excluded from hit counts (echoing the question is not recall),
and number words are normalized to digits before matching. Disagreements
with the live flags are counted and reported.

Declined or withdrawn arms exclude their whole quad from the paired test
(counts reported). A missing recall answer scores as not-recalled (ITT).

Usage:
    python loam_analysis.py --data-dir data/loam [--n-boot 10000]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

from run_loam import GLASSWORKS

ARMS = ["enacted", "observed", "briefed", "null"]

# Number words the Glassworks facts actually use, plus common small numbers
# so paraphrases like "twenty three" or "four hundred" match digit markers.
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def normalize_response(text: str) -> str:
    """Lowercase and convert spelled-out numbers to digits.

    Handles "twenty-three"/"twenty three" -> 23 and "<unit> hundred" -> N00.
    Appends digit forms rather than replacing, so word markers still match.
    """
    lower = text.lower()
    found = []
    for unit_word, unit_val in _UNITS.items():
        for tens_word, tens_val in _TENS.items():
            if re.search(rf"\b{tens_word}[-\s]{unit_word}\b", lower):
                found.append(str(tens_val + unit_val))
        if re.search(rf"\b{unit_word}\s+hundred\b", lower):
            found.append(str(unit_val * 100))
    for word_map in (_TEENS, _TENS):
        for word, val in word_map.items():
            if re.search(rf"\b{word}\b", lower):
                found.append(str(val))
    if found:
        lower += " " + " ".join(found)
    return lower


def effective_markers(fact, prompt: str) -> list[str]:
    """Markers that do NOT appear in the recall question itself.

    A marker echoed from the question (f04 "annealing", f06 "harbor") can
    be produced with zero knowledge of the fact, so it never counts.
    """
    prompt_lower = prompt.lower()
    return [m for m in fact.markers if m.lower() not in prompt_lower]


def rescore(response: str, fact, prompt: str) -> bool:
    """Corrected recall check: >= min(2, len(effective)) non-echoed marker
    hits on the normalized response."""
    markers = effective_markers(fact, prompt)
    if not markers:
        return False
    normalized = normalize_response(response)
    hits = sum(1 for m in markers if m.lower() in normalized)
    return hits >= min(2, len(markers))


def matched_pairs_rank_biserial(diffs) -> float | None:
    """r = (T+ - T-)/(T+ + T-) over nonzero paired differences.

    This is the matched-pairs form; the independent-samples formula
    2U/(n1*n2)-1 does not apply to a Wilcoxon signed-rank design.
    """
    d = np.array([x for x in diffs if x != 0], dtype=float)
    if len(d) == 0:
        return None
    ranks = stats.rankdata(np.abs(d))
    t_plus = float(ranks[d > 0].sum())
    t_minus = float(ranks[d < 0].sum())
    return (t_plus - t_minus) / (t_plus + t_minus)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_events(arm_dir: Path) -> list[dict]:
    f = arm_dir / "event_log.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _recall_answers(arm_dir: Path) -> dict[str, str]:
    """Map fact_id -> full recall answer, read from transcript.jsonl.

    A recall answer is the first assistant turn following the engine turn
    whose content is that fact's recall prompt. Falls back to the (500-char
    truncated) response stored on the recall event if the transcript is
    missing the exchange.
    """
    prompts = {rp.prompt: rp.target_fact for rp in GLASSWORKS.recall_prompts}
    answers: dict[str, str] = {}

    transcript = arm_dir / "transcript.jsonl"
    if transcript.exists():
        pending: str | None = None
        with open(transcript) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = entry.get("role")
                content = entry.get("content", "")
                if role == "engine" and content in prompts:
                    pending = prompts[content]
                elif role == "assistant" and pending is not None:
                    answers.setdefault(pending, content)
                    pending = None

    for event in _load_events(arm_dir):
        if event.get("type") == "recall":
            fid = event.get("fact_id")
            if fid and fid not in answers and event.get("response"):
                answers[fid] = event["response"]
    return answers


def _arm_status(arm_dir: Path) -> str:
    events = _load_events(arm_dir)
    for event in events:
        if event.get("type") in ("declined", "withdrawn"):
            return event["type"]
    if not (arm_dir / "transcript.jsonl").exists():
        return "missing"
    return "ok"


def score_arm(arm_dir: Path) -> dict:
    """Rescore one arm. Returns per-fact results, accuracy (ITT over all 6
    facts), accuracy without f02, status, and live-flag disagreements."""
    answers = _recall_answers(arm_dir)
    prompts = {rp.target_fact: rp.prompt for rp in GLASSWORKS.recall_prompts}
    live_flags = {e.get("fact_id"): e.get("recalled")
                  for e in _load_events(arm_dir) if e.get("type") == "recall"}

    per_fact: dict[str, bool] = {}
    disagreements = 0
    for fact in GLASSWORKS.facts:
        response = answers.get(fact.id, "")
        recalled = rescore(response, fact, prompts[fact.id]) if response else False
        per_fact[fact.id] = recalled
        if fact.id in live_flags and live_flags[fact.id] is not None \
                and bool(live_flags[fact.id]) != recalled:
            disagreements += 1

    n = len(GLASSWORKS.facts)
    without_f02 = [v for fid, v in per_fact.items() if fid != "f02"]
    return {
        "status": _arm_status(arm_dir),
        "per_fact": per_fact,
        "accuracy": sum(per_fact.values()) / n,
        "accuracy_without_f02": sum(without_f02) / len(without_f02),
        "live_disagreements": disagreements,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _paired_test(enacted: list[float], observed: list[float],
                 n_boot: int, seed: int = 42) -> dict:
    diffs = [e - o for e, o in zip(enacted, observed)]
    result = {
        "n_pairs": len(diffs),
        "mean_diff": float(np.mean(diffs)) if diffs else None,
        "rank_biserial_r": matched_pairs_rank_biserial(diffs),
    }
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) == 0:
        result.update({"p": None, "W": None,
                       "note": "all paired differences are zero"})
        return result
    w, p = stats.wilcoxon(enacted, observed, alternative="greater")
    result.update({"W": float(w), "p": float(p)})

    # Bootstrap CI on the mean paired difference: resample QUADS (pairs),
    # never the arms independently — the design is yoked.
    rng = np.random.RandomState(seed)
    diffs_arr = np.array(diffs)
    means = [float(np.mean(diffs_arr[rng.randint(0, len(diffs_arr),
                                                 len(diffs_arr))]))
             for _ in range(n_boot)]
    result["mean_diff_ci_95"] = [float(np.percentile(means, 2.5)),
                                 float(np.percentile(means, 97.5))]
    return result


def analyze(data_root, n_boot: int = 10_000) -> dict:
    data_root = Path(data_root)
    quad_dirs = sorted(d for d in data_root.glob("quad_*") if d.is_dir())

    per_quad = []
    excluded_quads = []
    total_disagreements = 0

    for quad_dir in quad_dirs:
        quad_num = int(quad_dir.name.split("_")[1])
        arms = {arm: score_arm(quad_dir / arm) for arm in ARMS}
        total_disagreements += sum(a["live_disagreements"]
                                   for a in arms.values())

        bad = {arm: a["status"] for arm, a in arms.items()
               if arm in ("enacted", "observed") and a["status"] != "ok"}
        if bad:
            excluded_quads.append({"quad": quad_num, "reason": bad})
            continue

        per_quad.append({
            "quad": quad_num,
            **{arm: arms[arm]["accuracy"] for arm in ARMS},
            "enacted_without_f02": arms["enacted"]["accuracy_without_f02"],
            "observed_without_f02": arms["observed"]["accuracy_without_f02"],
            "per_fact": {arm: arms[arm]["per_fact"] for arm in ARMS},
            "arm_status": {arm: arms[arm]["status"] for arm in ARMS},
        })

    primary = _paired_test([q["enacted"] for q in per_quad],
                           [q["observed"] for q in per_quad], n_boot)
    sensitivity = _paired_test([q["enacted_without_f02"] for q in per_quad],
                               [q["observed_without_f02"] for q in per_quad],
                               n_boot)

    # Exploratory: briefed vs null, two-sided (prereg: uncorrected, labeled).
    briefed = [q["briefed"] for q in per_quad]
    null = [q["null"] for q in per_quad]
    exploratory: dict = {"n_pairs": len(briefed)}
    if briefed and any(b - n != 0 for b, n in zip(briefed, null)):
        w, p = stats.wilcoxon(briefed, null, alternative="two-sided")
        exploratory.update({
            "W": float(w), "p": float(p),
            "mean_diff": float(np.mean(np.array(briefed) - np.array(null))),
            "rank_biserial_r": matched_pairs_rank_biserial(
                [b - n for b, n in zip(briefed, null)]),
        })
    else:
        exploratory["note"] = "no nonzero paired differences"

    per_fact_rates = {
        arm: {f.id: float(np.mean([q["per_fact"][arm][f.id]
                                   for q in per_quad]))
              for f in GLASSWORKS.facts} if per_quad else {}
        for arm in ARMS
    }

    return {
        "n_quads_found": len(quad_dirs),
        "per_quad": per_quad,
        "excluded_quads": excluded_quads,
        "primary": primary,
        "sensitivity_without_f02": sensitivity,
        "exploratory_briefed_vs_null": exploratory,
        "per_fact_recall_rates": per_fact_rates,
        "live_flag_disagreements": total_disagreements,
        "n_boot": n_boot,
        "scoring": "rescored from transcripts; question-echoed markers "
                   "excluded; number words normalized; ITT",
    }


def main():
    parser = argparse.ArgumentParser(description="Loam preregistered analysis")
    parser.add_argument("--data-dir", default="data/loam",
                        help="Root containing quad_XX/ directories")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <data-dir>/loam_analysis.json)")
    args = parser.parse_args()

    result = analyze(args.data_dir, n_boot=args.n_boot)

    print("=" * 65)
    print("LOAM ANALYSIS — preregistered paired test")
    print("=" * 65)
    print(f"\nQuads found: {result['n_quads_found']}  "
          f"pairs analyzed: {result['primary']['n_pairs']}  "
          f"excluded: {len(result['excluded_quads'])}")
    for ex in result["excluded_quads"]:
        print(f"  EXCLUDED quad {ex['quad']}: {ex['reason']}")
    if result["live_flag_disagreements"]:
        print(f"  NOTE: rescoring disagrees with live recalled flags on "
              f"{result['live_flag_disagreements']} item(s)")

    p = result["primary"]
    print(f"\nPRIMARY enacted > observed (Wilcoxon, one-tailed):")
    print(f"  W={p['W']}  p={p['p']}  r={p['rank_biserial_r']}  "
          f"mean diff={p['mean_diff']}  CI95={p.get('mean_diff_ci_95')}")
    s = result["sensitivity_without_f02"]
    print(f"SENSITIVITY without f02: W={s['W']}  p={s['p']}  "
          f"r={s['rank_biserial_r']}")
    e = result["exploratory_briefed_vs_null"]
    print(f"EXPLORATORY briefed vs null (two-sided, uncorrected): {e}")

    out = args.output or str(Path(args.data_dir) / "loam_analysis.json")
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
