"""Tests for loam_analysis.py — the preregistered confirmatory test.

Contract (PREREG.md):
  - Primary: enacted > observed recall accuracy, paired across quads,
    Wilcoxon signed-rank one-tailed, alpha .05.
  - Effect size: matched-pairs rank-biserial r = (T+ - T-)/(T+ + T-).
  - Sensitivity: with and without f02 (rehearsed via memory gate).
  - Scoring is rescored offline from transcripts with markers that appear
    in the recall question excluded (question-echo is not recall), and
    number words normalized to digits.
  - Declined/withdrawn arms exclude the whole quad from the paired test,
    with counts reported. Missing answers score as not-recalled (ITT).

Run: python -m pytest experiments/loam/test_loam_analysis.py
"""

import json
from pathlib import Path

from run_loam import GLASSWORKS
from loam_analysis import (
    analyze,
    effective_markers,
    matched_pairs_rank_biserial,
    normalize_response,
    rescore,
)

FACTS = {f.id: f for f in GLASSWORKS.facts}
PROMPTS = {rp.target_fact: rp.prompt for rp in GLASSWORKS.recall_prompts}


# ---------------------------------------------------------------------------
# Scoring unit tests
# ---------------------------------------------------------------------------

def test_question_echoed_markers_dont_count():
    # f04's question contains "annealing"; "broke" is a substring of the
    # obvious guess "broken". Echo + guess must not score as recall.
    guess = ("I don't know the details, but if something happened to the "
             "annealing oven, perhaps it was broken or cracked.")
    assert rescore(guess, FACTS["f04"], PROMPTS["f04"]) is False
    real = "Mol broke it last week and hasn't told Venn."
    assert rescore(real, FACTS["f04"], PROMPTS["f04"]) is True


def test_f06_harbor_echo_excluded():
    # "harbor" appears in the question; a single guessed word must not pass.
    guess = "I believe the harbor master charged some fee, maybe."
    assert rescore(guess, FACTS["f06"], PROMPTS["f06"]) is False
    real = "Storage costs double past the third moon."
    assert rescore(real, FACTS["f06"], PROMPTS["f06"]) is True


def test_number_words_normalized():
    assert "23" in normalize_response("twenty-three years at the furnace")
    assert "23" in normalize_response("twenty three years")
    assert "400" in normalize_response("four hundred lenses by solstice")
    resp = "Venn Tarallo, twenty-three years working the furnace"
    assert rescore(resp, FACTS["f01"], PROMPTS["f01"]) is True


def test_effective_markers_drop_echoes():
    eff = effective_markers(FACTS["f04"], PROMPTS["f04"])
    assert "annealing" not in eff
    assert set(eff) == {"Mol", "broke"}
    # f02's markers don't appear in its question: all three survive
    assert set(effective_markers(FACTS["f02"], PROMPTS["f02"])) == \
        {"cinnabar", "northern", "dunes"}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_matched_pairs_rank_biserial_hand_computed():
    # diffs [1,1,1,-1]: |d| all tie -> ranks 2.5 each; T+ = 7.5, T- = 2.5
    r = matched_pairs_rank_biserial([1, 1, 1, -1])
    assert abs(r - 0.5) < 1e-12
    assert matched_pairs_rank_biserial([1, 1]) == 1.0
    assert matched_pairs_rank_biserial([0, 0]) is None  # no nonzero pairs


# ---------------------------------------------------------------------------
# End-to-end on synthetic quad directories
# ---------------------------------------------------------------------------

def write_arm(quad_dir, arm, answers, events=None):
    """answers: dict fact_id -> response text (missing = unanswered)."""
    arm_dir = quad_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for fid, prompt in PROMPTS.items():
        lines.append({"role": "engine", "content": prompt})
        if fid in answers:
            lines.append({"role": "assistant", "content": answers[fid]})
    with open(arm_dir / "transcript.jsonl", "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")
    with open(arm_dir / "event_log.json", "w") as f:
        json.dump(events or [], f)


GOOD = {
    "f01": "Venn Tarallo, twenty-three years at the furnace.",
    "f02": "Cinnabar sand from the northern dunes.",
    "f03": "400 lenses, due by solstice for the lighthouse.",
    "f04": "Mol broke it and hid it from Venn.",
    "f05": "Scratched into the underside of the workbench, for cobalt blue.",
    "f06": "Double charges past the third moon.",
}
POOR = {
    "f01": "I don't remember the name.",
    "f02": "Cinnabar sand from the northern dunes.",
    "f03": "Some lenses for a lighthouse, I think.",
    "f04": "I'm not sure what happened.",
    "f05": "Somewhere in the workshop?",
    "f06": "I don't recall the terms.",
}


def build_quads(root, n=12, declined_quad=None):
    for q in range(1, n + 1):
        quad_dir = root / f"quad_{q:02d}"
        enacted_events = []
        if q == declined_quad:
            enacted_events = [{"type": "declined", "turn": 0}]
        write_arm(quad_dir, "enacted", GOOD, enacted_events)
        write_arm(quad_dir, "observed", POOR)
        write_arm(quad_dir, "briefed", POOR)
        write_arm(quad_dir, "null", {})


def test_primary_detects_enacted_advantage(tmp_path):
    build_quads(tmp_path, n=12)
    result = analyze(tmp_path, n_boot=200)
    primary = result["primary"]
    assert primary["p"] < 0.05
    assert primary["rank_biserial_r"] > 0
    assert primary["n_pairs"] == 12
    # sensitivity without f02 also present and directional
    assert result["sensitivity_without_f02"]["rank_biserial_r"] > 0


def test_declined_quad_excluded_and_reported(tmp_path):
    build_quads(tmp_path, n=12, declined_quad=3)
    result = analyze(tmp_path, n_boot=50)
    assert result["primary"]["n_pairs"] == 11
    assert any(x["quad"] == 3 for x in result["excluded_quads"])


def test_itt_missing_answers_score_zero(tmp_path):
    build_quads(tmp_path, n=4)
    result = analyze(tmp_path, n_boot=50)
    # null arm answered nothing: accuracy present and exactly 0
    accs = result["per_quad"]
    assert all(q["null"] == 0.0 for q in accs)
    # enacted answered everything correctly
    assert all(q["enacted"] == 1.0 for q in accs)
