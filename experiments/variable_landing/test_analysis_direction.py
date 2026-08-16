"""Directional tests for the VL analysis layer.

These encode the frozen prereg (v4) contract:
  - Primary metric is Jaccard DISTANCE (more geometric change = larger value).
  - P1: fictional > scrambled on distance; a true effect must yield p < .05
    on synthetic data with the preregistered direction.
  - Rank-biserial from scipy's U1 is r = 2U/(n1*n2) - 1 (favorable to
    sample 1 = positive).
  - Holm family is m=2 (PRIMARY + SECONDARY only); sanity tests uncorrected.

Run: python -m pytest experiments/variable_landing/test_analysis_direction.py
"""

import numpy as np
from scipy import stats

from variable_landing_analysis import (
    CONFIRMATORY_COMPARISONS,
    UNCORRECTED_COMPARISONS,
    analyze_trials,
    compute_metric,
    jaccard_distance,
    rank_biserial_r,
)


def make_trial(arm, snap1, snap2):
    return {"arm": arm, "snap1_data": snap1, "snap2_data": snap2}


def test_metric_is_distance():
    assert jaccard_distance(["a", "b"], ["a", "b"]) == 0.0
    assert jaccard_distance(["a", "b"], ["c", "d"]) == 1.0
    # half overlap: |intersection|=1, |union|=3 -> distance 2/3
    assert abs(jaccard_distance(["a", "b"], ["a", "c"]) - 2 / 3) < 1e-12


def test_compute_metric_excludes_dict_snapshots():
    # Dict-shaped snapshots are a different measurement; pooling a numeric
    # delta (higher = more change) with similarity/distance corrupts the arm.
    trial = make_trial("lived", {"x": 1.0}, {"x": 2.0})
    assert np.isnan(compute_metric(trial))


def test_rank_biserial_sign_convention():
    # x fully dominates y: scipy returns U1 = n1*n2 = 9, correct r = +1.
    u, _ = stats.mannwhitneyu([10, 11, 12], [1, 2, 3], alternative="greater")
    assert rank_biserial_r(u, 3, 3) == 1.0


def test_confirmatory_family_is_exactly_prereg():
    labels = [c[0] for c in CONFIRMATORY_COMPARISONS]
    assert len(labels) == 2
    assert any("fictional vs scrambled" in l for l in labels)
    assert any("lived vs fictional" in l for l in labels)
    # sanity comparisons exist but are outside the corrected family
    assert len(UNCORRECTED_COMPARISONS) >= 3


def test_true_prereg_effect_is_detected():
    """Synthetic data with the preregistered direction: fictional memories
    produce MORE workspace change (higher distance) than scrambled. The
    primary test must reject, with positive rank-biserial r."""
    rng = np.random.RandomState(7)
    base = [f"t{i}" for i in range(20)]
    trials = []
    for i in range(70):
        # fictional: snap2 keeps ~8/20 of snap1's tokens (large change)
        keep = rng.choice(20, size=8, replace=False)
        snap2 = [base[k] for k in keep] + [f"f{i}_{j}" for j in range(12)]
        trials.append(make_trial("fictional", list(base), snap2))
        # scrambled: snap2 keeps ~16/20 (small change)
        keep = rng.choice(20, size=16, replace=False)
        snap2 = [base[k] for k in keep] + [f"s{i}_{j}" for j in range(4)]
        trials.append(make_trial("scrambled", list(base), snap2))
        # secondary arms so the family runs
        trials.append(make_trial("lived", list(base),
                                 [base[k] for k in rng.choice(20, 6, replace=False)]
                                 + [f"l{i}_{j}" for j in range(14)]))
        trials.append(make_trial("no_intervention", list(base),
                                 [base[k] for k in rng.choice(20, 19, replace=False)]))

    result = analyze_trials({"trials": trials, "excluded": []}, n_boot=200)

    primary = next(c for c in result["confirmatory"]
                   if "fictional vs scrambled" in c["label"])
    assert primary["p_raw"] < 0.05, "true prereg effect must be detected"
    assert primary["rank_biserial_r"] > 0, "effect direction must read positive"
    assert primary["holm_rejected"]
    assert result["holm_family_size"] == 2


def test_skipped_comparison_does_not_inflate_family():
    """With no 'lived' arm data, the secondary is skipped; Holm must correct
    over the 1 test that ran, not inject placeholder p-values."""
    rng = np.random.RandomState(3)
    base = [f"t{i}" for i in range(20)]
    trials = []
    for i in range(30):
        keep = rng.choice(20, size=8, replace=False)
        trials.append(make_trial("fictional", list(base),
                                 [base[k] for k in keep] + [f"f{i}"]))
        keep = rng.choice(20, size=16, replace=False)
        trials.append(make_trial("scrambled", list(base),
                                 [base[k] for k in keep] + [f"s{i}"]))
    result = analyze_trials({"trials": trials, "excluded": []}, n_boot=50)
    assert result["holm_family_size"] == 1


def test_per_arm_exclusion_counts_reported():
    trials = [make_trial("fictional", ["a"], ["b"])]
    excluded = [{"arm": "fictional", "reason": "zero facts"},
                {"arm": "scrambled", "reason": "SIRA miss"},
                {"arm": "scrambled", "reason": "zero facts"}]
    result = analyze_trials({"trials": trials, "excluded": excluded}, n_boot=10)
    assert result["exclusions_by_arm"] == {"fictional": 1, "scrambled": 2}


# ---------------------------------------------------------------------------
# Unit-of-analysis amendments (2026-08-16, post-v3): the confirmatory unit is
# the MEMORY, paired across arms. Trial-level inference is refused whenever
# repeats are byte-identical duplicates (the v3 failure: 44/44 cells x7).
# ---------------------------------------------------------------------------

def make_mem_trial(arm, memory_id, snap1, snap2):
    return {"arm": arm, "memory_id": memory_id,
            "snap1_data": snap1, "snap2_data": snap2}


def synth_memories(n_mem=11, lived_hi=True, repeats=7, identical=True, seed=5):
    """Build trials where lived changes more than fictional per memory."""
    rng = np.random.RandomState(seed)
    trials = []
    base = [f"t{i}" for i in range(10)]
    for m in range(n_mem):
        mem = f"mem_{m:02d}"
        arm_snap2 = {
            "lived": [base[k] for k in rng.choice(10, 3, replace=False)] + [f"L{m}"],
            "fictional": [base[k] for k in rng.choice(10, 6, replace=False)] + [f"F{m}"],
            "scrambled": [base[k] for k in rng.choice(10, 6, replace=False)] + [f"S{m}"],
            "no_intervention": list(base),
        }
        for arm, snap2 in arm_snap2.items():
            for r in range(repeats):
                s2 = list(snap2) if identical else list(snap2) + [f"r{r}"]
                trials.append(make_mem_trial(arm, mem, list(base), s2))
    return trials


def test_duplicate_repeats_detected_and_trial_level_refused():
    trials = synth_memories(identical=True)
    result = analyze_trials({"trials": trials, "excluded": []}, n_boot=50)
    assert result["duplication_rate"] == 1.0
    assert result["confirmatory_unit"] == "memory"
    for comp in result["confirmatory"]:
        assert comp["n_pairs"] == 11
        assert "U" not in comp  # no trial-level Mann-Whitney in confirmatory


def test_memory_level_secondary_matches_direct_wilcoxon():
    trials = synth_memories(identical=True)
    result = analyze_trials({"trials": trials, "excluded": []}, n_boot=50)
    # hand-compute the same paired test
    from collections import defaultdict
    cell = defaultdict(dict)
    for t in trials:
        s1, s2 = set(t["snap1_data"]), set(t["snap2_data"])
        d = 1 - len(s1 & s2) / len(s1 | s2)
        cell[t["arm"]].setdefault(t["memory_id"], d)
    mems = sorted(cell["lived"])
    lived = [cell["lived"][m] for m in mems]
    fict = [cell["fictional"][m] for m in mems]
    w, p = stats.wilcoxon(lived, fict, alternative="greater")
    secondary = next(c for c in result["confirmatory"]
                     if "lived vs fictional" in c["label"])
    assert abs(secondary["p_raw"] - p) < 1e-12
    assert secondary["rank_biserial_r"] > 0  # lived built to dominate


def test_varying_repeats_still_confirm_at_memory_level():
    """Even with real (non-duplicate) repeats, the design is paired by
    memory: repeats aggregate to a cell mean before inference."""
    trials = synth_memories(identical=False)
    result = analyze_trials({"trials": trials, "excluded": []}, n_boot=50)
    assert result["duplication_rate"] < 1.0
    assert result["confirmatory_unit"] == "memory"
    assert all(c["n_pairs"] == 11 for c in result["confirmatory"])
