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
