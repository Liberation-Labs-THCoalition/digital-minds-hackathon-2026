"""Variable Landing Analysis — Holm-Bonferroni corrected comparisons.

Loads results from variable_landing_results.json, computes workspace Jaccard
similarity between snap1 and snap2 per trial, and runs Mann-Whitney U tests
with Holm-Bonferroni correction across the planned comparisons.

Usage:
    python variable_landing_analysis.py --input results/variable_landing_results.json
    python variable_landing_analysis.py --input results/variable_landing_results.json --output analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def jaccard_tokens(snap1: list, snap2: list) -> float:
    """Jaccard similarity between two token lists."""
    s1, s2 = set(snap1), set(snap2)
    union = s1 | s2
    if not union:
        return float("nan")
    return len(s1 & s2) / len(union)


def delta_numeric(snap1: dict, snap2: dict) -> float:
    """Mean absolute delta across shared numeric fields (fallback metric)."""
    shared_keys = set(snap1.keys()) & set(snap2.keys())
    deltas = []
    for k in shared_keys:
        v1, v2 = snap1[k], snap2[k]
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            deltas.append(abs(v2 - v1))
    if not deltas:
        return float("nan")
    return float(np.mean(deltas))


def compute_metric(trial: dict) -> float:
    """Compute the primary metric for a single trial.

    Priority:
      1. If snap data are lists -> Jaccard similarity
      2. If snap data are dicts -> mean absolute delta on numeric fields
      3. Otherwise -> NaN (excluded from analysis)
    """
    snap1 = trial.get("snap1_data")
    snap2 = trial.get("snap2_data")

    if snap1 is None or snap2 is None:
        return float("nan")

    # Token lists -> Jaccard
    if isinstance(snap1, list) and isinstance(snap2, list):
        return jaccard_tokens(snap1, snap2)

    # Dicts -> delta on numeric fields
    if isinstance(snap1, dict) and isinstance(snap2, dict):
        return delta_numeric(snap1, snap2)

    return float("nan")


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for rank-biserial r."""
    rng = np.random.RandomState(seed)
    rs = []
    for _ in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        u, _ = stats.mannwhitneyu(bx, by, alternative="two-sided")
        rs.append(rank_biserial_r(u, len(bx), len(by)))
    rs = np.array(rs)
    lo = float(np.percentile(rs, 100 * alpha / 2))
    hi = float(np.percentile(rs, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction.

    Args:
        p_values: list of raw p-values
        alpha: family-wise error rate

    Returns:
        list of booleans — True if null rejected at corrected alpha
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m

    for rank_k, (orig_idx, p_k) in enumerate(indexed, start=1):
        threshold = alpha / (m - rank_k + 1)
        if p_k < threshold:
            rejected[orig_idx] = True
        else:
            # Stop rejecting — all remaining are non-significant
            break

    return rejected


def descriptive_stats(values: np.ndarray) -> dict:
    """Median, IQR, n for an array of values."""
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return {"n": 0, "median": None, "iqr_25": None, "iqr_75": None}
    return {
        "n": int(len(clean)),
        "median": float(np.median(clean)),
        "iqr_25": float(np.percentile(clean, 25)),
        "iqr_75": float(np.percentile(clean, 75)),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

COMPARISONS = [
    ("PRIMARY: fictional vs scrambled", "fictional", "scrambled"),
    ("SECONDARY: lived vs fictional", "lived", "fictional"),
    ("lived vs scrambled", "lived", "scrambled"),
    ("lived vs no_intervention", "lived", "no_intervention"),
    ("fictional vs no_intervention", "fictional", "no_intervention"),
    ("scrambled vs no_intervention", "scrambled", "no_intervention"),
]


def run_analysis(input_path: str, output_path: str | None = None):
    data = json.loads(Path(input_path).read_text())
    trials = data.get("trials", [])

    if not trials:
        print("ERROR: No trials found in input file.")
        sys.exit(1)

    # Compute metric per trial and group by arm
    arm_values: dict[str, list[float]] = defaultdict(list)
    n_nan = 0

    for trial in trials:
        metric = compute_metric(trial)
        if np.isnan(metric):
            n_nan += 1
            continue
        arm_values[trial["arm"]].append(metric)

    print("=" * 65)
    print("VARIABLE LANDING ANALYSIS — Holm-Bonferroni corrected")
    print("=" * 65)
    print(f"\nTrials loaded: {len(trials)}  (NaN/excluded from metric: {n_nan})")

    # Descriptive stats per arm
    print("\n--- Descriptive Statistics ---\n")
    arm_stats = {}
    for arm in ["lived", "fictional", "scrambled", "no_intervention"]:
        vals = np.array(arm_values.get(arm, []))
        s = descriptive_stats(vals)
        arm_stats[arm] = s
        if s["n"] > 0:
            print(f"  {arm:20s}  n={s['n']:3d}  "
                  f"median={s['median']:.4f}  "
                  f"IQR=[{s['iqr_25']:.4f}, {s['iqr_75']:.4f}]")
        else:
            print(f"  {arm:20s}  n=  0  (no valid data)")

    # Mann-Whitney U tests (one-tailed: group1 > group2)
    print("\n--- Mann-Whitney U Tests (one-tailed) ---\n")

    comparison_results = []
    raw_p_values = []

    for label, arm_a, arm_b in COMPARISONS:
        vals_a = np.array(arm_values.get(arm_a, []))
        vals_b = np.array(arm_values.get(arm_b, []))

        if len(vals_a) < 2 or len(vals_b) < 2:
            print(f"  {label}: SKIPPED (insufficient data: "
                  f"n_{arm_a}={len(vals_a)}, n_{arm_b}={len(vals_b)})")
            comparison_results.append({
                "label": label,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "skipped": True,
                "reason": "insufficient data",
            })
            raw_p_values.append(1.0)  # conservative placeholder
            continue

        u_stat, p_two = stats.mannwhitneyu(vals_a, vals_b, alternative="greater")
        r = rank_biserial_r(u_stat, len(vals_a), len(vals_b))
        ci_lo, ci_hi = bootstrap_ci(vals_a, vals_b)

        raw_p_values.append(float(p_two))
        comparison_results.append({
            "label": label,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "n_a": int(len(vals_a)),
            "n_b": int(len(vals_b)),
            "U": float(u_stat),
            "p_raw": float(p_two),
            "rank_biserial_r": round(r, 4),
            "r_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "skipped": False,
        })

    # Apply Holm-Bonferroni
    rejected = holm_bonferroni(raw_p_values, alpha=0.05)

    for i, comp in enumerate(comparison_results):
        comp["holm_rejected"] = rejected[i]

        if comp.get("skipped"):
            continue

        sig_marker = " ***" if rejected[i] else ""
        print(f"  {comp['label']}")
        print(f"    U={comp['U']:.1f}  p={comp['p_raw']:.6f}  "
              f"r={comp['rank_biserial_r']:.4f}  "
              f"CI95=[{comp['r_ci_95'][0]:.4f}, {comp['r_ci_95'][1]:.4f}]"
              f"{sig_marker}")

    # Holm-Bonferroni summary
    print("\n--- Holm-Bonferroni Correction (alpha=0.05) ---\n")

    # Sort by raw p for display
    sorted_indices = sorted(range(len(raw_p_values)), key=lambda i: raw_p_values[i])
    m = len(raw_p_values)
    for rank_k, idx in enumerate(sorted_indices, start=1):
        comp = comparison_results[idx]
        threshold = 0.05 / (m - rank_k + 1)
        status = "REJECT" if comp["holm_rejected"] else "RETAIN"
        p_val = raw_p_values[idx]
        print(f"  rank {rank_k}: p={p_val:.6f} vs threshold={threshold:.6f}  "
              f"-> {status}  ({comp['label']})")

    print("\n" + "=" * 65)

    # Build output
    output = {
        "metadata": {
            "input_file": input_path,
            "n_trials": len(trials),
            "n_nan_excluded": n_nan,
            "alpha": 0.05,
            "correction": "Holm-Bonferroni",
            "n_comparisons": len(COMPARISONS),
            "bootstrap_resamples": 1000,
        },
        "descriptive_stats": arm_stats,
        "comparisons": comparison_results,
    }

    # Save output
    if output_path is None:
        output_path = str(Path(input_path).parent / "variable_landing_analysis.json")

    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to: {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Variable Landing Analysis with Holm-Bonferroni correction"
    )
    parser.add_argument("--input", required=True,
                        help="Path to variable_landing_results.json")
    parser.add_argument("--output", default=None,
                        help="Path for analysis output JSON (default: alongside input)")
    args = parser.parse_args()

    run_analysis(args.input, args.output)
