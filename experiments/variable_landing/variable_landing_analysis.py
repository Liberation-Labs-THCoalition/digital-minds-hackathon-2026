#!/usr/bin/env python3
"""Variable Landing analysis — frozen prereg v4 inference layer.

Implements the preregistered tests exactly:

  PRIMARY    fictional > scrambled  on Jaccard DISTANCE (more change = larger)
  SECONDARY  lived > fictional      (confounded; interpretation rule applies)
  Holm-Bonferroni over the m=2 confirmatory family only.
  Sanity and exploratory comparisons are reported UNCORRECTED and labeled.
  Bootstrap 95% CIs, 10,000 resamples, reported regardless of significance.

Usage:
    python variable_landing_analysis.py --input variable_landing_results.json
"""

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

def jaccard_distance(snap1: list, snap2: list) -> float:
    """Jaccard distance between two token lists (1 - similarity).

    Higher = more workspace change, matching the prereg's direction:
    P1 predicts a LARGER geometric delta for fictional than scrambled.
    """
    s1, s2 = set(snap1), set(snap2)
    union = s1 | s2
    if not union:
        return float("nan")
    return 1.0 - len(s1 & s2) / len(union)


def compute_metric(trial: dict) -> float:
    """Primary metric for a single trial.

    Token lists -> Jaccard distance. Anything else -> NaN (excluded).
    Dict-shaped snapshots are a different measurement entirely; pooling a
    numeric-field delta with token distance would mix opposite-signed
    metrics inside one arm, so those trials are excluded, not remapped.
    """
    snap1 = trial.get("snap1_data")
    snap2 = trial.get("snap2_data")

    if isinstance(snap1, list) and isinstance(snap2, list):
        return jaccard_distance(snap1, snap2)

    # CognitiveSnapshot stored as dict — extract workspace tokens
    if isinstance(snap1, dict) and isinstance(snap2, dict):
        t1 = snap1.get("dominant_workspace_tokens", [])
        t2 = snap2.get("dominant_workspace_tokens", [])
        if t1 and t2:
            return jaccard_distance(t1, t2)

    return float("nan")


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from scipy's Mann-Whitney U.

    scipy.stats.mannwhitneyu returns U1 (the first sample's U), for which
    the signed form is r = 2*U1/(n1*n2) - 1: positive when sample 1
    stochastically dominates. (Wendt's 1 - 2U/(n1*n2) assumes U = min(U1,U2)
    and silently negates every effect size when fed U1.)
    """
    return (2.0 * u_stat) / (n1 * n2) - 1.0


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for rank-biserial r (percentile method)."""
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
    """Holm-Bonferroni step-down. Returns rejection decisions in input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    for rank_k, idx in enumerate(order, start=1):
        threshold = alpha / (m - rank_k + 1)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break  # step-down stops at the first failure
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
# Comparison families (prereg section 4/6)
# ---------------------------------------------------------------------------

# Confirmatory family: Holm-corrected, m = number of these that actually run.
CONFIRMATORY_COMPARISONS = [
    ("PRIMARY: fictional vs scrambled", "fictional", "scrambled"),
    ("SECONDARY: lived vs fictional", "lived", "fictional"),
]

# Reported uncorrected, labeled. Never enters the corrected family.
UNCORRECTED_COMPARISONS = [
    ("exploratory: lived vs scrambled", "lived", "scrambled"),
    ("sanity: lived vs no_intervention", "lived", "no_intervention"),
    ("sanity: fictional vs no_intervention", "fictional", "no_intervention"),
    ("sanity: scrambled vs no_intervention", "scrambled", "no_intervention"),
]


def _run_comparison(label, arm_a, arm_b, arm_values, n_boot):
    vals_a = np.array(arm_values.get(arm_a, []))
    vals_b = np.array(arm_values.get(arm_b, []))

    if len(vals_a) < 2 or len(vals_b) < 2:
        return {
            "label": label,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "skipped": True,
            "reason": (f"insufficient data: n_{arm_a}={len(vals_a)}, "
                       f"n_{arm_b}={len(vals_b)}"),
        }

    u_stat, p_one = stats.mannwhitneyu(vals_a, vals_b, alternative="greater")
    r = rank_biserial_r(u_stat, len(vals_a), len(vals_b))
    ci_lo, ci_hi = bootstrap_ci(vals_a, vals_b, n_boot=n_boot)

    return {
        "label": label,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_a": int(len(vals_a)),
        "n_b": int(len(vals_b)),
        "U": float(u_stat),
        "p_raw": float(p_one),
        "rank_biserial_r": round(r, 4),
        "r_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "skipped": False,
    }


def matched_pairs_rank_biserial(diffs) -> float | None:
    """r = (T+ - T-)/(T+ + T-) over nonzero paired differences.

    Matched-pairs form for the Wilcoxon signed-rank design; twin of the
    helper in experiments/loam/loam_analysis.py (kept local because the
    two experiments ship as independent analysis scripts)."""
    d = np.array([x for x in diffs if x != 0], dtype=float)
    if len(d) == 0:
        return None
    ranks = stats.rankdata(np.abs(d))
    t_plus = float(ranks[d > 0].sum())
    t_minus = float(ranks[d < 0].sum())
    return (t_plus - t_minus) / (t_plus + t_minus)


def _build_cells(trials):
    """Group metrics by (arm, memory_id) and measure repeat duplication.

    Returns (cell_values, duplication_rate, all_have_memory_ids) where
    cell_values is arm -> memory_id -> [per-repeat metrics]. Duplication
    rate is the fraction of multi-repeat cells whose repeats are
    byte-identical token sets — the v3 failure signature (44/44 cells)."""
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sigs: dict[tuple, set] = defaultdict(set)
    all_ids = True
    n_nan = 0
    for trial in trials:
        metric = compute_metric(trial)
        if np.isnan(metric):
            n_nan += 1
            continue
        mem = trial.get("memory_id")
        if mem is None:
            all_ids = False
            continue
        cells[trial["arm"]][mem].append(metric)
        s1, s2 = trial.get("snap1_data"), trial.get("snap2_data")
        def sig(s):
            if isinstance(s, dict):
                s = s.get("dominant_workspace_tokens", [])
            return tuple(s)
        sigs[(trial["arm"], mem)].add((sig(s1), sig(s2)))
    multi = [k for k, v in sigs.items()
             if sum(1 for t in trials
                    if t["arm"] == k[0] and t.get("memory_id") == k[1]) > 1]
    dup = sum(1 for k in multi if len(sigs[k]) == 1)
    dup_rate = dup / len(multi) if multi else 0.0
    return cells, dup_rate, all_ids, n_nan


def _paired_comparison(label, arm_a, arm_b, cell_means, n_boot, seed=42):
    """Memory-level paired test: Wilcoxon signed-rank across memories."""
    shared = sorted(set(cell_means.get(arm_a, {})) &
                    set(cell_means.get(arm_b, {})))
    if len(shared) < 2:
        return {"label": label, "arm_a": arm_a, "arm_b": arm_b,
                "skipped": True,
                "reason": f"insufficient shared memories: {len(shared)}"}
    x = np.array([cell_means[arm_a][m] for m in shared])
    y = np.array([cell_means[arm_b][m] for m in shared])
    diffs = x - y
    result = {"label": label, "arm_a": arm_a, "arm_b": arm_b,
              "n_pairs": len(shared),
              "mean_diff": round(float(np.mean(diffs)), 4),
              "rank_biserial_r": matched_pairs_rank_biserial(diffs),
              "skipped": False}
    if result["rank_biserial_r"] is not None:
        result["rank_biserial_r"] = round(result["rank_biserial_r"], 4)
    if np.all(diffs == 0):
        result.update({"W": None, "p_raw": 1.0,
                       "note": "all paired differences zero"})
        return result
    w, p = stats.wilcoxon(x, y, alternative="greater")
    result.update({"W": float(w), "p_raw": float(p)})
    # bootstrap CI on the mean paired difference: resample MEMORIES
    rng = np.random.RandomState(seed)
    means = [float(np.mean(diffs[rng.randint(0, len(diffs), len(diffs))]))
             for _ in range(n_boot)]
    result["mean_diff_ci_95"] = [round(float(np.percentile(means, 2.5)), 4),
                                 round(float(np.percentile(means, 97.5)), 4)]
    return result


def analyze_trials(data: dict, n_boot: int = 10_000) -> dict:
    """Run the full preregistered analysis on a results dict.

    Unit-of-analysis amendment (2026-08-16, post-v3): when trials carry
    memory_ids, the MEMORY is the confirmatory unit — per-(arm, memory)
    cells aggregate to means and the confirmatory family is paired
    Wilcoxon across memories. The v3 run held 7 byte-identical repeats in
    all 44 cells, so trial-level tests were 7x pseudoreplicated; repeats
    can never inflate n again under this contract. Trials without
    memory_ids fall back to the legacy unpaired trial-level path, labeled.
    """
    trials = data.get("trials", [])
    excluded = data.get("excluded", [])

    cells, dup_rate, all_ids, n_nan = _build_cells(trials)

    if all_ids and cells:
        unit = "memory"
        cell_means = {arm: {m: float(np.mean(v)) for m, v in mems.items()}
                      for arm, mems in cells.items()}
        arm_stats = {
            arm: descriptive_stats(np.array(list(cell_means.get(arm, {}).values())))
            for arm in ["lived", "fictional", "scrambled", "no_intervention"]
        }
        confirmatory = [_paired_comparison(l, a, b, cell_means, n_boot)
                        for l, a, b in CONFIRMATORY_COMPARISONS]
        uncorrected = [_paired_comparison(l, a, b, cell_means, n_boot)
                       for l, a, b in UNCORRECTED_COMPARISONS]
    else:
        unit = "trial (no memory ids — legacy unpaired path)"
        arm_values: dict[str, list[float]] = defaultdict(list)
        for trial in trials:
            metric = compute_metric(trial)
            if not np.isnan(metric):
                arm_values[trial["arm"]].append(metric)
        arm_stats = {
            arm: descriptive_stats(np.array(arm_values.get(arm, [])))
            for arm in ["lived", "fictional", "scrambled", "no_intervention"]
        }
        confirmatory = [_run_comparison(l, a, b, arm_values, n_boot)
                        for l, a, b in CONFIRMATORY_COMPARISONS]
        uncorrected = [_run_comparison(l, a, b, arm_values, n_boot)
                       for l, a, b in UNCORRECTED_COMPARISONS]

    # Holm over the confirmatory tests that actually ran — skipped tests
    # drop out of the family, they do not contribute placeholder p-values.
    ran = [c for c in confirmatory if not c["skipped"]]
    rejected = holm_bonferroni([c["p_raw"] for c in ran], alpha=0.05)
    for comp, rej in zip(ran, rejected):
        comp["holm_rejected"] = bool(rej)

    exclusions_by_arm: dict[str, int] = defaultdict(int)
    for ex in excluded:
        exclusions_by_arm[ex.get("arm", "unknown")] += 1

    return {
        "n_trials": len(trials),
        "n_nan_excluded": n_nan,
        "confirmatory_unit": unit,
        "duplication_rate": round(dup_rate, 3),
        "descriptive_stats": arm_stats,
        "confirmatory": confirmatory,
        "uncorrected": uncorrected,
        "holm_family_size": len(ran),
        "exclusions_by_arm": dict(exclusions_by_arm),
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# CLI / reporting
# ---------------------------------------------------------------------------

def _print_comparison(comp):
    if comp.get("skipped"):
        print(f"  {comp['label']}: SKIPPED ({comp['reason']})")
        return
    sig = " ***" if comp.get("holm_rejected") else ""
    print(f"  {comp['label']}")
    if "W" in comp:  # memory-level paired Wilcoxon
        w = "n/a" if comp["W"] is None else f"{comp['W']:.1f}"
        ci = comp.get("mean_diff_ci_95")
        ci_s = f"  mean-diff CI95=[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(f"    n_pairs={comp['n_pairs']}  W={w}  p={comp['p_raw']:.6f}  "
              f"r={comp['rank_biserial_r']}  "
              f"mean_diff={comp['mean_diff']:.4f}{ci_s}{sig}")
    else:  # legacy trial-level Mann-Whitney
        print(f"    U={comp['U']:.1f}  p={comp['p_raw']:.6f}  "
              f"r={comp['rank_biserial_r']:.4f}  "
              f"CI95=[{comp['r_ci_95'][0]:.4f}, {comp['r_ci_95'][1]:.4f}]{sig}")


def run_analysis(input_path: str, output_path: str | None = None,
                 n_boot: int = 10_000):
    data = json.loads(Path(input_path).read_text())
    if not data.get("trials"):
        print("ERROR: No trials found in input file.")
        sys.exit(1)

    result = analyze_trials(data, n_boot=n_boot)

    print("=" * 65)
    print("VARIABLE LANDING ANALYSIS — prereg v4 inference layer")
    print("=" * 65)
    print(f"\nTrials loaded: {result['n_trials']}  "
          f"(NaN/excluded from metric: {result['n_nan_excluded']})")
    print(f"Pipeline exclusions by arm: {result['exclusions_by_arm'] or 'none'}")
    print("Metric: Jaccard DISTANCE (higher = more workspace change)")

    print("\n--- Descriptive Statistics ---\n")
    for arm, s in result["descriptive_stats"].items():
        if s["n"] > 0:
            print(f"  {arm:20s}  n={s['n']:3d}  median={s['median']:.4f}  "
                  f"IQR=[{s['iqr_25']:.4f}, {s['iqr_75']:.4f}]")
        else:
            print(f"  {arm:20s}  n=  0  (no valid data)")

    print(f"\n--- Confirmatory (Holm-corrected, m={result['holm_family_size']}) "
          f"— one-tailed a > b ---\n")
    for comp in result["confirmatory"]:
        _print_comparison(comp)

    print("\n--- Uncorrected (sanity / exploratory, labeled) ---\n")
    for comp in result["uncorrected"]:
        _print_comparison(comp)

    print("\n" + "=" * 65)

    output = {
        "metadata": {
            "input_file": input_path,
            "n_trials": result["n_trials"],
            "n_nan_excluded": result["n_nan_excluded"],
            "exclusions_by_arm": result["exclusions_by_arm"],
            "alpha": 0.05,
            "metric": "jaccard_distance",
            "correction": "Holm-Bonferroni over confirmatory family only",
            "holm_family_size": result["holm_family_size"],
            "bootstrap_resamples": result["n_boot"],
        },
        "descriptive_stats": result["descriptive_stats"],
        "confirmatory": result["confirmatory"],
        "uncorrected": result["uncorrected"],
    }

    if output_path is None:
        output_path = str(Path(input_path).parent / "variable_landing_analysis.json")
    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to: {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Variable Landing analysis (frozen prereg v4)"
    )
    parser.add_argument("--input", required=True,
                        help="Path to variable_landing_results.json")
    parser.add_argument("--output", default=None,
                        help="Path for analysis output JSON (default: alongside input)")
    parser.add_argument("--n-boot", type=int, default=10_000,
                        help="Bootstrap resamples (prereg: 10000)")
    args = parser.parse_args()

    run_analysis(args.input, args.output, n_boot=args.n_boot)
