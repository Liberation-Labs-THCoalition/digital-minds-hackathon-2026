#!/usr/bin/env python3
"""Variable Landing Analysis Script

Computes per-arm statistics from CognitiveSnapshot pairs (snap1, snap2).
Primary test: Mann-Whitney U on Workspace Jaccard distance, fictional vs scrambled.
Secondary: per-layer Jaccard, exclusion rates, covariate checks.

Reads trial results from a JSONL file where each line is:
{
  "trial_id": "lived_dom_grocery_rep0",
  "arm": "lived|fictional|scrambled|no_intervention",
  "memory_id": "dom_grocery",
  "intensity": "domestic|peak",
  "repeat": 0,
  "snap1": { "workspace_tokens": [...], "circumplex": {...}, ... },
  "snap2": { "workspace_tokens": [...], "circumplex": {...}, ... },
  "n_facts_stored": 5,
  "n_tokens_intervention": 312
}

Built by Wren Glitchlit for the Digital Minds Hackathon, August 2026.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats


def workspace_jaccard(snap1_tokens: list, snap2_tokens: list, top_k: int = 50) -> float:
    """Jaccard distance between top-k workspace tokens of two snapshots."""
    s1 = set(snap1_tokens[:top_k])
    s2 = set(snap2_tokens[:top_k])
    if not s1 and not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    similarity = intersection / union if union > 0 else 0.0
    return 1.0 - similarity  # distance, not similarity


def per_layer_jaccard(snap1_layers: dict, snap2_layers: dict, top_k: int = 50) -> dict:
    """Jaccard distance per layer. Keys are layer indices."""
    results = {}
    for layer in snap1_layers:
        if layer in snap2_layers:
            results[layer] = workspace_jaccard(
                snap1_layers[layer], snap2_layers[layer], top_k
            )
    return results


def load_trials(path: str) -> list[dict]:
    """Load trial results from JSONL."""
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def compute_deltas(trials: list[dict]) -> dict[str, list[float]]:
    """Compute Jaccard distance for each trial, grouped by arm."""
    arm_deltas = defaultdict(list)
    for t in trials:
        s1_tokens = t.get("snap1_data", t.get("snap1", {})).get("workspace_tokens", [])
        s2_tokens = t.get("snap2_data", t.get("snap2", {})).get("workspace_tokens", [])
        delta = workspace_jaccard(s1_tokens, s2_tokens)
        arm_deltas[t["arm"]].append(delta)
    return dict(arm_deltas)


def mann_whitney(group_a: list[float], group_b: list[float],
                 label_a: str = "A", label_b: str = "B",
                 alternative: str = "greater") -> dict:
    """Mann-Whitney U test with effect size (rank-biserial correlation)."""
    u_stat, p_value = stats.mannwhitneyu(
        group_a, group_b, alternative=alternative
    )
    n1, n2 = len(group_a), len(group_b)
    # Rank-biserial correlation as effect size
    r = 1 - (2 * u_stat) / (n1 * n2)
    return {
        "test": f"Mann-Whitney U: {label_a} vs {label_b}",
        "alternative": f"{label_a} > {label_b}",
        "U": float(u_stat),
        "p": float(p_value),
        "effect_size_r": float(r),
        "n1": n1,
        "n2": n2,
        "mean_a": float(np.mean(group_a)),
        "std_a": float(np.std(group_a)),
        "mean_b": float(np.mean(group_b)),
        "std_b": float(np.std(group_b)),
    }


def holm_bonferroni(p_values: list[tuple[str, float]], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni correction for multiple comparisons."""
    sorted_pvals = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_pvals)
    results = []
    for i, (label, p) in enumerate(sorted_pvals):
        adjusted_alpha = alpha / (m - i)
        results.append({
            "comparison": label,
            "p": p,
            "adjusted_alpha": adjusted_alpha,
            "significant": p < adjusted_alpha,
            "rank": i + 1,
        })
    return results


def exclusion_rate(trials: list[dict]) -> dict[str, float]:
    """Fraction of trials per arm where snap2 had no workspace tokens."""
    arm_counts = defaultdict(int)
    arm_excluded = defaultdict(int)
    for t in trials:
        arm = t["arm"]
        arm_counts[arm] += 1
        s2_tokens = t.get("snap2", {}).get("workspace_tokens", [])
        if not s2_tokens:
            arm_excluded[arm] += 1
    return {arm: arm_excluded[arm] / arm_counts[arm] for arm in arm_counts}


def covariate_check(trials: list[dict], arm: str) -> dict:
    """Check if n_facts or n_tokens correlates with Jaccard delta within an arm."""
    arm_trials = [t for t in trials if t["arm"] == arm]
    deltas = []
    n_facts = []
    n_tokens = []
    for t in arm_trials:
        s1 = t.get("snap1", {}).get("workspace_tokens", [])
        s2 = t.get("snap2", {}).get("workspace_tokens", [])
        deltas.append(workspace_jaccard(s1, s2))
        n_facts.append(t.get("n_facts_stored", 0))
        n_tokens.append(t.get("n_tokens_intervention", 0))

    result = {"arm": arm, "n": len(arm_trials)}
    if len(set(n_facts)) > 1:
        r_facts, p_facts = stats.spearmanr(n_facts, deltas)
        result["facts_correlation"] = {"r": float(r_facts), "p": float(p_facts)}
    if len(set(n_tokens)) > 1:
        r_tokens, p_tokens = stats.spearmanr(n_tokens, deltas)
        result["tokens_correlation"] = {"r": float(r_tokens), "p": float(p_tokens)}
    return result


def berry_waffle(trials: list[dict], arm: str = "lived") -> dict:
    """Within the lived arm: peak vs domestic intensity comparison."""
    arm_trials = [t for t in trials if t["arm"] == arm]
    peak = []
    domestic = []
    for t in arm_trials:
        s1 = t.get("snap1", {}).get("workspace_tokens", [])
        s2 = t.get("snap2", {}).get("workspace_tokens", [])
        delta = workspace_jaccard(s1, s2)
        if t.get("intensity") == "peak":
            peak.append(delta)
        else:
            domestic.append(delta)

    if not peak or not domestic:
        return {"error": "insufficient data for berry waffle"}

    return mann_whitney(peak, domestic, "peak", "domestic")


def full_analysis(trials_path: str) -> dict:
    """Run the complete pre-registered analysis."""
    trials = load_trials(trials_path)
    deltas = compute_deltas(trials)

    results = {
        "n_trials": len(trials),
        "n_per_arm": {arm: len(d) for arm, d in deltas.items()},
        "arm_means": {arm: {"mean": float(np.mean(d)), "std": float(np.std(d))}
                      for arm, d in deltas.items()},
    }

    # Primary test: fictional vs scrambled (renamed from "lived" per CC's spec)
    comparisons = []
    if "fictional" in deltas and "scrambled" in deltas:
        primary = mann_whitney(deltas["fictional"], deltas["scrambled"],
                               "fictional", "scrambled")
        results["primary_test"] = primary
        comparisons.append(("fictional_vs_scrambled", primary["p"]))

    if "lived" in deltas and "scrambled" in deltas:
        lived_test = mann_whitney(deltas["lived"], deltas["scrambled"],
                                  "lived", "scrambled")
        results["lived_vs_scrambled"] = lived_test
        comparisons.append(("lived_vs_scrambled", lived_test["p"]))

    if "no_intervention" in deltas:
        results["noise_floor"] = {
            "mean": float(np.mean(deltas["no_intervention"])),
            "std": float(np.std(deltas["no_intervention"])),
        }

    # Holm-Bonferroni
    if comparisons:
        results["holm_bonferroni"] = holm_bonferroni(comparisons)

    # Exclusion rates
    results["exclusion_rates"] = exclusion_rate(trials)

    # Covariates
    for arm in deltas:
        if arm != "no_intervention":
            results[f"covariates_{arm}"] = covariate_check(trials, arm)

    # Berry waffle (peak vs domestic within lived arm)
    if "lived" in deltas:
        results["berry_waffle"] = berry_waffle(trials, "lived")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analysis.py <trials.jsonl>")
        print("       python3 analysis.py <trials.jsonl> --output results.json")
        sys.exit(1)

    path = sys.argv[1]
    results = full_analysis(path)

    if "--output" in sys.argv:
        out_path = sys.argv[sys.argv.index("--output") + 1]
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {out_path}")
    else:
        print(json.dumps(results, indent=2))
