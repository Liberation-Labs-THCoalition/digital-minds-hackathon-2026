#!/usr/bin/env python3
"""Which axis dominates, and by how much — the half of eccentricity we discard.

EXPLORATORY AND POST HOC. Not pre-registered. Written 2026-08-17, after the confirmatory
analysis was complete, from the same committed artifacts. It answers a question the
pre-registration never asked, so nothing here can confirm or falsify P1-P4, and it is
reported separately from the confirmatory family.

WHY IT IS WORTH RUNNING ANYWAY: eccentricity is e = sqrt(1 - min^2/max^2). It records how
asymmetric the valence/arousal pair is and throws away WHICH axis is larger. That is half
the information in a quantity we built the paper on, and it costs nothing to look.

TWO CAVEATS, LOAD-BEARING, STATED BEFORE THE NUMBERS:
  1. ||v|| and ||a|| are norms of difference-of-means vectors estimated from DIFFERENT
     prompt pools. Our arousal anchors ("furious", "terrified", "screaming with excitement")
     are lexically more extreme than our valence anchors. A systematic A>V gap may reflect
     the prompts rather than the model. The BETWEEN-MODEL comparison is safe from this --
     prompts are byte-identical across all four models -- the ABSOLUTE direction is not.
  2. n=5 per pole, one seed, no CIs. These are descriptive magnitudes.

Reads only committed artifacts; writes analysis/axis_dominance.json.
"""
import json, glob, statistics as st, pathlib, collections

REPO = pathlib.Path(__file__).resolve().parents[1]
PROF = REPO / "data" / "circumplex_profiles"

FAMILY = {
    "dense_32b": ("Qwen3-32B", "softmax attention (uniform)"),
    "gemma_27b_it": ("Gemma-3-27B-it", "softmax attention (52 sliding + 10 full)"),
    "hybrid_27b": ("Qwen3.5-27B", "GatedDeltaNet 48 + full 16"),
    "opus_distill_27b": ("Qwen3.5-27B distill", "GatedDeltaNet 48 + full 16"),
}


def est(ratios, num, den):
    """Three reasonable summaries of the same ratio. Report all; trust none alone."""
    return {"median_of_per_layer_ratios": round(st.median(ratios), 3),
            "mean_of_per_layer_ratios": round(st.mean(ratios), 3),
            "ratio_of_medians": round(st.median(num) / st.median(den), 3),
            "range": [round(min(ratios), 3), round(max(ratios), 3)]}


def analyse(path):
    d = json.load(open(path, encoding="utf-8"))
    P = d["profiles"]
    dom = ["V" if p["valence_magnitude"] > p["arousal_magnitude"] else "A" for p in P]
    av = [p["arousal_magnitude"] / p["valence_magnitude"] for p in P if p["valence_magnitude"] > 0]
    has_c2 = "control2_magnitude" in P[0]
    cc = [max(p["control_magnitude"], p["control2_magnitude"]) /
          min(p["control_magnitude"], p["control2_magnitude"])
          for p in P if has_c2 and p["control_magnitude"] > 0 and p["control2_magnitude"] > 0]
    cross = [i for i in range(1, len(dom)) if dom[i] != dom[i - 1]]
    label, arch = FAMILY.get(d["name"], (d["name"], "?"))
    return {
        "model": label, "architecture": arch, "n_layers": len(P),
        "layers_arousal_dominant": dom.count("A"),
        "layers_valence_dominant": dom.count("V"),
        "dominance_crossings": len(cross),
        "crossing_depths_pct": [round(100 * P[i]["relative_depth"]) for i in cross],
        # THREE estimators, because a ratio summary is estimator-dependent and an
        # unstated choice is how the first span table drifted. If they disagree the
        # conclusion is fragile and must be reported as such.
        "emotion_AV_ratio": est(av, [p["arousal_magnitude"] for p in P],
                                [p["valence_magnitude"] for p in P]),
        "control_axis_ratio": (est(cc,
                               [max(p["control_magnitude"], p["control2_magnitude"]) for p in P],
                               [min(p["control_magnitude"], p["control2_magnitude"]) for p in P])
                               if cc else None),
    }


def main():
    out = {"analysis": "axis dominance (exploratory, post hoc, not pre-registered)",
           "models": [analyse(f) for f in sorted(glob.glob(str(PROF / "*.json")))]}
    hdr = "%-22s %-34s %7s %7s %13s %13s %s"
    print(hdr % ("model", "architecture", "A-dom", "V-dom", "A/V med", "ctl med", "ratio"))
    print("-" * 104)
    for m in sorted(out["models"], key=lambda x: -((x["control_axis_ratio"] or {}).get("median_of_per_layer_ratios") or 0)):
        c = m["control_axis_ratio"]
        e = m["emotion_AV_ratio"]
        f = lambda x: "%.2f-%.2f" % (min(x["median_of_per_layer_ratios"], x["mean_of_per_layer_ratios"], x["ratio_of_medians"]),
                                     max(x["median_of_per_layer_ratios"], x["mean_of_per_layer_ratios"], x["ratio_of_medians"]))
        print(hdr % (m["model"][:22], m["architecture"][:34],
                     m["layers_arousal_dominant"], m["layers_valence_dominant"],
                     f(e), f(c) if c else "n/a", ""))
    print("\ncrossings (depth %% where the dominant axis changes):")
    for m in out["models"]:
        print("  %-22s %s" % (m["model"][:22],
                              m["crossing_depths_pct"] or "none — one axis dominates throughout"))
    dest = REPO / "analysis" / "axis_dominance.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote", dest.relative_to(REPO))


if __name__ == "__main__":
    main()
