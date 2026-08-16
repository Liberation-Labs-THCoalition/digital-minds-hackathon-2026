#!/usr/bin/env python3
"""Compute the circumplex numbers from the four profile artifacts.

Everything reported in the paper must come from here. No hand-copied figures.
"""
import json, glob, os, statistics as st

FILES = sorted(glob.glob("data/circumplex_profiles/*.json"))

ARCH = {
    "gemma_27b_it": ("Gemma-3-27B-it", "dense"),
    "dense_32b": ("Qwen3-32B", "dense"),
    "hybrid_27b": ("Qwen3.5-27B", "hybrid"),
    "opus_distill_27b": ("Qwen3.5-27B Opus-distill", "hybrid"),
}

rows = []
for f in FILES:
    d = json.load(open(f))
    name = d["name"]
    profs = d["profiles"]
    ecc = [p["eccentricity"] for p in profs]
    ctl = [p.get("control_eccentricity") for p in profs]
    if any(c is None for c in ctl):
        print("!! %s has missing control_eccentricity" % name)
        continue

    ecc_span = max(ecc) - min(ecc)
    ctl_span = max(ctl) - min(ctl)
    ratio = ecc_span / ctl_span if ctl_span else float("nan")

    imin = min(range(len(ecc)), key=lambda i: ecc[i])
    imax = max(range(len(ecc)), key=lambda i: ecc[i])

    label, arch = ARCH.get(name, (name, "?"))
    rows.append(dict(
        key=name, label=label, arch=arch,
        n_layers=d["n_layers"], d_model=d["d_model"],
        ecc_span=ecc_span, ctl_span=ctl_span, ratio=ratio,
        min_layer=profs[imin]["layer"], min_depth=profs[imin]["relative_depth"], min_val=ecc[imin],
        max_layer=profs[imax]["layer"], max_val=ecc[imax],
        ecc=ecc, ctl=ctl, profs=profs,
        layer_types=d["layer_types"],
        has_anchor_count=any(k in d for k in ("n_anchors", "anchors_per_pole", "n_per_category")),
        keys=list(d.keys()),
    ))

print("=" * 100)
print("%-28s %-7s %5s  %-9s %-9s %-7s  %-18s" % (
    "model", "arch", "L", "ecc span", "ctl span", "ratio", "ecc minimum"))
print("=" * 100)
for r in sorted(rows, key=lambda r: (r["arch"], r["label"])):
    print("%-28s %-7s %5d  %-9.4f %-9.4f %-7.2fx  L%-3d (%.0f%% depth) ecc=%.3f" % (
        r["label"], r["arch"], r["n_layers"], r["ecc_span"], r["ctl_span"], r["ratio"],
        r["min_layer"], 100 * r["min_depth"], r["min_val"]))

# --- base vs distill agreement (controlled comparison: same arch, different training) ---
b = next((r for r in rows if r["key"] == "hybrid_27b"), None)
o = next((r for r in rows if r["key"] == "opus_distill_27b"), None)
if b and o:
    print("\n" + "=" * 100)
    print("BASE vs DISTILL (same architecture, different training)")
    print("  ratio:        %.4f vs %.4f   (diff %.5f)" % (b["ratio"], o["ratio"], abs(b["ratio"] - o["ratio"])))
    print("  ecc span:     %.5f vs %.5f   (diff %.6f)" % (b["ecc_span"], o["ecc_span"], abs(b["ecc_span"] - o["ecc_span"])))
    print("  ecc min layer: L%d vs L%d" % (b["min_layer"], o["min_layer"]))
    if len(b["ecc"]) == len(o["ecc"]):
        dev = [abs(x - y) for x, y in zip(b["ecc"], o["ecc"])]
        print("  per-layer |ecc| deviation: max %.5f, mean %.5f" % (max(dev), sum(dev) / len(dev)))

# --- pre-registered substrate confound test: period-4 structure in the hybrid ---
print("\n" + "=" * 100)
print("PRE-REGISTERED SUBSTRATE TEST (lag-4 autocorrelation of eccentricity)")
def acf(x, lag):
    n = len(x); m = sum(x) / n
    num = sum((x[i] - m) * (x[i + lag] - m) for i in range(n - lag))
    den = sum((v - m) ** 2 for v in x)
    return num / den if den else float("nan")

for r in sorted(rows, key=lambda r: (r["arch"], r["label"])):
    a = [acf(r["ecc"], L) for L in (1, 2, 3, 4, 5)]
    print("  %-28s lag1=%+.3f lag2=%+.3f lag3=%+.3f lag4=%+.3f lag5=%+.3f" % (r["label"], *a))

# does eccentricity differ by layer type in the hybrid?
for r in rows:
    lt = r["layer_types"]
    kinds = sorted(set(lt))
    if len(kinds) > 1:
        print("\n  %s layer types %s:" % (r["label"], kinds))
        for k in kinds:
            vals = [e for e, t in zip(r["ecc"], lt) if t == k]
            print("     %-18s n=%-3d mean ecc=%.4f  sd=%.4f" % (
                k, len(vals), st.mean(vals), st.pstdev(vals)))

# --- provenance: what the artifacts do and do not record ---
print("\n" + "=" * 100)
print("ARTIFACT PROVENANCE (#86)")
for r in rows:
    print("  %-28s keys=%s" % (r["label"], r["keys"]))
    print("  %-28s records anchor count? %s" % ("", r["has_anchor_count"]))
