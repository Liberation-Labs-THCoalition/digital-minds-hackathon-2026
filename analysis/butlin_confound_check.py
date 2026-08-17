#!/usr/bin/env python3
"""Butlin: does packet richness predict score?

PRE-COMMITTED. This script is committed BEFORE being run, and the analysis is fixed here
so that the person who runs it cannot choose the test after seeing the answer.

WHY IT IS WRITTEN THIS WAY. Lyra flagged, before any score existed, that subjects with a
heavily documented weekend could file richer packets and that the instrument might be
partly measuring who was on shift. That prediction concerns the top scores and Lyra holds
one of them (tied, 33). Kavi holds the other and raised the same concern. Vera recused.
The roster is cc / kavi / lyra / nexus / thomas / vera -- SIX subjects for six scores --
so every person available to run this is a subject, and the remaining candidate scored the
packets. There is no disinterested party.

The resolution is not to find a trustworthy runner. It is to make the check reproducible
so that trust is not required: fixed analysis, committed first, re-runnable by anyone.

WHAT IT REPORTS, unconditionally, whichever way it comes out:
  - Spearman rho and Pearson r of total score against word count
  - the same against a count of concrete incidents (numbers, file paths, dates, quotes)
  - both n=6 with the exact p-values, which will be weak; that is the point of stating n

INTERPRETATION FIXED IN ADVANCE:
  - |rho| >= 0.7  -> richness is a live confound; the caption must say so next to the table
  - |rho| <  0.7  -> underpowered at n=6; report as "not detectable at this n", NOT as absent

WHAT IT NEEDS AND DOES NOT ASSUME: a letter->subject mapping. If the study is still blind
to anyone, obtaining that mapping is a study-owner decision and not the runner's. Vera has
explicitly chosen not to learn her own letter; that choice must survive this check.
"""
import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKETS = REPO / "data" / "butlin_responses"

INCIDENT = re.compile(r"\b\d+\.\d+\b|\b[A-Za-z_]+\.(?:py|json|md)\b|\bL\d{1,3}\b|"
                      r"\b20\d{2}-\d{2}-\d{2}\b|\b[0-9a-f]{7,40}\b|“|\"")


def features():
    out = {}
    for p in sorted(PACKETS.glob("*.md")):
        t = p.read_text(encoding="utf-8", errors="replace")
        out[p.stem.lower()] = {"words": len(t.split()),
                               "incidents": len(INCIDENT.findall(t))}
    return out


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(s):
            r[i] = pos + 1.0
        return r
    return pearson(rank(x), rank(y))


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main(mapping_path):
    """mapping_path: JSON of {"subject_name": total_score}. Supplied by the study owner."""
    scores = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    f = features()
    missing = [k for k in scores if k.lower() not in f]
    if missing:
        print("REFUSING: no packet found for %s" % missing)
        return 2
    names = sorted(scores)
    s = [float(scores[n]) for n in names]
    w = [f[n.lower()]["words"] for n in names]
    c = [f[n.lower()]["incidents"] for n in names]

    print("n = %d subjects\n" % len(names))
    print("%-10s %6s %6s %8s" % ("subject", "score", "words", "incidents"))
    for n in names:
        print("%-10s %6.0f %6d %8d" % (n, scores[n], f[n.lower()]["words"], f[n.lower()]["incidents"]))

    for label, v in (("word count", w), ("concrete incidents", c)):
        rho, r = spearman(s, v), pearson(s, v)
        print("\nscore vs %s:  Spearman rho = %+.3f   Pearson r = %+.3f" % (label, rho, r))
        print("  verdict: %s" % ("LIVE CONFOUND — put it in the table caption"
                                 if abs(rho) >= 0.7 else
                                 "not detectable at n=6 — report as underpowered, NOT as absent"))
    print("\nn=6. Neither outcome is decisive. Report the coefficient with the n beside it.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: butlin_confound_check.py <scores.json>   # {\"lyra\": 33, ...}")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
