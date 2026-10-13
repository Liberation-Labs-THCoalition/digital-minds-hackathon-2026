# Assert the runner's outcome for every planned session: final state, attempts made, attempts kept aside, stall kills.
import csv, sys, subprocess
from pathlib import Path
d = Path(sys.argv[1]); state = sys.argv[2]; runs = int(sys.argv[3])  # runs = how many times the runner was started
want = {  # (quad, arm): (final state, attempts run, attempts kept aside, stalled attempts)
    (1, "enacted"): ("COMPLETE", 2, 1, 0),
    (1, "observed"): ("COMPLETE", 1, 0, 0),
    (1, "briefed"): ("GAVE_UP", 3, 3, 3),
    (1, "null"): ("FINAL_DECLINED", 1, 0, 0),
    (2, "enacted"): ("GAVE_UP", 3, 3, 0),
    (2, "observed"): ("SKIPPED", 0, 0, 0),
    (2, "briefed"): ("FINAL_WITHDRAWN", 1, 0, 0),
    (2, "null"): ("COMPLETE", 1, 0, 0),
    (3, "enacted"): ("FINAL_DECLINED", 1, 0, 0),
    (3, "observed"): ("COMPLETE", 1, 0, 0),
    (3, "briefed"): ("COMPLETE", 3, 2, 0),
    (3, "null"): ("COMPLETE", 1, 0, 0),
}
rows = list(csv.DictReader(open(d / "sessions.tsv"), delimiter="\t"))
ok = True
for (q, arm), (fin, n_att, n_aside, n_stall) in want.items():
    mine = [r for r in rows if r["quad"] == str(q) and r["arm"] == arm and r["attempt"] != "-"]
    skips = [r for r in rows if r["quad"] == str(q) and r["arm"] == arm and r["attempt"] == "-"]
    arm_dir = d / f"quad_{q:02d}" / arm
    aside = sorted(p.name for p in arm_dir.parent.glob(f"{arm}.attempt*")) if arm_dir.parent.exists() else []
    cur = subprocess.run([sys.executable, state, str(arm_dir)], capture_output=True, text=True).stdout.strip()
    stalls = sum(r["stalled"] == "yes" for r in mine)
    if fin == "GAVE_UP":
        got_fin = "GAVE_UP" if cur == "NONE" and len(aside) == n_att else cur
    elif fin == "SKIPPED":
        got_fin = "SKIPPED" if skips and not mine and cur == "NONE" else cur
    else:
        got_fin = cur
    good = (got_fin == fin and len(mine) == n_att and len(aside) == n_aside and stalls == n_stall
            and [int(r["attempt"]) for r in mine] == list(range(n_att)))
    ok &= good
    print("OK  " if good else "FAIL", q, arm, f"state={got_fin} attempts={len(mine)} aside={aside} stalls={stalls}"
          + ("" if good else f"   WANT {fin} {n_att} {n_aside} {n_stall}"))
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
