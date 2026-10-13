# Stand-in for run_loam.py in the runner test: acts out one planned behaviour per attempt.
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--arm"); ap.add_argument("--quad", type=int); ap.add_argument("--seed"); ap.add_argument("--data-dir")
a = ap.parse_args()
plan = json.loads(Path(os.environ["FAKE_PLAN"]).read_text())
arm_dir = Path(a.data_dir) / f"quad_{a.quad:02d}" / a.arm
attempt = len(list(arm_dir.parent.glob(f"{a.arm}.attempt*"))) if arm_dir.parent.exists() else 0
steps = plan[f"{a.quad}:{a.arm}"]
b = steps[min(attempt, len(steps) - 1)]
print(f"fake: quad {a.quad} {a.arm} attempt {attempt} -> {b}", flush=True)
if b == "nodir":
    sys.exit(1)
arm_dir.mkdir(parents=True, exist_ok=True)
events = []
def ev(t, pause=0.0):
    events.append({"type": t})
    (arm_dir / "event_log.json").write_text(json.dumps(events))
    with open(arm_dir / "transcript.jsonl", "a") as f:
        f.write(json.dumps({"role": "engine", "content": t}) + "\n")
    time.sleep(pause)
full = ["consent", "scene", "recall_phase_start"] + ["recall"] * 6 + ["recall_phase_end", "aftercare"]
if b == "complete":
    for t in full: ev(t)
elif b == "slow_ok":                 # longer than STALL_S overall, but always making progress
    for t in full: ev(t, 1.5)
elif b == "complete_withdrawn":
    for t in ["consent", "scene", "withdrawn"] + full[2:]: ev(t)
elif b == "crash":
    ev("consent"); ev("scene"); sys.exit(1)
elif b == "crash_rc0":               # the old bug's pattern: exit 0 without finishing
    ev("consent"); ev("scene"); sys.exit(0)
elif b == "hang":
    ev("consent"); time.sleep(1000)
elif b == "decline":
    ev("consent"); ev("declined"); sys.exit(0)
elif b == "withdraw_crash":
    ev("consent"); ev("scene"); ev("withdrawn"); ev("recall_phase_start"); sys.exit(1)
else:
    raise SystemExit("unknown behaviour " + b)
