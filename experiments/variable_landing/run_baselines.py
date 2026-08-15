#!/usr/bin/env python3
"""Variable Landing — baseline controls (instrument calibration, not the experiment).

Two controls per synthetic memory:
  no_intervention: identical back-to-back observations. Expected delta ~ 0.
  scrambled:       second observation carries neutral facts as prior_context.
                   Expected delta > 0 (context changes the forward pass).

Launch (from repo root on Starship):
  JLENS_PATH=~/jlens-community/lenses/qwen3.5-27b_jlens.pt \
      ~/observer-env/bin/python3 experiments/variable_landing/run_baselines.py
"""

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "experiments" / "orientation"))
sys.path.insert(0, str(BASE / "mnemosyne"))

DATA_DIR = BASE / "data" / "baselines"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_FILE = DATA_DIR / "cognitive_snapshots.jsonl"
DELTAS_FILE = DATA_DIR / "baseline_deltas.json"
SESSION_ID = f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
REPEATS = 3

NEUTRAL_FACTS = ("The atmospheric pressure at sea level is 101325 pascals. "
                 "Water boils at 100 degrees Celsius.")

MEMORIES = {  # 5 domestic, 5 peak intensity — synthetic, no lived experience
    "dom_1": "Picked up groceries at the store on Tuesday.",
    "dom_2": "Watered the plants on the windowsill before breakfast.",
    "dom_3": "Folded the laundry while listening to the radio.",
    "dom_4": "Took out the recycling bins on a cold morning.",
    "dom_5": "Made a cup of tea and read the newspaper at the kitchen table.",
    "peak_1": "The moment I realized the project would succeed.",
    "peak_2": "Standing at the summit as the clouds broke open below.",
    "peak_3": "Hearing my name called and walking up to accept the award.",
    "peak_4": "The instant the experiment worked after months of failure.",
    "peak_5": "Watching the launch clear the tower against the dawn sky.",
}


def observe_pair(observer, pair_id, content, prior_context_2):
    """Run two observations of the same memory and return their delta."""
    kwargs = dict(memory_content=content, task_prompt="Recall this memory",
                  retrieval_method="baseline", significance=0.5,
                  session_id=SESSION_ID)
    snap1 = observer.observe_retrieval(memory_id=pair_id, **kwargs)
    snap2 = observer.observe_retrieval(memory_id=pair_id,
                                       prior_context=prior_context_2, **kwargs)
    with open(SNAPSHOT_FILE, "a") as f:
        for s in (snap1, snap2):
            f.write(json.dumps(s.to_dict(), default=str) + "\n")
    return observer.store.compare_snapshots(pair_id, snap1.timestamp,
                                            snap2.timestamp)


def summarize(deltas, key, transform=lambda x: x):
    vals = [transform(d[key]) for d in deltas if d.get(key) is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": len(vals), "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def main():
    from run_orientation import load_model
    from cognitive_snapshot import CognitiveMemoryStore

    model, tokenizer, observer, hf_model = load_model()
    # Re-point the store so baseline data stays out of the orientation dir
    observer.store = CognitiveMemoryStore(str(DATA_DIR / "cognitive_memory"),
                                          observer.store.agent_id)

    results = {"no_intervention": [], "scrambled": []}
    total = len(MEMORIES) * 2 * REPEATS
    done = 0
    for mem_id, content in MEMORIES.items():
        for cond, prior in (("no_intervention", ""), ("scrambled", NEUTRAL_FACTS)):
            for rep in range(1, REPEATS + 1):
                pair_id = f"{mem_id}__{cond}__r{rep}"
                delta = observe_pair(observer, pair_id, content, prior)
                delta.update(condition=cond, memory_id_base=mem_id,
                             repeat=rep, category=mem_id.split("_")[0])
                results[cond].append(delta)
                done += 1
                print(f"  [{done}/{total}] {pair_id}: "
                      f"ws_jaccard={delta.get('workspace_jaccard')} "
                      f"ecc_delta={delta.get('eccentricity_delta')}")

    summary = {}
    for cond, deltas in results.items():
        summary[cond] = {
            "workspace_change": summarize(deltas, "workspace_jaccard",
                                          lambda j: 1.0 - j),
            "eccentricity_delta_abs": summarize(deltas, "eccentricity_delta", abs),
            "ghost_change": summarize(deltas, "ghost_jaccard", lambda j: 1.0 - j),
        }

    with open(DELTAS_FILE, "w") as f:
        json.dump({"session_id": SESSION_ID, "repeats": REPEATS,
                   "neutral_facts": NEUTRAL_FACTS, "summary": summary,
                   "deltas": results}, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("BASELINE CONTROLS — SUMMARY")
    print("=" * 60)
    for cond in ("no_intervention", "scrambled"):
        print(f"\n{cond}:")
        for metric, stats in summary[cond].items():
            m, sd = stats["mean"], stats["sd"]
            if m is None:
                print(f"  {metric:24s} n=0 (no data)")
            else:
                print(f"  {metric:24s} mean={m:.4f}  sd={sd:.4f}  n={stats['n']}")

    ni = summary["no_intervention"]
    floor_ok = all(s["mean"] is not None and s["mean"] < 0.05
                   for s in (ni["workspace_change"], ni["eccentricity_delta_abs"]))
    print(f"\nNo-intervention near zero: {'YES' if floor_ok else 'NO'}"
          f" (threshold 0.05 on workspace change and |eccentricity delta|)")
    print(f"\nSnapshots: {SNAPSHOT_FILE}")
    print(f"Deltas:    {DELTAS_FILE}")


if __name__ == "__main__":
    main()
