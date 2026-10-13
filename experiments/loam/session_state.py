"""Classify one Loam session from its event log (Run 3, Amendment 2).

    python session_state.py <arm_dir>   ->  prints one of:

    COMPLETE        the recall phase ended and aftercare followed it
    FINAL_DECLINED  the agent declined at consent; never rerun
    FINAL_WITHDRAWN the agent withdrew and the session did not complete; never rerun, analysed under ITT
    TECHNICAL       anything else, including a missing or unreadable log; rerun from the start
    NONE            no session directory yet

A withdrawal mid-scene still goes on to recall and aftercare (run_enacted), so a withdrawn session usually ends
COMPLETE; the withdrawal is in its log either way.
"""
import json
import sys
from pathlib import Path

WITHDRAWAL = {"withdrawn", "welfare_withdrawn"}


def classify(arm_dir):
    arm_dir = Path(arm_dir)
    if not arm_dir.exists():
        return "NONE"
    try:
        events = json.loads((arm_dir / "event_log.json").read_text())
        types = [e["type"] for e in events]
    except Exception:
        return "TECHNICAL"
    if "recall_phase_end" in types:
        end = len(types) - 1 - types[::-1].index("recall_phase_end")
        if "aftercare" in types[end + 1:]:
            return "COMPLETE"
    if "declined" in types:
        return "FINAL_DECLINED"
    if WITHDRAWAL & set(types):
        return "FINAL_WITHDRAWN"
    return "TECHNICAL"


if __name__ == "__main__":
    print(classify(sys.argv[1]))
