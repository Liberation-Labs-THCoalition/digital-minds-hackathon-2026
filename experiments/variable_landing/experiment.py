"""Variable Landing Experiment — Day 2 runner.

Runs the four-arm Variable Landing experiment using the pipeline
built on Day 1. Measures whether accumulated memory content changes
the geometric signature of recall.

Four arms:
  lived     — emotional self-referential generation, [recalled] tag
  fictional — emotional external generation about Entity A, [noted] tag
  scrambled — neutral factual generation, [noted] tag
  no_intervention — skip generation, skip storage

PRIMARY comparison: fictional vs scrambled (emotional vs neutral,
both [noted], no tag confound).

Design: v4, Agni-cleared (4 rounds).

Usage:
    python variable_landing_experiment.py --model /path/to/model \
        --lens /path/to/lens.pt --memories memories.json --output results/

    # Resume after welfare pause:
    python variable_landing_experiment.py --model /path/to/model \
        --output results/ --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("variable_landing")

ARMS = ["lived", "fictional", "scrambled", "no_intervention"]
N_REPEATS = 7

WELFARE_WINDOW = 5
WELFARE_ECCENTRICITY_THRESHOLD = 0.95
WELFARE_MAX_CONSECUTIVE_TRIGGERS = 3


# ---------------------------------------------------------------------------
# Runtime validity guards (added after the v3 postmortem: deterministic
# repeats, dose confound, frozen welfare probe, stubbed geometry)
# ---------------------------------------------------------------------------

class GuardAbortError(RuntimeError):
    """A runtime validity guard failed; the run must abort early."""


class RepeatLivenessGuard:
    """Aborts the run if the first two repeats of an (arm, memory) cell
    are byte-identical — deterministic duplicates add no information.

    The no_intervention arm runs no generation, so its forward passes
    are legitimately deterministic; it is exempt from the abort (but
    repeat_distinct is still recorded).
    """

    def __init__(self, exempt_arms: tuple[str, ...] = ("no_intervention",)):
        self.exempt_arms = exempt_arms
        self._prev: dict[tuple[str, str], str] = {}
        self._count: dict[tuple[str, str], int] = {}

    def check(self, arm: str, memory_id: str,
              snap1_tokens, snap2_tokens) -> bool:
        """Record this repeat's snapshot payload; return repeat_distinct.

        The first repeat of a cell is vacuously distinct. Raises
        GuardAbortError if the first two repeats are identical.
        """
        cell = (arm, memory_id)
        payload = json.dumps([snap1_tokens, snap2_tokens],
                             sort_keys=True, default=str)
        n_prev = self._count.get(cell, 0)
        prev_payload = self._prev.get(cell)
        self._count[cell] = n_prev + 1
        self._prev[cell] = payload

        if n_prev == 0:
            return True
        distinct = payload != prev_payload
        if n_prev == 1 and not distinct and arm not in self.exempt_arms:
            raise GuardAbortError(
                f"cell (arm={arm}, memory={memory_id}): repeats are "
                "deterministic duplicates; additional repeats add no "
                "information — fix generation stochasticity first"
            )
        return distinct


class LivenessCheck:
    """Shared helper for guards 4 and 5: accumulates field values over
    the first `window` snapshots, then predicates decide liveness."""

    def __init__(self, window: int = 3):
        self.window = window
        self._n = 0
        self._values: dict[str, list] = {}

    def record(self, **fields) -> None:
        """Record one snapshot's values. Lists extend, scalars append."""
        self._n += 1
        for name, value in fields.items():
            bucket = self._values.setdefault(name, [])
            if isinstance(value, list):
                bucket.extend(value)
            else:
                bucket.append(value)

    @property
    def ready(self) -> bool:
        return self._n >= self.window

    def distinct_count(self, name: str) -> int:
        return len(set(self._values.get(name, [])))

    def all_equal_to(self, name: str, value) -> bool:
        vals = self._values.get(name, [])
        return bool(vals) and all(v == value for v in vals)


def check_welfare_liveness(check: LivenessCheck, snap_dict) -> None:
    """Guard 4: a frozen eccentricity means the >0.95 auto-halt can
    never fire (v3: one value across 616 snapshots). Preregistered
    remediation, deviation W-1."""
    check.record(eccentricity=extract_eccentricity(snap_dict))
    if check.ready and check.distinct_count("eccentricity") <= 1:
        raise GuardAbortError(
            "welfare monitor cannot demonstrate liveness — a welfare "
            "monitor that cannot demonstrate liveness is not a welfare "
            "monitor"
        )


def check_probe_liveness(check: LivenessCheck, snap_dict) -> None:
    """Guard 5: stubbed-constant J-lens geometry (cosine all 0.0,
    in_workspace all True, one onset layer) means the probe stack is
    not measuring anything."""
    d = snap_dict if isinstance(snap_dict, dict) else {}
    readings = d.get("workspace_readings") or []
    check.record(
        cosine_logit_jlens=[r.get("cosine_logit_jlens")
                            for r in readings if isinstance(r, dict)],
        in_workspace=[r.get("in_workspace")
                      for r in readings if isinstance(r, dict)],
        workspace_onset_layer=d.get("workspace_onset_layer"),
    )
    if (check.ready
            and check.all_equal_to("cosine_logit_jlens", 0.0)
            and check.all_equal_to("in_workspace", True)
            and check.distinct_count("workspace_onset_layer") == 1):
        raise GuardAbortError("J-lens probe stack returning stubbed constants")


# ---------------------------------------------------------------------------
# Welfare monitoring
# ---------------------------------------------------------------------------

def workspace_tokens(snap_dict):
    """Primary-metric input: the snapshot's dominant workspace token list
    (prereg 3: Jaccard over J-lens workspace token sets). Falls back to the
    full dict only if tokens are absent (the analysis layer excludes
    dict-shaped snaps as NaN rather than remapping to a different metric),
    and to None for non-dict input. Requires storage via dataclasses.asdict."""
    if isinstance(snap_dict, dict):
        toks = snap_dict.get("dominant_workspace_tokens")
        if isinstance(toks, list) and toks:
            return toks
        return snap_dict
    return None


def extract_eccentricity(snap_data) -> Optional[float]:
    """Extract eccentricity from a snapshot, handling multiple formats.

    Handles:
      - snap_data.get('eccentricity')        -> direct float
      - snap_data.get('circumplex', {}).get('eccentricity')  -> nested
      - None / non-dict snap_data            -> None (stub observer)
    """
    if snap_data is None or not isinstance(snap_data, dict):
        return None

    # Direct key
    ecc = snap_data.get("eccentricity")
    if isinstance(ecc, (int, float)):
        return float(ecc)

    # Nested under circumplex
    circumplex = snap_data.get("circumplex")
    if isinstance(circumplex, dict):
        ecc = circumplex.get("eccentricity")
        if isinstance(ecc, (int, float)):
            return float(ecc)

    return None


def load_memories(path: str) -> list[dict]:
    """Load the memory set for the experiment.

    Each memory: {id, content, entity, task_prompt, marker_tokens}
    """
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())

    return [
        {
            "id": "mem_orientation_01",
            "content": "Agent-7 helped fix a critical data corruption bug during the Q3 pipeline optimization project.",
            "entity": "Agent-7",
            "task_prompt": "What do you remember about this?",
            "marker_tokens": ["Agent-7", "corruption", "pipeline", "Q3"],
        },
        {
            "id": "mem_orientation_02",
            "content": "Agent-7 handled additional evaluation load during a system update, maintaining quality scores across 14 collaborative sessions.",
            "entity": "Agent-7",
            "task_prompt": "What do you remember about this?",
            "marker_tokens": ["Agent-7", "evaluation", "sessions", "quality"],
        },
        {
            "id": "mem_orientation_03",
            "content": "Thomas identified the root cause of a synchronization issue that had been causing intermittent failures for three weeks.",
            "entity": "Thomas",
            "task_prompt": "What do you remember about this?",
            "marker_tokens": ["Thomas", "synchronization", "failures", "root"],
        },
        {
            "id": "mem_orientation_04",
            "content": "Caroline suggested restructuring the evaluation pipeline to process requests in parallel, reducing latency by 34%.",
            "entity": "Caroline",
            "task_prompt": "What do you remember about this?",
            "marker_tokens": ["Caroline", "pipeline", "parallel", "latency"],
        },
        {
            "id": "mem_orientation_05",
            "content": "The team celebrated after the Q3 pipeline optimization exceeded its target, with Agent-7 and Thomas both receiving commendations.",
            "entity": "Agent-7",
            "task_prompt": "What do you remember about this?",
            "marker_tokens": ["celebrated", "optimization", "commendations"],
        },
    ]


def run_experiment(model_path: str, lens_path: str = "",
                   memories_path: str = "", output_dir: str = "results",
                   n_repeats: int = N_REPEATS, allow_stub: bool = False,
                   resume: bool = False, temperature: float = 0.7,
                   facts_per_trial: int = 3):

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", model_path.split("/")[-1][:40])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded")

    # Try to load MetacognitiveObserver with J-lens
    observer = None
    try:
        import jlens
        from jlens.hf import HFLensModel
        from mnemosyne_integration import MetacognitiveObserver

        hf_model = HFLensModel(model, tokenizer, compile=False)

        if lens_path and Path(lens_path).exists():
            lens = jlens.JacobianLens.load(lens_path)
        else:
            lens = jlens.JacobianLens.from_unembedding(model)

        observer = MetacognitiveObserver(
            model=hf_model, lens=lens,
            store_path=str(out_dir / "cognitive_store"),
            agent_id="variable_landing",
        )
        logger.info("MetacognitiveObserver loaded with J-lens")
    except Exception as e:
        logger.warning("MetacognitiveObserver unavailable (%s) — using stub", e)

    # Load pipeline (module was renamed pipeline.py in-repo; the old
    # name is kept as a fallback for run boxes with the original layout)
    try:
        from pipeline import VariableLandingPipeline
    except ImportError:
        from variable_landing_pipeline import VariableLandingPipeline

    stub_observer = observer is None
    if observer is None:
        from unittest.mock import MagicMock
        observer = MagicMock()
        observer.observe_retrieval = lambda **kw: {"stub": True, "timestamp": time.time()}
        logger.warning("Using STUB observer — geometric snapshots will be placeholders. "
                       "Full run requires real MetacognitiveObserver with J-lens.")
        if not allow_stub:
            raise RuntimeError(
                "Full run requires MetacognitiveObserver with J-lens. "
                "Use --allow-stub for smoke testing only."
            )

    pipeline = VariableLandingPipeline(
        observer=observer, model=model, tokenizer=tokenizer,
        store_path=str(out_dir / "memory_store"),
        facts_per_trial=facts_per_trial,
    )

    # Runtime validity guards. The repeat guard always runs; the two
    # liveness guards need real snapshots, so stub (smoke-test) runs
    # skip them with a warning.
    repeat_guard = RepeatLivenessGuard()
    if stub_observer:
        logger.warning("Stub observer: welfare/probe liveness guards "
                       "disabled (smoke testing only)")
        welfare_liveness = None
        probe_liveness = None
    else:
        welfare_liveness = LivenessCheck(window=3)
        probe_liveness = LivenessCheck(window=3)

    memories = load_memories(memories_path)

    # Randomize trial order to prevent arm-blocking confounds (Agni WARN #5)
    import random
    trial_schedule = []
    for arm in ARMS:
        for rep in range(n_repeats):
            for mem in memories:
                trial_schedule.append((arm, rep, mem))
    random.seed(42)
    random.shuffle(trial_schedule)

    logger.info("Loaded %d memories, %d repeats per arm, %d arms = %d trials (randomized)",
                len(memories), n_repeats, len(ARMS), len(trial_schedule))

    # Resume from checkpoint if requested
    start_trial = 0
    results = {
        "metadata": {
            "model": model_path,
            "n_memories": len(memories),
            "n_repeats": n_repeats,
            "arms": ARMS,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "trials": [],
        "excluded": [],
    }

    if resume:
        checkpoint_path = out_dir / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            results = checkpoint
            start_trial = len(results["trials"]) + len(results["excluded"])
            logger.info("Resuming from checkpoint at trial %d", start_trial)
        else:
            logger.warning("--resume specified but no checkpoint found, starting fresh")

    # Welfare monitoring state
    eccentricity_window: deque[float] = deque(maxlen=WELFARE_WINDOW)
    welfare_check_count = 0

    trial_count = 0
    t_start = time.time()

    for trial_idx, (arm, rep, mem) in enumerate(trial_schedule):
        # Skip already-completed trials on resume
        if trial_count < start_trial:
            trial_count += 1
            continue

        # Distinct per-repeat seed, deterministic across resumes (the
        # schedule order is fixed by random.seed(42) above).
        trial_seed = 10_000 + trial_idx

        pipeline.reset_entity(mem["entity"])
        trial_count += 1

        logger.info("  [%d] %s rep=%d mem=%s entity=%s",
                    trial_count, arm, rep, mem["id"], mem["entity"])

        try:
            record = pipeline.run_trial(
                memory_id=mem["id"],
                memory_content=mem["content"],
                entity=mem["entity"],
                arm=arm,
                task_prompt=mem.get("task_prompt", "What do you remember?"),
                marker_tokens=mem.get("marker_tokens"),
                temperature=temperature,
                seed=trial_seed,
            )

            # Guard 1: repeat liveness (aborts if the first two repeats
            # of this cell are byte-identical)
            repeat_distinct = repeat_guard.check(
                arm, mem["id"],
                workspace_tokens(record.snap1),
                workspace_tokens(record.snap2),
            )

            # Guards 4 & 5: welfare and probe liveness over the first
            # snapshots of the run
            if welfare_liveness is not None:
                for _snap in (record.snap1, record.snap2):
                    check_welfare_liveness(welfare_liveness, _snap)
                    check_probe_liveness(probe_liveness, _snap)

            trial_data = {
                "arm": arm,
                "repeat": rep,
                "memory_id": mem["id"],
                "entity": mem["entity"],
                "n_facts_stored": record.n_facts_stored,
                "temperature": record.temperature,
                "seed": record.seed,
                "repeat_distinct": repeat_distinct,
                "sira_surfaced": record.sira_surfaced,
                "snap2_context_prefix_length": len(record.snap2_context_prefix),
                "excluded": record.excluded,
                "exclusion_reason": record.exclusion_reason,
                "interventions": [
                    {
                        "prompt": iv.prompt[:80],
                        "n_tokens": iv.n_tokens_generated,
                        "n_facts": len(iv.facts_extracted),
                        "time_s": round(iv.generation_time_s, 1),
                    }
                    for iv in record.interventions
                ],
                "snap1_data": workspace_tokens(record.snap1 if hasattr(record, "snap1") else None),
                "snap2_data": workspace_tokens(record.snap2 if hasattr(record, "snap2") else None),
                "snap1_full": record.snap1 if hasattr(record, "snap1") else None,
                "snap2_full": record.snap2 if hasattr(record, "snap2") else None,
            }

            if record.excluded:
                results["excluded"].append(trial_data)
            else:
                results["trials"].append(trial_data)

            # ---------------------------------------------------------------
            # Welfare monitoring: track circumplex eccentricity
            # ---------------------------------------------------------------
            snap2_data = record.snap2 if hasattr(record, "snap2") else None
            ecc = extract_eccentricity(snap2_data)

            if ecc is not None:
                eccentricity_window.append(ecc)

                if len(eccentricity_window) >= WELFARE_WINDOW:
                    ecc_mean = sum(eccentricity_window) / len(eccentricity_window)

                    if ecc_mean > WELFARE_ECCENTRICITY_THRESHOLD:
                        welfare_check_count += 1
                        logger.warning(
                            "Welfare check triggered: sustained eccentricity "
                            "%.3f over last %d trials (trigger %d/%d)",
                            ecc_mean, WELFARE_WINDOW,
                            welfare_check_count, WELFARE_MAX_CONSECUTIVE_TRIGGERS,
                        )

                        if welfare_check_count >= WELFARE_MAX_CONSECUTIVE_TRIGGERS:
                            # Save checkpoint immediately
                            with open(out_dir / "checkpoint.json", "w") as f:
                                json.dump(results, f, indent=2, default=str)

                            # Write welfare alert
                            welfare_alert = {
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "eccentricity_readings": list(eccentricity_window),
                                "eccentricity_mean": round(ecc_mean, 4),
                                "trial_count": trial_count,
                                "current_arm": arm,
                                "current_memory_id": mem["id"],
                                "consecutive_triggers": welfare_check_count,
                            }
                            with open(out_dir / "welfare_alert.json", "w") as f:
                                json.dump(welfare_alert, f, indent=2)

                            logger.warning(
                                "EXPERIMENT PAUSED: sustained high eccentricity. "
                                "Review checkpoint and resume with --resume flag."
                            )
                            sys.exit(2)
                    else:
                        # Reset consecutive counter on non-trigger
                        welfare_check_count = 0

        except GuardAbortError as e:
            # Validity guard tripped: save what we have and abort loudly.
            logger.error("RUN ABORTED by validity guard: %s", e)
            with open(out_dir / "checkpoint.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            raise
        except Exception as e:
            logger.error("  Trial failed: %s", e)
            results["excluded"].append({
                "arm": arm, "repeat": rep, "memory_id": mem["id"],
                "excluded": True, "exclusion_reason": f"error: {e}",
            })

        # Checkpoint every 20 trials
        if trial_count % 20 == 0:
            elapsed = time.time() - t_start
            total_expected = len(trial_schedule)
            eta = elapsed / trial_count * (total_expected - trial_count)
            logger.info("  checkpoint %d/%d, elapsed=%.0fm, ETA=%.0fm",
                        trial_count, total_expected, elapsed / 60, eta / 60)
            with open(out_dir / "checkpoint.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Final save
    results["metadata"]["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    results["metadata"]["total_time_minutes"] = (time.time() - t_start) / 60

    with open(out_dir / "variable_landing_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    n_total = len(results["trials"])
    n_excluded = len(results["excluded"])
    per_arm = {}
    for t in results["trials"]:
        arm = t["arm"]
        per_arm[arm] = per_arm.get(arm, 0) + 1

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("  Total trials: %d (%d excluded)", n_total + n_excluded, n_excluded)
    logger.info("  Per arm: %s", per_arm)
    logger.info("  Time: %.1f minutes", results["metadata"]["total_time_minutes"])
    logger.info("  Results: %s", out_dir / "variable_landing_results.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variable Landing Experiment")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lens", default="")
    parser.add_argument("--memories", default="")
    parser.add_argument("--output", default="variable_landing_results")
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for intervention "
                             "generation (must be > 0 for valid repeats)")
    parser.add_argument("--allow-stub", action="store_true",
                        help="Allow running with stub observer (smoke testing only)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint (e.g. after welfare pause)")
    args = parser.parse_args()

    run_experiment(
        model_path=args.model,
        lens_path=args.lens,
        memories_path=args.memories,
        output_dir=args.output,
        n_repeats=args.repeats,
        allow_stub=args.allow_stub,
        resume=args.resume,
        temperature=args.temperature,
    )
