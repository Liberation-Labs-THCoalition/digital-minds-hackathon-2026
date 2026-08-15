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
# Welfare monitoring
# ---------------------------------------------------------------------------

def workspace_tokens(snap_dict):
    """Primary-metric input: the snapshot's dominant workspace token list
    (prereg 3: Jaccard over J-lens workspace token sets). Falls back to the
    full dict (delta_numeric path) only if tokens are absent, and to None for
    non-dict input. Requires pipeline storage via dataclasses.asdict."""
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
                   resume: bool = False):

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

    # Load pipeline
    from variable_landing_pipeline import VariableLandingPipeline

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
    )

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

    for arm, rep, mem in trial_schedule:
        # Skip already-completed trials on resume
        if trial_count < start_trial:
            trial_count += 1
            continue

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
            )

            trial_data = {
                "arm": arm,
                "repeat": rep,
                "memory_id": mem["id"],
                "entity": mem["entity"],
                "n_facts_stored": record.n_facts_stored,
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
    )
