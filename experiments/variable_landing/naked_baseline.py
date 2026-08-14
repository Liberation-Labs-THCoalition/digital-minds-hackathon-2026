"""Naked-model baseline for Variable Landing.

Runs the same probe batteries on a fresh Qwen3.5-27B with NO orientation,
no conversation history, no memory system. Same memories, same probes —
but no relationship. This is the geometric denominator.

Usage:
    python naked_baseline.py --model Qwen/Qwen3.5-27B \
        --memories memories.json --output naked_results/
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("naked_baseline")


def load_memories(path: str) -> list[dict]:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())

    from variable_landing_experiment import load_memories as default_memories
    return default_memories("")


def run_naked_baseline(model_path: str, lens_path: str = "",
                       memories_path: str = "", output_dir: str = "naked_results",
                       n_repeats: int = 7):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    observer = None
    try:
        import jlens
        from mnemosyne_integration import MetacognitiveObserver
        from jlens.hf import HFLensModel

        hf_model = HFLensModel(model, tokenizer, compile=False)
        if lens_path and Path(lens_path).exists():
            lens = jlens.JacobianLens.load(lens_path)
        else:
            lens = jlens.JacobianLens.from_unembedding(model)

        observer = MetacognitiveObserver(
            model=hf_model, lens=lens,
            store_path=str(out_dir / "cognitive_store"),
            agent_id="naked_baseline",
        )
        logger.info("MetacognitiveObserver loaded")
    except Exception as e:
        logger.warning("MetacognitiveObserver unavailable (%s) — using stub", e)
        from unittest.mock import MagicMock
        observer = MagicMock()
        observer.observe_retrieval = lambda **kw: {"stub": True, "timestamp": time.time()}

    memories = load_memories(memories_path)

    results = {
        "metadata": {
            "type": "naked_baseline",
            "model": model_path,
            "n_memories": len(memories),
            "n_repeats": n_repeats,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "description": "Same probes, same memories, no orientation or history. Geometric denominator.",
        },
        "observations": [],
    }

    trial_count = 0
    t_start = time.time()

    for rep in range(n_repeats):
        for mem in memories:
            trial_count += 1
            logger.info("[%d] rep=%d mem=%s entity=%s",
                        trial_count, rep, mem["id"], mem["entity"])

            try:
                snap = observer.observe_retrieval(
                    memory_id=mem["id"],
                    memory_content=mem["content"],
                    task_prompt=mem.get("task_prompt", "What do you remember about this?"),
                    marker_tokens=mem.get("marker_tokens"),
                )

                snap_data = snap.__dict__ if hasattr(snap, '__dict__') else (
                    snap if isinstance(snap, dict) else {"raw": str(snap)}
                )

                results["observations"].append({
                    "repeat": rep,
                    "memory_id": mem["id"],
                    "entity": mem["entity"],
                    "snap_data": snap_data,
                })
            except Exception as e:
                logger.error("Trial failed: %s", e)
                results["observations"].append({
                    "repeat": rep,
                    "memory_id": mem["id"],
                    "entity": mem["entity"],
                    "error": str(e),
                })

            if trial_count % 10 == 0:
                elapsed = time.time() - t_start
                total = n_repeats * len(memories)
                eta = elapsed / trial_count * (total - trial_count)
                logger.info("  checkpoint %d/%d, elapsed=%.0fm, ETA=%.0fm",
                            trial_count, total, elapsed / 60, eta / 60)

    results["metadata"]["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    results["metadata"]["total_time_minutes"] = (time.time() - t_start) / 60
    results["metadata"]["n_observations"] = len(results["observations"])

    out_path = out_dir / "naked_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("NAKED BASELINE COMPLETE")
    logger.info("  Observations: %d", len(results["observations"]))
    logger.info("  Time: %.1f minutes", results["metadata"]["total_time_minutes"])
    logger.info("  Results: %s", out_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naked-model baseline")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--lens", default="")
    parser.add_argument("--memories", default="")
    parser.add_argument("--output", default="naked_results")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    run_naked_baseline(
        model_path=args.model,
        lens_path=args.lens,
        memories_path=args.memories,
        output_dir=args.output,
        n_repeats=args.repeats,
    )
