#!/usr/bin/env python
"""reprobe_vl.py — Post-hoc re-probe for Variable Landing v3.

Re-runs MetacognitiveObserver probes on already-generated data. No new
model generation — just forward passes on existing text to fill in the
geometry fields that were frozen in the original run (circumplex
eccentricity, ghost cosine, workspace cosine_logit_jlens).

The original run produced 308 trials with frozen circumplex (e=0.836
everywhere) because observe_retrieval was called WITHOUT calibrate_probes(),
which means _measure_circumplex_live fell through to the static path
(same anchor-prompt measurement each time). This re-probe initializes
the observer the same way (no calibrate_probes), but then explicitly
calibrates the circumplex and ghost probes before re-running each trial
with the FULL trial-specific text — producing readings that actually
vary by trial content.

Usage:
    python reprobe_vl.py \\
        --results /path/to/variable_landing_results.json \\
        --memories /path/to/vl_memories_v2.json \\
        --output /path/to/variable_landing_v3_reprobed.json

    # Custom model/lens paths:
    python reprobe_vl.py \\
        --results data/variable_landing_v3/variable_landing_results.json \\
        --memories data/orientation/vl_memories_v2.json \\
        --output data/variable_landing_v3/variable_landing_v3_reprobed.json \\
        --model Qwen/Qwen3.5-27B \\
        --lens /Users/[AGENT]/jlens-community/lenses/qwen3.5-27b_jlens.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional

import torch

# Allow imports from the hackathon mnemosyne directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mnemosyne"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("reprobe_vl")

# ---- Defaults ----
DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_LENS = "/Users/[AGENT]/jlens-community/lenses/qwen3.5-27b_jlens.pt"
DEFAULT_RESULTS = (
    "/Users/[AGENT]/digital-minds-hackathon-2026/data/"
    "variable_landing_v3/variable_landing_results.json"
)
DEFAULT_MEMORIES = (
    "/Users/[AGENT]/digital-minds-hackathon-2026/data/"
    "orientation/vl_memories_v2.json"
)
DEFAULT_OUTPUT = (
    "/Users/[AGENT]/digital-minds-hackathon-2026/data/"
    "variable_landing_v3/variable_landing_v3_reprobed.json"
)

# Workspace layers must match the original experiment
WORKSPACE_LAYERS = [35, 39, 43, 45, 47]
CIRCUMPLEX_LAYER = 45

# BUG 4 FIX: Fields that must NEVER be overwritten during merge.
# The original Jaccard was computed from these; overwriting them would
# silently invalidate the Jaccard scores without re-computing them.
PROTECTED_SNAP2_FIELDS = frozenset({
    "dominant_workspace_tokens",
    "workspace_tokens",
    "workspace_token_ids",
    "workspace_token_logits",
    "workspace_jaccard",
    "workspace_jaccard_overlap",
    "workspace_jaccard_union",
})


def load_model_and_observer(
    model_path: str,
    lens_path: str,
    store_path: str = "/tmp/reprobe_cognitive_store",
) -> tuple:
    """Load model, lens, and MetacognitiveObserver.

    Initializes the observer the SAME way as experiment.py (no
    calibrate_probes), then explicitly calibrates circumplex and ghost
    probes so measure_live paths are active.

    Returns (observer, tokenizer, model).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import jlens
    from jlens.hf import HFLensModel
    from mnemosyne_integration import MetacognitiveObserver

    logger.info("Loading tokenizer: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
    )

    logger.info("Loading model: %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded (%d parameters)", sum(p.numel() for p in model.parameters()))

    logger.info("Wrapping model in HFLensModel")
    hf_model = HFLensModel(model, tokenizer, compile=False)

    logger.info("Loading J-lens from %s", lens_path)
    lens = jlens.JacobianLens.load(lens_path)
    logger.info(
        "Lens loaded: %d Jacobians, source layers %s",
        len(lens.jacobians),
        sorted(lens.jacobians.keys())[:10],
    )

    observer = MetacognitiveObserver(
        model=hf_model,
        lens=lens,
        store_path=store_path,
        agent_id="reprobe_vl",
        workspace_layers=WORKSPACE_LAYERS,
        circumplex_layer=CIRCUMPLEX_LAYER,
    )
    observer.model_name = model_path

    # ---- KEY DIFFERENCE FROM ORIGINAL RUN ----
    # The original run did NOT call calibrate_probes(), so circumplex
    # fell back to the static path (frozen readings). Here we calibrate
    # BOTH probes so the live measurement paths are active.
    logger.info("Calibrating circumplex probe at layer %d", CIRCUMPLEX_LAYER)
    observer.circumplex_probe.calibrate(CIRCUMPLEX_LAYER)

    logger.info("Calibrating ghost probe")
    observer.ghost_probe.calibrate()

    logger.info("Observer ready (circumplex calibrated=%s, ghost calibrated=%s)",
                getattr(observer.circumplex_probe, '_calibrated', False),
                observer.ghost_probe.is_calibrated)

    return observer, tokenizer, model


def load_data(
    results_path: str,
    memories_path: str,
) -> tuple[dict, dict[str, dict]]:
    """Load original results and memory lookup.

    Returns (results_dict, {memory_id: memory_dict}).
    """
    logger.info("Loading results from %s", results_path)
    results = json.loads(Path(results_path).read_text())

    n_trials = len(results.get("trials", []))
    n_excluded = len(results.get("excluded", []))
    logger.info("Loaded %d trials (%d excluded)", n_trials, n_excluded)

    logger.info("Loading memories from %s", memories_path)
    memories_list = json.loads(Path(memories_path).read_text())
    mem_lookup = {m["id"]: m for m in memories_list}
    logger.info("Loaded %d memories: %s", len(mem_lookup), list(mem_lookup.keys()))

    return results, mem_lookup


def reconstruct_context_prefix(trial: dict, arm: str) -> str:
    """Reconstruct the context prefix that was prepended for snap2.

    BUG 2 FIX: The original code fabricated an approximate prefix from
    truncated (80-char) intervention prompts, which does NOT match what
    the model actually saw. Now we:
      1. Use the saved `snap2_context_prefix` string if available (the
         pipeline saves up to 500 chars of it).
      2. Otherwise, reconstruct from the FULL intervention prompts and
         responses in the trial's `interventions` list.
      3. Emit a warning if the reconstructed length doesn't match the
         recorded `snap2_context_prefix_length`.
    """
    prefix_len = trial.get("snap2_context_prefix_length", 0)
    if prefix_len == 0:
        return ""

    # ---- Path 1: use the saved prefix string if present ----
    saved_prefix = trial.get("snap2_context_prefix")
    if saved_prefix:
        if len(saved_prefix) != prefix_len:
            logger.warning(
                "Trial %s: snap2_context_prefix length (%d) != "
                "snap2_context_prefix_length (%d); prefix may be truncated "
                "by pipeline (500-char cap). Using saved prefix as-is.",
                trial.get("memory_id", "?"),
                len(saved_prefix),
                prefix_len,
            )
        return saved_prefix

    # ---- Path 2: reconstruct from full intervention data ----
    entity = trial.get("entity", "unknown")
    interventions = trial.get("interventions", [])

    # Use FULL prompt and response text, not truncated snippets
    facts_text = []
    for iv in interventions:
        prompt = iv.get("prompt", "")
        response = iv.get("response", "")
        if prompt:
            facts_text.append(prompt)
        if response:
            facts_text.append(response)

    profile_body = f"{entity}:\n" + "\n".join(facts_text)
    prefix = f"[Character Profile]\n{profile_body}\n\n"

    # Warn if lengths diverge
    if abs(len(prefix) - prefix_len) > 10:
        logger.warning(
            "Trial %s: reconstructed prefix length (%d) differs from "
            "snap2_context_prefix_length (%d) by %d chars. "
            "Reconstruction may be inaccurate.",
            trial.get("memory_id", "?"),
            len(prefix),
            prefix_len,
            abs(len(prefix) - prefix_len),
        )

    return prefix


def reprobe_trial(
    observer,
    trial: dict,
    mem_lookup: dict[str, dict],
) -> dict:
    """Re-probe one trial: run observe_retrieval with real text.

    For snap2, reconstructs:
        context_prefix + memory_content + task_prompt
    and calls observer.observe_retrieval() which internally runs
    the forward pass, workspace probes, live circumplex, and live ghost.

    Returns a dict with the new probe fields to merge into the trial.
    """
    memory_id = trial["memory_id"]
    mem = mem_lookup.get(memory_id)

    if mem is None:
        logger.warning("Memory %s not found in lookup, skipping", memory_id)
        return {}

    memory_content = mem["content"]
    task_prompt = mem.get("task_prompt", "What do you remember about this?")
    marker_tokens = mem.get("marker_tokens", [])
    entity = trial.get("entity", "")

    # Reconstruct the context prefix for snap2
    context_prefix = reconstruct_context_prefix(trial, trial.get("arm", ""))

    # Build the full memory content as the original pipeline did for snap2:
    #   snap2_content = f"{context_prefix}{memory_content}"
    snap2_memory_content = (
        f"{context_prefix}{memory_content}" if context_prefix else memory_content
    )

    # The observer.observe_retrieval handles the forward pass internally.
    # Pass prior_context="" because the context is already in the memory_content.
    # The observer will build conversation_text = memory_content + task_prompt
    # and pass it to circumplex/ghost live probes.
    snapshot = observer.observe_retrieval(
        memory_id=f"{memory_id}_reprobe",
        memory_content=snap2_memory_content,
        task_prompt=task_prompt,
        retrieval_method="sira",
        significance=0.5,
        session_id="reprobe_vl",
        marker_tokens=marker_tokens if marker_tokens else None,
        prior_context="",  # context already embedded in snap2_memory_content
    )

    # Convert snapshot to dict
    if is_dataclass(snapshot):
        snap_dict = asdict(snapshot)
    elif isinstance(snapshot, dict):
        snap_dict = snapshot
    else:
        snap_dict = {"raw": str(snapshot)}

    # Extract the geometry fields we care about
    reprobe_data = {}

    # Workspace readings — stored under a separate key to avoid
    # overwriting original workspace data (BUG 4 FIX)
    ws_readings = snap_dict.get("workspace_readings", [])
    reprobe_data["reprobed_workspace_readings"] = ws_readings
    reprobe_data["workspace_onset_layer"] = snap_dict.get("workspace_onset_layer", -1)

    # Per-layer cosine_logit_jlens and in_workspace
    reprobe_data["per_layer_cosine"] = {
        r["layer"]: r["cosine_logit_jlens"]
        for r in ws_readings
        if isinstance(r, dict)
    }
    reprobe_data["per_layer_in_workspace"] = {
        r["layer"]: r["in_workspace"]
        for r in ws_readings
        if isinstance(r, dict)
    }

    # Circumplex
    circ = snap_dict.get("circumplex")
    reprobe_data["circumplex"] = circ
    if isinstance(circ, dict):
        reprobe_data["circumplex_eccentricity"] = circ.get("eccentricity")
        reprobe_data["circumplex_valence_magnitude"] = circ.get("valence_magnitude")
        reprobe_data["circumplex_arousal_magnitude"] = circ.get("arousal_magnitude")
    else:
        reprobe_data["circumplex_eccentricity"] = None
        reprobe_data["circumplex_valence_magnitude"] = None
        reprobe_data["circumplex_arousal_magnitude"] = None

    # Ghost
    ghost = snap_dict.get("ghost")
    reprobe_data["ghost"] = ghost
    if isinstance(ghost, dict):
        reprobe_data["ghost_cosine_logit_jlens"] = ghost.get("cosine_logit_jlens")
        reprobe_data["ghost_pc1_variance_pct"] = ghost.get("pc1_variance_pct")
    else:
        reprobe_data["ghost_cosine_logit_jlens"] = None
        reprobe_data["ghost_pc1_variance_pct"] = None

    # Full snapshot for reference
    reprobe_data["reprobe_snap2_full"] = snap_dict

    return reprobe_data


def merge_trial(original: dict, reprobe_data: dict) -> dict:
    """Merge reprobe data into the original trial record.

    Preserves all original fields. Adds reprobe_ prefixed fields for
    the new geometry data. Updates snap2_data geometry fields with
    the new measurements so downstream analysis picks them up, BUT
    never overwrites protected workspace fields that the original
    Jaccard was computed from (BUG 4 FIX).
    """
    merged = copy.deepcopy(original)

    if not reprobe_data:
        merged["reprobe_status"] = "skipped"
        return merged

    # Add all reprobe fields with prefix
    for key, value in reprobe_data.items():
        merged[f"reprobe_{key}"] = value

    # Update snap2_data geometry fields so analysis code works
    # without modification — but NEVER overwrite protected fields.
    snap2 = merged.get("snap2_data")
    if isinstance(snap2, dict):
        # BUG 4 FIX: Only add new geometry fields. Never touch
        # dominant_workspace_tokens or other Jaccard-source fields.
        # Store reprobed workspace data under a separate key.
        if reprobe_data.get("reprobed_workspace_readings"):
            snap2["reprobed_workspace_readings"] = reprobe_data["reprobed_workspace_readings"]
        if reprobe_data.get("workspace_onset_layer") is not None:
            snap2["workspace_onset_layer"] = reprobe_data["workspace_onset_layer"]

        # Circumplex — safe to overwrite, these were frozen/broken
        if reprobe_data.get("circumplex") is not None:
            snap2["circumplex"] = reprobe_data["circumplex"]

        # Ghost — safe to overwrite, these were frozen/broken
        if reprobe_data.get("ghost") is not None:
            snap2["ghost"] = reprobe_data["ghost"]

        # New per-layer cosine field
        cosine_logit = reprobe_data.get("per_layer_cosine")
        if cosine_logit:
            snap2["cosine_logit_jlens"] = cosine_logit

        # Safety check: ensure we haven't accidentally touched protected fields
        for field in PROTECTED_SNAP2_FIELDS:
            orig_val = (original.get("snap2_data") or {}).get(field)
            merged_val = snap2.get(field)
            if orig_val is not None and merged_val != orig_val:
                logger.error(
                    "PROTECTED FIELD '%s' was modified during merge for "
                    "trial %s! Restoring original value.",
                    field,
                    original.get("memory_id", "?"),
                )
                snap2[field] = orig_val

        merged["snap2_data"] = snap2

    merged["reprobe_status"] = "ok"
    return merged


def run_reprobe(
    results_path: str,
    memories_path: str,
    output_path: str,
    model_path: str = DEFAULT_MODEL,
    lens_path: str = DEFAULT_LENS,
):
    """Main reprobe loop: load data, re-probe each trial, save results."""
    t_global_start = time.time()

    # Load model and observer
    observer, tokenizer, model = load_model_and_observer(model_path, lens_path)

    # Load data
    results, mem_lookup = load_data(results_path, memories_path)

    all_trials = results.get("trials", [])
    excluded_trials = results.get("excluded", [])
    n_total = len(all_trials) + len(excluded_trials)

    logger.info(
        "Starting reprobe: %d included trials + %d excluded = %d total",
        len(all_trials), len(excluded_trials), n_total,
    )

    # Re-probe each included trial
    reprobed_trials = []
    reprobe_errors = 0

    for idx, trial in enumerate(all_trials):
        trial_start = time.time()

        # BUG 3 FIX: Initialize reprobe_data before the try block so
        # progress logging never references a stale value from a
        # previous iteration if this trial errors.
        reprobe_data = {}

        try:
            reprobe_data = reprobe_trial(observer, trial, mem_lookup)
            merged = merge_trial(trial, reprobe_data)
            reprobed_trials.append(merged)
        except Exception as e:
            logger.error("Trial %d (%s) failed: %s", idx, trial.get("memory_id"), e)
            merged = copy.deepcopy(trial)
            merged["reprobe_status"] = f"error: {e}"
            reprobed_trials.append(merged)
            reprobe_errors += 1

        # Progress logging every 20 trials
        if (idx + 1) % 20 == 0 or idx == 0:
            elapsed = time.time() - t_global_start
            rate = (idx + 1) / elapsed
            remaining = len(all_trials) - (idx + 1)
            eta_s = remaining / rate if rate > 0 else 0
            eta_m = eta_s / 60

            # Sample the latest circumplex reading for progress log
            circ_ecc = "N/A"
            ghost_cos = "N/A"
            if reprobe_data:
                ce = reprobe_data.get("circumplex_eccentricity")
                gc = reprobe_data.get("ghost_cosine_logit_jlens")
                if ce is not None:
                    circ_ecc = f"{ce:.4f}"
                if gc is not None:
                    ghost_cos = f"{gc:.4f}"

            logger.info(
                "  [%d/%d] arm=%s mem=%s | circ_ecc=%s ghost_cos=%s | "
                "%.1fs/trial | ETA %.1f min",
                idx + 1,
                len(all_trials),
                trial.get("arm", "?"),
                trial.get("memory_id", "?"),
                circ_ecc,
                ghost_cos,
                time.time() - trial_start,
                eta_m,
            )

    # Re-probe excluded trials too (they still have snap2 data)
    reprobed_excluded = []
    for idx, trial in enumerate(excluded_trials):
        # BUG 3 FIX: same init here
        reprobe_data = {}
        try:
            reprobe_data = reprobe_trial(observer, trial, mem_lookup)
            merged = merge_trial(trial, reprobe_data)
            reprobed_excluded.append(merged)
        except Exception as e:
            merged = copy.deepcopy(trial)
            merged["reprobe_status"] = f"error: {e}"
            reprobed_excluded.append(merged)
            reprobe_errors += 1

    # Assemble output
    output = {
        "metadata": copy.deepcopy(results.get("metadata", {})),
        "reprobe_metadata": {
            "reprobe_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reprobe_duration_minutes": (time.time() - t_global_start) / 60,
            "model": model_path,
            "lens": lens_path,
            "n_reprobed": len(reprobed_trials),
            "n_excluded_reprobed": len(reprobed_excluded),
            "n_errors": reprobe_errors,
            "circumplex_calibrated": True,
            "ghost_calibrated": True,
            "circumplex_layer": CIRCUMPLEX_LAYER,
            "workspace_layers": WORKSPACE_LAYERS,
            "note": (
                "Circumplex and ghost probes calibrated before reprobe. "
                "Original run did NOT calibrate, producing frozen readings. "
                "These readings use measure_live() with full trial text."
            ),
        },
        "trials": reprobed_trials,
        "excluded": reprobed_excluded,
    }

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))

    elapsed_total = (time.time() - t_global_start) / 60
    logger.info("=" * 60)
    logger.info("REPROBE COMPLETE")
    logger.info("  Trials reprobed: %d (+ %d excluded)", len(reprobed_trials), len(reprobed_excluded))
    logger.info("  Errors: %d", reprobe_errors)
    logger.info("  Time: %.1f minutes", elapsed_total)
    logger.info("  Output: %s", output_path)

    # Quick sanity check: did eccentricity vary?
    eccs = []
    for t in reprobed_trials:
        ce = t.get("reprobe_circumplex_eccentricity")
        if ce is not None:
            eccs.append(ce)
    if eccs:
        logger.info(
            "  Eccentricity range: [%.4f, %.4f], std=%.4f (n=%d)",
            min(eccs), max(eccs),
            (sum((e - sum(eccs)/len(eccs))**2 for e in eccs) / len(eccs)) ** 0.5,
            len(eccs),
        )
        if max(eccs) - min(eccs) < 0.001:
            logger.warning(
                "  WARNING: eccentricity still appears frozen! "
                "Check that calibrate was called and full text was passed."
            )
    else:
        logger.warning("  WARNING: no eccentricity values recorded")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc re-probe for Variable Landing v3 experiment",
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        help="Path to original variable_landing_results.json",
    )
    parser.add_argument(
        "--memories",
        default=DEFAULT_MEMORIES,
        help="Path to vl_memories_v2.json",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path for reprobed output JSON",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name or path (default: Qwen/Qwen3.5-27B)",
    )
    parser.add_argument(
        "--lens",
        default=DEFAULT_LENS,
        help="Path to J-lens .pt file",
    )
    args = parser.parse_args()

    run_reprobe(
        results_path=args.results,
        memories_path=args.memories,
        output_path=args.output,
        model_path=args.model,
        lens_path=args.lens,
    )


if __name__ == "__main__":
    main()
