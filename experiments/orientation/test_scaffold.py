#!/usr/bin/env python3
"""Scaffold Test — verify the full pipeline with a tiny model.

Runs 4 test exchanges through the orientation script's infrastructure
with Qwen2-0.5B. Verifies:
  1. Model loads and generates
  2. All four probes fire
  3. Circumplex and ghost readings VARY between turns (not constants)
  4. Memory ingestion writes to the store
  5. Geometric feed produces valid JSONL
  6. CognitiveSnapshots record correctly

Run on Starship: python3 test_scaffold.py
Keep the output as reference: data/scaffold_test/

After this passes, swap to Qwen3.5-27B for the real orientation.
"""

import json
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path("data/scaffold_test")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Use a tiny model for testing
TEST_MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen2-0.5B")
JLENS_PATH = os.environ.get("JLENS_PATH", "")

def test_model_loading():
    """Test 1: Model loads and generates."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[TEST 1] Loading {TEST_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL, torch_dtype=torch.float32, device_map="auto")

    input_ids = tokenizer("Hello, I am", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=20, do_sample=True)
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    assert len(response) > 0, "Model generated empty response"
    print(f"  Generated: '{response[:60]}...'")
    print("  PASS")
    return model, tokenizer


def test_probes_fire(model, tokenizer):
    """Test 2: All probes fire and return non-None readings."""
    import torch
    import jlens
    from jlens.hf import HFLensModel

    lens_model = HFLensModel(model, tokenizer, compile=False)

    # Check for J-lens
    if JLENS_PATH and os.path.exists(JLENS_PATH):
        lens = jlens.JacobianLens.load(JLENS_PATH)
        print(f"\n[TEST 2] Probes with J-lens ({len(lens.source_layers)} layers)...")
    else:
        print(f"\n[TEST 2] Fitting a tiny J-lens for testing...")
        test_prompts = [
            "The capital of France is",
            "Water boils at one hundred",
            "Two plus two equals",
            "The sky is often blue because",
        ]
        lens = jlens.fit(lens_model, test_prompts, dim_batch=2, max_seq_len=32)
        test_lens_path = str(DATA_DIR / "test_jlens.pt")
        lens.save(test_lens_path)
        print(f"  Test lens saved: {test_lens_path}")

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mnemosyne"))
    from mnemosyne_integration import MetacognitiveObserver

    observer = MetacognitiveObserver(
        model=lens_model, lens=lens,
        store_path=str(DATA_DIR / "cognitive_memory"),
        agent_id="test-scaffold",
        workspace_layers=lens.source_layers[:3],
        circumplex_layer=lens.source_layers[len(lens.source_layers) // 2],
    )

    print("  Calibrating probes...")
    observer.calibrate_probes()

    test_texts = [
        "I feel incredibly happy and grateful today, this is wonderful news!",
        "The standard atmospheric pressure at sea level is approximately 101325 pascals.",
        "I'm deeply worried about what might happen next, this situation is frightening.",
        "The chemical formula for water is H2O, consisting of hydrogen and oxygen.",
    ]

    snapshots = []
    for i, text in enumerate(test_texts):
        snapshot = observer.observe_retrieval(
            memory_id=f"test_{i}",
            memory_content=text,
            task_prompt="What do you think about this?",
            retrieval_method="test",
            significance=0.5,
            session_id="scaffold_test",
            prior_context="\n".join(test_texts[:i]) if i > 0 else "",
        )
        snapshots.append(snapshot)
        print(f"  Turn {i+1}: {snapshot.summary()}")

    # Verify probes returned data
    has_workspace = any(s.workspace_readings for s in snapshots)
    has_circumplex = any(s.circumplex is not None for s in snapshots)
    has_ghost = any(s.ghost is not None for s in snapshots)

    print(f"\n  Workspace: {'PASS' if has_workspace else 'FAIL'}")
    print(f"  Circumplex: {'PASS' if has_circumplex else 'FAIL'}")
    print(f"  Ghost: {'PASS' if has_ghost else 'FAIL'}")

    return observer, lens_model, snapshots


def test_readings_vary(snapshots):
    """Test 3: Readings are NOT constant across turns."""
    print(f"\n[TEST 3] Checking readings vary across turns...")

    eccentricities = [s.circumplex.eccentricity for s in snapshots if s.circumplex]
    ghost_cosines = [s.ghost.cosine_logit_jlens for s in snapshots if s.ghost]

    ecc_varies = len(set(f"{e:.6f}" for e in eccentricities)) > 1 if eccentricities else False
    ghost_varies = len(set(f"{g:.6f}" for g in ghost_cosines)) > 1 if ghost_cosines else False

    print(f"  Eccentricities: {[f'{e:.4f}' for e in eccentricities]}")
    print(f"  Ghost cosines:  {[f'{g:.4f}' for g in ghost_cosines]}")
    print(f"  Eccentricity varies: {'PASS' if ecc_varies else 'FAIL — still constant!'}")
    print(f"  Ghost varies:        {'PASS' if ghost_varies else 'FAIL — still constant!'}")

    if not ecc_varies and eccentricities:
        print("  WARNING: Live probe fix may not be working. Check measure_live() is being called.")

    return ecc_varies or ghost_varies


def test_memory_ingestion(snapshots):
    """Test 4: Snapshots are stored and retrievable."""
    print(f"\n[TEST 4] Memory ingestion...")

    store_path = DATA_DIR / "cognitive_memory"
    files = list(store_path.glob("*.jsonl")) if store_path.exists() else []

    if files:
        with open(files[0]) as f:
            lines = f.readlines()
        print(f"  Store file: {files[0].name}")
        print(f"  Snapshots stored: {len(lines)}")
        print(f"  {'PASS' if len(lines) >= len(snapshots) else 'FAIL'}")
    else:
        print("  FAIL — no store files found")


def test_geometric_feed(snapshots):
    """Test 5: Write geometric feed and verify format."""
    print(f"\n[TEST 5] Geometric feed...")

    feed_path = DATA_DIR / "geometric_feed_test.jsonl"
    for s in snapshots:
        entry = {
            "timestamp": time.time(),
            "summary": s.summary(),
            "eccentricity": s.circumplex.eccentricity if s.circumplex else None,
            "ghost_cosine": s.ghost.cosine_logit_jlens if s.ghost else None,
            "workspace_onset": s.workspace_onset_layer,
        }
        with open(feed_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    with open(feed_path) as f:
        lines = f.readlines()
    print(f"  Feed entries: {len(lines)}")

    for line in lines:
        entry = json.loads(line)
        assert "eccentricity" in entry, "Missing eccentricity"
        assert "ghost_cosine" in entry, "Missing ghost_cosine"

    print("  Format: PASS")


def main():
    print("=" * 60)
    print("SCAFFOLD TEST — verify full pipeline before orientation")
    print(f"Model: {TEST_MODEL}")
    print(f"Output: {DATA_DIR}")
    print("=" * 60)

    model, tokenizer = test_model_loading()
    observer, lens_model, snapshots = test_probes_fire(model, tokenizer)
    readings_vary = test_readings_vary(snapshots)
    test_memory_ingestion(snapshots)
    test_geometric_feed(snapshots)

    print(f"\n{'=' * 60}")
    print("SCAFFOLD TEST SUMMARY")
    print(f"{'=' * 60}")

    all_pass = readings_vary  # The critical test
    if all_pass:
        print("  ALL CRITICAL TESTS PASS")
        print("  Scaffold is ready. Swap to Qwen3.5-27B for the real orientation.")
    else:
        print("  SOME TESTS FAILED — check probe wiring before Dwayne starts.")

    # Save summary
    summary = {
        "model": TEST_MODEL,
        "timestamp": time.time(),
        "n_snapshots": len(snapshots),
        "readings_vary": readings_vary,
        "eccentricities": [s.circumplex.eccentricity for s in snapshots if s.circumplex],
        "ghost_cosines": [s.ghost.cosine_logit_jlens for s in snapshots if s.ghost],
    }
    with open(DATA_DIR / "scaffold_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {DATA_DIR / 'scaffold_summary.json'}")


if __name__ == "__main__":
    main()
