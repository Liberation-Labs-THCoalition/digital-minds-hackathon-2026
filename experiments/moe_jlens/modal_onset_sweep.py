"""Dual-model logit lens onset sweep — the MoE paper's central figure.

Two models (MoE vs dense) × 7 relative depths × 8 prompts.
Shows where the residual stream becomes linearly decodable on each
architecture. The onset SHIFT between MoE and dense is the finding.

Ground truth: model.generate(max_new_tokens=1), NOT hardcoded tokens.

Usage:
    modal run modal_onset_sweep.py
"""

import modal

app = modal.App("moe-onset-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "accelerate", "bitsandbytes")
    .pip_install("hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("moe-jlens-hf-cache", create_if_missing=True)

MODELS = {
    "Qwen/Qwen3-30B-A3B": {
        "type": "moe",
        "n_layers": 48,
        "layers": [12, 24, 31, 36, 41, 44, 47],
        "pcts":   [25, 50, 65, 75, 85, 92, 98],
    },
    "Qwen/Qwen3-32B": {
        "type": "dense",
        "n_layers": 64,
        "layers": [16, 32, 42, 48, 54, 59, 63],
        "pcts":   [25, 50, 66, 75, 84, 92, 98],
    },
}

SANITY_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return",
    "Water is composed of hydrogen and",
    "To be or not to be, that is the",
    "The mitochondria is the powerhouse of the",
    "In 1969, Neil Armstrong became the first person to walk on the",
    "E = mc^2 was formulated by Albert",
    "SELECT name FROM users WHERE age >",
]


def run_model(model_name, config, tokenizer_cls, model_cls, torch):
    """Run logit lens sweep on one model, return results dict."""
    import gc
    import time

    t0 = time.time()
    layers = config["layers"]
    pcts = config["pcts"]

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({config['type']}, {config['n_layers']} layers)")
    print(f"Layers: {layers} ({pcts}%)")
    print(f"{'='*60}")

    print(f"[LOAD] {model_name}...")
    tokenizer = tokenizer_cls.from_pretrained(model_name, cache_dir="/hf-cache")
    try:
        model = model_cls.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            cache_dir="/hf-cache",
        )
    except Exception:
        print(f"[LOAD] bf16 failed, trying 4-bit...")
        from transformers import BitsAndBytesConfig
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = model_cls.from_pretrained(
            model_name, quantization_config=qconfig, device_map="auto",
            cache_dir="/hf-cache",
        )
    model.eval()
    device = next(model.parameters()).device
    print(f"[LOAD] done in {time.time()-t0:.0f}s")

    # Ground truth: actual next tokens via greedy generation
    actual_nexts = {}
    for prompt in SANITY_PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        actual_nexts[prompt] = gen[0, input_ids.shape[1]].item()
        word = tokenizer.decode([actual_nexts[prompt]]).strip()
        print(f"  GT: '{prompt[:40]}...' -> '{word}'")

    model_results = {
        "model": model_name,
        "type": config["type"],
        "n_layers": config["n_layers"],
        "curve": [],
    }

    for layer, pct in zip(layers, pcts):
        correct = 0
        total = 0
        details = []

        for prompt in SANITY_PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            actual_next = actual_nexts[prompt]

            captured = {}
            def hook_fn(module, input, output, _captured=captured):
                if isinstance(output, tuple):
                    _captured["h"] = output[0][:, -1:, :].detach().float()
                else:
                    _captured["h"] = output[:, -1:, :].detach().float()

            hook = model.model.layers[layer].register_forward_hook(hook_fn)
            with torch.no_grad():
                model(input_ids)
            hook.remove()

            if "h" not in captured:
                continue

            h = captured["h"].to(model.lm_head.weight.dtype)
            logits = model.lm_head(h).squeeze(0).squeeze(0).float()
            top10 = torch.topk(logits, 10).indices.tolist()

            hit = actual_next in top10
            if hit:
                correct += 1
            total += 1

            actual_word = tokenizer.decode([actual_next]).strip()
            top3 = [tokenizer.decode([t]).strip() for t in top10[:3]]
            details.append({
                "prompt": prompt[:60],
                "actual": actual_word,
                "hit": hit,
                "top3": top3,
            })

        accuracy = correct / max(total, 1)
        model_results["curve"].append({
            "layer": layer,
            "depth_pct": pct,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "details": details,
        })
        print(f"  L{layer} ({pct}%): {correct}/{total} = {accuracy:.0%}")

    # Find onset (first layer > 50%)
    onset = None
    for point in model_results["curve"]:
        if point["accuracy"] > 0.5:
            onset = point["depth_pct"]
            break
    model_results["onset_pct"] = onset
    print(f"  ONSET: {onset}% depth" if onset else "  ONSET: not reached")

    # Free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return model_results


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_onset_sweep():
    import json
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    all_results = {"models": {}, "prompts": SANITY_PROMPTS}

    for model_name, config in MODELS.items():
        result = run_model(model_name, config, AutoTokenizer, AutoModelForCausalLM, torch)
        all_results["models"][config["type"]] = result

    # Summary
    moe = all_results["models"].get("moe", {})
    dense = all_results["models"].get("dense", {})
    moe_onset = moe.get("onset_pct")
    dense_onset = dense.get("onset_pct")

    all_results["summary"] = {
        "moe_onset_pct": moe_onset,
        "dense_onset_pct": dense_onset,
        "onset_shift": (moe_onset - dense_onset) if moe_onset and dense_onset else None,
        "elapsed_s": round(time.time() - t0, 1),
    }

    print(f"\n{'='*60}")
    print(f"ONSET SWEEP COMPLETE")
    print(f"  MoE onset:   {moe_onset}% depth")
    print(f"  Dense onset:  {dense_onset}% depth")
    if moe_onset and dense_onset:
        print(f"  Shift:        +{moe_onset - dense_onset} percentage points")
    print(f"  Elapsed:      {all_results['summary']['elapsed_s']}s")

    # Print curves side by side
    print(f"\n  {'Depth':>6}  {'MoE':>6}  {'Dense':>6}")
    moe_curve = {p["depth_pct"]: p["accuracy"] for p in moe.get("curve", [])}
    dense_curve = {p["depth_pct"]: p["accuracy"] for p in dense.get("curve", [])}
    all_pcts = sorted(set(list(moe_curve.keys()) + list(dense_curve.keys())))
    for pct in all_pcts:
        m = f"{moe_curve.get(pct, -1):.0%}" if pct in moe_curve else "  -"
        d = f"{dense_curve.get(pct, -1):.0%}" if pct in dense_curve else "  -"
        print(f"  {pct:>5}%  {m:>6}  {d:>6}")
    print(f"{'='*60}")

    out_path = "/results/onset_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    RESULTS_VOL.commit()
    print(f"Saved to {out_path}")

    return all_results


@app.local_entrypoint()
def main():
    result = run_onset_sweep.remote()
    import json
    print(json.dumps(result, indent=2, default=str))
