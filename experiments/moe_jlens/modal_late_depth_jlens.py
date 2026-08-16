"""Late-depth J-lens fit on MoE — the experiment that might crack it.

The onset sweep showed Qwen3-30B-A3B becomes logit-lens readable at L41+.
The J-lens was previously fitted at L24 where nothing is readable. This
fits a STANDARD (unconditioned) J-lens at the readable regime.

If J-lens outperforms logit lens at L41-L44: depth was the problem, not MoE.
If J-lens matches or underperforms: the fitting adds nothing even where
the residual is readable.

Usage:
    modal run modal_late_depth_jlens.py
"""

import modal

app = modal.App("moe-late-depth-jlens")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "numpy", "scipy", "accelerate")
    .pip_install("jlens @ git+https://github.com/anthropics/jacobian-lens.git")
    .pip_install("datasets", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("moe-jlens-hf-cache", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
TARGET_LAYERS = [41, 42, 44]

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


@app.function(
    image=image,
    gpu="H100",
    timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def fit_late_depth():
    import json, time
    import torch
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import HFLensModel
    from jlens.examples import load_wikitext_prompts

    t0 = time.time()
    print(f"=== LATE-DEPTH J-LENS FIT ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Target layers: {TARGET_LAYERS}")

    # Load model
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir="/hf-cache")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/hf-cache")
    model = HFLensModel(hf_model, tokenizer, compile=False)
    print(f"  Model loaded: {model.n_layers} layers, d={model.d_model}")
    print(f"  GPU: {torch.cuda.get_device_name()} ({torch.cuda.memory_allocated()/1e9:.1f}GB)")

    # Precompute ground truth via model.generate (NOT " the" — that bug is dead)
    print(f"\n--- Ground truth (model.generate) ---")
    actual_nexts = {}
    device = next(hf_model.parameters()).device
    for prompt in SANITY_PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = hf_model.generate(input_ids, max_new_tokens=1, do_sample=False)
        actual_nexts[prompt] = gen[0, input_ids.shape[1]].item()
        actual_word = tokenizer.decode([actual_nexts[prompt]]).strip()
        print(f"  '{prompt[:40]}...' -> '{actual_word}'")

    # Load fitting prompts
    fit_prompts = load_wikitext_prompts(n_prompts=200)
    print(f"\n--- Fitting with {len(fit_prompts)} wikitext prompts ---")

    results = {
        "model": MODEL_NAME,
        "target_layers": TARGET_LAYERS,
        "n_fit_prompts": len(fit_prompts),
        "per_layer": {},
    }

    for layer in TARGET_LAYERS:
        print(f"\n{'='*60}")
        print(f"LAYER {layer} ({layer/model.n_layers*100:.0f}% depth)")
        print(f"{'='*60}")

        # === PHASE 1: Fit standard J-lens at this layer ===
        print(f"  Fitting standard J-lens...")
        fit_t0 = time.time()
        try:
            torch.cuda.empty_cache()
            lens = jlens.fit(model, fit_prompts,
                             source_layers=[layer],
                             dim_batch=4, max_seq_len=128)
            fit_time = time.time() - fit_t0
            lens_path = f"/results/late_depth_lens_L{layer}.pt"
            lens.save(lens_path)
            RESULTS_VOL.commit()
            print(f"  Fitted in {fit_time:.0f}s ({fit_time/60:.1f}min). Saved.")
        except Exception as e:
            fit_time = time.time() - fit_t0
            print(f"  FIT FAILED after {fit_time:.0f}s: {e}")
            results["per_layer"][f"L{layer}"] = {
                "fit_status": "failed",
                "fit_time": fit_time,
                "error": str(e),
            }
            continue

        # === PHASE 2: Sanity gate — J-lens vs logit lens ===
        print(f"  Running sanity gate...")
        jlens_correct = 0
        logit_correct = 0
        gate_details = []

        for prompt in SANITY_PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            actual_next = actual_nexts[prompt]

            # Hook to capture residual
            captured = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured["h"] = output[0][:, -1:, :].detach()
                else:
                    captured["h"] = output[:, -1:, :].detach()

            hook = hf_model.model.layers[layer].register_forward_hook(hook_fn)
            with torch.no_grad():
                hf_model(input_ids)
            hook.remove()

            if "h" not in captured:
                continue

            h = captured["h"].float()  # [1, 1, d_model]

            # Logit lens: direct unembed
            h_cast = h.to(hf_model.lm_head.weight.dtype)
            logit_logits = hf_model.lm_head(h_cast).squeeze(0).squeeze(0).float()
            logit_top10 = torch.topk(logit_logits, 10).indices.tolist()
            logit_hit = actual_next in logit_top10

            # J-lens: transport then unembed
            h_squeezed = h.squeeze(0).squeeze(0)  # [d_model]
            transported = lens.transport(h_squeezed, layer)
            t_cast = transported.unsqueeze(0).to(hf_model.lm_head.weight.dtype)
            jlens_logits = hf_model.lm_head(t_cast).squeeze(0).float()
            jlens_top10 = torch.topk(jlens_logits, 10).indices.tolist()
            jlens_hit = actual_next in jlens_top10

            if logit_hit:
                logit_correct += 1
            if jlens_hit:
                jlens_correct += 1

            actual_word = tokenizer.decode([actual_next]).strip()
            jl_top3 = [tokenizer.decode([t]).strip() for t in jlens_top10[:3]]
            ll_top3 = [tokenizer.decode([t]).strip() for t in logit_top10[:3]]
            print(f"    '{prompt[:30]}' actual='{actual_word}' "
                  f"jlens={'HIT' if jlens_hit else 'miss'}({jl_top3}) "
                  f"logit={'HIT' if logit_hit else 'miss'}({ll_top3})")

            gate_details.append({
                "prompt": prompt[:60],
                "actual": actual_word,
                "jlens_hit": jlens_hit,
                "logit_hit": logit_hit,
                "jlens_top3": jl_top3,
                "logit_top3": ll_top3,
            })

        n_prompts = len(SANITY_PROMPTS)
        jlens_acc = jlens_correct / n_prompts
        logit_acc = logit_correct / n_prompts
        improvement = jlens_acc - logit_acc

        results["per_layer"][f"L{layer}"] = {
            "fit_status": "ok",
            "fit_time": fit_time,
            "jlens_accuracy": jlens_acc,
            "jlens_correct": jlens_correct,
            "logit_accuracy": logit_acc,
            "logit_correct": logit_correct,
            "improvement": improvement,
            "n_prompts": n_prompts,
            "details": gate_details,
            "verdict": (
                "JLENS_WINS" if jlens_acc > logit_acc + 0.05 else
                "MATCH" if abs(jlens_acc - logit_acc) <= 0.05 else
                "LOGIT_WINS"
            ),
        }

        print(f"\n  L{layer} RESULTS:")
        print(f"    J-lens:     {jlens_correct}/{n_prompts} = {jlens_acc:.1%}")
        print(f"    Logit lens: {logit_correct}/{n_prompts} = {logit_acc:.1%}")
        print(f"    Improvement: {improvement:+.1%}")
        print(f"    Verdict: {results['per_layer'][f'L{layer}']['verdict']}")

    # Overall summary
    results["elapsed_s"] = round(time.time() - t0, 1)

    any_wins = any(v.get("verdict") == "JLENS_WINS" for v in results["per_layer"].values())
    results["overall_verdict"] = (
        "CRACKED" if any_wins else
        "NEGATIVE — J-lens adds nothing even at readable depth"
    )

    print(f"\n{'='*60}")
    print(f"OVERALL: {results['overall_verdict']}")
    print(f"Total time: {results['elapsed_s']/60:.1f} min")
    print(f"{'='*60}")

    # Save
    with open("/results/late_depth_jlens_results.json", "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()

    return results


@app.local_entrypoint()
def main():
    result = fit_late_depth.remote()
    import json
    print(json.dumps(result, indent=2, default=str))
