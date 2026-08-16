"""Logit lens control — Lyra's missing baseline for the MoE sanity gate.

Does the PLAIN logit lens (unembed the mid-layer residual directly, no
transport, no Jacobian, no fitting) pass the same sanity gate that the
J-lens fails at 1/8?

If logit lens also fails: the residual stream is not linearly decodable
at these layers by ANY method. J-lens failure is about the representation,
not the Jacobian.

If logit lens passes: the information IS there but the J-lens transport
destroys it. The problem is the fitting, not the architecture.

Usage:
    modal run modal_logit_lens_control.py
"""

import modal

app = modal.App("moe-logit-lens-control")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "accelerate")
    .pip_install("hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("moe-jlens-hf-cache", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
TARGET_LAYERS = [12, 24, 36, 42, 47]

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
    timeout=1800,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_logit_lens_control():
    import json
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print(f"[LOAD] {MODEL_NAME} in bf16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/hf-cache")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir="/hf-cache",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[LOAD] done in {time.time()-t0:.0f}s, device={device}")

    n_layers = model.config.num_hidden_layers
    print(f"[INFO] Model has {n_layers} layers, all MoE (no dense FFN fallback)")
    print(f"[INFO] Testing logit lens at layers {TARGET_LAYERS}")

    results = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "target_layers": TARGET_LAYERS,
        "per_layer": {},
        "per_prompt": [],
    }

    # Precompute actual next tokens via greedy generation (the REAL ground truth)
    actual_nexts = {}
    for prompt in SANITY_PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        actual_nexts[prompt] = gen[0, input_ids.shape[1]].item()
        actual_word = tokenizer.decode([actual_nexts[prompt]]).strip()
        print(f"  Ground truth: '{prompt[:40]}...' -> '{actual_word}'")

    for layer in TARGET_LAYERS:
        correct = 0
        total = 0
        layer_details = []

        for prompt in SANITY_PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            actual_next = actual_nexts[prompt]

            # Hook to capture residual at target layer
            captured = {}
            def hook_fn(module, input, output):
                # Qwen3MoE decoder layer returns (hidden_states, ...) tuple
                if isinstance(output, tuple):
                    captured["h"] = output[0][:, -1:, :].detach().float()
                else:
                    captured["h"] = output[:, -1:, :].detach().float()

            hook = model.model.layers[layer].register_forward_hook(hook_fn)
            with torch.no_grad():
                model(input_ids)
            hook.remove()

            if "h" not in captured:
                print(f"  WARNING: no activation captured at layer {layer}")
                continue

            h = captured["h"]  # [1, 1, d_model]

            # PLAIN LOGIT LENS: unembed directly, no transport
            # Cast to model dtype for lm_head, then back to float for topk
            h_cast = h.to(model.lm_head.weight.dtype)
            logits = model.lm_head(h_cast).squeeze(0).squeeze(0).float()  # [vocab]
            top10 = torch.topk(logits, 10).indices.tolist()

            hit = actual_next in top10
            if hit:
                correct += 1
            total += 1

            actual_word = tokenizer.decode([actual_next]).strip()
            top5_words = ", ".join(tokenizer.decode([t]).strip() for t in top10[:5])
            status = "HIT" if hit else "miss"
            print(f"  L{layer} '{prompt[:40]}' -> [{top5_words}] actual='{actual_word}' {status}")

            layer_details.append({
                "prompt": prompt[:60],
                "actual_token": actual_word,
                "hit": hit,
                "top5": [tokenizer.decode([t]).strip() for t in top10[:5]],
            })

        accuracy = correct / max(total, 1)
        results["per_layer"][f"L{layer}"] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "details": layer_details,
        }
        print(f"  L{layer} GATE: {correct}/{total} = {accuracy:.1%}")

    # Overall
    all_correct = sum(v["correct"] for v in results["per_layer"].values())
    all_total = sum(v["total"] for v in results["per_layer"].values())
    results["overall_accuracy"] = all_correct / max(all_total, 1)
    results["overall"] = f"{all_correct}/{all_total}"
    results["elapsed_s"] = round(time.time() - t0, 1)

    # Compare to J-lens gate
    results["comparison"] = {
        "jlens_gate": "1/8 = 12.5% (GATE_FAIL)",
        "logit_lens_gate": f"{all_correct}/{all_total} = {results['overall_accuracy']:.1%}",
        "interpretation": (
            "logit_lens_also_fails" if results["overall_accuracy"] < 0.20
            else "logit_lens_passes"
        ),
    }

    print(f"\n{'='*60}")
    print(f"LOGIT LENS CONTROL: {all_correct}/{all_total} = {results['overall_accuracy']:.1%}")
    print(f"J-LENS GATE:        1/8 = 12.5%")
    if results["overall_accuracy"] < 0.20:
        print("INTERPRETATION: Residual NOT linearly decodable at these layers.")
        print("J-lens failure is about the representation, not the Jacobian.")
    else:
        print("INTERPRETATION: Information IS there. J-lens transport destroys it.")
        print("Problem is the fitting, not the architecture.")
    print(f"{'='*60}")

    out_path = "/results/logit_lens_control_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    RESULTS_VOL.commit()
    print(f"Saved to {out_path}")

    return results


@app.local_entrypoint()
def main():
    result = run_logit_lens_control.remote()
    import json
    print(json.dumps(result, indent=2, default=str))
