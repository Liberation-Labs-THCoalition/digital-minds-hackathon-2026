"""Gurnee-currency evaluation of already-fitted MoE lenses.

Re-evaluates the "failed" lenses using Gurnee et al.'s actual metrics
instead of the output-cosine gate they were designed to fail:

  1. Ablation KL: zero the residual component along the lens direction,
     measure KL divergence in the output. Non-zero = causally relevant.
  2. Swap success: for prompt pairs, swap lens-space coordinates, check
     if top-1 predictions flip. Higher = content-bearing dimensions.

This is EVAL ONLY — no new fitting. Uses lenses already on the volume.

Usage:
    modal run modal_gurnee_eval.py
"""

import modal

app = modal.App("moe-gurnee-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "numpy", "scipy", "accelerate")
    .pip_install("jlens @ git+https://github.com/anthropics/jacobian-lens.git")
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)
MODEL_NAME = "Qwen/Qwen3-30B-A3B"

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

# Additional prompts for swap pairs (need diverse content)
SWAP_PROMPTS = [
    "The chemical formula for salt is",
    "The largest planet in our solar system is",
    "In Python, a list comprehension looks like",
    "Mozart composed his first symphony at age",
    "The boiling point of water in Celsius is",
    "The speed of light in meters per second is approximately",
    "Shakespeare wrote Romeo and",
    "The square root of 144 is",
]


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def run_gurnee_eval():
    import json
    import os
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import HFLensModel
    from jlens.hooks import ActivationRecorder

    os.environ["HF_HOME"] = "/hf-cache"
    device = "cuda"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model = HFLensModel(hf_model, tokenizer, compile=False)
    print(f"Model loaded: {model.n_layers} layers, d={model.d_model}")

    # Discover available lenses on the volume
    lens_dir = "/results"
    available_lenses = {}
    for fname in os.listdir(lens_dir):
        if fname.endswith(".pt") and "lens" in fname.lower():
            available_lenses[fname] = os.path.join(lens_dir, fname)
    print(f"Available lenses: {list(available_lenses.keys())}")

    # Also check for the weighted transport
    for fname in os.listdir(lens_dir):
        if "weighted_transport" in fname and fname.endswith(".pt"):
            available_lenses[fname] = os.path.join(lens_dir, fname)

    results = {"model": MODEL_NAME, "lenses_evaluated": [], "prompts": SANITY_PROMPTS}

    # Evaluate each lens
    for lens_name, lens_path in sorted(available_lenses.items()):
        print(f"\n{'='*60}\nEvaluating: {lens_name}\n{'='*60}")
        try:
            import jlens
            lens = jlens.JacobianLens.load(lens_path)
        except Exception as e:
            print(f"  Failed to load {lens_name}: {e}")
            results["lenses_evaluated"].append({
                "lens": lens_name, "error": str(e)})
            continue

        if not lens.source_layers:
            print(f"  No source layers in {lens_name}")
            continue

        mid_layer = lens.source_layers[len(lens.source_layers) // 2]
        print(f"  Using layer {mid_layer}")

        # ============================================================
        # 1. Ablation KL
        # ============================================================
        print(f"\n  [Ablation KL] Zeroing lens-direction component...")
        ablation_kls = []

        for prompt in SANITY_PROMPTS:
            input_ids = model.encode(prompt, max_length=64)

            with torch.no_grad():
                with ActivationRecorder(model.layers, at=[mid_layer]) as rec:
                    model.forward(input_ids)
                    h = rec.activations[mid_layer][0, -1:].detach().float()

                # Original logits at the output
                original_output = hf_model(input_ids).logits[0, -1].float()
                original_probs = torch.softmax(original_output, dim=-1)

                # Transport to get the lens direction
                transported = lens.transport(h.cpu(), mid_layer)
                lens_dir_vec = (transported - h.cpu()).squeeze(0)

                if lens_dir_vec.norm() < 1e-8:
                    ablation_kls.append(0.0)
                    continue

                lens_dir_unit = lens_dir_vec / lens_dir_vec.norm()

                # Zero the component along lens direction in the residual
                h_ablated = h.cpu().squeeze(0) - (
                    h.cpu().squeeze(0) @ lens_dir_unit) * lens_dir_unit

                # Get ablated output by unembedding
                ablated_logits = model.unembed(
                    h_ablated.unsqueeze(0).to(device)).squeeze(0).float()
                ablated_probs = torch.softmax(ablated_logits, dim=-1)

                # Also get unablated unembed for fair comparison
                unablated_logits = model.unembed(
                    h.cpu().unsqueeze(0).to(device) if h.dim() == 1
                    else h.to(device)).squeeze(0).float()
                unablated_probs = torch.softmax(unablated_logits, dim=-1)

                # KL(unablated || ablated)
                kl = torch.sum(unablated_probs * (
                    torch.log(unablated_probs + 1e-10) -
                    torch.log(ablated_probs + 1e-10))).item()
                ablation_kls.append(kl)
                print(f"    '{prompt[:35]}...' KL={kl:.4f}")

        mean_kl = float(np.mean(ablation_kls)) if ablation_kls else 0.0
        print(f"  Mean ablation KL: {mean_kl:.4f}")

        # ============================================================
        # 2. Swap success rate
        # ============================================================
        print(f"\n  [Swap Success] Testing lens-coordinate swaps...")
        all_prompts = SANITY_PROMPTS + SWAP_PROMPTS
        swap_successes = 0
        swap_total = 0

        # Get activations + original top-1 for all prompts
        prompt_data = []
        for prompt in all_prompts:
            input_ids = model.encode(prompt, max_length=64)
            with torch.no_grad():
                with ActivationRecorder(model.layers, at=[mid_layer]) as rec:
                    model.forward(input_ids)
                    h = rec.activations[mid_layer][0, -1:].detach().float()

                transported = lens.transport(h.cpu(), mid_layer)
                jl_logits = model.unembed(
                    transported.to(device)).squeeze(0).float()
                top1 = torch.argmax(jl_logits).item()
                prompt_data.append({
                    "prompt": prompt, "h": h.cpu(), "transported": transported,
                    "top1": top1,
                })

        # Swap pairs: for each pair, swap their lens-space coordinates
        for i in range(len(prompt_data)):
            for j in range(i + 1, min(i + 4, len(prompt_data))):
                pd_i, pd_j = prompt_data[i], prompt_data[j]
                if pd_i["top1"] == pd_j["top1"]:
                    continue

                # Lens-space coords: difference from original to transported
                delta_i = pd_i["transported"] - pd_i["h"]
                delta_j = pd_j["transported"] - pd_j["h"]

                # Swap: apply j's delta to i's hidden state and vice versa
                swapped_i = pd_i["h"] + delta_j
                swapped_j = pd_j["h"] + delta_i

                swapped_i_logits = model.unembed(
                    swapped_i.to(device)).squeeze(0).float()
                swapped_j_logits = model.unembed(
                    swapped_j.to(device)).squeeze(0).float()

                new_top1_i = torch.argmax(swapped_i_logits).item()
                new_top1_j = torch.argmax(swapped_j_logits).item()

                # Success = top-1 flipped (changed from original)
                if new_top1_i != pd_i["top1"]:
                    swap_successes += 1
                if new_top1_j != pd_j["top1"]:
                    swap_successes += 1
                swap_total += 2

        swap_rate = swap_successes / max(swap_total, 1)
        print(f"  Swap success rate: {swap_successes}/{swap_total} = {swap_rate:.3f}")

        # NOTE: an `original gate` block stood here. It constructed ground truth as
        # `prompt + " the"` and read back the appended token, so it scored whether the
        # token " the" was in the lens top-10 -- a constant, not the model's
        # continuation. Removed 2026-08-16. Superseded by modal_onset_sweep.py, which
        # takes ground truth from model.generate(max_new_tokens=1).

        lens_result = {
            "lens": lens_name,
            "layer": mid_layer,
            "ablation_kl_mean": round(mean_kl, 6),
            "ablation_kl_per_prompt": [round(k, 6) for k in ablation_kls],
            "swap_success_rate": round(swap_rate, 4),
            "swap_successes": swap_successes,
            "swap_total": swap_total,
        }
        results["lenses_evaluated"].append(lens_result)
        print(f"\n  Summary: KL={mean_kl:.4f}, swap={swap_rate:.3f}")

    # Save results
    out_path = "/results/gurnee_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nResults saved to {out_path}")
    return results


@app.local_entrypoint()
def main():
    result = run_gurnee_eval.remote()
    import json
    print(json.dumps(result, indent=2, default=str))
