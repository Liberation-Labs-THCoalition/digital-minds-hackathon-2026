"""Tuned lens on MoE — learns a per-layer affine readout.

The plain logit lens applies the raw unembedding matrix to mid-layer
residuals and gets 0-25% on Qwen3-30B-A3B at layers 12-36. The tuned
lens (Belrose et al. 2023, arXiv:2303.08112) learns a per-layer affine
transform (linear + bias) mapping residuals → logits. If it recovers
readability at mid-depth, the information IS there but needs a learned
basis — a real architectural finding.

Includes a shuffled-label control at each layer: if the random-fit
matches the real fit, the transform is memorizing, not learning.

Usage:
    modal run modal_tuned_lens.py
"""

import modal

app = modal.App("moe-tuned-lens")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "numpy", "scipy", "accelerate")
    .pip_install("datasets", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("moe-jlens-hf-cache", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
TARGET_LAYERS = [12, 24, 31, 36, 41, 44]
N_FIT = 500
N_EVAL = 8

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
def run_tuned_lens():
    import json, time
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    t0 = time.time()
    print("=== TUNED LENS ON MoE ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {TARGET_LAYERS}")
    print(f"Fit prompts: {N_FIT}, Eval prompts: {N_EVAL}")

    # Load model
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir="/hf-cache")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/hf-cache")
    device = next(hf_model.parameters()).device
    d_model = hf_model.config.hidden_size
    vocab_size = hf_model.config.vocab_size
    n_layers = hf_model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers, d={d_model}, vocab={vocab_size}")
    print(f"  GPU: {torch.cuda.get_device_name()} ({torch.cuda.memory_allocated()/1e9:.1f}GB)")

    # Get ground truth for eval prompts
    print("\n--- Ground truth (model.generate) ---")
    actual_nexts = {}
    for prompt in SANITY_PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = hf_model.generate(input_ids, max_new_tokens=1, do_sample=False)
        tok_id = gen[0, input_ids.shape[1]].item()
        actual_nexts[prompt] = tok_id
        print(f"  '{prompt[:40]}...' -> '{tokenizer.decode([tok_id]).strip()}'")

    # Load fitting corpus
    print(f"\n--- Loading {N_FIT} wikitext prompts ---")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train",
                      cache_dir="/hf-cache")
    fit_texts = []
    for row in ds:
        text = row["text"].strip()
        if len(text) > 100:
            fit_texts.append(text[:256])
        if len(fit_texts) >= N_FIT:
            break
    print(f"  Loaded {len(fit_texts)} prompts")

    # Hook to capture residual stream at each layer
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            captured[layer_idx] = h.detach()
        return hook_fn

    # Collect (residual, target_logits) pairs for fitting
    print(f"\n--- Collecting activations ---")
    layer_data = {l: {"residuals": [], "logits": []} for l in TARGET_LAYERS}

    # Register hooks on transformer layers
    layers_module = hf_model.model.layers
    hooks = []
    for layer_idx in TARGET_LAYERS:
        h = layers_module[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    batch_size = 4
    for i in range(0, len(fit_texts), batch_size):
        batch = fit_texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = hf_model(**inputs)
            final_logits = outputs.logits  # [B, seq, vocab]

        for layer_idx in TARGET_LAYERS:
            h = captured[layer_idx]  # [B, seq, d_model]
            # Take last non-padding position per sample
            for b in range(h.shape[0]):
                mask = inputs.attention_mask[b]
                last_pos = mask.sum().item() - 1
                layer_data[layer_idx]["residuals"].append(
                    h[b, last_pos].float().cpu())
                layer_data[layer_idx]["logits"].append(
                    final_logits[b, last_pos].float().cpu())

        if (i // batch_size) % 25 == 0:
            print(f"  {i+batch_size}/{len(fit_texts)}")

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"  Collected {len(layer_data[TARGET_LAYERS[0]]['residuals'])} samples per layer")

    # Fit tuned lens at each layer + evaluate
    results = {
        "model": MODEL_NAME,
        "method": "tuned_lens_ridge",
        "n_fit": len(fit_texts),
        "n_eval": N_EVAL,
        "d_model": d_model,
        "per_layer": {},
    }

    for layer_idx in TARGET_LAYERS:
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx} ({layer_idx/n_layers*100:.0f}% depth)")
        print(f"{'='*60}")

        R = torch.stack(layer_data[layer_idx]["residuals"])  # [N, d_model]
        L = torch.stack(layer_data[layer_idx]["logits"])    # [N, vocab]

        # We can't do full vocab regression (vocab=151k is too big).
        # Instead: project logits to top-k subspace, or just check if
        # the learned transform's top-10 includes the actual next token.
        #
        # Approach: learn a low-rank affine mapping R -> L via SVD of R,
        # then project. Equivalent to Ridge regression in the principal
        # component basis.

        # SVD of residuals for dimensionality reduction
        print(f"  SVD of residuals ({R.shape})...")
        U, S, Vt = torch.linalg.svd(R - R.mean(0), full_matrices=False)
        # Keep top-k components (enough to capture structure)
        k = min(256, R.shape[0] - 1, d_model)
        R_reduced = U[:, :k] * S[:k]  # [N, k]
        print(f"  Reduced to {k} components, explained var: {(S[:k]**2).sum()/(S**2).sum():.3f}")

        # Ridge regression: R_reduced -> L
        # L_hat = R_reduced @ W + b
        # This is a k -> vocab linear map
        print(f"  Fitting Ridge regression ({k} -> {vocab_size})...")
        ridge_alpha = 1.0
        R_aug = torch.cat([R_reduced, torch.ones(R_reduced.shape[0], 1)], dim=1)
        # Solve: (R^T R + alpha I) W = R^T L
        RtR = R_aug.T @ R_aug
        RtR.diagonal().add_(ridge_alpha)
        RtL = R_aug.T @ L
        W = torch.linalg.solve(RtR, RtL)  # [k+1, vocab]
        print(f"  Fit complete. W shape: {W.shape}")

        # Also fit shuffled control
        print(f"  Fitting shuffled control...")
        perm = torch.randperm(L.shape[0])
        L_shuffled = L[perm]
        RtL_shuf = R_aug.T @ L_shuffled
        W_shuffled = torch.linalg.solve(RtR, RtL_shuf)

        # Evaluate on held-out prompts
        print(f"  Evaluating on {N_EVAL} prompts...")
        tuned_correct = 0
        logit_correct = 0
        shuffled_correct = 0
        details = []

        for prompt in SANITY_PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            captured.clear()

            # Re-register hook for this layer only
            hook = layers_module[layer_idx].register_forward_hook(make_hook(layer_idx))
            with torch.no_grad():
                out = hf_model(input_ids)
            hook.remove()

            h = captured[layer_idx][0, -1].float().cpu()  # [d_model]

            # Project through SVD basis
            h_centered = h - R.mean(0)
            h_reduced = (h_centered @ Vt[:k].T)  # [k]
            h_aug = torch.cat([h_reduced, torch.ones(1)])  # [k+1]

            # Tuned lens prediction
            tuned_logits = h_aug @ W  # [vocab]
            tuned_top10 = tuned_logits.topk(10).indices.tolist()

            # Plain logit lens prediction
            logit_logits = hf_model.lm_head(
                hf_model.model.norm(h.to(device).to(torch.bfloat16))
            ).float().cpu()
            logit_top10 = logit_logits.topk(10).indices.tolist()

            # Shuffled control prediction
            shuf_logits = h_aug @ W_shuffled
            shuf_top10 = shuf_logits.topk(10).indices.tolist()

            actual_id = actual_nexts[prompt]
            actual_word = tokenizer.decode([actual_id]).strip()

            tuned_hit = actual_id in tuned_top10
            logit_hit = actual_id in logit_top10
            shuf_hit = actual_id in shuf_top10

            if tuned_hit:
                tuned_correct += 1
            if logit_hit:
                logit_correct += 1
            if shuf_hit:
                shuffled_correct += 1

            details.append({
                "prompt": prompt,
                "actual": actual_word,
                "tuned_hit": tuned_hit,
                "logit_hit": logit_hit,
                "shuffled_hit": shuf_hit,
                "tuned_top3": [tokenizer.decode([t]).strip() for t in tuned_top10[:3]],
                "logit_top3": [tokenizer.decode([t]).strip() for t in logit_top10[:3]],
            })

            print(f"    '{prompt[:30]}...' actual='{actual_word}' "
                  f"tuned={'HIT' if tuned_hit else 'miss'} "
                  f"logit={'HIT' if logit_hit else 'miss'} "
                  f"shuf={'HIT' if shuf_hit else 'miss'}")

        tuned_acc = tuned_correct / N_EVAL
        logit_acc = logit_correct / N_EVAL
        shuf_acc = shuffled_correct / N_EVAL

        verdict = "TUNED_WINS" if tuned_acc > logit_acc and tuned_acc > shuf_acc else \
                  "LOGIT_WINS" if logit_acc > tuned_acc else \
                  "SHUFFLED_MATCHES" if shuf_acc >= tuned_acc else "MATCH"

        results["per_layer"][f"L{layer_idx}"] = {
            "depth_pct": round(layer_idx / n_layers * 100, 1),
            "tuned_accuracy": tuned_acc,
            "logit_accuracy": logit_acc,
            "shuffled_accuracy": shuf_acc,
            "tuned_correct": tuned_correct,
            "logit_correct": logit_correct,
            "shuffled_correct": shuffled_correct,
            "n_prompts": N_EVAL,
            "svd_components": k,
            "explained_variance": float((S[:k]**2).sum()/(S**2).sum()),
            "verdict": verdict,
            "details": details,
        }

        print(f"\n  RESULT L{layer_idx}: tuned={tuned_acc*100:.0f}% "
              f"logit={logit_acc*100:.0f}% shuffled={shuf_acc*100:.0f}% "
              f"-> {verdict}")

        # Free memory
        del R, L, R_reduced, R_aug, W, W_shuffled, U, S, Vt
        torch.cuda.empty_cache()

    # Overall verdict
    mid_layers = [l for l in TARGET_LAYERS if l <= 36]
    mid_tuned = [results["per_layer"][f"L{l}"]["tuned_accuracy"] for l in mid_layers]
    mid_logit = [results["per_layer"][f"L{l}"]["logit_accuracy"] for l in mid_layers]
    mid_shuf = [results["per_layer"][f"L{l}"]["shuffled_accuracy"] for l in mid_layers]

    avg_tuned = sum(mid_tuned) / len(mid_tuned) if mid_tuned else 0
    avg_logit = sum(mid_logit) / len(mid_logit) if mid_logit else 0
    avg_shuf = sum(mid_shuf) / len(mid_shuf) if mid_shuf else 0

    if avg_tuned > avg_logit + 0.1 and avg_tuned > avg_shuf + 0.1:
        overall = "CRACKED"
    elif avg_tuned > avg_logit and avg_tuned <= avg_shuf:
        overall = "SHUFFLED_MATCHES_TUNED"
    else:
        overall = "NEGATIVE"

    results["overall"] = {
        "verdict": overall,
        "mid_depth_tuned_mean": avg_tuned,
        "mid_depth_logit_mean": avg_logit,
        "mid_depth_shuffled_mean": avg_shuf,
    }
    results["elapsed_minutes"] = round((time.time() - t0) / 60, 1)

    print(f"\n{'='*60}")
    print(f"OVERALL: {overall}")
    print(f"  Mid-depth (L12-L36): tuned {avg_tuned*100:.0f}% vs "
          f"logit {avg_logit*100:.0f}% vs shuffled {avg_shuf*100:.0f}%")
    print(f"  Elapsed: {results['elapsed_minutes']:.1f} min")
    print(f"{'='*60}")

    # Save
    with open("/results/tuned_lens_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    RESULTS_VOL.commit()
    print(f"\nSaved to /results/tuned_lens_results.json")


@app.local_entrypoint()
def main():
    run_tuned_lens.remote()
