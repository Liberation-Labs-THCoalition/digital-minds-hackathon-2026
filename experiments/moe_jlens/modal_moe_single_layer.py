"""Single-layer conditioned fit — one layer per Modal run, 4h timeout.

Usage:
    modal run modal_moe_single_layer.py --layer 12
    modal run modal_moe_single_layer.py --layer 24
    modal run modal_moe_single_layer.py --layer 36
"""

import modal

app = modal.App("moe-jlens-layer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "numpy", "scikit-learn", "scipy", "accelerate")
    .pip_install("jlens @ git+https://github.com/anthropics/jacobian-lens.git")
    .pip_install("datasets")
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)
MODEL_NAME = "Qwen/Qwen3-30B-A3B"


@app.function(image=image, gpu="H100", timeout=14400, volumes={"/results": RESULTS_VOL})
def fit_single_layer(target_layer: int = 24, n_fit: int = 672):
    import json, time, pickle, os
    import torch
    import numpy as np
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import HFLensModel
    from jlens.examples import load_wikitext_prompts

    print(f"=== CONDITIONED FIT: Layer {target_layer} ===")

    # Load model
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HFLensModel(hf_model, tokenizer, compile=False)
    print(f"  Model: {model.n_layers} layers, d={model.d_model}")
    print(f"  GPU: {torch.cuda.get_device_name()} ({torch.cuda.memory_allocated()/1e9:.1f}GB)")

    # Load routing data from Stage 1
    with open("/results/chunked_routing.json") as f:
        routing_data = json.load(f)

    layer_str = str(target_layer)
    if layer_str not in routing_data["clusters"]:
        print(f"  ERROR: Layer {target_layer} not in routing data")
        return {"error": f"Layer {target_layer} not found"}

    cluster_info = routing_data["clusters"][layer_str]
    labels = np.array(cluster_info["labels"])
    unique_labels = np.unique(labels)
    sizes = cluster_info["sizes"]
    print(f"  Clusters: k={cluster_info['k']}, sizes={sizes}")

    # Load prompts
    fit_prompts = load_wikitext_prompts(n_prompts=n_fit)
    print(f"  Fitting prompts: {len(fit_prompts)}")

    # Fit conditioned lenses — one per cluster
    results = {}
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_prompts = [p for p, m in zip(fit_prompts, mask) if m]
        n = len(cluster_prompts)

        if n < 20:
            print(f"\n  Cluster {cluster_id}: {n} prompts (skipping, < 20)")
            results[int(cluster_id)] = {"n_prompts": n, "status": "skipped"}
            continue

        print(f"\n  Cluster {cluster_id}: fitting on {n} prompts...", flush=True)
        t0 = time.time()
        try:
            torch.cuda.empty_cache()
            lens = jlens.fit(model, cluster_prompts,
                             source_layers=[target_layer],
                             dim_batch=4, max_seq_len=128)
            elapsed = time.time() - t0
            lens_path = f"/results/conditioned_lens_L{target_layer}_C{cluster_id}.pt"
            lens.save(lens_path)
            RESULTS_VOL.commit()
            print(f"    Done in {elapsed:.0f}s ({elapsed/60:.1f}min). Saved + committed.")
            results[int(cluster_id)] = {
                "n_prompts": n, "time": elapsed, "path": lens_path, "status": "ok"}
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    FAILED after {elapsed:.0f}s: {e}")
            results[int(cluster_id)] = {
                "n_prompts": n, "time": elapsed, "error": str(e), "status": "failed"}

    # Save manifest for this layer
    manifest = {
        "layer": target_layer,
        "clusters": results,
        "total_time": sum(r.get("time", 0) for r in results.values()),
    }
    manifest_path = f"/results/conditioned_manifest_L{target_layer}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    RESULTS_VOL.commit()

    print(f"\n  Layer {target_layer} complete. {sum(1 for r in results.values() if r['status']=='ok')}/{len(results)} clusters fitted.")
    print(f"  Total time: {manifest['total_time']/60:.1f} min")
    return manifest


@app.local_entrypoint()
def main(layer: int = 24):
    result = fit_single_layer.remote(target_layer=layer)
    print(f"\nLayer {layer} result: {result}")
