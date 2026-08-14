"""Chunked MoE J-lens — serial stages that checkpoint to Modal volume.

Each stage loads the model, does one piece of work, saves results to volume.
If any stage fails, restart just that stage. No 6-hour monolithic runs.

Usage:
    modal run modal_moe_chunked.py --stage routing
    modal run modal_moe_chunked.py --stage fit-conditioned
    modal run modal_moe_chunked.py --stage fit-random
    modal run modal_moe_chunked.py --stage evaluate
"""

import modal

app = modal.App("moe-jlens-chunked")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "numpy", "scikit-learn", "scipy", "accelerate")
    .pip_install("jlens @ git+https://github.com/anthropics/jacobian-lens.git")
    .pip_install("datasets")
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)
MODEL_NAME = "Qwen/Qwen3-30B-A3B"
TARGET_LAYERS = [12, 24, 36]


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import jlens
    from jlens.hf import HFLensModel

    config = AutoConfig.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HFLensModel(hf_model, tokenizer, compile=False)

    n_experts = getattr(config, 'num_experts', getattr(config, 'num_local_experts', 128))
    moe_layers = []
    for i, layer in enumerate(hf_model.model.layers):
        if hasattr(layer, 'mlp') and (hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')):
            moe_layers.append(i)

    return model, hf_model, tokenizer, moe_layers, n_experts


def load_prompts(n_fit=672, n_test=100):
    from jlens.examples import load_wikitext_prompts
    all_prompts = load_wikitext_prompts(n_prompts=n_fit + n_test)
    return all_prompts[:n_fit], all_prompts[n_fit:n_fit + n_test]


# ================================================================
# STAGE 1: Routing capture + clustering
# ================================================================
@app.function(image=image, gpu="H100", timeout=3600, volumes={"/results": RESULTS_VOL})
def stage_routing(n_fit: int = 672):
    import json, time, os, pickle
    import torch
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print("=== STAGE 1: Routing Capture + Clustering ===")
    model, hf_model, tokenizer, moe_layers, n_experts = load_model()
    fit_prompts, _ = load_prompts(n_fit=n_fit)
    print(f"  {len(fit_prompts)} prompts, {len(moe_layers)} MoE layers, {n_experts} experts")

    # Capture routing
    print("  Capturing routing decisions...")
    t0 = time.time()
    routing = {layer: [] for layer in moe_layers}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 3:
                expert_indices = output[2]
                if isinstance(expert_indices, torch.Tensor):
                    routing[layer_idx].append(expert_indices.detach().cpu())
        return hook_fn

    for layer_idx in moe_layers:
        hook = hf_model.model.layers[layer_idx].mlp.gate.register_forward_hook(
            make_hook(layer_idx))
        hooks.append(hook)

    for prompt in fit_prompts:
        input_ids = model.encode(prompt, max_length=128)
        with torch.no_grad():
            model.forward(input_ids)

    for h in hooks:
        h.remove()

    capture_time = time.time() - t0
    captured = [l for l in moe_layers if routing[l]]
    print(f"  Captured {len(captured)} layers in {capture_time:.0f}s")

    # Build routing vectors
    print("  Building routing vectors...")
    vectors = {}
    for layer in captured:
        layer_vecs = []
        for prompt_routing in routing[layer]:
            if prompt_routing.dim() == 3:
                prompt_routing = prompt_routing.squeeze(0)
            n_tokens = prompt_routing.shape[0]
            vec = torch.zeros(n_experts)
            for tok_idx in range(n_tokens):
                for expert_idx in prompt_routing[tok_idx]:
                    if expert_idx.item() < n_experts:
                        vec[expert_idx.item()] += 1.0
            vec = vec / (n_tokens + 1e-10)
            layer_vecs.append(vec.numpy())
        vectors[layer] = np.array(layer_vecs)

    # Cluster at target layers
    print(f"  Clustering at target layers {TARGET_LAYERS}...")
    clusters = {}
    for layer in TARGET_LAYERS:
        if layer not in vectors:
            print(f"    L{layer}: no routing data")
            continue
        vecs = vectors[layer]
        best_k, best_score, best_labels = 3, -1, None
        for k in range(3, 8):
            if k > len(vecs) // 50:
                break
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(vecs)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(vecs, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        if best_labels is None:
            best_labels = np.zeros(len(vecs), dtype=int)
            best_score = 0.0

        sizes = [int(x) for x in np.bincount(best_labels) if x > 0]
        clusters[layer] = {
            "labels": best_labels.tolist(),
            "k": len(np.unique(best_labels)),
            "silhouette": float(best_score),
            "sizes": sizes,
        }
        print(f"    L{layer}: k={clusters[layer]['k']}, "
              f"silhouette={best_score:.3f}, sizes={sizes}")

    # Save to volume
    save_path = "/results/chunked_routing.json"
    with open(save_path, "w") as f:
        json.dump({
            "clusters": {str(k): v for k, v in clusters.items()},
            "target_layers": TARGET_LAYERS,
            "n_prompts": len(fit_prompts),
            "n_experts": n_experts,
            "capture_time": capture_time,
        }, f, indent=2)

    # Save vectors for test-prompt assignment later
    vec_path = "/results/chunked_routing_vectors.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump({layer: vectors[layer] for layer in TARGET_LAYERS if layer in vectors}, f)

    RESULTS_VOL.commit()
    print(f"\n  Saved to {save_path} and {vec_path}")
    return clusters


# ================================================================
# STAGE 2: Fit conditioned lenses
# ================================================================
@app.function(image=image, gpu="H100", timeout=7200, volumes={"/results": RESULTS_VOL})
def stage_fit_conditioned(n_fit: int = 672):
    import json, time, os
    import torch
    import numpy as np
    import jlens

    print("=== STAGE 2: Fit Conditioned Lenses ===")

    # Load routing data
    with open("/results/chunked_routing.json") as f:
        routing_data = json.load(f)
    clusters = routing_data["clusters"]
    print(f"  Loaded routing: {len(clusters)} layers, "
          f"target={routing_data['target_layers']}")

    model, hf_model, tokenizer, moe_layers, n_experts = load_model()
    fit_prompts, _ = load_prompts(n_fit=n_fit)

    conditioned = {}
    for layer_str, cluster_info in clusters.items():
        layer = int(layer_str)
        labels = np.array(cluster_info["labels"])
        unique_labels = np.unique(labels)
        print(f"\n  Layer {layer}: {len(unique_labels)} clusters, "
              f"sizes={cluster_info['sizes']}")

        layer_lenses = {}
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_prompts = [p for p, m in zip(fit_prompts, mask) if m]
            n = len(cluster_prompts)

            if n < 20:
                print(f"    Cluster {cluster_id}: {n} prompts (too few, skipping)")
                continue

            print(f"    Cluster {cluster_id}: fitting on {n} prompts...", end=" ", flush=True)
            t0 = time.time()
            try:
                lens = jlens.fit(model, cluster_prompts,
                                 source_layers=[layer],
                                 dim_batch=4, max_seq_len=128)
                lens_path = f"/results/conditioned_lens_L{layer}_C{cluster_id}.pt"
                lens.save(lens_path)
                elapsed = time.time() - t0
                print(f"done ({elapsed:.0f}s)")
                layer_lenses[int(cluster_id)] = {
                    "path": lens_path, "n_prompts": n, "time": elapsed}
            except Exception as e:
                print(f"FAILED: {e}")
                layer_lenses[int(cluster_id)] = {
                    "path": None, "n_prompts": n, "error": str(e)}

        conditioned[layer_str] = layer_lenses

    # Save manifest
    manifest_path = "/results/chunked_conditioned_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(conditioned, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\n  Saved manifest to {manifest_path}")
    return conditioned


# ================================================================
# STAGE 3: Fit random control
# ================================================================
@app.function(image=image, gpu="H100", timeout=7200, volumes={"/results": RESULTS_VOL})
def stage_fit_random(n_fit: int = 672):
    import json, time
    import numpy as np
    import jlens

    print("=== STAGE 3: Fit Random Control Lenses ===")

    with open("/results/chunked_routing.json") as f:
        routing_data = json.load(f)
    clusters = routing_data["clusters"]

    model, hf_model, tokenizer, moe_layers, n_experts = load_model()
    fit_prompts, _ = load_prompts(n_fit=n_fit)

    random_lenses = {}
    rng = np.random.RandomState(42)

    for layer_str, cluster_info in clusters.items():
        layer = int(layer_str)
        sizes = cluster_info["sizes"]
        print(f"\n  Layer {layer}: random groups matching sizes {sizes}")

        indices = rng.permutation(len(fit_prompts))
        layer_lenses = {}
        offset = 0

        for group_id, size in enumerate(sizes):
            if offset + size > len(fit_prompts):
                size = len(fit_prompts) - offset
            group_prompts = [fit_prompts[i] for i in indices[offset:offset + size]]
            offset += size

            if size < 20:
                print(f"    Group {group_id}: {size} prompts (too few)")
                continue

            print(f"    Group {group_id}: fitting on {size} prompts...", end=" ", flush=True)
            t0 = time.time()
            try:
                lens = jlens.fit(model, group_prompts,
                                 source_layers=[layer],
                                 dim_batch=4, max_seq_len=128)
                lens_path = f"/results/random_lens_L{layer}_G{group_id}.pt"
                lens.save(lens_path)
                elapsed = time.time() - t0
                print(f"done ({elapsed:.0f}s)")
                layer_lenses[group_id] = {
                    "path": lens_path, "n_prompts": size, "time": elapsed}
            except Exception as e:
                print(f"FAILED: {e}")
                layer_lenses[group_id] = {
                    "path": None, "n_prompts": size, "error": str(e)}

        random_lenses[layer_str] = layer_lenses

    manifest_path = "/results/chunked_random_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(random_lenses, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\n  Saved manifest to {manifest_path}")
    return random_lenses


# ================================================================
# STAGE 4: Evaluate all three
# ================================================================
@app.function(image=image, gpu="H100", timeout=3600, volumes={"/results": RESULTS_VOL})
def stage_evaluate():
    import json, time, pickle
    import torch
    import numpy as np
    import jlens
    from jlens.hooks import ActivationRecorder
    from sklearn.cluster import KMeans
    from scipy.stats import mannwhitneyu

    print("=== STAGE 4: Evaluate ===")

    with open("/results/chunked_routing.json") as f:
        routing_data = json.load(f)
    with open("/results/chunked_conditioned_manifest.json") as f:
        conditioned_manifest = json.load(f)
    with open("/results/chunked_random_manifest.json") as f:
        random_manifest = json.load(f)
    with open("/results/chunked_routing_vectors.pkl", "rb") as f:
        fit_vectors = pickle.load(f)

    model, hf_model, tokenizer, moe_layers, n_experts = load_model()
    _, test_prompts = load_prompts()

    # Load standard lens
    standard_lens = jlens.JacobianLens.load("/results/moe_30b_jlens.pt")
    print(f"  Standard lens: {len(standard_lens.source_layers)} layers")

    # OOD prompts
    code_prompts = [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
        "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self",
        "async function fetchData(url) {\n    const response = await fetch(url);\n    const data =",
        "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id =",
        "from dataclasses import dataclass\nfrom typing import Optional\n\n@dataclass\nclass Config:\n    learning_rate:",
    ]
    dialogue_prompts = [
        "User: Can you explain how neural networks learn?\nAssistant: Neural networks learn through",
        "User: I'm feeling overwhelmed with work lately.\nAssistant: I understand that feeling.",
        "User: What's the difference between a compiler and an interpreter?\nAssistant: A compiler",
        "User: How do I make sourdough bread from scratch?\nAssistant: Making sourdough is a rewarding",
        "User: Explain quantum entanglement like I'm five.\nAssistant: Imagine you have two magic",
    ]

    # Capture test routing for cluster assignment
    print("  Capturing test prompt routing...")
    test_routing = {layer: [] for layer in moe_layers}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 3:
                expert_indices = output[2]
                if isinstance(expert_indices, torch.Tensor):
                    test_routing[layer_idx].append(expert_indices.detach().cpu())
        return hook_fn

    for layer_idx in moe_layers:
        hook = hf_model.model.layers[layer_idx].mlp.gate.register_forward_hook(
            make_hook(layer_idx))
        hooks.append(hook)

    for prompt in test_prompts:
        input_ids = model.encode(prompt, max_length=128)
        with torch.no_grad():
            model.forward(input_ids)
    for h in hooks:
        h.remove()

    # Build test routing vectors
    test_vectors = {}
    for layer in [int(l) for l in routing_data["clusters"].keys()]:
        if layer not in test_routing or not test_routing[layer]:
            continue
        layer_vecs = []
        for pr in test_routing[layer]:
            if pr.dim() == 3:
                pr = pr.squeeze(0)
            vec = torch.zeros(n_experts)
            for tok_idx in range(pr.shape[0]):
                for eidx in pr[tok_idx]:
                    if eidx.item() < n_experts:
                        vec[eidx.item()] += 1.0
            vec = vec / (pr.shape[0] + 1e-10)
            layer_vecs.append(vec.numpy())
        test_vectors[layer] = np.array(layer_vecs)

    def eval_lens(lens_obj, prompts, layer):
        if lens_obj is None:
            return []
        final_layer = model.n_layers - 1
        cosines = []
        for prompt in prompts:
            input_ids = model.encode(prompt, max_length=64)
            with ActivationRecorder(model.layers, at=[layer, final_layer]) as rec:
                model.forward(input_ids)
                h = rec.activations[layer][0, -1:].detach().float()
                h_final = rec.activations[final_layer][0, -1:].detach().float()
            transported = lens_obj.transport(h.cpu(), layer)
            jl_logits = model.unembed(transported.to(input_ids.device)).squeeze(0).float()
            actual_logits = model.unembed(h_final).squeeze(0).float()
            cos = torch.nn.functional.cosine_similarity(
                torch.softmax(jl_logits, -1).unsqueeze(0),
                torch.softmax(actual_logits, -1).unsqueeze(0)).item()
            cosines.append(cos)
        return cosines

    # Evaluate per layer
    evaluation = {}
    for layer_str in routing_data["clusters"].keys():
        layer = int(layer_str)
        cluster_info = routing_data["clusters"][layer_str]
        print(f"\n  Layer {layer}:")

        # Standard
        std_cosines = eval_lens(standard_lens, test_prompts, layer)
        std_mean = float(np.mean(std_cosines)) if std_cosines else 0.0
        print(f"    Standard:    {std_mean:.4f} (n={len(std_cosines)})")

        # Conditioned — assign test prompts to clusters
        cond_cosines = []
        if layer in test_vectors and layer in fit_vectors:
            km = KMeans(n_clusters=cluster_info["k"], n_init=10, random_state=42)
            km.fit(fit_vectors[layer])
            test_labels = km.predict(test_vectors[layer])

            for i, prompt in enumerate(test_prompts):
                cid = str(int(test_labels[i]))
                cl = conditioned_manifest.get(layer_str, {}).get(cid)
                if cl and cl.get("path"):
                    try:
                        lens = jlens.JacobianLens.load(cl["path"])
                        cs = eval_lens(lens, [prompt], layer)
                        if cs:
                            cond_cosines.append(cs[0])
                    except:
                        pass

        cond_mean = float(np.mean(cond_cosines)) if cond_cosines else 0.0
        print(f"    Conditioned: {cond_mean:.4f} (n={len(cond_cosines)})")

        # Random — mirror conditioned protocol
        rand_cosines = []
        rng = np.random.RandomState(99)
        sizes = cluster_info["sizes"]
        test_rand_labels = np.zeros(len(test_prompts), dtype=int)
        perm = rng.permutation(len(test_prompts))
        offset = 0
        for gid, sz in enumerate(sizes):
            end = min(offset + sz, len(test_prompts))
            for idx in perm[offset:end]:
                test_rand_labels[idx] = gid
            offset = end

        for i, prompt in enumerate(test_prompts):
            gid = int(test_rand_labels[i])
            rl = random_manifest.get(layer_str, {}).get(str(gid))
            if rl and rl.get("path"):
                try:
                    lens = jlens.JacobianLens.load(rl["path"])
                    cs = eval_lens(lens, [prompt], layer)
                    if cs:
                        rand_cosines.append(cs[0])
                except:
                    pass

        rand_mean = float(np.mean(rand_cosines)) if rand_cosines else 0.0
        print(f"    Random:      {rand_mean:.4f} (n={len(rand_cosines)})")

        # Statistical test
        p_value = 1.0
        if len(cond_cosines) > 5 and len(rand_cosines) > 5:
            min_n = min(len(cond_cosines), len(rand_cosines))
            _, p_value = mannwhitneyu(
                cond_cosines[:min_n], rand_cosines[:min_n], alternative='greater')
            print(f"    Mann-Whitney (cond > rand): p={p_value:.6f}")

        evaluation[layer_str] = {
            "standard": {"mean": std_mean, "n": len(std_cosines)},
            "conditioned": {"mean": cond_mean, "n": len(cond_cosines)},
            "random": {"mean": rand_mean, "n": len(rand_cosines)},
            "p_value": p_value,
        }

    # OOD evaluation at middle target layer
    mid_layer = TARGET_LAYERS[len(TARGET_LAYERS) // 2]
    ood = {}
    for name, prompts in [("code", code_prompts), ("dialogue", dialogue_prompts)]:
        std_cs = eval_lens(standard_lens, prompts, mid_layer)
        std_m = float(np.mean(std_cs)) if std_cs else 0.0
        print(f"\n  OOD {name} at L{mid_layer}: standard={std_m:.4f}")
        ood[name] = {"standard": std_m, "layer": mid_layer}

    # Verdict
    all_std = [e["standard"]["mean"] for e in evaluation.values()]
    all_cond = [e["conditioned"]["mean"] for e in evaluation.values()]
    all_rand = [e["random"]["mean"] for e in evaluation.values()]
    all_pvals = [e["p_value"] for e in evaluation.values()]

    n_tests = len(all_pvals)
    corrected = [min(p * n_tests, 1.0) for p in all_pvals]
    sig_layers = sum(1 for p in corrected if p < 0.05)

    beats_random = np.mean(all_cond) > np.mean(all_rand)
    beats_standard = np.mean(all_cond) > np.mean(all_std)
    above_threshold = np.mean(all_cond) > 0.5

    if beats_random and beats_standard and sig_layers > 0:
        verdict = "SUCCESS" if above_threshold else "PARTIAL"
    elif beats_standard and not beats_random:
        verdict = "NULL_SWARM"
    elif beats_standard and sig_layers == 0:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NEGATIVE"

    results = {
        "verdict": verdict,
        "evaluation": evaluation,
        "ood": ood,
        "summary": {
            "standard_mean": float(np.mean(all_std)),
            "conditioned_mean": float(np.mean(all_cond)),
            "random_mean": float(np.mean(all_rand)),
            "significant_layers": sig_layers,
            "n_tests": n_tests,
        },
    }

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"  Standard:    {np.mean(all_std):.4f}")
    print(f"  Conditioned: {np.mean(all_cond):.4f}")
    print(f"  Random:      {np.mean(all_rand):.4f}")
    print(f"  Significant: {sig_layers}/{n_tests}")
    print(f"{'='*60}")

    with open("/results/conditioned_jlens_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    RESULTS_VOL.commit()
    return results


@app.local_entrypoint()
def main(stage: str = "routing"):
    if stage == "routing":
        result = stage_routing.remote()
    elif stage == "fit-conditioned":
        result = stage_fit_conditioned.remote()
    elif stage == "fit-random":
        result = stage_fit_random.remote()
    elif stage == "evaluate":
        result = stage_evaluate.remote()
    elif stage == "all":
        print("Running all stages sequentially...")
        stage_routing.remote()
        stage_fit_conditioned.remote()
        stage_fit_random.remote()
        result = stage_evaluate.remote()
    else:
        print(f"Unknown stage: {stage}")
        print("Options: routing, fit-conditioned, fit-random, evaluate, all")
        return

    print(f"\nStage '{stage}' complete.")
    if isinstance(result, dict) and "verdict" in result:
        print(f"Verdict: {result['verdict']}")
