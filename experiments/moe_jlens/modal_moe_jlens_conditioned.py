"""Path-Conditioned MoE J-lens on Modal — extends the standard fitting with routing-aware lenses.

Builds on the baseline result (12.5% gate accuracy with standard J-lens on Qwen3-30B-A3B).
Adds path-conditioned fitting: capture routing decisions, cluster prompts by expert trajectory,
fit separate J-lenses per cluster. Includes random-conditioned control for null swarm check.

Runs on Modal H100 (80GB). Reuses the existing fitted standard lens from the volume.

Paper 5: "Path-Conditioned Jacobian Lenses for Mixture-of-Experts Models"
"""

import modal

app = modal.App("moe-jlens-conditioned")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "scikit-learn",
        "scipy",
        "accelerate",
    )
    .pip_install("jlens @ git+https://github.com/anthropics/jacobian-lens.git")
    .pip_install("datasets")
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)


def capture_routing_decisions(model, hf_model, prompts, moe_layers,
                              max_seq_len=128, top_k_experts=None):
    """Hook MoE routers to capture expert assignments for each prompt.

    top_k_experts: how many experts to capture per token. Read from
    config.num_experts_per_tok if not specified. Captures the FULL
    routing decision, not just top-2.

    Returns: dict mapping layer_idx -> list of tensors, one per prompt.
    """
    import torch

    if top_k_experts is None:
        config = hf_model.config
        top_k_experts = getattr(config, 'num_experts_per_tok',
                                getattr(config, 'num_selected_experts', 8))

    routing_decisions = {layer: [] for layer in moe_layers}
    hooks = []

    for layer_idx in moe_layers:
        layer = hf_model.model.layers[layer_idx]
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            hook = layer.mlp.gate.register_forward_hook(
                make_gate_hook(layer_idx, routing_decisions, top_k=top_k_experts))
            hooks.append(hook)
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            hook = layer.mlp.gate.register_forward_hook(
                make_gate_hook(layer_idx, routing_decisions, top_k=top_k_experts))
            hooks.append(hook)

    for prompt in prompts:
        input_ids = model.encode(prompt, max_length=max_seq_len)
        with torch.no_grad():
            model.forward(input_ids)

    for h in hooks:
        h.remove()

    return routing_decisions


def make_gate_hook(layer_idx, routing_decisions, top_k=None):
    """Hook for the gate/router module directly to capture routing logits.

    top_k: number of experts per token to capture. If None, captures all
    experts that the model actually routes to (from config.num_experts_per_tok).
    """
    import torch

    def hook_fn(module, input, output):
        k = top_k or 8
        if isinstance(output, torch.Tensor):
            actual_k = min(k, output.shape[-1])
            topk = torch.topk(output.float(), actual_k, dim=-1).indices
            routing_decisions[layer_idx].append(topk.detach().cpu())
        elif isinstance(output, tuple):
            logits = output[0] if isinstance(output[0], torch.Tensor) else output
            if isinstance(logits, torch.Tensor) and logits.dim() >= 2:
                actual_k = min(k, logits.shape[-1])
                topk = torch.topk(logits.float(), actual_k, dim=-1).indices
                routing_decisions[layer_idx].append(topk.detach().cpu())

    return hook_fn


def build_routing_vectors(routing_decisions, n_experts, moe_layers):
    """Convert per-prompt routing decisions into binary expert-activation vectors.

    For each MoE layer, each prompt gets a vector of length n_experts where
    entry j = fraction of tokens routed to expert j. This captures the
    prompt's "routing fingerprint" at that layer.
    """
    import torch
    import numpy as np

    vectors = {}
    for layer in moe_layers:
        if layer not in routing_decisions or not routing_decisions[layer]:
            continue
        layer_vecs = []
        for prompt_routing in routing_decisions[layer]:
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
    return vectors


def cluster_by_routing(routing_vectors, k_range=(3, 8), min_cluster_size=50):
    """Cluster prompts by routing pattern at each layer.

    Uses k-means with silhouette-guided k selection.
    Merges clusters smaller than min_cluster_size into nearest neighbor.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np

    clusters = {}
    for layer, vectors in routing_vectors.items():
        n_prompts = len(vectors)
        if n_prompts < min_cluster_size * 2:
            clusters[layer] = {"labels": np.zeros(n_prompts, dtype=int), "k": 1,
                               "silhouette": 0.0, "note": "too few prompts for clustering"}
            continue

        best_k = k_range[0]
        best_score = -1
        best_labels = None

        max_k = min(k_range[1], n_prompts // min_cluster_size)
        if max_k < k_range[0]:
            max_k = k_range[0]

        for k in range(k_range[0], max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(vectors)

            cluster_sizes = np.bincount(labels)
            if cluster_sizes.min() < min_cluster_size:
                merged_labels = merge_small_clusters(labels, vectors, km.cluster_centers_,
                                                      min_cluster_size)
                actual_k = len(np.unique(merged_labels))
                if actual_k < 2:
                    continue
                labels = merged_labels

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(vectors, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        if best_labels is None:
            best_labels = np.zeros(n_prompts, dtype=int)
            best_score = 0.0
            best_k = 1

        actual_k = len(np.unique(best_labels))
        clusters[layer] = {"labels": best_labels, "k": actual_k,
                           "k_pre_merge": best_k,
                           "silhouette": best_score,
                           "sizes": [int(x) for x in np.bincount(best_labels) if x > 0]}

    return clusters


def merge_small_clusters(labels, vectors, centers, min_size):
    """Merge clusters smaller than min_size into their nearest neighbor."""
    import numpy as np
    from scipy.spatial.distance import cdist

    labels = labels.copy()
    while True:
        sizes = np.bincount(labels)
        small = np.where((sizes < min_size) & (sizes > 0))[0]
        if len(small) == 0:
            break
        smallest = small[np.argmin(sizes[small])]
        mask = labels == smallest
        if not mask.any():
            break
        other_centers = []
        other_ids = []
        for c in range(len(sizes)):
            if c != smallest and sizes[c] >= min_size:
                other_centers.append(vectors[labels == c].mean(axis=0))
                other_ids.append(c)
        if not other_ids:
            break
        small_center = vectors[mask].mean(axis=0)
        dists = cdist([small_center], other_centers)[0]
        nearest = other_ids[np.argmin(dists)]
        labels[mask] = nearest

    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[l] for l in labels])


def fit_conditioned_lenses(model, prompts, clusters, layer, lens_kwargs):
    """Fit a separate J-lens for each cluster at a given layer.

    Only fits the target layer (source_layers=[layer]) rather than all
    layers — critical for compute budget (~47x speedup per fit).
    """
    import jlens
    import numpy as np

    labels = clusters[layer]["labels"]
    unique_labels = np.unique(labels)
    conditioned_lenses = {}

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_prompts = [p for p, m in zip(prompts, mask) if m]
        n = len(cluster_prompts)

        if n < 20:
            conditioned_lenses[int(cluster_id)] = {
                "lens": None, "n_prompts": n, "note": "too few prompts"}
            continue

        try:
            lens = jlens.fit(model, cluster_prompts,
                             source_layers=[layer], **lens_kwargs)
            conditioned_lenses[int(cluster_id)] = {"lens": lens, "n_prompts": n}
        except Exception as e:
            conditioned_lenses[int(cluster_id)] = {
                "lens": None, "n_prompts": n, "error": str(e)}

    return conditioned_lenses


def fit_random_conditioned_lenses(model, prompts, cluster_sizes, layer, lens_kwargs):
    """Fit J-lenses on random groups matching the cluster sizes — null swarm control.

    Only fits the target layer (source_layers=[layer]).
    """
    import jlens
    import numpy as np

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(prompts))
    random_lenses = {}
    random_assignments = {}
    offset = 0

    for cluster_id, size in enumerate(cluster_sizes):
        if offset + size > len(prompts):
            size = len(prompts) - offset
        group_indices = indices[offset:offset + size]
        random_assignments[cluster_id] = group_indices.tolist()

        if size < 20:
            random_lenses[cluster_id] = {"lens": None, "n_prompts": size, "note": "too few"}
            offset += size
            continue

        group_prompts = [prompts[i] for i in group_indices]
        try:
            lens = jlens.fit(model, group_prompts,
                             source_layers=[layer], **lens_kwargs)
            random_lenses[cluster_id] = {"lens": lens, "n_prompts": size}
        except Exception as e:
            random_lenses[cluster_id] = {"lens": None, "n_prompts": size, "error": str(e)}
        offset += size

    return random_lenses, random_assignments


def evaluate_lens(model, lens_obj, test_prompts, layer, tokenizer):
    """Compute transport cosine for a fitted lens on test prompts.

    Transport cosine: cosine similarity between the lens-transported
    residual stream (at target layer) and the model's actual final-layer
    hidden state, both projected through the unembedding matrix.
    """
    import torch
    from jlens.hooks import ActivationRecorder

    if lens_obj is None:
        return {"transport_cosine": 0.0, "n_prompts": 0, "note": "no lens"}

    final_layer = max(model.layers) if hasattr(model, 'layers') else model.n_layers - 1
    cosines = []
    for prompt in test_prompts:
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
            torch.softmax(actual_logits, -1).unsqueeze(0)
        ).item()
        cosines.append(cos)

    if not cosines:
        return {"transport_cosine": 0.0, "n_prompts": 0}

    import numpy as np
    return {
        "transport_cosine": float(np.mean(cosines)),
        "transport_std": float(np.std(cosines)),
        "n_prompts": len(cosines),
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=21600,
    volumes={"/results": RESULTS_VOL},
)
def run_conditioned_jlens(fit_prompts: int = 672, test_prompts_n: int = 100):
    """Main experiment: standard vs path-conditioned vs random-conditioned J-lens on MoE."""
    import json
    import os
    import time

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import jlens
    from jlens.hf import HFLensModel
    from jlens.examples import load_wikitext_prompts

    results_dir = "/results"
    device = "cuda"
    model_name = "Qwen/Qwen3-30B-A3B"

    print("=" * 65)
    print("PATH-CONDITIONED MoE J-LENS")
    print("Standard vs Conditioned vs Random-Conditioned")
    print(f"Device: {device} ({torch.cuda.get_device_name()})")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")
    print("=" * 65)

    # Load model
    print(f"\n[LOAD] Loading {model_name} in bf16...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFLensModel(hf_model, tokenizer, compile=False)
    print(f"  Loaded in {time.time()-t0:.0f}s: {model.n_layers} layers, d={model.d_model}")

    # Identify MoE layers
    moe_layers = []
    dense_layers = []
    for i, layer in enumerate(hf_model.model.layers):
        if hasattr(layer, 'mlp') and (hasattr(layer.mlp, 'experts') or hasattr(layer.mlp, 'gate')):
            moe_layers.append(i)
        else:
            dense_layers.append(i)
    n_experts = getattr(config, 'num_experts', getattr(config, 'num_local_experts', 8))
    print(f"  MoE layers: {len(moe_layers)}, Dense layers: {len(dense_layers)}, Experts: {n_experts}")

    # Load prompts — WikiText for fitting, plus code and dialogue for OOD eval
    print(f"\n[DATA] Loading prompts...")
    all_prompts = load_wikitext_prompts(n_prompts=fit_prompts + test_prompts_n)
    fitting_prompts = all_prompts[:fit_prompts]
    test_prompts_wiki = all_prompts[fit_prompts:fit_prompts + test_prompts_n]

    test_prompts_code = [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
        "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self",
        "async function fetchData(url) {\n    const response = await fetch(url);\n    const data =",
        "fn main() {\n    let mut vec: Vec<i32> = Vec::new();\n    for i in 0..10 {\n        vec.push(",
        "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id =",
        "class BinarySearchTree:\n    def insert(self, val):\n        if self.root is None:\n            self.root =",
        "#include <stdio.h>\n#include <stdlib.h>\n\nint* merge_sort(int* arr, int n) {\n    if (n <=",
        "from dataclasses import dataclass\nfrom typing import Optional\n\n@dataclass\nclass Config:\n    learning_rate:",
        "kubectl apply -f deployment.yaml\n# Check pod status\nkubectl get pods -n production | grep",
        "def train_epoch(model, dataloader, optimizer):\n    model.train()\n    total_loss = 0\n    for batch in",
    ]

    test_prompts_dialogue = [
        "User: Can you explain how neural networks learn?\nAssistant: Neural networks learn through",
        "User: I'm feeling overwhelmed with work lately.\nAssistant: I understand that feeling. It sounds like",
        "User: What's the difference between a compiler and an interpreter?\nAssistant: Great question. A compiler",
        "User: My cat keeps knocking things off the table.\nAssistant: That's actually a very common behavior in cats.",
        "User: How do I make sourdough bread from scratch?\nAssistant: Making sourdough is a rewarding process. First",
        "User: Can you help me debug this error: TypeError: cannot unpack non-sequence NoneType\nAssistant: This error",
        "User: What are the ethical implications of AI in healthcare?\nAssistant: There are several important considerations",
        "User: I need to give a presentation tomorrow and I'm nervous.\nAssistant: Presentation anxiety is very normal.",
        "User: Explain quantum entanglement like I'm five.\nAssistant: Imagine you have two magic coins that are",
        "User: What's your opinion on remote work vs office work?\nAssistant: Both have distinct advantages. Remote work",
    ]

    test_prompts = test_prompts_wiki
    print(f"  Fitting: {len(fitting_prompts)}")
    print(f"  Test: {len(test_prompts_wiki)} wiki, {len(test_prompts_code)} code, {len(test_prompts_dialogue)} dialogue")

    # ================================================================
    # STEP 1: Load or fit standard J-lens (baseline)
    # ================================================================
    standard_lens_path = os.path.join(results_dir, "moe_30b_jlens.pt")
    if os.path.exists(standard_lens_path):
        print(f"\n[STEP 1] Loading existing standard lens from volume")
        standard_lens = jlens.JacobianLens.load(standard_lens_path)
    else:
        print(f"\n[STEP 1] Fitting standard J-lens (baseline)...")
        t0 = time.time()
        standard_lens = jlens.fit(
            model, fitting_prompts, dim_batch=4, max_seq_len=128,
            checkpoint_path=os.path.join(results_dir, "standard_checkpoint.pt"),
            checkpoint_every=5,
        )
        print(f"  Fitted in {time.time()-t0:.0f}s")
        standard_lens.save(standard_lens_path)

    # ================================================================
    # STEP 2: Capture routing decisions
    # ================================================================
    print(f"\n[STEP 2] Capturing routing decisions for {len(fitting_prompts)} prompts...")
    t0 = time.time()
    routing_decisions = capture_routing_decisions(
        model, hf_model, fitting_prompts, moe_layers, max_seq_len=128)
    capture_time = time.time() - t0
    captured_layers = [l for l in moe_layers if l in routing_decisions and routing_decisions[l]]
    print(f"  Captured {len(captured_layers)} layers in {capture_time:.0f}s")

    if not captured_layers:
        print("\n  FATAL: No routing decisions captured. Hook may need adjustment.")
        print("  Dumping MoE layer structure for debugging:")
        for i in moe_layers[:3]:
            layer = hf_model.model.layers[i]
            print(f"    L{i}: {type(layer.mlp).__name__}")
            for name, mod in layer.mlp.named_children():
                print(f"      .{name}: {type(mod).__name__}")
        result = {"verdict": "ROUTING_CAPTURE_FAILED", "model": model_name}
        with open(os.path.join(results_dir, "conditioned_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        RESULTS_VOL.commit()
        return result

    # ================================================================
    # STEP 3: Build routing vectors and cluster
    # ================================================================
    print(f"\n[STEP 3] Building routing vectors and clustering...")
    routing_vectors = build_routing_vectors(routing_decisions, n_experts, captured_layers)
    clusters = cluster_by_routing(routing_vectors, k_range=(3, 8), min_cluster_size=50)

    for layer in sorted(clusters.keys())[:5]:
        c = clusters[layer]
        print(f"  L{layer}: k={c['k']}, silhouette={c['silhouette']:.3f}, sizes={c.get('sizes', '?')}")

    # ================================================================
    # STEP 4: Select target layers for conditioned fitting
    # ================================================================
    # Focus on layers where clustering found structure (silhouette > 0.1)
    # and that are in the workspace band (middle 60% of layers)
    workspace_start = int(model.n_layers * 0.2)
    workspace_end = int(model.n_layers * 0.8)
    target_layers = [
        l for l in sorted(clusters.keys())
        if clusters[l]["k"] > 1
        and clusters[l]["silhouette"] > 0.1
        and workspace_start <= l <= workspace_end
    ]

    if not target_layers:
        target_layers = sorted(clusters.keys())[:5]
        print(f"\n  WARN: No layers with good clustering. Using first 5: {target_layers}")
    else:
        if len(target_layers) > 10:
            step = len(target_layers) // 10
            target_layers = target_layers[::step]
        print(f"\n  Target layers for conditioned fitting: {target_layers}")

    # ================================================================
    # STEP 5: Fit conditioned lenses
    # ================================================================
    print(f"\n[STEP 5] Fitting conditioned J-lenses ({len(target_layers)} layers)...")
    lens_kwargs = {"dim_batch": 4, "max_seq_len": 128}
    conditioned_results = {}

    for layer in target_layers:
        print(f"\n  Layer {layer} (k={clusters[layer]['k']}, "
              f"sizes={clusters[layer].get('sizes', '?')}):")
        t0 = time.time()
        conditioned_lenses = fit_conditioned_lenses(
            model, fitting_prompts, clusters, layer, lens_kwargs)
        elapsed = time.time() - t0

        for cid, cl in conditioned_lenses.items():
            status = f"fitted ({cl['n_prompts']} prompts)" if cl.get("lens") else cl.get("note", cl.get("error", "?"))
            print(f"    Cluster {cid}: {status}")

        conditioned_results[layer] = {
            "conditioned_lenses": conditioned_lenses,
            "elapsed": elapsed,
        }

    # ================================================================
    # STEP 6: Fit random-conditioned control
    # ================================================================
    print(f"\n[STEP 6] Fitting random-conditioned control lenses...")
    random_results = {}
    all_random_assignments = {}
    for layer in target_layers:
        sizes = clusters[layer].get("sizes", [len(fitting_prompts)])
        print(f"  Layer {layer} (random groups matching sizes {sizes}):")
        t0 = time.time()
        random_lenses, random_assignments = fit_random_conditioned_lenses(
            model, fitting_prompts, sizes, layer, lens_kwargs)
        elapsed = time.time() - t0
        random_results[layer] = {"random_lenses": random_lenses, "elapsed": elapsed}
        all_random_assignments[layer] = random_assignments

    # ================================================================
    # STEP 7: Evaluate all three approaches
    # ================================================================
    print(f"\n[STEP 7] Evaluating on {len(test_prompts)} test prompts...")

    print("  Capturing test prompt routing...")
    test_routing = capture_routing_decisions(
        model, hf_model, test_prompts, moe_layers, max_seq_len=128)
    test_routing_vectors = build_routing_vectors(test_routing, n_experts, captured_layers)

    evaluation = {}
    for layer in target_layers:
        print(f"\n  Layer {layer}:")

        # Standard lens
        std_result = evaluate_lens(model, standard_lens, test_prompts, layer, tokenizer)
        print(f"    Standard:    cos={std_result['transport_cosine']:.4f} "
              f"± {std_result.get('transport_std', 0):.4f}")

        # Conditioned lens — assign test prompts to clusters, use matching lens
        cond_cosines = []
        if layer in test_routing_vectors and layer in clusters:
            from sklearn.cluster import KMeans
            fit_vectors = routing_vectors[layer]
            test_vectors = test_routing_vectors[layer]
            km = KMeans(n_clusters=clusters[layer]["k"], n_init=10, random_state=42)
            km.fit(fit_vectors)
            test_labels = km.predict(test_vectors)

            for i, prompt in enumerate(test_prompts):
                cluster_id = int(test_labels[i])
                cl = conditioned_results.get(layer, {}).get("conditioned_lenses", {}).get(cluster_id)
                if cl and cl.get("lens"):
                    result = evaluate_lens(model, cl["lens"], [prompt], layer, tokenizer)
                    if result["n_prompts"] > 0:
                        cond_cosines.append(result["transport_cosine"])

        cond_mean = float(np.mean(cond_cosines)) if cond_cosines else 0.0
        cond_std = float(np.std(cond_cosines)) if cond_cosines else 0.0
        print(f"    Conditioned: cos={cond_mean:.4f} ± {cond_std:.4f} (n={len(cond_cosines)})")

        # Random-conditioned control — MIRRORS conditioned protocol exactly:
        # assign each test prompt to a random group, evaluate with that group's lens
        rand_cosines = []
        rng_test = np.random.RandomState(99)
        sizes = clusters[layer].get("sizes", [len(test_prompts)])
        test_rand_labels = np.zeros(len(test_prompts), dtype=int)
        test_perm = rng_test.permutation(len(test_prompts))
        offset = 0
        for gid, size in enumerate(sizes):
            end = min(offset + size, len(test_prompts))
            actual_end = min(end, len(test_perm))
            for idx in test_perm[offset:actual_end]:
                test_rand_labels[idx] = gid
            offset = actual_end

        for i, prompt in enumerate(test_prompts):
            group_id = int(test_rand_labels[i])
            rl = random_results.get(layer, {}).get("random_lenses", {}).get(group_id)
            if rl and rl.get("lens"):
                result = evaluate_lens(model, rl["lens"], [prompt], layer, tokenizer)
                if result["n_prompts"] > 0:
                    rand_cosines.append(result["transport_cosine"])

        rand_mean = float(np.mean(rand_cosines)) if rand_cosines else 0.0
        rand_std = float(np.std(rand_cosines)) if rand_cosines else 0.0
        print(f"    Random:      cos={rand_mean:.4f} ± {rand_std:.4f} (n={len(rand_cosines)})")

        # Per-layer statistical test (Wilcoxon signed-rank, conditioned vs random)
        p_value = 1.0
        if len(cond_cosines) > 5 and len(rand_cosines) > 5:
            from scipy.stats import mannwhitneyu
            min_n = min(len(cond_cosines), len(rand_cosines))
            stat, p_value = mannwhitneyu(
                cond_cosines[:min_n], rand_cosines[:min_n], alternative='greater')
            print(f"    Mann-Whitney U (cond > rand): p={p_value:.6f}")

        improvement = cond_mean - std_result["transport_cosine"]
        over_random = cond_mean - rand_mean
        print(f"    Improvement over standard: {improvement:+.4f}")
        print(f"    Improvement over random:   {over_random:+.4f}")

        evaluation[layer] = {
            "standard": std_result,
            "conditioned": {"transport_cosine": cond_mean, "transport_std": cond_std,
                            "n_prompts": len(cond_cosines),
                            "per_prompt_cosines": cond_cosines},
            "random": {"transport_cosine": rand_mean, "transport_std": rand_std,
                       "n_prompts": len(rand_cosines),
                       "per_prompt_cosines": rand_cosines},
            "p_value_cond_vs_rand": p_value,
            "improvement_over_standard": improvement,
            "improvement_over_random": over_random,
        }

    # ================================================================
    # STEP 7b: Cross-domain evaluation (OOD generalization)
    # ================================================================
    print(f"\n[STEP 7b] Cross-domain evaluation...")
    ood_evaluation = {}
    mid_target = target_layers[len(target_layers) // 2] if target_layers else None

    if mid_target and mid_target in conditioned_results:
        for domain_name, domain_prompts in [("code", test_prompts_code),
                                             ("dialogue", test_prompts_dialogue)]:
            std_r = evaluate_lens(model, standard_lens, domain_prompts, mid_target, tokenizer)

            cond_cosines_ood = []
            ood_routing = capture_routing_decisions(
                model, hf_model, domain_prompts, moe_layers, max_seq_len=128)
            ood_vectors = build_routing_vectors(ood_routing, n_experts, captured_layers)
            if mid_target in ood_vectors:
                km = KMeans(n_clusters=clusters[mid_target]["k"], n_init=10, random_state=42)
                km.fit(routing_vectors[mid_target])
                ood_labels = km.predict(ood_vectors[mid_target])
                for i, prompt in enumerate(domain_prompts):
                    cid = int(ood_labels[i])
                    cl = conditioned_results.get(mid_target, {}).get(
                        "conditioned_lenses", {}).get(cid)
                    if cl and cl.get("lens"):
                        r = evaluate_lens(model, cl["lens"], [prompt], mid_target, tokenizer)
                        if r["n_prompts"] > 0:
                            cond_cosines_ood.append(r["transport_cosine"])

            cond_ood = float(np.mean(cond_cosines_ood)) if cond_cosines_ood else 0.0
            print(f"  {domain_name} at L{mid_target}: standard={std_r['transport_cosine']:.4f}, "
                  f"conditioned={cond_ood:.4f} (n={len(cond_cosines_ood)})")
            ood_evaluation[domain_name] = {
                "layer": mid_target,
                "standard": std_r["transport_cosine"],
                "conditioned": cond_ood,
                "n_prompts": len(cond_cosines_ood),
            }

    # ================================================================
    # STEP 8: Final results
    # ================================================================
    if not evaluation:
        print("\n  FATAL: No layers produced valid evaluations.")
        result = {"verdict": "NO_VALID_LAYERS", "model": model_name}
        with open(os.path.join(results_dir, "conditioned_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        RESULTS_VOL.commit()
        return result

    print(f"\n{'=' * 65}")
    print(f"PATH-CONDITIONED MoE J-LENS RESULTS")
    print(f"{'=' * 65}")

    all_std = [e["standard"]["transport_cosine"] for e in evaluation.values()]
    all_cond = [e["conditioned"]["transport_cosine"] for e in evaluation.values()]
    all_rand = [e["random"]["transport_cosine"] for e in evaluation.values()]
    all_pvals = [e.get("p_value_cond_vs_rand", 1.0) for e in evaluation.values()]

    print(f"  Mean transport cosine across {len(evaluation)} layers:")
    print(f"    Standard:    {np.mean(all_std):.4f}")
    print(f"    Conditioned: {np.mean(all_cond):.4f}")
    print(f"    Random:      {np.mean(all_rand):.4f}")

    # Bonferroni correction for multiple layer comparisons
    n_tests = len(all_pvals)
    corrected_pvals = [min(p * n_tests, 1.0) for p in all_pvals]
    significant_layers = sum(1 for p in corrected_pvals if p < 0.05)
    print(f"\n  Bonferroni-corrected significance ({n_tests} tests):")
    for layer, p_raw, p_corr in zip(evaluation.keys(), all_pvals, corrected_pvals):
        sig = "*" if p_corr < 0.05 else ""
        print(f"    L{layer}: p_raw={p_raw:.6f}, p_corrected={p_corr:.6f} {sig}")
    print(f"  Significant layers: {significant_layers}/{n_tests}")

    # Standing Committee validation
    for layer in target_layers:
        sizes = clusters[layer].get("sizes", [])
        if sizes:
            total = sum(sizes)
            top3 = sum(sorted(sizes, reverse=True)[:3])
            print(f"  L{layer} routing mass: top-3 clusters = {top3}/{total} = {top3/total:.1%}")

    conditioned_beats_random = np.mean(all_cond) > np.mean(all_rand)
    conditioned_beats_standard = np.mean(all_cond) > np.mean(all_std)
    conditioned_above_threshold = np.mean(all_cond) > 0.5
    has_significant = significant_layers > 0

    if conditioned_beats_random and conditioned_beats_standard and has_significant:
        if conditioned_above_threshold:
            verdict = "SUCCESS: Path-conditioned J-lens works on MoE"
        else:
            verdict = "PARTIAL: Conditioned beats baselines but below 0.5 threshold"
    elif conditioned_beats_standard and not conditioned_beats_random:
        verdict = "NULL SWARM: Improvement is subset overfitting, not routing-specific"
    elif conditioned_beats_standard and not has_significant:
        verdict = "INCONCLUSIVE: Conditioned numerically better but no layer survives Bonferroni"
    else:
        verdict = "NEGATIVE: Conditioned fitting does not improve on standard"

    print(f"\n  VERDICT: {verdict}")

    full_results = {
        "experiment": "Path-Conditioned MoE J-lens",
        "model": model_name,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "n_experts": n_experts,
        "moe_layers": len(moe_layers),
        "dense_layers": len(dense_layers),
        "fit_prompts": len(fitting_prompts),
        "test_prompts": len(test_prompts),
        "target_layers": target_layers,
        "clustering": {str(l): {k: v for k, v in c.items() if k != "labels"}
                       for l, c in clusters.items()},
        "evaluation": {str(k): v for k, v in evaluation.items()},
        "ood_evaluation": ood_evaluation,
        "summary": {
            "standard_mean": float(np.mean(all_std)),
            "conditioned_mean": float(np.mean(all_cond)),
            "random_mean": float(np.mean(all_rand)),
            "conditioned_beats_random": bool(conditioned_beats_random),
            "conditioned_beats_standard": bool(conditioned_beats_standard),
            "conditioned_above_threshold": bool(conditioned_above_threshold),
            "significant_layers": significant_layers,
            "n_tests": n_tests,
            "corrected_pvals": {str(l): p for l, p in zip(evaluation.keys(), corrected_pvals)},
        },
        "verdict": verdict,
        "gpu": torch.cuda.get_device_name(),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }

    output_path = os.path.join(results_dir, "conditioned_jlens_results.json")
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")

    RESULTS_VOL.commit()
    return full_results


@app.local_entrypoint()
def main():
    result = run_conditioned_jlens.remote(fit_prompts=672, test_prompts_n=100)
    print("\n" + "=" * 65)
    print("REMOTE EXECUTION COMPLETE")
    print("=" * 65)
    if isinstance(result, dict):
        print(f"  Verdict: {result.get('verdict', 'unknown')}")
        import json
        print(json.dumps(result.get("summary", {}), indent=2))
    else:
        print(f"  Result: {result}")
