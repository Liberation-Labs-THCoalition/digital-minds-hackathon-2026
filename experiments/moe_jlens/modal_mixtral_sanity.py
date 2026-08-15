"""Mixtral J-lens sanity gate — the fast falsifier for MoE J-lens.

Mixtral-8x7B is the easiest MoE case: 8 experts, top-2 routing, C(8,2) = 28
possible expert paths. If a standard J-lens passes the sanity gate here, and
routing-conditioned lenses beat a random control on transport cosine, the
method works and the Qwen3-30B-A3B negative (128 experts, top-8, ~1.4e12
paths) is cardinality-specific. If it fails here too, the approach is dead on
the easiest possible instance. See:
  digital-minds-hackathon-2026/infrastructure/moe_jlens_next_steps_litreview.md
  ("Hours 0-6 — Sanity gate on Mixtral")

Pipeline (single run, checkpoints to volume after every phase):
  1. Load Mixtral-8x7B in 4-bit NF4 (46.7B params = ~93GB bf16, does not fit
     an H100 80GB; 4-bit is ~24GB and bnb supports backward through frozen
     quantized weights, which is all the Jacobian fit needs).
  2. Fit a STANDARD J-lens (50 WikiText prompts, 3 target layers).
  3. Sanity gate: 8 test prompts, next-token top-10 accuracy. Pass = >20%.
  4. Capture routing (hook on block_sparse_moe.gate — output is raw router
     logits (batch*seq, 8); top-2 computed from softmax, matching
     MixtralSparseMoeBlock.forward exactly).
  5. k-means (k=3-7, silhouette-selected) on 8-dim per-prompt routing vectors.
  6. Fit conditioned lenses per cluster at the 3 target layers.
  7. Fit random control (same group sizes, shuffled assignment).
  8. Evaluate standard vs conditioned vs random transport cosine + verdict.

Usage:
    modal run modal_mixtral_sanity.py
    modal run modal_mixtral_sanity.py --stop-on-gate-fail   # abort after step 3 if gate fails
    modal run modal_mixtral_sanity.py --n-fit 50 --n-test 40
"""

import modal

app = modal.App("mixtral-jlens-sanity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "numpy", "scikit-learn", "scipy",
                 "accelerate", "bitsandbytes")
    .pip_install("jlens @ git+https://github.com/anthropics/jacobian-lens.git")
    .pip_install("datasets", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("moe-jlens-hf-cache", create_if_missing=True)

MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"  # base model — fit data is WikiText
N_EXPERTS = 8
TOP_K = 2
TARGET_LAYERS = [8, 16, 24]  # Mixtral has 32 layers, all MoE
GATE_THRESHOLD = 0.20  # sanity gate: next-token top-10 accuracy must exceed this

# 8 diverse sanity-gate prompts with unambiguous continuations
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
    timeout=7200,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_sanity(n_fit: int = 50, n_test: int = 40, stop_on_gate_fail: bool = False):
    import json, time
    import torch
    import numpy as np
    import jlens
    from jlens.hf import HFLensModel
    from jlens.hooks import ActivationRecorder
    from jlens.examples import load_wikitext_prompts
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from scipy.stats import mannwhitneyu

    t_start = time.time()
    results = {"model": MODEL_NAME, "n_experts": N_EXPERTS, "top_k": TOP_K,
               "target_layers": TARGET_LAYERS, "quantization": "4bit-nf4"}

    def checkpoint():
        with open("/results/mixtral_sanity_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        RESULTS_VOL.commit()

    # ------------------------------------------------------------
    # 1. Load model (4-bit — 93GB bf16 does not fit H100 80GB)
    # ------------------------------------------------------------
    print("=== 1. Loading Mixtral-8x7B (4-bit NF4) ===")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HFLensModel(hf_model, tokenizer, compile=False)
    HF_CACHE_VOL.commit()  # persist downloaded weights for reruns
    print(f"  Model: {model.n_layers} layers, d={model.d_model}")
    print(f"  GPU: {torch.cuda.get_device_name()} "
          f"({torch.cuda.memory_allocated()/1e9:.1f}GB allocated)")

    # Verify the routing module is where we expect it
    moe0 = hf_model.model.layers[0].block_sparse_moe
    assert moe0.num_experts == N_EXPERTS and moe0.top_k == TOP_K, \
        f"Unexpected MoE config: {moe0.num_experts} experts, top-{moe0.top_k}"

    fit_prompts = load_wikitext_prompts(n_prompts=n_fit + n_test)
    fit_prompts, test_prompts = fit_prompts[:n_fit], fit_prompts[n_fit:n_fit + n_test]
    print(f"  Prompts: {len(fit_prompts)} fit, {len(test_prompts)} test")

    # ------------------------------------------------------------
    # 2. Fit standard J-lens (quick fit, all 3 layers in one call)
    # ------------------------------------------------------------
    print("\n=== 2. Fitting standard J-lens ===")
    t0 = time.time()
    standard_lens = jlens.fit(model, fit_prompts,
                              source_layers=TARGET_LAYERS,
                              dim_batch=4, max_seq_len=128)
    standard_path = "/results/mixtral_standard_jlens.pt"
    standard_lens.save(standard_path)
    results["standard_fit_time"] = time.time() - t0
    checkpoint()
    print(f"  Done in {results['standard_fit_time']:.0f}s. Saved {standard_path}")

    # ------------------------------------------------------------
    # Shared eval helpers
    # ------------------------------------------------------------
    final_layer = model.n_layers - 1

    def lens_and_actual_logits(lens_obj, prompt, layer, max_length=64):
        input_ids = model.encode(prompt, max_length=max_length)
        with ActivationRecorder(model.layers, at=[layer, final_layer]) as rec:
            with torch.no_grad():
                model.forward(input_ids)
            h = rec.activations[layer][0, -1:].detach().float()
            h_final = rec.activations[final_layer][0, -1:].detach().float()
        transported = lens_obj.transport(h.cpu(), layer)
        jl_logits = model.unembed(transported.to(input_ids.device)).squeeze(0).float()
        actual_logits = model.unembed(h_final).squeeze(0).float()
        return jl_logits, actual_logits

    def transport_cosine(lens_obj, prompts, layer):
        cosines = []
        for prompt in prompts:
            jl_logits, actual_logits = lens_and_actual_logits(lens_obj, prompt, layer)
            cos = torch.nn.functional.cosine_similarity(
                torch.softmax(jl_logits, -1).unsqueeze(0),
                torch.softmax(actual_logits, -1).unsqueeze(0)).item()
            cosines.append(cos)
        return cosines

    # ------------------------------------------------------------
    # 3. Sanity gate: 8 prompts, next-token top-10 accuracy
    # ------------------------------------------------------------
    print("\n=== 3. Sanity gate (next-token top-10 accuracy) ===")
    gate_per_layer = {}
    for layer in TARGET_LAYERS:
        hits = 0
        for prompt in SANITY_PROMPTS:
            jl_logits, actual_logits = lens_and_actual_logits(
                standard_lens, prompt, layer)
            actual_top1 = actual_logits.argmax().item()
            lens_top10 = torch.topk(jl_logits, 10).indices.tolist()
            hits += int(actual_top1 in lens_top10)
        acc = hits / len(SANITY_PROMPTS)
        gate_per_layer[layer] = acc
        print(f"  L{layer}: {acc:.1%} ({hits}/{len(SANITY_PROMPTS)})")

    gate_acc = float(np.mean(list(gate_per_layer.values())))
    gate_pass = gate_acc > GATE_THRESHOLD
    results["sanity_gate"] = {
        "per_layer": {str(k): v for k, v in gate_per_layer.items()},
        "mean_accuracy": gate_acc,
        "threshold": GATE_THRESHOLD,
        "pass": gate_pass,
    }

    # Standard transport cosine on the test set (reported either way; on a
    # gate pass this is the "standard J-lens works on this MoE" number)
    std_cosines = {}
    for layer in TARGET_LAYERS:
        cs = transport_cosine(standard_lens, test_prompts, layer)
        std_cosines[layer] = cs
        print(f"  L{layer} standard transport cosine: {np.mean(cs):.4f}")
    results["standard_transport_cosine"] = {
        str(l): float(np.mean(cs)) for l, cs in std_cosines.items()}
    checkpoint()

    print(f"\n  GATE {'PASS' if gate_pass else 'FAIL'}: "
          f"{gate_acc:.1%} vs threshold {GATE_THRESHOLD:.0%}")
    if not gate_pass:
        print("  Standard J-lens fails even on the easiest MoE (8 experts, top-2).")
        print("  The Qwen3 12% gate failure is NOT cardinality-specific.")
        if stop_on_gate_fail:
            results["verdict"] = "GATE_FAIL_STOPPED"
            checkpoint()
            return results
        print("  Continuing anyway — conditioned-vs-random is still informative.")

    # ------------------------------------------------------------
    # 4. Capture routing for all prompts
    #    Mixtral: layers[i].block_sparse_moe.gate is nn.Linear(d, 8);
    #    forward-hook output = raw router logits (batch*seq, 8).
    #    Top-2 from softmax, exactly as MixtralSparseMoeBlock.forward does.
    # ------------------------------------------------------------
    print("\n=== 4. Capturing routing (fit + test prompts) ===")
    t0 = time.time()

    def capture_routing(prompts):
        """Per prompt, per target layer: 8-dim expert-frequency vector
        (fraction of tokens whose top-2 includes each expert)."""
        captured = {layer: [] for layer in TARGET_LAYERS}
        current = {layer: None for layer in TARGET_LAYERS}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                logits = output  # (batch*seq, n_experts)
                weights = torch.softmax(logits.detach().float(), dim=-1)
                _, top2 = torch.topk(weights, TOP_K, dim=-1)  # (n_tokens, 2)
                current[layer_idx] = top2.cpu()
            return hook_fn

        for layer_idx in TARGET_LAYERS:
            hooks.append(
                hf_model.model.layers[layer_idx].block_sparse_moe.gate
                .register_forward_hook(make_hook(layer_idx)))

        try:
            for prompt in prompts:
                input_ids = model.encode(prompt, max_length=128)
                with torch.no_grad():
                    model.forward(input_ids)
                for layer_idx in TARGET_LAYERS:
                    top2 = current[layer_idx]
                    n_tokens = top2.shape[0]
                    vec = np.zeros(N_EXPERTS, dtype=np.float64)
                    idx, counts = np.unique(top2.numpy().ravel(), return_counts=True)
                    vec[idx] = counts
                    captured[layer_idx].append(vec / max(n_tokens, 1))
        finally:
            for h in hooks:
                h.remove()
        return {layer: np.array(v) for layer, v in captured.items()}

    fit_vectors = capture_routing(fit_prompts)
    test_vectors = capture_routing(test_prompts)
    results["routing_capture_time"] = time.time() - t0
    for layer in TARGET_LAYERS:
        mean_load = fit_vectors[layer].mean(axis=0)
        print(f"  L{layer} expert load: "
              + " ".join(f"{x:.2f}" for x in mean_load))

    # ------------------------------------------------------------
    # 5. Cluster by routing (k-means, k=3-7, silhouette-selected)
    # ------------------------------------------------------------
    print("\n=== 5. Clustering routing vectors ===")
    clusters = {}
    for layer in TARGET_LAYERS:
        vecs = fit_vectors[layer]
        best_k, best_score, best_labels, best_km = 3, -1.0, None, None
        for k in range(3, 8):
            if k >= len(vecs):
                break
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(vecs)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(vecs, labels)
            if score > best_score:
                best_k, best_score, best_labels, best_km = k, score, labels, km
        if best_labels is None:
            best_labels = np.zeros(len(vecs), dtype=int)
            best_km = None
        sizes = [int(x) for x in np.bincount(best_labels) if x > 0]
        clusters[layer] = {"labels": best_labels, "km": best_km,
                           "k": int(len(np.unique(best_labels))),
                           "silhouette": float(best_score), "sizes": sizes}
        print(f"  L{layer}: k={clusters[layer]['k']}, "
              f"silhouette={best_score:.3f}, sizes={sizes}")

    results["clusters"] = {
        str(l): {"k": c["k"], "silhouette": c["silhouette"], "sizes": c["sizes"],
                 "labels": c["labels"].tolist()}
        for l, c in clusters.items()}
    checkpoint()

    # ------------------------------------------------------------
    # 6 + 7. Fit conditioned lenses per cluster, and random control
    # ------------------------------------------------------------
    MIN_GROUP = 6  # 50 prompts / k<=7 gives small groups; keep the floor low

    def fit_groups(tag, layer, labels):
        """Fit one lens per group label; returns {group_id: lens_path|None}."""
        out = {}
        for gid in np.unique(labels):
            group_prompts = [p for p, l in zip(fit_prompts, labels) if l == gid]
            n = len(group_prompts)
            if n < MIN_GROUP:
                print(f"    {tag} L{layer} G{gid}: {n} prompts (< {MIN_GROUP}, skip)")
                out[int(gid)] = None
                continue
            print(f"    {tag} L{layer} G{gid}: fitting on {n} prompts...",
                  end=" ", flush=True)
            t0 = time.time()
            try:
                torch.cuda.empty_cache()
                lens = jlens.fit(model, group_prompts, source_layers=[layer],
                                 dim_batch=4, max_seq_len=128)
                path = f"/results/mixtral_{tag}_lens_L{layer}_G{gid}.pt"
                lens.save(path)
                print(f"done ({time.time()-t0:.0f}s)")
                out[int(gid)] = path
            except Exception as e:
                print(f"FAILED: {e}")
                out[int(gid)] = None
        return out

    print("\n=== 6. Fitting conditioned lenses ===")
    conditioned_paths = {}
    for layer in TARGET_LAYERS:
        conditioned_paths[layer] = fit_groups("cond", layer, clusters[layer]["labels"])
        RESULTS_VOL.commit()

    print("\n=== 7. Fitting random control ===")
    rng = np.random.RandomState(42)
    random_labels = {}
    random_paths = {}
    for layer in TARGET_LAYERS:
        # Same group sizes as the real clusters, random assignment
        sizes = clusters[layer]["sizes"]
        labels = np.zeros(len(fit_prompts), dtype=int)
        perm = rng.permutation(len(fit_prompts))
        offset = 0
        for gid, sz in enumerate(sizes):
            labels[perm[offset:offset + sz]] = gid
            offset += sz
        random_labels[layer] = labels
        random_paths[layer] = fit_groups("rand", layer, labels)
        RESULTS_VOL.commit()

    # ------------------------------------------------------------
    # 8. Evaluate: standard vs conditioned vs random
    # ------------------------------------------------------------
    print("\n=== 8. Evaluation ===")
    lens_cache = {}

    def load_lens(path):
        if path not in lens_cache:
            lens_cache[path] = jlens.JacobianLens.load(path)
        return lens_cache[path]

    def eval_grouped(paths, layer, test_labels):
        cosines = []
        for prompt, gid in zip(test_prompts, test_labels):
            path = paths.get(int(gid))
            if not path:
                continue
            cs = transport_cosine(load_lens(path), [prompt], layer)
            if cs:
                cosines.append(cs[0])
        return cosines

    evaluation = {}
    eval_rng = np.random.RandomState(99)
    for layer in TARGET_LAYERS:
        print(f"\n  Layer {layer}:")
        std_cs = std_cosines[layer]  # computed in step 3
        std_mean = float(np.mean(std_cs))
        print(f"    Standard:    {std_mean:.4f} (n={len(std_cs)})")

        # Conditioned: assign test prompts to nearest routing cluster
        km = clusters[layer]["km"]
        if km is not None:
            test_labels = km.predict(test_vectors[layer])
        else:
            test_labels = np.zeros(len(test_prompts), dtype=int)
        cond_cs = eval_grouped(conditioned_paths[layer], layer, test_labels)
        cond_mean = float(np.mean(cond_cs)) if cond_cs else 0.0
        print(f"    Conditioned: {cond_mean:.4f} (n={len(cond_cs)})")

        # Random: mirror the protocol with random test assignment
        sizes = clusters[layer]["sizes"]
        test_rand = np.zeros(len(test_prompts), dtype=int)
        perm = eval_rng.permutation(len(test_prompts))
        offset = 0
        for gid, sz in enumerate(sizes):
            end = min(offset + max(1, round(sz / len(fit_prompts) * len(test_prompts))),
                      len(test_prompts))
            test_rand[perm[offset:end]] = gid
            offset = end
        rand_cs = eval_grouped(random_paths[layer], layer, test_rand)
        rand_mean = float(np.mean(rand_cs)) if rand_cs else 0.0
        print(f"    Random:      {rand_mean:.4f} (n={len(rand_cs)})")

        p_value = 1.0
        if len(cond_cs) > 5 and len(rand_cs) > 5:
            _, p_value = mannwhitneyu(cond_cs, rand_cs, alternative="greater")
            print(f"    Mann-Whitney (cond > rand): p={p_value:.6f}")

        evaluation[str(layer)] = {
            "standard": {"mean": std_mean, "n": len(std_cs)},
            "conditioned": {"mean": cond_mean, "n": len(cond_cs)},
            "random": {"mean": rand_mean, "n": len(rand_cs)},
            "p_value": float(p_value),
        }

    results["evaluation"] = evaluation

    # ------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------
    all_std = [e["standard"]["mean"] for e in evaluation.values()]
    all_cond = [e["conditioned"]["mean"] for e in evaluation.values()]
    all_rand = [e["random"]["mean"] for e in evaluation.values()]
    pvals = [e["p_value"] for e in evaluation.values()]
    corrected = [min(p * len(pvals), 1.0) for p in pvals]  # Bonferroni
    sig_layers = sum(1 for p in corrected if p < 0.05)

    cond_beats_rand = np.mean(all_cond) > np.mean(all_rand)

    if gate_pass and cond_beats_rand and sig_layers > 0:
        verdict = "METHOD_WORKS"  # Qwen3 negative is cardinality-specific
    elif gate_pass and not cond_beats_rand:
        verdict = "GATE_PASS_CONDITIONING_NULL"  # clustering fails even at C(8,2)=28
    elif gate_pass:
        verdict = "GATE_PASS_INCONCLUSIVE"  # cond > rand but not significant
    else:
        verdict = "GATE_FAIL"  # standard J-lens broken even on easiest MoE

    results["verdict"] = verdict
    results["summary"] = {
        "gate_accuracy": gate_acc,
        "gate_pass": gate_pass,
        "standard_mean": float(np.mean(all_std)),
        "conditioned_mean": float(np.mean(all_cond)),
        "random_mean": float(np.mean(all_rand)),
        "significant_layers": sig_layers,
        "n_layers": len(pvals),
        "total_time_min": (time.time() - t_start) / 60,
    }
    checkpoint()

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"  Sanity gate: {gate_acc:.1%} ({'PASS' if gate_pass else 'FAIL'}, "
          f"threshold {GATE_THRESHOLD:.0%})")
    print(f"  Standard:    {np.mean(all_std):.4f}")
    print(f"  Conditioned: {np.mean(all_cond):.4f}")
    print(f"  Random:      {np.mean(all_rand):.4f}")
    print(f"  Significant layers (Bonferroni): {sig_layers}/{len(pvals)}")
    print(f"  Total: {results['summary']['total_time_min']:.1f} min")
    print(f"{'='*60}")
    if verdict == "METHOD_WORKS":
        print("Standard J-lens works on Mixtral and conditioning beats random:")
        print("the Qwen3 negative is cardinality-specific. Proceed to the")
        print("weighted mixture-of-transports build (lit review, hours 6-48).")
    elif verdict == "GATE_FAIL":
        print("Standard J-lens fails on 8 experts / top-2. The Qwen3 gate")
        print("failure is not about cardinality — the transport itself breaks")
        print("on MoE layers. Go straight to the exact Jacobian decomposition.")
    return results


@app.local_entrypoint()
def main(n_fit: int = 50, n_test: int = 40, stop_on_gate_fail: bool = False):
    result = run_sanity.remote(
        n_fit=n_fit, n_test=n_test, stop_on_gate_fail=stop_on_gate_fail)
    print(f"\nVerdict: {result.get('verdict')}")
    print(f"Summary: {result.get('summary')}")
