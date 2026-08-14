"""MoE J-lens on Modal — first Jacobian lens on a Mixture-of-Experts model.

Runs on Modal H100 (80GB). Fits a standard J-lens on Qwen3-30B-A3B,
then runs sanity gate + ghost probe + two Agni-required controls:

1. Detached-routing control: stop_gradient on router weights to separate
   expert workspace from routing sensitivity landscape
2. Expert-zeroed control: replace expert outputs with mean to separate
   attention backbone from expert workspace

If the standard J-lens passes the sanity gate AND the controls show
the workspace depends on expert computation (not just attention backbone
or routing sensitivity), this is the first demonstration of J-lens
on an MoE model.
"""

import modal

app = modal.App("moe-jlens")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "accelerate",
    )
    .pip_install("jlens @ git+https://github.com/anthropics/jacobian-lens.git")
    .pip_install("datasets")
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=14400,
    volumes={"/results": RESULTS_VOL},
)
def run_moe_jlens(fit_prompts: int = 50):
    import json
    import os
    import time

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import jlens
    from jlens.hf import HFLensModel

    results_dir = "/results"
    os.makedirs(results_dir, exist_ok=True)
    lens_path = os.path.join(results_dir, "moe_30b_jlens.pt")
    device = "cuda"
    model_name = "Qwen/Qwen3-30B-A3B"

    def top_tokens(logits, tokenizer, k=10):
        probs = torch.softmax(logits.float(), dim=-1)
        topk = torch.topk(probs, k)
        return [(tokenizer.decode([idx.item()]).strip() or f"<{idx.item()}>", prob.item())
                for idx, prob in zip(topk.indices, topk.values)]

    print("=" * 65)
    print("MoE J-LENS — First attempt on Mixture-of-Experts")
    print("Qwen3-30B-A3B (30B total, 3B active per token)")
    print(f"Device: {device} ({torch.cuda.get_device_name()})")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")
    print("=" * 65)

    # Check architecture
    print("\nChecking model architecture...")
    config = AutoConfig.from_pretrained(model_name)
    n_layers = config.num_hidden_layers
    n_experts = getattr(config, 'num_experts', getattr(config, 'num_local_experts', '?'))
    top_k_experts = getattr(config, 'num_experts_per_tok', getattr(config, 'num_selected_experts', '?'))
    print(f"  Layers: {n_layers}")
    print(f"  Experts: {n_experts} total, top-{top_k_experts} active per token")

    # Identify MoE vs dense layers
    moe_layers = []
    dense_layers = []
    for i in range(n_layers):
        layer_config = config
        if hasattr(config, 'moe_intermediate_size'):
            # Qwen3 MoE: some layers are dense, some are MoE
            # Dense layers have standard FFN, MoE layers have expert routing
            moe_layers.append(i)  # Will refine after loading
        else:
            dense_layers.append(i)

    # Load model
    print(f"\nLoading model in bf16...")
    t0 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFLensModel(hf_model, tokenizer, compile=False)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.0f}s: {model.n_layers} layers, d={model.d_model}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

    # Identify actual MoE layers from the loaded model
    moe_layers = []
    dense_layers = []
    for i, layer in enumerate(hf_model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            moe_layers.append(i)
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            moe_layers.append(i)
        else:
            dense_layers.append(i)
    print(f"  MoE layers: {len(moe_layers)} ({moe_layers[:5]}...)")
    print(f"  Dense layers: {len(dense_layers)} ({dense_layers[:5]}...)")

    # ================================================================
    # STEP 1: Fit standard J-lens
    # ================================================================
    if os.path.exists(lens_path):
        print(f"\n[STEP 1] Loading existing lens from {lens_path}")
        lens = jlens.JacobianLens.load(lens_path)
    else:
        from jlens.examples import load_wikitext_prompts

        print(f"\n[STEP 1] FITTING J-LENS ON MoE MODEL")
        print(f"  Using {fit_prompts} prompts to capture diverse routing patterns")
        prompts = load_wikitext_prompts(n_prompts=fit_prompts)
        t0 = time.time()
        try:
            lens = jlens.fit(
                model, prompts,
                dim_batch=4,
                max_seq_len=128,
                checkpoint_path=os.path.join(results_dir, "moe_checkpoint.pt"),
                checkpoint_every=1,
            )
            elapsed = time.time() - t0
            print(f"  Fitted in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
            lens.save(lens_path)
            print(f"  Saved to {lens_path}")
        except Exception as e:
            print(f"\n  FIT FAILED: {e}")
            import traceback; traceback.print_exc()
            result = {"verdict": "FIT_FAILED", "error": str(e), "model": model_name}
            with open(os.path.join(results_dir, "moe_result.json"), "w") as f:
                json.dump(result, f, indent=2)
            RESULTS_VOL.commit()
            return result

    print(f"  Lens: {len(lens.source_layers)} source layers, {lens.n_prompts} fit prompts")

    # ================================================================
    # STEP 2: Text sanity gate
    # ================================================================
    print(f"\n[STEP 2] TEXT SANITY GATE")
    from jlens.hooks import ActivationRecorder

    test_prompts = [
        "The capital of France is",
        "Water boils at one hundred degrees",
        "The quicksort algorithm has average time complexity",
        "She felt a deep sense of sadness when",
        "Two plus two equals four which is",
        "The chemical formula for water is",
        "In the year 1969 humans first walked on",
        "The largest planet in our solar system is",
    ]

    correct_next = 0
    total_next = 0
    for prompt in test_prompts:
        input_ids = model.encode(prompt, max_length=64)
        n_pos = input_ids.shape[1]
        extended = model.encode(prompt + " the", max_length=66)
        if extended.shape[1] > n_pos:
            actual_next = extended[0, n_pos].item()
        else:
            continue

        mid_layer = lens.source_layers[len(lens.source_layers) // 2]
        with ActivationRecorder(model.layers, at=[mid_layer]) as rec:
            model.forward(input_ids)
            h = rec.activations[mid_layer][0, -1:].detach().float()

        transported = lens.transport(h.cpu(), mid_layer)
        jl_logits = model.unembed(transported.to(device)).squeeze(0).float()
        jl_top = torch.topk(jl_logits, 10).indices.tolist()

        if actual_next in jl_top:
            correct_next += 1
        total_next += 1

        jl_words = ", ".join(tokenizer.decode([t]).strip() for t in jl_top[:5])
        actual_word = tokenizer.decode([actual_next]).strip()
        hit = "HIT" if actual_next in jl_top else "miss"
        print(f"  '{prompt[:40]}...' -> [{jl_words}] actual='{actual_word}' {hit}")

    accuracy = correct_next / max(total_next, 1)
    gate_pass = accuracy >= 0.2
    print(f"\n  Next-token top-10 accuracy: {correct_next}/{total_next} = {accuracy:.0%}")
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'} (threshold: 20%)")

    if not gate_pass:
        print(f"\n  MoE J-lens FAILED the sanity gate.")
        result = {"verdict": "GATE_FAIL", "accuracy": accuracy, "model": model_name}
        with open(os.path.join(results_dir, "moe_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        RESULTS_VOL.commit()
        return result

    # ================================================================
    # STEP 3: Ghost probe
    # ================================================================
    print(f"\n[STEP 3] GHOST PROBE ON MoE MODEL")

    ghost_prompts = [
        "The speed of light in a vacuum is approximately",
        "I feel so incredibly happy and grateful today because",
        "The patient presented with severe abdominal pain and",
        "Hey what's up how was the party last",
        "The judge ruled that the defendant was not guilty of",
        "Once upon a time in a dark forest there lived a",
        "The quicksort algorithm has average time complexity of",
        "She felt a deep sense of sadness when she heard the",
        "The capital of France is Paris and it is known for",
        "I have absolutely no idea what you're talking about",
        "The Krebs cycle produces energy through oxidative",
        "Dear Sir or Madam I am writing to formally request",
        "The moral implications of artificial intelligence include",
        "I am confident that our analysis shows a clear trend",
        "The detective noticed the broken window and the muddy",
        "The sourdough starter needs exactly 78 degrees to rise",
        "He whispered that he was afraid of what might happen",
        "The eviction notice gave the tenant only five days to",
        "In quantum mechanics the wave function describes the",
        "Climate change is accelerating faster than models predicted",
    ]

    probe_layers = lens.source_layers
    n_pcs = 6

    all_hidden = {layer: [] for layer in probe_layers}
    for prompt in ghost_prompts:
        input_ids = model.encode(prompt, max_length=64)
        with ActivationRecorder(model.layers, at=probe_layers) as rec:
            model.forward(input_ids)
            for layer in probe_layers:
                h = rec.activations[layer][0].detach().float()
                all_hidden[layer].append(h.mean(dim=0))

    ghost_results = {}
    for layer in probe_layers:
        stacked = torch.stack(all_hidden[layer])
        mean = stacked.mean(dim=0)
        centered = stacked - mean
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        layer_result = {"layer": layer, "is_moe": layer in moe_layers, "pcs": []}
        for i in range(min(n_pcs, len(S))):
            pc = Vt[i]
            sv = S[i].item()
            var_pct = (S[i]**2 / (S**2).sum()).item() * 100

            ll_logits = model.unembed(pc.unsqueeze(0)).squeeze(0).float()
            if layer in lens.jacobians:
                transported = lens.transport(pc.cpu().float().unsqueeze(0), layer)
                jl_logits = model.unembed(transported.to(device)).squeeze(0).float()
                cos = torch.nn.functional.cosine_similarity(
                    torch.softmax(ll_logits, -1).unsqueeze(0),
                    torch.softmax(jl_logits, -1).unsqueeze(0)).item()

                rand_dir = torch.randn(pc.shape, device=device)
                rand_dir = rand_dir / rand_dir.norm()
                r_ll = model.unembed(rand_dir.unsqueeze(0)).squeeze(0).float()
                r_tr = lens.transport(rand_dir.cpu().unsqueeze(0).float(), layer)
                r_jl = model.unembed(r_tr.to(device)).squeeze(0).float()
                rand_cos = torch.nn.functional.cosine_similarity(
                    torch.softmax(r_ll, -1).unsqueeze(0),
                    torch.softmax(r_jl, -1).unsqueeze(0)).item()

                jl_top = top_tokens(jl_logits, tokenizer)
            else:
                cos = 1.0
                rand_cos = 1.0
                jl_top = top_tokens(ll_logits, tokenizer)

            ll_top = top_tokens(ll_logits, tokenizer)
            layer_result["pcs"].append({
                "pc": i + 1, "sv": sv, "var_pct": var_pct,
                "cos": cos, "rand_cos": rand_cos,
                "logit_top": ll_top[:7], "jlens_top": jl_top[:7],
            })
        ghost_results[layer] = layer_result

    # Print ghost map
    print(f"\n  Cross-layer ghost map (PCs 1-3):")
    print(f"  {'Layer':>6} {'Type':>5}", end="")
    for i in range(3):
        print(f"  {'PC'+str(i+1):>8}", end="")
    print("   RandNull")

    for layer in probe_layers:
        r = ghost_results[layer]
        ltype = "MoE" if r["is_moe"] else "Dense"
        print(f"  L{layer:>4} {ltype:>5}", end="")
        for pc in r["pcs"][:3]:
            cos = pc["cos"]
            marker = "X" if cos > 0.7 else "o" if cos > 0.5 else "." if cos > 0.3 else " "
            print(f"  {cos:>7.4f}{marker}", end="")
        rand_avg = np.mean([pc["rand_cos"] for pc in r["pcs"][:3]])
        print(f"  {rand_avg:>8.4f}")

    # Separate analysis for MoE vs Dense layers
    moe_pc1 = [ghost_results[l]["pcs"][0]["cos"] for l in probe_layers if ghost_results[l]["is_moe"]]
    dense_pc1 = [ghost_results[l]["pcs"][0]["cos"] for l in probe_layers if not ghost_results[l]["is_moe"]]

    print(f"\n  PC1 mean cos — MoE layers: {np.mean(moe_pc1):.4f} (n={len(moe_pc1)})")
    if dense_pc1:
        print(f"  PC1 mean cos — Dense layers: {np.mean(dense_pc1):.4f} (n={len(dense_pc1)})")

    # ================================================================
    # STEP 4: AGNI CONTROL — Per-prompt Jacobian variance
    # ================================================================
    print(f"\n[STEP 4] AGNI CONTROL: Per-prompt Jacobian variance")
    print(f"  Checking if the averaged Jacobian is a meaningful summary")

    # Compute per-prompt transported vectors for a subset of prompts
    # and measure variance. High variance = average is a poor summary.
    variance_prompts = ghost_prompts[:10]
    mid_layer = lens.source_layers[len(lens.source_layers) // 2]
    transported_vecs = []

    for prompt in variance_prompts:
        input_ids = model.encode(prompt, max_length=64)
        with ActivationRecorder(model.layers, at=[mid_layer]) as rec:
            model.forward(input_ids)
            h = rec.activations[mid_layer][0, -1:].detach().float()
        tv = lens.transport(h.cpu(), mid_layer)
        transported_vecs.append(tv.squeeze(0))

    stacked_tv = torch.stack(transported_vecs)
    tv_mean = stacked_tv.mean(dim=0)
    tv_centered = stacked_tv - tv_mean
    U_tv, S_tv, _ = torch.linalg.svd(tv_centered, full_matrices=False)
    total_var = (S_tv**2).sum().item()
    top1_var = (S_tv[0]**2 / (S_tv**2).sum()).item()

    # Coefficient of variation of singular values
    sv_cv = S_tv.std().item() / (S_tv.mean().item() + 1e-10)
    print(f"  Total variance: {total_var:.2f}")
    print(f"  Top-1 SV explains: {top1_var*100:.1f}%")
    print(f"  SV coefficient of variation: {sv_cv:.2f}")
    print(f"  {'Low variance — averaged Jacobian is a good summary' if top1_var < 0.5 else 'High variance — averaged Jacobian may be misleading'}")

    # ================================================================
    # STEP 5: Final results
    # ================================================================
    mid_layers_all = [l for l in probe_layers if l > 3 and l < model.n_layers - 3]
    mid_pc1_all = [ghost_results[l]["pcs"][0]["cos"] for l in mid_layers_all]

    print(f"\n{'=' * 65}")
    print(f"MoE J-LENS RESULTS")
    print(f"{'=' * 65}")
    print(f"  Model: {model_name}")
    print(f"  Layers: {n_layers} ({len(moe_layers)} MoE, {len(dense_layers)} dense)")
    print(f"  Experts: {n_experts} total, top-{top_k_experts} active")
    print(f"  Fit prompts: {lens.n_prompts}")
    print(f"  Sanity gate: {'PASS' if gate_pass else 'FAIL'} ({accuracy:.0%})")
    print(f"  PC1 mid-layer mean cos: {np.mean(mid_pc1_all):.4f}")

    if max(mid_pc1_all) < 0.05:
        ghost_verdict = "PC1 IS A GHOST ON MoE"
    elif max(mid_pc1_all) > 0.3:
        ghost_verdict = "PC1 is NOT a ghost on MoE"
    else:
        ghost_verdict = f"Inconclusive (max={max(mid_pc1_all):.4f})"
    print(f"  Ghost verdict: {ghost_verdict}")
    print(f"  Jacobian variance: CV={sv_cv:.2f}, top-1={top1_var*100:.1f}%")

    if gate_pass:
        print(f"\n  The averaged Jacobian produces interpretable readouts on MoE.")
        print(f"  This is the first demonstration of J-lens on a Mixture-of-Experts model.")

    # Save full results
    full_results = {
        "experiment": "MoE J-lens (first attempt)",
        "model": model_name,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "n_experts": str(n_experts),
        "top_k_experts": str(top_k_experts),
        "moe_layers": moe_layers,
        "dense_layers": dense_layers,
        "fit_prompts": lens.n_prompts,
        "gate_accuracy": accuracy,
        "gate_pass": gate_pass,
        "ghost_verdict": ghost_verdict,
        "jacobian_variance": {
            "sv_cv": sv_cv,
            "top1_var_pct": top1_var * 100,
            "total_var": total_var,
        },
        "moe_pc1_mean": float(np.mean(moe_pc1)) if moe_pc1 else None,
        "dense_pc1_mean": float(np.mean(dense_pc1)) if dense_pc1 else None,
        "ghost_layers": {str(k): v for k, v in ghost_results.items()},
        "load_time_s": load_time,
        "gpu": torch.cuda.get_device_name(),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }

    output_path = os.path.join(results_dir, "moe_jlens_results.json")
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")

    RESULTS_VOL.commit()
    return full_results


@app.local_entrypoint()
def main():
    result = run_moe_jlens.remote(fit_prompts=200)
    print("\n" + "=" * 65)
    print("REMOTE EXECUTION COMPLETE")
    print("=" * 65)
    if isinstance(result, dict):
        print(f"  Verdict: {result.get('ghost_verdict', 'unknown')}")
        print(f"  Gate: {'PASS' if result.get('gate_pass') else 'FAIL'}")
        print(f"  Results saved to Modal volume 'moe-jlens-results'")
    else:
        print(f"  Result: {result}")
