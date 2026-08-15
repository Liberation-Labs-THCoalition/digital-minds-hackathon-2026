"""Weighted Mixture-of-Transports for MoE J-lens on Qwen3-30B-A3B.

The chain-rule fix for the path-conditioned negative result. The MoE layer
Jacobian decomposes EXACTLY as:

    J(x) = sum_e g_e(x) * J_expert_e(x)  +  gating term

so the transport operator should be a routing-weighted mixture of per-expert
transports — with the weights known per token, for free, from the forward pass:

    T(x) = T_shared + sum_e w_e(x) * U_e V_e^T

- T_shared (full rank d x d) absorbs the standing committee's shared computation
- w_e(x) = the model's own normalized top-8 routing weights (0 elsewhere)
- U_e V_e^T = low-rank per-expert transport deltas
- Rank budget (CommitteeAudit): committee experts (top-5 by routing mass) r=64,
  tail experts r=16, sub-0.1%-mass experts pooled into a single delta

This is ridge regression on logged (h_in, h_final, w) triples — the model is
linear in the parameters given logged weights. No iterative J-lens fitting.
Per-expert deltas are fit by GREEDY BACKFITTING (sequential ridge on the
running residual, ordered by routing mass) because committee experts co-occur
on nearly every token and independent fits would be collinear.

Controls:
- standard J-lens (existing baseline from volume)
- k-means conditioned lenses (existing negative result from volume, if present)
- T_shared-only ablation (do the per-expert deltas add anything?)
- shuffled-weights control: identical pipeline, identical capacity, routing
  weight vectors permuted across tokens (destroys the x <-> w correspondence
  while preserving every marginal). This is the null swarm check: if the
  weighted mixture only beats standard because of extra parameters, the
  shuffled model will match it.

Lit review: infrastructure/moe_jlens_next_steps_litreview.md
Prior negative: conditioned_jlens_results.json (k-means clustering, no effect)

Usage:
    modal run modal_weighted_transport.py
    modal run modal_weighted_transport.py --layer 24 --n-fit 672 --n-test 100
    modal run modal_weighted_transport.py --recapture  # ignore cached activations
"""

import modal

app = modal.App("moe-jlens-weighted-transport")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "numpy", "scikit-learn", "scipy", "accelerate")
    .pip_install("jlens @ git+https://github.com/anthropics/jacobian-lens.git")
    .pip_install("datasets")
)

RESULTS_VOL = modal.Volume.from_name("moe-jlens-results", create_if_missing=True)
MODEL_NAME = "Qwen/Qwen3-30B-A3B"

COMMITTEE_SIZE = 5          # top experts by routing mass get the big rank budget
COMMITTEE_RANK = 64
TAIL_RANK = 16
POOLED_RANK = 16
POOL_MASS_THRESHOLD = 1e-3  # experts below 0.1% of total routing mass are pooled
MIN_EXPERT_TOKENS = 64      # experts with fewer tokens than this contribute nothing useful
CHUNK = 16384               # rows per GPU chunk for normal-equation accumulation


# ================================================================
# Model / data loading (same conventions as modal_moe_chunked.py)
# ================================================================
def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from jlens.hf import HFLensModel

    config = AutoConfig.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HFLensModel(hf_model, tokenizer, compile=False)

    n_experts = getattr(config, 'num_experts', getattr(config, 'num_local_experts', 128))
    top_k = getattr(config, 'num_experts_per_tok', 8)
    return model, hf_model, tokenizer, n_experts, top_k


def load_prompts(n_fit=672, n_test=100):
    from jlens.examples import load_wikitext_prompts
    all_prompts = load_wikitext_prompts(n_prompts=n_fit + n_test)
    return all_prompts[:n_fit], all_prompts[n_fit:n_fit + n_test]


# ================================================================
# Capture: per-token (h_layer, h_final, routing weights) triples
# ================================================================
def capture_triples(model, hf_model, prompts, layer, final_layer, n_experts, top_k,
                    max_seq_len=128):
    """One forward pass per prompt with an ActivationRecorder at [layer, final]
    plus a gate hook on the target layer's router.

    Qwen3MoeTopKRouter output is a 3-tuple:
      [0] raw router logits (n_tokens, n_experts)
      [1] top-k gate weights (n_tokens, top_k) — softmaxed/normalized
      [2] top-k expert indices (n_tokens, top_k)

    Returns dict of CPU tensors:
      X (N, d) fp16        — hidden state at target layer, all token positions
      Y (N, d) fp16        — hidden state at final layer, same positions
      w_idx (N, top_k) i64 — top-k expert indices per token
      w_val (N, top_k) f32 — top-k routing weights per token
      prompt_id (N,) i64   — which prompt each token came from
      is_last (N,) bool    — last-token position per prompt (used for eval)
    """
    import torch
    from jlens.hooks import ActivationRecorder

    gate_capture = []

    def gate_hook(module, inp, output):
        if isinstance(output, tuple) and len(output) >= 3:
            gate_capture.append((output[1].detach().float().cpu(),
                                 output[2].detach().cpu()))
        else:
            logits = output[0] if isinstance(output, tuple) else output
            vals, idx = torch.topk(logits.detach().float(), top_k, dim=-1)
            w = torch.softmax(vals, dim=-1)
            gate_capture.append((w.cpu(), idx.cpu()))

    hook = hf_model.model.layers[layer].mlp.gate.register_forward_hook(gate_hook)

    Xs, Ys, Wis, Wvs, Pids, Lasts = [], [], [], [], [], []
    try:
        for p_i, prompt in enumerate(prompts):
            input_ids = model.encode(prompt, max_length=max_seq_len)
            gate_capture.clear()
            with torch.no_grad():
                with ActivationRecorder(model.layers, at=[layer, final_layer]) as rec:
                    model.forward(input_ids)
                    h = rec.activations[layer][0].detach().to(torch.float16).cpu()
                    h_final = rec.activations[final_layer][0].detach().to(torch.float16).cpu()

            if not gate_capture:
                continue
            w_val, w_idx = gate_capture[0]
            if w_idx.dim() == 3:
                w_idx = w_idx.squeeze(0)
                w_val = w_val.squeeze(0)

            n_tok = min(h.shape[0], h_final.shape[0], w_idx.shape[0])
            if n_tok == 0:
                continue
            Xs.append(h[:n_tok])
            Ys.append(h_final[:n_tok])
            Wis.append(w_idx[:n_tok].long())
            Wvs.append(w_val[:n_tok].float())
            Pids.append(torch.full((n_tok,), p_i, dtype=torch.long))
            last = torch.zeros(n_tok, dtype=torch.bool)
            last[-1] = True
            Lasts.append(last)

            if (p_i + 1) % 100 == 0:
                print(f"    captured {p_i + 1}/{len(prompts)} prompts", flush=True)
    finally:
        hook.remove()

    return {
        "X": torch.cat(Xs), "Y": torch.cat(Ys),
        "w_idx": torch.cat(Wis), "w_val": torch.cat(Wvs),
        "prompt_id": torch.cat(Pids), "is_last": torch.cat(Lasts),
    }


# ================================================================
# Routing statistics: committee / tail / pooled partition
# ================================================================
def expert_stats(w_idx, w_val, n_experts):
    """Per-expert total routing mass, token counts, Gini, partition."""
    import torch

    mass = torch.zeros(n_experts, dtype=torch.float64)
    counts = torch.zeros(n_experts, dtype=torch.long)
    flat_idx = w_idx.reshape(-1)
    flat_val = w_val.reshape(-1).double()
    mass.scatter_add_(0, flat_idx, flat_val)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))

    share = (mass / mass.sum()).numpy()
    order = share.argsort()[::-1].copy()

    committee = [int(e) for e in order[:COMMITTEE_SIZE]]
    tail, pooled = [], []
    for e in order[COMMITTEE_SIZE:]:
        e = int(e)
        if share[e] >= POOL_MASS_THRESHOLD and int(counts[e]) >= MIN_EXPERT_TOKENS:
            tail.append(e)
        else:
            pooled.append(e)

    # Gini of routing-mass distribution
    s = sorted(share)
    n = len(s)
    cum = 0.0
    for i, v in enumerate(s):
        cum += (2 * (i + 1) - n - 1) * v
    gini = cum / (n * sum(s)) if sum(s) > 0 else 0.0

    return {
        "share": share, "counts": counts.numpy(),
        "committee": committee, "tail": tail, "pooled": pooled,
        "gini": float(gini),
        "committee_mass": float(sum(share[e] for e in committee)),
    }


# ================================================================
# Fitting: T_shared ridge + greedy per-expert low-rank backfitting
# ================================================================
def _accum_normal_eq(A, B, device, weights=None):
    """Accumulate A^T A and A^T B in chunks (A, B fp16 CPU; result fp32 GPU).

    Optional per-row weights scale the FEATURE side only: with z_i = w_i * a_i
    this solves min ||Z D - B||^2, i.e. the model b_i ~= w_i * a_i D — exactly
    the per-expert term of the mixture. (Scaling B too would fit a_i D ~= b_i
    with w^2 sample weighting, which mis-scales the delta by the typical w.)"""
    import torch
    d_a, d_b = A.shape[1], B.shape[1]
    AtA = torch.zeros(d_a, d_a, device=device)
    AtB = torch.zeros(d_a, d_b, device=device)
    for i in range(0, A.shape[0], CHUNK):
        a = A[i:i + CHUNK].to(device).float()
        b = B[i:i + CHUNK].to(device).float()
        if weights is not None:
            a = a * weights[i:i + CHUNK].to(device).float().unsqueeze(1)
        AtA += a.T @ a
        AtB += a.T @ b
    return AtA, AtB


def _ridge_solve(AtA, AtB, alpha):
    """Solve (AtA + lam I) T = AtB with lam scaled to the data: alpha * mean diag."""
    import torch
    d = AtA.shape[0]
    lam = alpha * AtA.diagonal().mean().clamp(min=1e-8)
    return torch.linalg.solve(AtA + lam * torch.eye(d, device=AtA.device), AtB)


def fit_weighted_transport(X, Y, w_idx, w_val, prompt_id, n_experts, device,
                           alphas=(1e-4, 1e-3, 1e-2, 1e-1), tag=""):
    """Fit T_shared (ridge, alpha chosen on a held-out prompt split) then
    per-expert low-rank deltas by greedy backfitting on the residual.

    Returns (transport_dict, fit_report).
    """
    import torch

    N, d = X.shape
    stats = expert_stats(w_idx, w_val, n_experts)

    # --- T_shared: alpha sweep on a 10% held-out prompt split ---
    val_mask = (prompt_id % 10 == 0)
    tr = ~val_mask
    AtA_tr, AtY_tr = _accum_normal_eq(X[tr], Y[tr], device)
    Xv = X[val_mask]
    Yv = Y[val_mask]

    best_alpha, best_err = alphas[0], float("inf")
    for a in alphas:
        T = _ridge_solve(AtA_tr, AtY_tr, a)
        err = 0.0
        for i in range(0, Xv.shape[0], CHUNK):
            xv = Xv[i:i + CHUNK].to(device).float()
            yv = Yv[i:i + CHUNK].to(device).float()
            err += ((xv @ T - yv) ** 2).sum().item()
        if err < best_err:
            best_err, best_alpha = err, a
    print(f"  [{tag}] T_shared alpha={best_alpha} (val MSE/token={best_err / max(1, Xv.shape[0]):.4f})")

    # Refit T_shared on ALL tokens with chosen alpha
    AtA, AtY = _accum_normal_eq(X, Y, device)
    T_shared = _ridge_solve(AtA, AtY, best_alpha)
    del AtA, AtY, AtA_tr, AtY_tr

    # --- Running residual R = Y - X @ T_shared (fp16 CPU) ---
    R = torch.empty_like(Y)
    ss_res_shared = 0.0
    for i in range(0, N, CHUNK):
        x = X[i:i + CHUNK].to(device).float()
        y = Y[i:i + CHUNK].to(device).float()
        r = y - x @ T_shared
        ss_res_shared += (r ** 2).sum().item()
        R[i:i + CHUNK] = r.to(torch.float16).cpu()

    # Per-token weight lookup: dense (N,) weight of expert e where selected, else 0
    def expert_weight_vector(experts):
        w = torch.zeros(N)
        for e in experts:
            sel = (w_idx == e)
            w += (w_val * sel.float()).sum(dim=1)
        return w

    # --- Greedy backfitting, ordered by routing mass ---
    deltas = {}
    fit_order = ([("expert", e, COMMITTEE_RANK) for e in stats["committee"]]
                 + [("expert", e, TAIL_RANK) for e in stats["tail"]]
                 + ([("pooled", stats["pooled"], POOLED_RANK)] if stats["pooled"] else []))

    ss_res = ss_res_shared
    for kind, e, rank in fit_order:
        experts = e if kind == "pooled" else [e]
        w = expert_weight_vector(experts)
        rows = (w > 0).nonzero(as_tuple=True)[0]
        if rows.numel() < MIN_EXPERT_TOKENS:
            continue

        Xe, Re, we = X[rows], R[rows], w[rows]
        # Ridge on features z = w_e * x, target r  (linear-in-params given logged w)
        ZtZ, ZtR = _accum_normal_eq(Xe, Re, device, weights=we)
        D = _ridge_solve(ZtZ, ZtR, best_alpha)
        del ZtZ, ZtR

        # Low-rank truncation via SVD
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        r_eff = min(rank, d)
        U_r = (U[:, :r_eff] * S[:r_eff]).contiguous()
        Vh_r = Vh[:r_eff, :].contiguous()

        # Update running residual on this expert's rows (chunked)
        removed = 0.0
        for i in range(0, rows.numel(), CHUNK):
            ridx = rows[i:i + CHUNK]
            x = X[ridx].to(device).float()
            wv = we[i:i + CHUNK].to(device).float().unsqueeze(1)
            pred = wv * ((x @ U_r) @ Vh_r)
            r_old = R[ridx].to(device).float()
            r_new = r_old - pred
            removed += ((r_old ** 2).sum() - (r_new ** 2).sum()).item()
            R[ridx] = r_new.to(torch.float16).cpu()
        ss_res -= removed

        key = "pooled" if kind == "pooled" else int(e)
        deltas[key] = {"U": U_r.to(torch.float32).cpu(), "Vh": Vh_r.to(torch.float32).cpu(),
                       "experts": [int(x) for x in experts], "rank": r_eff,
                       "n_tokens": int(rows.numel()),
                       "variance_removed": float(removed)}

    ss_tot = 0.0
    for i in range(0, N, CHUNK):
        y = Y[i:i + CHUNK].to(device).float()
        ss_tot += ((y - y.mean(0, keepdim=True)) ** 2).sum().item()

    report = {
        "alpha": best_alpha,
        "n_tokens": int(N),
        "r2_shared": float(1.0 - ss_res_shared / ss_tot),
        "r2_full": float(1.0 - ss_res / ss_tot),
        "n_deltas": len(deltas),
        "committee": stats["committee"],
        "committee_mass": stats["committee_mass"],
        "gini": stats["gini"],
        "n_tail": len(stats["tail"]),
        "n_pooled": len(stats["pooled"]),
    }
    print(f"  [{tag}] R2 shared={report['r2_shared']:.4f}  full={report['r2_full']:.4f}  "
          f"deltas={len(deltas)} (committee mass {stats['committee_mass']:.1%}, "
          f"Gini {stats['gini']:.4f})")

    transport = {"T_shared": T_shared.to(torch.float32).cpu(), "deltas": deltas,
                 "pooled_experts": stats["pooled"], "d": d}
    return transport, report


def move_transport(tm, device):
    """Return a copy of the transport dict with all tensors on device (do this
    once before the eval loop instead of shipping T_shared per prompt)."""
    return {
        "T_shared": tm["T_shared"].to(device),
        "deltas": {k: {**dl, "U": dl["U"].to(device), "Vh": dl["Vh"].to(device)}
                   for k, dl in tm["deltas"].items()},
        "pooled_experts": set(tm["pooled_experts"]),
        "d": tm["d"],
    }


def apply_weighted_transport(h, w_idx_tok, w_val_tok, tm, device):
    """T(x) = T_shared + sum_e w_e(x) * U_e Vh_e applied to h (1, d) fp32 GPU.
    tm must already be on device (see move_transport)."""
    y = h @ tm["T_shared"]
    pooled_w = 0.0
    for j in range(w_idx_tok.shape[0]):
        e = int(w_idx_tok[j])
        w = float(w_val_tok[j])
        if e in tm["deltas"]:
            dl = tm["deltas"][e]
            y = y + w * ((h @ dl["U"]) @ dl["Vh"])
        elif e in tm["pooled_experts"]:
            pooled_w += w
    if pooled_w > 0 and "pooled" in tm["deltas"]:
        dl = tm["deltas"]["pooled"]
        y = y + pooled_w * ((h @ dl["U"]) @ dl["Vh"])
    return y


# ================================================================
# Diagnostics: principal angles between committee deltas
# ================================================================
def committee_principal_angles(tm, committee):
    """Mean principal angle (degrees) between input subspaces (Vh rows) of
    committee expert deltas — quantifies the near-orthogonality claim."""
    import torch
    import numpy as np

    angles = {}
    for i, e1 in enumerate(committee):
        for e2 in committee[i + 1:]:
            if e1 not in tm["deltas"] or e2 not in tm["deltas"]:
                continue
            V1 = torch.linalg.qr(tm["deltas"][e1]["Vh"].T.float())[0]
            V2 = torch.linalg.qr(tm["deltas"][e2]["Vh"].T.float())[0]
            s = torch.linalg.svdvals(V1.T @ V2).clamp(-1, 1)
            ang = torch.rad2deg(torch.acos(s)).mean().item()
            angles[f"{e1}-{e2}"] = float(ang)
    mean_angle = float(np.mean(list(angles.values()))) if angles else None
    return {"pairwise_deg": angles, "mean_deg": mean_angle}


# ================================================================
# Evaluation
# ================================================================
def transport_cosine(model, transported, h_final, device):
    """Softmax cosine between unembedded transported state and actual final state.
    Same metric as modal_moe_chunked.stage_evaluate."""
    import torch
    jl_logits = model.unembed(transported.to(device)).squeeze(0).float()
    actual_logits = model.unembed(h_final.to(device)).squeeze(0).float()
    return torch.nn.functional.cosine_similarity(
        torch.softmax(jl_logits, -1).unsqueeze(0),
        torch.softmax(actual_logits, -1).unsqueeze(0)).item()


# ================================================================
# Main experiment
# ================================================================
@app.function(
    image=image,
    gpu="H100",
    timeout=14400,
    volumes={"/results": RESULTS_VOL},
)
def run_weighted_transport(layer: int = 24, n_fit: int = 672, n_test: int = 100,
                           recapture: bool = False):
    import json
    import os
    import pickle
    import time

    import numpy as np
    import torch
    import jlens
    from scipy.stats import wilcoxon

    device = "cuda"
    t_start = time.time()

    print("=" * 65)
    print("WEIGHTED MIXTURE-OF-TRANSPORTS — MoE J-LENS")
    print(f"T(x) = T_shared + sum_e w_e(x) * U_e V_e^T   (layer {layer})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 65)

    # ------------------------------------------------------------
    # Load model + prompts
    # ------------------------------------------------------------
    print(f"\n[LOAD] {MODEL_NAME} in bf16...")
    t0 = time.time()
    model, hf_model, tokenizer, n_experts, top_k = load_model()
    final_layer = model.n_layers - 1
    print(f"  {model.n_layers} layers, d={model.d_model}, {n_experts} experts top-{top_k} "
          f"({time.time() - t0:.0f}s)")

    fit_prompts, test_prompts = load_prompts(n_fit=n_fit, n_test=n_test)

    # ------------------------------------------------------------
    # Load existing routing data from the volume (prior negative result)
    # ------------------------------------------------------------
    routing_data, fit_vectors = None, None
    try:
        with open("/results/chunked_routing.json") as f:
            routing_data = json.load(f)
        with open("/results/chunked_routing_vectors.pkl", "rb") as f:
            fit_vectors = pickle.load(f)
        print(f"  Loaded prior routing data: layers {list(routing_data['clusters'].keys())}")
    except Exception as e:
        print(f"  WARN: prior routing data unavailable ({e}) — "
              f"conditioned baseline will be skipped")

    # ------------------------------------------------------------
    # Capture (h_L24, h_final, w) triples for fitting — cached on volume
    # ------------------------------------------------------------
    cap_path = f"/results/weighted_capture_L{layer}.pt"
    if os.path.exists(cap_path) and not recapture:
        print(f"\n[CAPTURE] Loading cached fit capture from {cap_path}")
        cap = torch.load(cap_path, map_location="cpu")
    else:
        print(f"\n[CAPTURE] Capturing {len(fit_prompts)} fit prompts at L{layer} + L{final_layer}...")
        t0 = time.time()
        cap = capture_triples(model, hf_model, fit_prompts, layer, final_layer,
                              n_experts, top_k, max_seq_len=128)
        print(f"  {cap['X'].shape[0]} tokens in {time.time() - t0:.0f}s")
        torch.save(cap, cap_path)
        RESULTS_VOL.commit()

    N = cap["X"].shape[0]
    print(f"  Fit tokens: {N} across {int(cap['prompt_id'].max()) + 1} prompts")

    # Cross-check committee against the prior routing-frequency vectors
    stats = expert_stats(cap["w_idx"], cap["w_val"], n_experts)
    print(f"\n[COMMITTEE] top-{COMMITTEE_SIZE} experts by routing mass: {stats['committee']}")
    print(f"  committee mass: {stats['committee_mass']:.1%}  Gini: {stats['gini']:.4f}")
    print(f"  tail (r={TAIL_RANK}): {len(stats['tail'])} experts, "
          f"pooled (<{POOL_MASS_THRESHOLD:.1%} mass): {len(stats['pooled'])} experts")
    if fit_vectors is not None and layer in fit_vectors:
        freq_committee = fit_vectors[layer].mean(axis=0).argsort()[::-1][:COMMITTEE_SIZE]
        overlap = len(set(stats["committee"]) & set(int(x) for x in freq_committee))
        print(f"  overlap with prior frequency-vector committee: {overlap}/{COMMITTEE_SIZE}")

    # ------------------------------------------------------------
    # Fit the weighted mixture-of-transports (real weights)
    # ------------------------------------------------------------
    print(f"\n[FIT] Weighted mixture-of-transports (ridge + greedy backfitting)...")
    t0 = time.time()
    tm_real, report_real = fit_weighted_transport(
        cap["X"], cap["Y"], cap["w_idx"], cap["w_val"], cap["prompt_id"],
        n_experts, device, tag="real")
    print(f"  fitted in {time.time() - t0:.0f}s")
    torch.save({"transport": tm_real, "report": report_real, "layer": layer},
               f"/results/weighted_transport_L{layer}.pt")
    RESULTS_VOL.commit()

    # ------------------------------------------------------------
    # Fit the shuffled-weights control (identical capacity, meaningless weights)
    # ------------------------------------------------------------
    print(f"\n[FIT] Shuffled-weights control (null swarm check)...")
    t0 = time.time()
    rng = np.random.RandomState(42)
    perm = torch.from_numpy(rng.permutation(N))
    tm_shuf, report_shuf = fit_weighted_transport(
        cap["X"], cap["Y"], cap["w_idx"][perm], cap["w_val"][perm], cap["prompt_id"],
        n_experts, device, tag="shuffled")
    print(f"  fitted in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------
    # Diagnostics: near-orthogonality of committee deltas
    # ------------------------------------------------------------
    angles = committee_principal_angles(tm_real, stats["committee"])
    if angles["mean_deg"] is not None:
        print(f"\n[DIAG] Mean principal angle between committee delta input subspaces: "
              f"{angles['mean_deg']:.1f} deg")

    # ------------------------------------------------------------
    # Load baselines: standard lens + conditioned lenses (prior negative)
    # ------------------------------------------------------------
    standard_lens = None
    try:
        standard_lens = jlens.JacobianLens.load("/results/moe_30b_jlens.pt")
        print(f"\n[BASELINE] Standard lens loaded")
    except Exception as e:
        print(f"\n[BASELINE] WARN: standard lens unavailable ({e})")

    conditioned_manifest = None
    for path in (f"/results/conditioned_manifest_L{layer}.json",
                 "/results/chunked_conditioned_manifest.json"):
        try:
            with open(path) as f:
                d = json.load(f)
            conditioned_manifest = d.get("clusters", d.get("lenses", d)) \
                if "layer" in d else d.get(str(layer), {})
            print(f"[BASELINE] Conditioned manifest loaded from {path}")
            break
        except Exception:
            continue
    if not conditioned_manifest:
        print(f"[BASELINE] WARN: no conditioned manifest for L{layer}")

    # ------------------------------------------------------------
    # Capture test prompts
    # ------------------------------------------------------------
    print(f"\n[EVAL] Capturing {len(test_prompts)} test prompts...")
    tcap = capture_triples(model, hf_model, test_prompts, layer, final_layer,
                           n_experts, top_k, max_seq_len=128)
    last_rows = tcap["is_last"].nonzero(as_tuple=True)[0]
    n_eval = last_rows.numel()
    print(f"  {n_eval} evaluation positions (last token per prompt)")

    # Conditioned baseline needs test routing-frequency vectors + KMeans assignment
    cond_labels = None
    if conditioned_manifest and routing_data and fit_vectors is not None \
            and layer in fit_vectors and str(layer) in routing_data["clusters"]:
        from sklearn.cluster import KMeans
        test_freq = np.zeros((len(test_prompts), n_experts))
        for row in range(tcap["w_idx"].shape[0]):
            pid = int(tcap["prompt_id"][row])
            for e in tcap["w_idx"][row].tolist():
                test_freq[pid, e] += 1.0
        counts = np.bincount(tcap["prompt_id"].numpy(), minlength=len(test_prompts))
        test_freq /= (counts[:, None] + 1e-10)
        k = routing_data["clusters"][str(layer)]["k"]
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(fit_vectors[layer])
        cond_labels = km.predict(test_freq)

    # Shuffled control eval: permute test weight vectors across prompts
    rng_t = np.random.RandomState(99)
    test_perm = rng_t.permutation(n_eval)

    # ------------------------------------------------------------
    # Evaluate all methods (paired, same prompts)
    # ------------------------------------------------------------
    cos = {"weighted": [], "shared_only": [], "shuffled_control": [],
           "standard": [], "conditioned": []}
    cond_lens_cache = {}
    tm_real_dev = move_transport(tm_real, device)
    tm_shuf_dev = move_transport(tm_shuf, device)

    for j in range(n_eval):
        row = int(last_rows[j])
        h = tcap["X"][row].to(device).float().unsqueeze(0)
        h_final = tcap["Y"][row].float().unsqueeze(0)
        wi, wv = tcap["w_idx"][row], tcap["w_val"][row]

        # (d) weighted mixture-of-transports
        y = apply_weighted_transport(h, wi, wv, tm_real_dev, device)
        cos["weighted"].append(transport_cosine(model, y, h_final, device))

        # ablation: T_shared only
        y = h @ tm_real_dev["T_shared"]
        cos["shared_only"].append(transport_cosine(model, y, h_final, device))

        # (e) shuffled control: shuffled-fit model + permuted test weights
        prow = int(last_rows[int(test_perm[j])])
        y = apply_weighted_transport(h, tcap["w_idx"][prow], tcap["w_val"][prow],
                                     tm_shuf_dev, device)
        cos["shuffled_control"].append(transport_cosine(model, y, h_final, device))

        # (a) standard J-lens
        if standard_lens is not None:
            transported = standard_lens.transport(h.cpu(), layer)
            cos["standard"].append(transport_cosine(model, transported, h_final, device))

        # (b) k-means conditioned (prior negative result)
        if cond_labels is not None:
            pid = int(tcap["prompt_id"][row])
            cid = str(int(cond_labels[pid]))
            entry = conditioned_manifest.get(cid)
            if entry and entry.get("path"):
                try:
                    if cid not in cond_lens_cache:
                        cond_lens_cache[cid] = jlens.JacobianLens.load(entry["path"])
                    transported = cond_lens_cache[cid].transport(h.cpu(), layer)
                    cos["conditioned"].append(
                        transport_cosine(model, transported, h_final, device))
                except Exception:
                    pass

    # ------------------------------------------------------------
    # Statistics + verdict
    # ------------------------------------------------------------
    def summarize(name):
        c = cos[name]
        if not c:
            return {"transport_cosine": None, "n": 0}
        return {"transport_cosine": float(np.mean(c)), "std": float(np.std(c)),
                "n": len(c)}

    def paired_p(a_name, b_name):
        a, b = cos[a_name], cos[b_name]
        if len(a) < 6 or len(a) != len(b):
            return None
        try:
            diffs = np.array(a) - np.array(b)
            if np.allclose(diffs, 0):
                return 1.0
            return float(wilcoxon(a, b, alternative="greater")[1])
        except Exception:
            return None

    summary = {name: summarize(name) for name in cos}
    p_vs_shuffled = paired_p("weighted", "shuffled_control")
    p_vs_standard = paired_p("weighted", "standard")
    p_vs_shared = paired_p("weighted", "shared_only")

    print(f"\n{'=' * 65}")
    print(f"RESULTS — transport cosine at L{layer} (n={n_eval} test prompts)")
    print(f"{'=' * 65}")
    for name in ("weighted", "shared_only", "shuffled_control", "standard", "conditioned"):
        s = summary[name]
        if s["transport_cosine"] is not None:
            print(f"  {name:18s} {s['transport_cosine']:.4f} ± {s['std']:.4f} (n={s['n']})")
        else:
            print(f"  {name:18s} unavailable")
    print(f"\n  Wilcoxon (weighted > shuffled control): p={p_vs_shuffled}")
    print(f"  Wilcoxon (weighted > standard):         p={p_vs_standard}")
    print(f"  Wilcoxon (weighted > shared-only):      p={p_vs_shared}")

    w_mean = summary["weighted"]["transport_cosine"]
    std_mean = summary["standard"]["transport_cosine"]
    shuf_mean = summary["shuffled_control"]["transport_cosine"]

    beats_shuffled = (p_vs_shuffled is not None and p_vs_shuffled < 0.05
                      and w_mean > shuf_mean)
    beats_standard = (std_mean is None) or (w_mean > std_mean)
    deltas_matter = (p_vs_shared is not None and p_vs_shared < 0.05)

    if beats_shuffled and beats_standard and deltas_matter:
        verdict = ("SUCCESS: weighted mixture-of-transports works — routing-weight "
                   "conditioning fixes what cluster conditioning could not"
                   if w_mean > 0.5 else
                   "PARTIAL: weighted mixture beats all controls but below 0.5 threshold")
    elif beats_standard and not beats_shuffled:
        verdict = ("NULL_SWARM: weighted mixture beats standard but not the "
                   "shuffled-weights control — improvement is capacity, not routing")
    elif beats_shuffled and not deltas_matter:
        verdict = ("INCONCLUSIVE: beats shuffled control but per-expert deltas add "
                   "nothing over T_shared alone")
    else:
        verdict = ("NEGATIVE: MoE transport is not routing-decomposable even with "
                   "exact weights — second well-controlled negative")

    print(f"\n  VERDICT: {verdict}")

    results = {
        "experiment": "Weighted Mixture-of-Transports MoE J-lens",
        "model": MODEL_NAME,
        "layer": layer,
        "final_layer": final_layer,
        "n_experts": n_experts,
        "top_k": top_k,
        "n_fit_prompts": len(fit_prompts),
        "n_fit_tokens": int(N),
        "n_test_prompts": n_eval,
        "rank_budget": {"committee": COMMITTEE_RANK, "tail": TAIL_RANK,
                        "pooled": POOLED_RANK, "pool_threshold": POOL_MASS_THRESHOLD},
        "committee": stats["committee"],
        "committee_mass": stats["committee_mass"],
        "routing_gini": stats["gini"],
        "fit_report_real": report_real,
        "fit_report_shuffled": report_shuf,
        "committee_principal_angles": angles,
        "evaluation": summary,
        "per_prompt_cosines": {k: v for k, v in cos.items() if v},
        "p_values": {"weighted_vs_shuffled": p_vs_shuffled,
                     "weighted_vs_standard": p_vs_standard,
                     "weighted_vs_shared_only": p_vs_shared},
        "verdict": verdict,
        "runtime_s": time.time() - t_start,
        "gpu": torch.cuda.get_device_name(),
    }

    out_path = "/results/weighted_transport_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    RESULTS_VOL.commit()
    print(f"\n  Results saved to {out_path}")
    return results


@app.local_entrypoint()
def main(layer: int = 24, n_fit: int = 672, n_test: int = 100,
         recapture: bool = False):
    result = run_weighted_transport.remote(
        layer=layer, n_fit=n_fit, n_test=n_test, recapture=recapture)
    print("\n" + "=" * 65)
    print("REMOTE EXECUTION COMPLETE")
    print("=" * 65)
    if isinstance(result, dict):
        print(f"  Verdict: {result.get('verdict')}")
        import json
        print(json.dumps(result.get("evaluation", {}), indent=2))
        print(json.dumps(result.get("p_values", {}), indent=2))
