"""Dense positive control for Track 6 — on Modal, protocol-matched to the MoE arms.

WHY THIS EXISTS
  Track 6 measured 4-9% transport cosine on Qwen3-30B-A3B across standard, conditioned and
  random arms. Two things remain unestablished:
    H_pipeline  can this fitting pipeline fit anything at all? A pipeline that fits nothing
                produces exactly the observed signature (everything low, nothing separating).
    H_metric    what does softmax-probability cosine over a 152k vocabulary score on a model
                where the J-lens is known to work?
  No published dense figure exists under this metric — Gurnee et al. report none and their
  reference implementation computes no cosine at all (verified, full text + repo, two
  independent extractions). The number has to be measured or the paper stays uncalibrated,
  which is what it currently and honestly says.

WHY MODAL RATHER THAN STARSHIP
  Two prior runs died on shared hardware:
    run 1  device_map="auto" mmapped the safetensors and never materialised a weight.
           6h01m, RSS 1.4GB against a 61GB model, 373M pageins, zero layers. MY CODE BUG.
           Fixed here (explicit device_map + residency gate). Modal would not have saved it.
    run 2  loaded correctly (65.5GB resident, verified) then thrashed against three other
           tenants. Killed at 358 min, swap 1GB -> 21.4GB, 37% duty, still on layer 1 of 3.
           Dedicated GPU removes every term in that sentence.
  The stronger reason: the MoE arms ran on Modal H100 (modal_moe_chunked.py:60). Running the
  control on the SAME hardware retires pre-registration threat T2 (device/dtype mismatch
  between control and arms), which no amount of Starship time could close. This makes the
  control properly matched, not merely completable.

PROTOCOL — byte-for-byte the MoE arms, verified against modal_moe_chunked.py:404-422
  fit    jlens.fit(source_layers=[L], dim_batch=4, max_seq_len=128)     [MoE:224,292]
  eval   model.encode(prompt, max_length=64)   <- 64, not 128, their asymmetry  [MoE:410]
  record ActivationRecorder at [layer, final_layer], last token only [0,-1:]    [MoE:411-414]
  score  cos(softmax(unembed(transport(h))), softmax(unembed(h_final)))         [MoE:415-421]
  corpus load_wikitext_prompts, N_FIT=672 / 100 held out — MATCHES THE ARMS EXACTLY.
         Starship ran 200 because 672 was unaffordable there; on dedicated hardware 672 is
         affordable and removes the pre-registered escalation branch entirely.

PRE-REGISTERED, WRITTEN BEFORE ANY OUTPUT (unchanged from the Starship attempts)
  PASS       mean > 0.7    pipeline + metric validated; the MoE null is strong
  AMBIGUOUS  0.15 - 0.7    pipeline fits but the metric does not reach the scale the paper
                           compares against; MoE numbers are not comparable to published work
  FAIL       mean <= 0.15  HALT. Dense scores in the MoE range: the 4-9% regime is the
                           METRIC, not the architecture, and no Track 6 number is
                           interpretable as a measurement of MoE.
  All three outcomes get reported. This is not run to confirm anything.

USAGE
  modal run modal_dense_control.py --stage fit --layer 16     # smoke test FIRST
  modal run modal_dense_control.py --stage fit --layer 32
  modal run modal_dense_control.py --stage fit --layer 48
  modal run modal_dense_control.py --stage evaluate
"""
import modal

app = modal.App("dense-control-jlens")

# Image copied verbatim from modal_moe_chunked.py:17-23. A control built on a different
# image than its arms is a weaker control, and this one is proven on this exact workload.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "numpy", "scikit-learn", "scipy", "accelerate")
    .pip_install("jlens @ git+https://github.com/anthropics/jacobian-lens.git")
    .pip_install("datasets")
)

RESULTS_VOL = modal.Volume.from_name("dense-control-results", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen3-32B"      # DENSE. model_type qwen3, 64 layers, d_model 5120.
REL_DEPTHS = [0.25, 0.50, 0.75]    # -> L16/32/48 of 64, depth-matched to MoE L12/24/36 of 48
N_FIT = 672                        # matches the MoE arms exactly
N_EVAL = 100
FIT_SEQ_LEN = 128
EVAL_MAX_LEN = 64
SEED = 42
PASS_T, FAIL_T = 0.7, 0.15

WEIGHTS_GB = 61.0
RESIDENT_MIN_GB = 45.0     # below this the weights were never materialised (run 1)
LAYER_BUDGET_MIN = 240.0   # enforced from the heartbeat thread, see _heartbeat


def _heartbeat(stop_evt, state, log):
    """Liveness AND progress AND the deadline.

    Run 2's budget check was written AFTER the jlens.fit() it was meant to bound, so it
    could only fire once the thing it was interrupting had already returned. It never ran:
    358 minutes elapsed against a 150-minute budget. The only clock still running during a
    blocked fit is this thread, so the deadline lives here and this thread kills the process.

    Modal's own timeout will also kill the container, but it writes no diagnosis. This says
    WHY, into the log, before dying.
    """
    import os
    import time
    t0 = time.time()
    while not stop_evt.wait(60):
        import torch
        alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        resv = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
        el = (time.time() - t0) / 60
        log(f"HEARTBEAT elapsed={el:.1f}min phase={state.get('phase')} "
            f"cuda_alloc={alloc:.1f}GB reserved={resv:.1f}GB")
        started = state.get("phase_started_at")
        if started and (time.time() - started) / 60 > LAYER_BUDGET_MIN:
            log(f"HALT (budget): phase {state.get('phase')} ran "
                f"{(time.time()-started)/60:.0f}min against a {LAYER_BUDGET_MIN:.0f}min "
                f"budget, cuda_alloc={alloc:.1f}GB. Killing from the heartbeat thread — "
                f"run 2 sat here for 358 minutes because this check lived after the call "
                f"it was timing.")
            os._exit(2)


def _load_model(log):
    """Load with an EXPLICIT device map and verify residency before returning."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from jlens.hf import HFLensModel

    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    log(f"config: model_type={cfg.model_type} layers={cfg.num_hidden_layers} "
        f"hidden={cfg.hidden_size}")
    if cfg.model_type != "qwen3" or getattr(cfg, "num_experts", None):
        raise SystemExit(f"FATAL: not the dense model this control requires "
                         f"(model_type={cfg.model_type}). Refusing to report a bad control.")

    free, total = torch.cuda.mem_get_info()
    log(f"GATE 1 preflight: {free/1e9:.0f}GB free of {total/1e9:.0f}GB, "
        f"need ~{WEIGHTS_GB*1.15:.0f}GB")
    if free / 1e9 < WEIGHTS_GB * 1.05:
        raise SystemExit(f"HALT (preflight): {free/1e9:.0f}GB free, need "
                         f"{WEIGHTS_GB*1.05:.0f}GB. Request a larger GPU rather than "
                         f"reducing dim_batch or max_seq_len — those are protocol "
                         f"parameters matched to the MoE arms.")

    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    load_min = (time.time() - t0) / 60

    # GATE 2 — residency. Run 1's exact failure: from_pretrained returned in 24 seconds
    # having only established a memory map. "loaded in 0.4 min" is a RECORD of a load.
    alloc = torch.cuda.memory_allocated() / 1e9
    log(f"GATE 2 residency: from_pretrained returned in {load_min:.1f}min, "
        f"cuda_allocated={alloc:.1f}GB")
    dmap = getattr(hf, "hf_device_map", None) or {}
    bad = {k: v for k, v in dmap.items() if v in ("disk", "meta", "cpu")}
    if bad:
        raise SystemExit(f"HALT (residency): {len(bad)} modules offloaded, "
                         f"e.g. {list(bad.items())[:3]}")
    if alloc < RESIDENT_MIN_GB:
        raise SystemExit(
            f"HALT (residency): only {alloc:.1f}GB allocated for a {WEIGHTS_GB:.0f}GB "
            f"model. The weights were not materialised — this is run 1's failure, caught "
            f"at second {int(load_min*60)} instead of hour six.")
    log(f"GATE 2 PASS — {alloc:.1f}GB genuinely resident")
    return HFLensModel(hf, tok, compile=False), cfg


def _corpus(log):
    from jlens.examples import load_wikitext_prompts
    texts = load_wikitext_prompts(n_prompts=N_FIT + N_EVAL)
    fit, ev = texts[:N_FIT], texts[N_FIT:N_FIT + N_EVAL]
    log(f"corpus: fit=[0:{N_FIT}] eval=[{N_FIT}:{N_FIT+N_EVAL}] — MATCHES the MoE arms "
        f"(modal_moe_chunked.py:51-54)")
    return fit, ev


@app.function(image=image, gpu="H100", timeout=14400, volumes={"/results": RESULTS_VOL})
def stage_fit(layer: int):
    """Fit the lens at one layer and checkpoint it. One layer per stage so no single
    failure costs more than one layer's work — the pattern modal_moe_chunked.py uses."""
    import threading
    import time
    import torch
    import numpy as np
    import jlens

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log(f"DENSE POSITIVE CONTROL — {MODEL_NAME} — FIT L{layer}")
    log(f"prereg: PASS>{PASS_T} AMBIGUOUS {FAIL_T}-{PASS_T} FAIL<={FAIL_T} | seed={SEED}")

    model, cfg = _load_model(log)
    fit_prompts, _ = _corpus(log)

    state = {"phase": f"fit_L{layer}", "phase_started_at": time.time()}
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(stop, state, log), daemon=True).start()
    try:
        t0 = time.time()
        lens = jlens.fit(model, fit_prompts, source_layers=[layer],
                         dim_batch=4, max_seq_len=FIT_SEQ_LEN)
        fit_min = (time.time() - t0) / 60
        log(f"fitted in {fit_min:.1f} min")
    finally:
        stop.set()

    # Persist via the library's own save, DEFAULT dtype — exactly what the MoE arms do
    # (modal_moe_chunked.py:226,294). The default is float16; saving float32 would be
    # "better" and would make this control LESS comparable to the arms it calibrates.
    # Matching beats improving, for a control.
    path = f"/results/dense_lens_L{layer}.pt"
    lens.save(path)
    with open(f"/results/dense_lens_L{layer}.meta.json", "w") as f:
        import json as _json
        _json.dump({"layer": layer, "n_fit": N_FIT, "seed": SEED,
                    "model": MODEL_NAME, "fit_minutes": round(fit_min, 1)}, f, indent=2)
    RESULTS_VOL.commit()
    log(f"saved {path}")
    return {"layer": layer, "fit_minutes": round(fit_min, 1)}


@app.function(image=image, gpu="H100", timeout=7200, volumes={"/results": RESULTS_VOL})
def stage_evaluate():
    """Load every fitted lens and score held-out prompts. Byte-for-byte the MoE eval_lens."""
    import json
    import os
    import time
    import torch
    import numpy as np
    import jlens
    from jlens import ActivationRecorder

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, cfg = _load_model(log)
    _, eval_prompts = _corpus(log)
    n_layers = cfg.num_hidden_layers
    layers = [int(round(r * n_layers)) for r in REL_DEPTHS]
    final_layer = model.n_layers - 1

    def eval_lens(lens_obj, prompts, layer):
        """modal_moe_chunked.py:404-422, verbatim. One deliberate deviation: no_grad,
        which is numerically inert (verified bit-identical through the full jlens path)
        and only prevents 100 autograd graphs per layer."""
        cosines = []
        with torch.no_grad():
            for prompt in prompts:
                input_ids = model.encode(prompt, max_length=EVAL_MAX_LEN)
                with ActivationRecorder(model.layers, at=[layer, final_layer]) as rec:
                    model.forward(input_ids)
                    h = rec.activations[layer][0, -1:].detach().float()
                    h_final = rec.activations[final_layer][0, -1:].detach().float()
                transported = lens_obj.transport(h.cpu(), layer)
                jl = model.unembed(transported.to(input_ids.device)).squeeze(0).float()
                actual = model.unembed(h_final).squeeze(0).float()
                cosines.append(torch.nn.functional.cosine_similarity(
                    torch.softmax(jl, -1).unsqueeze(0),
                    torch.softmax(actual, -1).unsqueeze(0)).item())
        return cosines

    results, missing = {}, []
    for L in layers:
        p = f"/results/dense_lens_L{L}.pt"
        if not os.path.exists(p):
            missing.append(L)
            continue
        # jlens.JacobianLens.load — same call the MoE arms use (modal_moe_chunked.py:342).
        # NOTE: JacobianLens.__init__ requires keyword-only n_prompts and d_model, so
        # reconstructing from a raw jacobians dict raises TypeError. Verified against the
        # installed library before this script ever reached a GPU.
        lens = jlens.JacobianLens.load(p)
        cs = eval_lens(lens, eval_prompts, L)
        m = float(np.mean(cs))
        results[str(L)] = {"mean": m, "n": len(cs), "cosines": [round(c, 6) for c in cs],
                           "relative_depth": round(L / n_layers, 3)}
        log(f"L{L}: transport cosine = {m:.4f} (n={len(cs)})")

    if missing:
        raise SystemExit(f"HALT: no fitted lens for layers {missing}. Run stage_fit for "
                         f"each before evaluating — a partial control is not a control.")

    mean = float(np.mean([v["mean"] for v in results.values()]))
    verdict = "PASS" if mean > PASS_T else ("FAIL_METRIC" if mean <= FAIL_T else "AMBIGUOUS")
    boundary = min(abs(mean - PASS_T), abs(mean - FAIL_T))
    note = None
    if boundary < 0.05:
        note = (f"BOUNDARY-ADJACENT: mean is {boundary:.4f} from a threshold. Per prereg T3 "
                f"a bootstrap CI is required before any paper claim. Raw cosines saved.")

    out = {"experiment": "dense_positive_control_modal", "model": MODEL_NAME,
           "model_type": cfg.model_type, "n_layers": n_layers,
           "hidden_size": cfg.hidden_size, "n_fit": N_FIT, "n_eval": N_EVAL, "seed": SEED,
           "protocol": "byte-matched to modal_moe_chunked.py:404-422; corpus matches "
                       "MoE arms exactly (fit[0:672], eval[672:772])",
           "prereg": {"pass": PASS_T, "fail": FAIL_T},
           "per_layer": results, "mean_transport_cosine": mean, "verdict": verdict,
           "boundary_note": note,
           "moe_reference": {"standard": 0.05246, "conditioned_2layer": 0.06710,
                             "random_2layer": 0.06443,
                             "PROVENANCE": "hardcoded from prior runs; not re-verified "
                                           "against a primary file for this run"}}
    with open("/results/dense_control_result.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()

    log("=" * 60)
    log(f"MEAN TRANSPORT COSINE: {mean:.4f}   VERDICT: {verdict}")
    log("MoE for comparison: standard 0.0525 conditioned 0.0671 random 0.0644")
    if note:
        log("!! " + note)
    if verdict == "FAIL_METRIC":
        log("HALT — dense scores in the MoE range. The 4-9% regime is the METRIC.")
    return out


@app.local_entrypoint()
def main(stage: str = "fit", layer: int = 16):
    if stage == "fit":
        print(stage_fit.remote(layer=layer))
    elif stage == "evaluate":
        import json
        print(json.dumps(stage_evaluate.remote(), indent=2))
    else:
        raise SystemExit("stage must be 'fit' or 'evaluate'")
