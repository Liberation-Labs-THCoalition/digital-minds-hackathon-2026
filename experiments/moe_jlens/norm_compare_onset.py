#!/usr/bin/env python3
"""Onset sweep WITH the final RMSNorm. Local re-run of modal_onset_sweep.py.

THE BUG: modal_onset_sweep.py:129 does `model.lm_head(h)`. The model does
`lm_head(self.norm(h))` (modeling_qwen3_moe.py:519). Every Table 0 cell unembedded an
un-normalised residual, so S4.0's "the readout works / approaching the tautology" was false
and the paper's own unexplained empty-token domination at dense L63 is the signature.

Ground truth is model.generate(max_new_tokens=1), unchanged. Both variants are computed in
the SAME forward pass so the comparison is exact.
"""
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return",
    "Water is composed of hydrogen and",
    "To be or not to be, that is the",
    "The mitochondria is the powerhouse of the",
    "In 1969, Neil Armstrong became the first person to walk on t",
    "E = mc^2 was formulated by Albert",
    "SELECT name FROM users WHERE age >",
]
GRIDS = {"moe": [12, 24, 31, 36, 41, 44, 47], "dense": [16, 32, 42, 48, 54, 59, 63]}

def run(model_id, kind):
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="mps")
    m.eval()
    core = m.model
    n_layers = len(core.layers)
    truth = {}
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(m.device)
        with torch.no_grad():
            g = m.generate(ids, max_new_tokens=1, do_sample=False)
        truth[p] = g[0, ids.shape[1]].item()
    out = []
    for layer in GRIDS[kind]:
        raw = normed = 0
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(m.device)
            cap = {}
            def hook(mod, inp, o, _c=cap):
                _c["h"] = (o[0] if isinstance(o, tuple) else o)[:, -1:, :].detach()
            hd = core.layers[layer].register_forward_hook(hook)
            with torch.no_grad():
                m(ids)
            hd.remove()
            h = cap["h"].to(m.lm_head.weight.dtype)
            if truth[p] in torch.topk(m.lm_head(h).squeeze().float(), 10).indices.tolist():
                raw += 1
            if truth[p] in torch.topk(m.lm_head(core.norm(h)).squeeze().float(), 10).indices.tolist():
                normed += 1
        pct = round(100 * layer / (n_layers - 1))
        out.append({"layer": layer, "depth_pct": pct, "no_norm": raw, "with_norm": normed, "total": len(PROMPTS)})
        print("  L%-3d %3d%%   no-norm %d/8   WITH-NORM %d/8" % (layer, pct, raw, normed), flush=True)
    del m
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return {"model": model_id, "n_layers": n_layers, "curve": out}

if __name__ == "__main__":
    res = {}
    import os
    only = os.environ.get("ONLY")
    pairs = [("moe", "Qwen/Qwen3-30B-A3B"), ("dense", "/Users/[AGENT]/models/qwen3-32b")]
    for kind, mid in [p for p in pairs if not only or p[0] == only]:
        print("=== %s: %s ===" % (kind, mid), flush=True)
        try:
            res[kind] = run(mid, kind)
        except Exception as e:
            print("  FAILED: %r" % (e,), flush=True)
            res[kind] = {"error": repr(e)}
    with open("/tmp/onset_sweep_normfix_%s.json" % (only or "all"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved /tmp/onset_sweep_normfix.json")
