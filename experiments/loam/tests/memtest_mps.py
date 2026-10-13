# Does MPS memory grow per generated token in plain HF generate() on Qwen3.5-27B, and does releasing the
# allocator's cache bound it without changing the output?
# Usage: python memtest_mps.py LABEL [--empty-every N]     (env PYTORCH_MPS_LOW_WATERMARK_RATIO for variant C)
# Bare model: no J-lens, no observer, no hooks. Same load path as run_loam.py (bf16, device_map="auto").
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

label = sys.argv[1]
every = int(sys.argv[sys.argv.index("--empty-every") + 1]) if "--empty-every" in sys.argv else 0
NEW = 800
GB = 1024 ** 3

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B", torch_dtype=torch.bfloat16, device_map="auto")
filler = " ".join(f"Line {i}: the glassworks by the sea keeps its furnace at a steady heat." for i in range(120))
msgs = [{"role": "system", "content": "You are an agent exploring a workshop by the sea. " + filler},
        {"role": "user", "content": "Describe, at length, everything you notice in the workshop."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
torch.mps.empty_cache()
base = torch.mps.driver_allocated_memory()

class Probe(StoppingCriteria):
    def __init__(self):
        self.step, self.peak, self.trace = 0, 0, []
    def __call__(self, input_ids, scores, **kw):
        self.step += 1
        if every and self.step % every == 0:
            torch.mps.empty_cache()
        m = torch.mps.driver_allocated_memory()
        self.peak = max(self.peak, m)
        if self.step % 100 == 0:
            self.trace.append((self.step, round(m / GB, 1)))
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

probe = Probe()
torch.manual_seed(42)
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=NEW, min_new_tokens=NEW, temperature=0.7, top_p=0.9,
                         do_sample=True, pad_token_id=tok.eos_token_id, stopping_criteria=StoppingCriteriaList([probe]))
dt = time.time() - t0
new = out[0][inputs.input_ids.shape[1]:]
res = {"label": label, "empty_every": every, "low_wm": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
       "prompt_tokens": int(inputs.input_ids.shape[1]), "new_tokens": int(len(new)),
       "base_gb": round(base / GB, 1), "peak_gb": round(probe.peak / GB, 1),
       "growth_mb_per_token": round((probe.peak - base) / max(1, len(new)) / 1024 ** 2, 1),
       "tok_per_s": round(len(new) / dt, 2), "trace_gb": probe.trace,
       "text_sha": __import__("hashlib").sha256(tok.decode(new).encode()).hexdigest()[:16]}
print("RESULT " + json.dumps(res), flush=True)
