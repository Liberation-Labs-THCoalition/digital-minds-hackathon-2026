# `moe_result.json` is INVALID — do not cite

Withdrawn 2026-08-16.

`moe_result.json` records `{"verdict": "GATE_FAIL", "accuracy": 0.125}` for
Qwen3-30B-A3B. **That accuracy measures nothing.**

The gate that produced it, `experiments/moe_jlens/modal_moe_jlens_baseline.py:176`,
built its ground truth by appending a fixed string to each prompt:

```python
extended = model.encode(prompt + " the", max_length=66)
actual_next = extended[0, n_pos].item()   # always the token for " the"
```

`actual_next` is a constant. The gate scored whether the token `" the"` appeared in the
lens's top ten — not whether the model's continuation did. The 1-of-8 result is therefore
uninformative about the lens, and the `GATE_FAIL` verdict is unfounded.

The file is left unmodified because it is the accurate record of what the broken run
emitted. It is the *interpretation* that was wrong, not the serialization.

**Supersedes it:** `onset_sweep_results.json`, produced by
`experiments/moe_jlens/modal_onset_sweep.py`, which takes ground truth from
`model.generate(max_new_tokens=1)` — the model's own greedy continuation. See paper S4.0.

**Not affected:** the three-arm transport-cosine comparison (`conditioned_jlens_results.json`)
never went through this gate, and the ablation-KL and coordinate-swap metrics in
`modal_gurnee_eval.py` do not use this construction either.

Found by the Agni adversarial review, 2026-08-16.
