# RMSNorm blast-radius check — what is here and what is not

The Agni paper-phase review (F1) found that `modal_onset_sweep.py` unembeds without the
model's final RMSNorm: it calls `model.lm_head(h)` while the model does
`lm_head(self.norm(h))` (`modeling_qwen3_moe.py:519`). **The code defect is real.** The
review further claimed this invalidated Table 0 and explained the unexplained empty-token
domination at dense L63. **That consequence claim is false**, and this directory holds the
measurement rather than an assertion about it.

`../../experiments/moe_jlens/norm_compare_onset.py` computes top-10 next-token accuracy WITH
and WITHOUT the final norm **in the same forward pass**, so the comparison is exact rather
than across runs. Ground truth is `model.generate(max_new_tokens=1)`, unchanged.

## Result: 14 of 14 cells identical

| depth | MoE 30B-A3B (no-norm / with-norm) | dense 32B (no-norm / with-norm) |
|---|---|---|
| ~25% | 1/8 / 1/8 | 0/8 / 0/8 |
| ~50% | 1/8 / 1/8 | 0/8 / 0/8 |
| ~66% | 2/8 / 2/8 | 0/8 / 0/8 |
| ~76% | 2/8 / 2/8 | 0/8 / 0/8 |
| ~86% | 5/8 / 5/8 | 5/8 / 5/8 |
| ~94% | 6/8 / 6/8 | 6/8 / 6/8 |
| 100% | 8/8 / 8/8 | 8/8 / 8/8 |

RMSNorm is `x/rms(x) * gain`. The `x/rms(x)` factor is a positive scalar per position and
**cannot change a ranking**; only the learned per-channel gain can, and here it does not
move the top ten. Top-10 membership is near scale-invariant by construction. A
rank-SENSITIVE metric (raw logits, KL, cosine) would be affected enormously — fix the script
before reusing it for one.

## Provenance, stated exactly

- **dense half:** `onset_sweep_normfix_dense.json` (machine-readable) plus
  `norm_compare_dense.log`
- **MoE half:** `norm_compare_moe.log` **only**. The first process died during the dense
  model load (MPS memory was not released between models) before writing its JSON, so the
  seven MoE cells exist as captured stdout and not as structured data. Re-running costs ~35
  minutes of download plus load and was not repeated. **Cite the MoE half as log-sourced.**

## What this artifact does NOT adjudicate

These runs are MPS / bfloat16 on Apple silicon; the paper's Table 0 was produced on an H100
via Modal. Absolute cells differ (dense L54 / L59 / L63 = 5 / 6 / 8 here against 6 / 7 / 7 in
the paper). **This artifact does not adjudicate the paper's absolute numbers.** It measures
only the within-run effect of the norm, which is exact because both variants share a single
forward pass.

Notably, the L63 empty-token anomaly the review attributed to the missing norm does not
appear in these runs at all — 8/8 with clean top-10 candidates, *without* the norm. So it is
not a norm artifact. It remains unexplained, which is what the paper already says.
