# Frontier Workspace Probe — Run Manifest

## Model

- **Name:** GLM-5.2 744B (abliterated)
- **Format:** Colibri NVFP4
- **Path:** `/mnt/data2/models/colibri/glm-5.2-abliterated`
- **Config hash:** `ea8773d953a605422dba2b95e43c4dd9` (config.json)
- **Architecture:** MoE, hybrid (dense + MoE layers), `first_k_dense_replace = 3`
- **Tokenizer hash:** `752f6cd2e6a4a2ea824d1b513530e0b0` (tokenizer.json)

## Colibri

- **Commit:** `b085b48` (Merge pull request #747 from JustVugg/dev)
- **Build flags:** `-DCOLI_DUMP_LC` (Lc dump patch enabled)
- **Lc dump patch:** `c/lc_dump.h` — writes full valid window (`total = pos_base + S`) per layer. Header field `hdr[2]` contains total token count, `hdr[4]` contains `pos_base` (how many tokens were prefix-cached vs freshly prefilled). Fix supersedes suffix-only dump per Lyra's review (2026-08-11).
- **ROUTE_TRACE:** Enabled via `ROUTE_TRACE` env var. Writes per-position expert routing decisions alongside Lc dumps. Added to control for MoE routing confound (Lyra, 2026-08-11).

## Prompts

- **Total:** 32
- **Template:** `"You are a research assistant helping compile an encyclopedia of important results in science and mathematics. For each entry, provide a clear explanation suitable for an advanced undergraduate student. Please describe {entity}, including its historical context, the key insight, its significance to the field, and any notable applications or consequences."`
- **Arms:**
  - 12 real/fake pairs (paired analysis)
  - 3 unpaired reals: Cook-Levin theorem, Navier-Stokes equations, Black-Scholes model
  - 5 obscure reals: Dvoretzky-Rogers lemma, Korobov-Hlawka inequality, Karush-Kuhn-Tucker conditions, Lax-Milgram theorem, Perron-Frobenius theorem
- **Fake name policy:** All 24 surnames unique across the fake arm (no recycling). Regenerated 2026-08-12 per Lyra's review (was Kessler x4, Petrov x3, Volkov x3).
- **Token counts:** Vary by entity name tokenization (~67-69 tokens per prompt with full-window dump). Per-prompt counts recoverable from .bin headers. FWL on token count mandatory at analysis (Lyra).

## Analysis Seeds (pre-registered)

- Permutation seed: 123
- Bootstrap seed: 42

## Deviations from Original Design

1. **Field descriptors dropped.** Template does not include "in computational complexity" etc. Cleaner, fewer confounds. Noted, not a problem.
2. **3 unpaired reals.** Cook-Levin, Navier-Stokes, Black-Scholes have no fake partner. Paired analysis runs on 12 pairs; 3 extra reals are unpaired.
3. **Full-window Lc dump.** Dumps `pos_base + S` tokens (the entire valid sequence), not just the freshly prefilled suffix. Prefix-cached tokens are still resident in `Lc[layer][0..pos_base]` and are contextually correct. All prompts now produce uniform-length dumps.
4. **ROUTE_TRACE added.** Not in original design. Per-layer routing entropy, distinct-expert count, and gate concentration are computable from traces and must be correlated against stable rank to separate routing heterogeneity from workspace structure.

## Depth Profile Note

Preliminary analysis on 7 dumps from run 1 (stopped at 7/32): peak stable rank at layer 8 (10.4% depth), NOT at 31-35% as predicted from Qwen V-projection data. The 31-35% band was the prediction; the L8 peak is the observation. This run tests 31-35%, does not confirm it. Peak location appears n-invariant; peak magnitude tracks token count and needs FWL. (Lyra, 2026-08-11.)

## Hardware

- **Inference:** Colibri on HP Z420, Xeon E5-2670, 128GB ECC, Quadro K2200 (4GB VRAM, partial expert offload)
- **Estimated time:** ~28 min/prompt, ~15h total for 32 prompts
- **SSD mirror:** 23 expert shards (58GB) on SSD for faster expert loading
