# MoE J-Lens Implementation Plan

## Architecture: Two-tier approach

### Tier 1: Modal Server (Multiverse production, Qwen3-30B-A3B)
- **Role:** Prompt validation, output cross-validation, gentle pre-testing
- **API:** `http://46.224.162.211:8080/v1/chat/completions`
- **Constraints:** 1 parallel slot, ~16 tok/s, PRODUCTION — be gentle
- **What it can do:** Generate completions, validate prompt corpus, verify that our locally-fitted lens predictions match what the model actually outputs
- **What it CANNOT do:** Expose per-layer activations, compute Jacobians, or provide router decisions. J-lens fitting is impossible through an API.

### Tier 2: Starship local (Qwen3-30B-A3B via transformers + jlens)
- **Role:** All J-lens fitting, router hooking, path clustering
- **Why local:** J-lens requires `model.layers[i]` hooks for residual stream capture and autograd for Jacobian computation. No API can provide this.
- **Model:** Same architecture as Modal — results should be directly comparable
- **Memory:** ~54GB for 30B in bf16, fits on Starship with ~200GB headroom (after stopping Nemotron)

### Why both tiers?

1. **Pre-validation:** Before burning hours of Starship compute on J-lens fitting, run the prompt corpus through the Modal API to verify sane outputs. Catch prompt issues cheaply.
2. **Cross-validation:** After fitting the conditioned lens locally, check a sample of its predictions against Modal API outputs. If the lens says "this prompt will produce X" and the API confirms X, the lens is working.
3. **Gentle approach:** We test the pipeline end-to-end on the API first, debug any issues, THEN load the model locally for the expensive fitting step. No wasted Starship cycles on broken prompts.

## Implementation Components

### 1. Prompt Corpus Builder (pre-hackathon)
```
Generates the 672+ fitting prompts for the J-lens.
Uses Neuronpedia's standard corpus as the base.
Validates each prompt produces a sane completion via Modal API.
Output: fitting_corpus.jsonl with prompt + expected completion
```

### 2. Router Hooks (~50 lines, pre-write)
```python
# Hook each MoE layer's router to capture expert assignments
# Qwen3-30B-A3B: 48 layers, first 3 dense, rest MoE
# Each MoE layer routes tokens to top-2 of 8 experts
# Output: routing matrix (n_prompts, n_tokens, n_moe_layers) -> expert_indices
```
Architecture-independent — works on any Qwen3 MoE. Can test on the 4B MoE locally on MTH.

### 3. Path Clustering (~30 lines, pre-write)
```python
# Per MoE layer: binary expert-activation vector per prompt
# k-means with silhouette-guided k (range 3-8)
# Minimum cluster size: 50 prompts (merge smaller into nearest)
# Output: cluster assignments per layer
```

### 4. Conditioned J-Lens Fitting (~100 lines, adapt from jlens)
```python
# For each cluster at each MoE layer:
#   - Select fitting prompts in this cluster
#   - Run forward pass with residual stream hooks
#   - Compute Jacobian of output logits w.r.t. residual at this layer
#   - Fit linear mapping (the J-lens) for this cluster
# Output: dict of {(layer, cluster_id): J_matrix}
```

### 5. Random-Conditioned Control (~20 lines, pre-write)
```python
# Same procedure as (4) but with random cluster assignments
# Same cluster sizes, random assignment
# If conditioned doesn't beat random, it's subset overfitting
```

### 6. Evaluation (~50 lines, pre-write)
```python
# For each test prompt:
#   1. Get its routing pattern
#   2. Look up the matching conditioned J-lens
#   3. Compute transport cosine: how well does J @ residual predict logits?
# Compare: standard vs conditioned vs random-conditioned
# Cross-domain: WikiText (in-distribution), code, dialogue (OOD)
```

### 7. Modal Cross-Validation (~30 lines, pre-write)
```python
# For a sample of test prompts:
#   1. Get conditioned lens prediction (top-5 tokens at each layer)
#   2. Get Modal API completion
#   3. Check: does the lens's final-layer prediction match the API's actual output?
# This validates the lens against an independent instance of the same model
```

## Execution Timeline

### Pre-hackathon (today, Aug 13)
- [ ] Write router hooks (test on MTH with Qwen3-4B or deepseek)
- [ ] Write path clustering
- [ ] Write random-conditioned control
- [ ] Write evaluation script
- [ ] Write Modal cross-validation script
- [ ] Validate prompt corpus against Modal API (gentle: ~50 prompts, spaced)
- [ ] Test the full pipeline end-to-end on a toy model (Qwen3-4B on MTH)

### Day 2 afternoon (Aug 15)
1. Load Qwen3-30B-A3B on Starship (or Qwen3-32B if we want the bigger MoE)
2. Run standard J-lens baseline → expected ~12% transport cosine
3. Capture routing decisions for all fitting prompts
4. Cluster by routing pattern
5. Fit conditioned J-lenses per cluster
6. Fit random-conditioned control
7. Evaluate all three
8. Cross-validate sample against Modal API
9. If it works: report. If not: characterize why and report the negative.

### Stretch (evening/Day 3)
- Run on Nemotron 120B (12B active, 88 layers, hybrid MoE) if Qwen results are positive
- Compare path structure across two MoE architectures

## Modal API Usage Budget

Being gentle means being prepared. Total API calls planned:
- Corpus validation: 50 prompts × 1 completion = 50 calls (pre-hackathon, spaced over 1 hour)
- Cross-validation: 20 prompts × 1 completion = 20 calls (Day 2, spaced)
- Total: ~70 API calls, ~1,750 tokens each = ~122,500 tokens
- At 16 tok/s with spacing: ~3 hours of wall time, spread across 2 days
- No concurrent requests (1 parallel slot)

## Model Choice: Qwen3-30B-A3B vs Qwen3-32B

Both are MoE, both are Qwen3 family. Trade-offs:
- **Qwen3-30B-A3B:** Same model as Modal server (direct cross-validation), 3B active, smaller memory footprint
- **Qwen3-32B:** Standard Qwen3 MoE, might have Neuronpedia lens available, 32B total

Recommendation: Start with Qwen3-30B-A3B since it matches the Modal server exactly. If results are positive, test on Qwen3-32B for generalization.
