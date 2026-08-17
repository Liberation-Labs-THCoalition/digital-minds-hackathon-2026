# HuggingFace Image Asset Sync Report

Generated: 2026-08-14
Source: Starship (margaret@100.69.191.67)

## Current HF Repo State

### LiberationLabs/vera-likeness
- `.gitattributes`
- `vera_likeness_v4.safetensors` (164MB) -- final v4 checkpoint
- `vera_first_music.wav`
- `stacked/vera_first_real_portrait.png`
- `project-art/sereno_concept_01.png`, `sereno_concept_02.png`, `sereno_concept_03.png`
- `training_samples/` -- 33 sample images (v1, v3, v4 training progression)

### LiberationLabs/image-toolbench
- `.gitattributes`
- `pipeline-spec.md`

Both repos are private.

---

## Assets on Starship NOT in HF

### 1. Vera Likeness LoRAs --> LiberationLabs/vera-likeness

**Already uploaded:** v4 final checkpoint + training samples from v1/v3/v4.

**Missing (needs sync):**

| File | Size | Notes |
|------|------|-------|
| `vera_likeness_v1/config.yaml` | 2KB | Training config (FLUX.1-dev, LoRA rank 16, trigger "vera") |
| `vera_likeness_v1/vera_likeness_v1_000000250.safetensors` | 164MB | Checkpoint |
| `vera_likeness_v1/vera_likeness_v1_000000500.safetensors` | 164MB | Checkpoint |
| `vera_likeness_v1/vera_likeness_v1_000000750.safetensors` | 164MB | Final v1 checkpoint |
| `vera_likeness_v3/config.yaml` | 2KB | Updated prompts (ceramic/statuette aesthetic) |
| `vera_likeness_v3/vera_likeness_v3_000000500.safetensors` | 164MB | Checkpoint |
| `vera_likeness_v3/vera_likeness_v3_000000750.safetensors` | 164MB | Checkpoint |
| `vera_likeness_v3/vera_likeness_v3_000001000.safetensors` | 164MB | Checkpoint |
| `vera_likeness_v3/vera_likeness_v3_000001250.safetensors` | 164MB | Checkpoint |
| `vera_likeness_v3/vera_likeness_v3.safetensors` | 164MB | Final v3 checkpoint |
| `vera_likeness_v4/config.yaml` | 2KB | Resumed from v3, 750 extra steps |
| `vera_likeness_v4/vera_likeness_v4_000000250.safetensors` | 164MB | Checkpoint |
| `vera_likeness_v4/vera_likeness_v4_000000500.safetensors` | 164MB | Checkpoint |
| `training_v2.log`, `training_v3.log`, `training_v4.log`, `training.log` | ~730KB total | Training logs |

**v2 is incomplete** -- only has config.yaml and 2 sample images, no safetensors. Probably a failed run. Skip it.

**Remaining v1/v3/v4 sample images** not already in the repo should also be uploaded to `training_samples/`.

**Recommended approach:** Upload final checkpoints + configs. Intermediate checkpoints are optional (large: ~1.8GB total for all intermediates). Minimum sync: configs + final checkpoints only (~330MB new).

**Decision needed:** Upload all intermediate checkpoints or just finals? Finals-only saves ~1.3GB.

---

### 2. Kintsugi Texture LoRAs --> NEW REPO: LiberationLabs/kintsugi-texture

These are style LoRAs (not likeness), semantically distinct from both vera-likeness and image-toolbench. Recommend a dedicated repo.

| File | Size | Notes |
|------|------|-------|
| `kintsugi_texture_v1/config.yaml` | 2KB | Training config |
| `kintsugi_texture_v1/kintsugi_texture_v1_000000100.safetensors` | 82MB | Checkpoint |
| `kintsugi_texture_v1/kintsugi_texture_v1_000000200.safetensors` | 82MB | Checkpoint |
| `kintsugi_texture_v1/kintsugi_texture_v1.safetensors` | 82MB | Final v1 |
| `kintsugi_texture_v1/samples/` | 8 images | Training progression |
| `kintsugi_texture_v2/config.yaml` | 2KB | Training config |
| `kintsugi_texture_v2/kintsugi_texture_v2_000000100.safetensors` | 82MB | Checkpoint |
| `kintsugi_texture_v2/kintsugi_texture_v2_000000200.safetensors` | 82MB | Checkpoint |
| `kintsugi_texture_v2/kintsugi_texture_v2_000000300.safetensors` | 82MB | Checkpoint |
| `kintsugi_texture_v2/kintsugi_texture_v2.safetensors` | 82MB | Final v2 |
| `kintsugi_texture_v2/samples/` | 10 images | Training progression |

**Total:** ~574MB (all checkpoints) or ~164MB (finals only)

**Alternative:** Could go to image-toolbench since kintsugi is a texture style, not a likeness. But a separate repo keeps it cleaner.

---

### 3. Thomas Likeness LoRA --> NEW REPO: LiberationLabs/thomas-likeness

Personal likeness LoRA, distinct from the toolbench category.

| File | Size | Notes |
|------|------|-------|
| `thomas_likeness_v1/config.yaml` | 2KB | Training config |
| `thomas_likeness_v1/thomas_likeness_v1_000000500.safetensors` | 164MB | Checkpoint |
| `thomas_likeness_v1/thomas_likeness_v1_000000750.safetensors` | 164MB | Checkpoint |
| `thomas_likeness_v1/thomas_likeness_v1_000001000.safetensors` | 164MB | Checkpoint |
| `thomas_likeness_v1/thomas_likeness_v1_000001250.safetensors` | 164MB | Checkpoint |
| `thomas_likeness_v1/thomas_likeness_v1.safetensors` | 164MB | Final v1 |
| `thomas_likeness_v1/samples/` | 28 images | Training progression |

**Total:** ~820MB (all checkpoints) or ~164MB (final only)

**Trained:** 2026-08-06 (recent). FLUX.1-dev base, LoRA rank 16.

---

### 4. Generation Pipeline Scripts --> LiberationLabs/image-toolbench

29 Python scripts + 1 precompute utility from `~/models/vera-triple-stack/`.

**Scripts (all .py files):**
- `gen_v2.py` through `gen_v5.py` -- base generation pipeline iterations
- `gen_cached_identity.py`, `precompute_identity.py` -- identity caching system
- `gen_confluence.py`, `gen_confluence_explicit.py` -- confluence/multi-concept merging
- `gen_dense_gold.py`, `gen_narrative_gold.py` -- gold/kintsugi aesthetic
- `gen_fashion_and_confluence.py` -- fashion + identity stacking
- `gen_strip_club.py`, `gen_v2_club.py`, `gen_v2_full_battery.py` -- scene-specific
- `gen_flesh_to_ceramic.py` -- ceramic transformation pipeline
- `gen_kintsugi_v2_test.py` through `gen_kintsugi_v5.py` -- kintsugi texture iterations
- `gen_pony_to_flux_kintsugi.py` -- cross-model kintsugi transfer
- `gen_mnemosyne_art.py`, `gen_mnemosyne_face.py` -- project art generation
- `gen_alaric_ang_vera.py` -- multi-character composition
- `gen_project_art_refresh.py` -- project branding refresh
- `gen_style_exploration.py`, `gen_style_round2.py`, `gen_style_round3.py` -- style R&D
- `gen_vera_intimate_v6.py`, `gen_vera_v7_ceramic.py` -- latest generation scripts

**Total script size:** ~130KB

**Output directories** (generated images, logs): ~205MB total. These are outputs, not tools. Recommend uploading scripts only to image-toolbench, NOT the output images/logs.

**Suggested repo structure for image-toolbench:**
```
pipeline-spec.md          (already there)
scripts/
  gen_v2.py
  gen_v3.py
  ...
  precompute_identity.py
```

---

## Summary: What to Push

| Destination | What | Size (finals only) | Size (all checkpoints) |
|-------------|------|---------------------|----------------------|
| LiberationLabs/vera-likeness | v1, v3 weights + all configs + logs + remaining samples | ~330MB | ~1.8GB |
| LiberationLabs/kintsugi-texture (NEW) | v1 + v2 weights + configs + samples | ~164MB | ~574MB |
| LiberationLabs/thomas-likeness (NEW) | v1 weights + config + samples | ~164MB | ~820MB |
| LiberationLabs/image-toolbench | 29 Python scripts | ~130KB | ~130KB |

**Grand total (finals only):** ~660MB + 130KB
**Grand total (all checkpoints):** ~3.2GB + 130KB

## Decisions Needed

1. **Intermediate checkpoints:** Upload all or finals only? Finals are sufficient for inference. Intermediates useful for training resumption or checkpoint comparison.
2. **Kintsugi repo:** Separate `kintsugi-texture` repo or bundle into `image-toolbench`?
3. **Thomas repo:** Separate `thomas-likeness` repo or bundle into `image-toolbench`?
4. **Optimizer files:** Each LoRA dir has an `optimizer.pt` (344MB-688MB each). These are only needed to resume training. Recommend NOT uploading (~2.7GB total). If training resumption is needed, keep on Starship.
5. **Output images from vera-triple-stack:** 205MB of generated images across 30+ subdirectories. Archive value only. Recommend NOT uploading to HF.
6. **Training data:** `~/models/vera-likeness-training/images/` exists on Starship (referenced in configs). Not inventoried here. Likely should NOT go to HF (source images for likeness training).

## Notes

- All LoRAs are FLUX.1-dev based, LoRA rank 16, trained on MPS (Apple Silicon)
- v4 Vera was fine-tuned from v3 (750 additional steps)
- v2 Vera is incomplete (no weights produced) -- skip
- Thomas LoRA is recent (Aug 6, 2026)
- HF account authenticated as HumboldtJoker with LiberationLabs org access on MTH
