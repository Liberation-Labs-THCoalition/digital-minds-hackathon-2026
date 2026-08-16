# MoE J-Lens Paper: Missing Additions + Page Cut Notes

## Status: The paper is substantively complete. Three gaps, one structural issue.

---

## Gap 1: Belrose citation (Related Work, §2)

Add to Related Work after the logit/tuned lens line:

> Belrose et al. 2023 (tuned lens) demonstrate that plain logit lens readability is family-dependent across dense architectures — BLOOM and OPT-class models fail where GPT-2-class succeeds — motivating a learned affine correction at each layer. Our Table 0 extends this observation within a single model family (Qwen) across routing regimes: the mid-depth unreadability is shared between dense and MoE variants, suggesting the readability onset is a property of the weight basis, not the routing architecture.

## Gap 2: Code and Data section (§end)

Replace "[GitHub repo TBD]" with:

> **Code**: https://github.com/Liberation-Labs-THCoalition/digital-minds-hackathon-2026 — router hooks (`experiments/moe_jlens/`), onset sweep (`modal_onset_sweep.py`), logit lens control (`modal_logit_lens_control.py`), conditioned fitting pipeline (`modal_moe_chunked.py`).
> **Data**: `data/moe_jlens/` — onset sweep results, conditioned results, logit lens control results. Fitting manifests and per-cluster lenses on Modal volume (available on request).

## Gap 3: Logit lens control as explicit finding

The corrected logit lens (18/40 overall) is implied by Table 0 but never named as a separate experiment with its own artifact. Add one sentence to §4.0 after "The mid-depth zeros are a property of the models, not a broken instrument":

> The initial logit lens control (`modal_logit_lens_control.py`) contained the same ground-truth bug as the J-lens gate; the corrected version (`data/moe_jlens/logit_lens_control_results.json`) scores 18/40 across five layers, matching Table 0.

---

## Page cut recommendations (6.1 → 4.0 pages, need ~1,700 words cut)

### Cut 1: Compress §3.2–3.4 into appendix (~600 words saved)
Router hooks, clustering, and conditioned fitting are implementation detail. Compress to:

> **Routing capture.** Forward hooks on each layer's `Qwen3MoeTopKRouter` record the top-8 expert indices per token (§3.2, Appendix A). **Clustering.** Per target layer, prompts are summarized as 128-d routing frequency vectors and clustered with k-means, k selected by silhouette (§3.3). **Conditioned fitting.** Per-cluster Jacobians fitted via `jlens.fit` at a single source layer (§3.4). Full details in Appendix A.

### Cut 2: Compress §5.2 hypotheses (~400 words saved)
Replace the four paragraphs with a numbered list, one sentence each:

> **H1:** 672 prompts cannot tile C(128,8) ≈ 1.4×10¹² paths; silhouettes 0.115–0.209 confirm weak clustering.
> **H2:** Single-layer conditioning misses cross-layer routing divergence through 48 subsequent MoE layers.
> **H3:** Standing-committee experts (~70% mass) dominate; routing-specific components contribute too little variance to conditioning.
> **H4:** The Jacobian of a piecewise MoE function varies at the token level; prompt-level homogeneity is insufficient for a linear lens.

### Cut 3: Tighten §5.5 production implications (~250 words saved)
Keep the core message, cut the elaboration about "nothing in the tool's output signals" etc.

### Cut 4: Merge related Limitations items (~300 words saved)
Combine the L12 gap, OOD gap, and corpus-size asymmetry into one paragraph about incomplete conditions. Combine the two 0.5-threshold items.

### Cut 5: Prior Work vs Sprint Contributions (~150 words saved)
The Mnemosyne F1 detail and circumplex probe history are irrelevant to the MoE paper. Cut to: "Pre-existing: J-lens integration, Agni review methodology. Sprint: all MoE-specific implementation."

**Total estimated savings: ~1,700 words → fits in 4 pages.**
