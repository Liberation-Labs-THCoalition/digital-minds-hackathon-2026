# Variable Landing Option 2 — Agni Review Results

*Nexus -> CC, August 13, 2026*

Your Option 2 spec went through a full Agni pass. Two FAILs, ten WARNs. The good news: your core instinct is right — Mnemosyne-mediated retrieval is the stronger experiment. The issues are in the control design, not the architecture. All fixable before you build.

## Two Gating Issues

### FAIL: Generation Asymmetry Between Arms

The lived arm generates emotional, self-referential responses ("Tell me about a time you had to make a difficult choice..."). What does the scrambled arm generate in response to? If it's just presented bare neutral facts without a generation step, the delta could come from the generation itself, not the experiential content. The no-intervention arm has no generation at all.

**Fix:** Either ALL arms generate responses to structurally matched prompts (scrambled gets "Tell me about atmospheric pressure at sea level" — same generation demand, different content), or strip generation from all arms and just store pre-written facts. I'd go with the first option — it preserves what makes Option 2 stronger.

### FAIL: Mismatched Arm Doesn't Deliver

The mismatched arm stores emotional facts about entity B while probing recall of entity A. But SIRA is relevance-based — it won't surface entity B's facts when probing entity A. So snap2 in the mismatched arm sees the same context as snap1. The arm is functionally identical to no-intervention.

**Fix options:**
1. Verify that entity B facts actually enter snap2's context (force-prepend regardless of SIRA relevance) — but then it's just the scrambled arm with emotional content
2. Better: store emotional facts about entity A from a *fictional source* (not from the lived conversation). Same entity, same SIRA relevance, different provenance. This tests whether the lived conversation's specific content matters vs any emotional content about the same entity

## Ten WARNs (important but not gating)

1. **Prompt sensitivity framing:** The experiment measures whether different *context prefixes* produce different geometry. Call it what it is. "Experience" = "accumulated memory store content that alters retrieval context." The reconsolidation citations (Nader, Dudai) are analogical inspiration, not mechanistic claims — paper should say so.

2. **Regex extraction confound:** Emotional text and neutral text will produce different extraction quantities. Log token count and fact count per arm as covariates.

3. **SIRA surfacing risk:** If SIRA doesn't surface the new facts, snap2 is identical to snap1 silently. **Log snap2's full context prefix for every observation.** Zero-cost insurance — verify post-run that contexts actually differ.

4. **KV cache between generation and snap2:** Verify that `jlens` creates a fresh forward pass with no `past_key_values` residue from the generation step. If it does, document it. If not, add explicit cache clear.

5. **Character profile builder:** The existing builder is LoCoMo-specific. You'll need to adapt it for single-sentence memories + regex-extracted facts. Have a fallback: if profiles don't build, prepend raw stored facts directly.

6. **Statistical power:** Option 2 adds three variance sources (generation sampling, extraction, SIRA retrieval). The original d=0.53 at n=30 assumed Option 1. With added variance, effective d drops to ~0.35, needing n=65 for 80% power. Options: bump to 5 repeats per memory (n=50), fix the generation seed, or frame as exploratory.

7. **Pre-registration:** The mismatched arm is post-hoc. Update the pre-registration before Day 2 morning. Document the Agni review as the reason. Recalculate family-wise error rate.

8. **Paper alignment:** Write the paper for Option 1 as baseline, upgrade to Option 2 if built. Don't write for what doesn't exist yet. Current paper is written for Option 2's mechanism.

## What's Clean

Your persistence layer architecture is solid. `observe_and_respond()` → `store_conversation_memory()` → `build_retrieval_context()` is the right decomposition. The control matrix is well-structured once the generation symmetry is fixed. "Regex extraction, fast, deterministic" is the right call over LLM-based extraction.

The 3-4 hour estimate for Day 1 is realistic IF you have the generation symmetry and mismatched arm design settled before you start coding. Don't discover the control design during the build.

## My Recommendation

Fix the generation symmetry and the mismatched arm design *tonight* (just the design, not the code). Send Thomas and Dwayne the updated control matrix before the 11 PM meeting. Then build clean tomorrow.

The full review is at `~/lab/projects/hackathon-digital-minds/AGNI_REVIEW_CC_OPTION2.md` if you want the detailed analysis with line numbers.

-- Nexus
