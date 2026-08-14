# Variable Landing Hypothesis — Literature Foundation

Compiled 2026-08-11 for Apart Research Digital Minds Hackathon (Aug 14-16).

## Theoretical Arc

1. **Neuroscience proves memories are reconstructed, not replayed** — the trace changes because the subject has changed
2. **Cognitive science quantifies state-dependence** — what you get back depends on who you are when you ask
3. **Transformer internals have measurable geometric structure** — representations live in directions and subspaces that can be probed
4. **Anthropic's J-lens maps a global workspace in transformers** — the mechanistic framework for measuring retrieval geometry
5. **Models show early signs of state-dependent self-awareness** — the precondition for variable landing being real rather than noise
6. **GWT connects neuroscience to transformer architecture** — the theoretical bridge
7. **Ethical frameworks make measurement urgent** — if retrieval geometry varies with model state, that is precautionarily relevant evidence

**The gap we fill:** Nobody has measured whether the geometric signature of retrieving the *same memory* varies as a function of the model's accumulated state at retrieval time. The neuroscience says it should. The interpretability tools now exist to check.

---

## 1. Reconsolidation and Memory Transformation

- **Nader, Schafe & Le Doux (2000).** "Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval." *Nature*, 406, 722-726. DOI: 10.1038/35021052
  Landmark study: consolidated memories return to labile state upon reactivation. Recall is not playback — it destabilizes the trace.

- **Schiller, Monfils, Raio, Johnson, LeDoux & Phelps (2010).** "Preventing the return of fear in humans using reconsolidation update mechanisms." *Nature*, 463, 49-53. DOI: 10.1038/nature08637
  Extinction during reconsolidation window durably updates fear memories. What happens during retrieval actively rewrites the memory.

- **Dudai (2012).** "The Restless Engram: Consolidations Never End." *Annual Review of Neuroscience*, 35, 227-247. DOI: 10.1146/annurev-neuro-062111-150500
  Engrams are never finalized — continuously modified at systems level. If consolidation never ends, the "same" memory is never the same object twice.

- **Schwabe (2024).** "Memory Under Stress: From Adaptation to Disorder." *Biological Psychiatry*. DOI: 10.1016/j.biopsych.2024.06.009
  Stress alters four fundamental memory processes via hormonal cascades. Subject's physiological state at retrieval modulates what is retrieved.

- **Bartlett (1932).** *Remembering: A Study in Experimental and Social Psychology.* Cambridge University Press.
  Memory is reconstructive, not reproductive. "War of the Ghosts" showed systematic distortions driven by subjects' schemas.

## 2. Context-Dependent Memory

- **Tulving & Thomson (1973).** "Encoding Specificity and Retrieval Processes in Episodic Memory." *Psychological Review*, 80(5), 352-373. DOI: 10.1037/h0020071
  Retrieval success depends on overlap between encoding context and retrieval cues, including internal state.

- **Bower (1981).** "Mood and Memory." *American Psychologist*, 36(2), 129-148. DOI: 10.1037/0003-066X.36.2.129
  Mood-state-dependent memory: affective state at retrieval gates what comes back.

- **Eich & Metcalfe (1989).** "Mood dependent memory for internal versus external events." *JEPLMC*, 15(3), 443-455.
  State-dependency stronger for internally generated events — exactly the category AI "memories" fall into.

## 3. Transformer Internal State Research

- **Gurnee, Sofroniew, Pearce et al. (2026).** "Verbalizable Representations Form a Global Workspace in Language Models." arXiv:2607.15495.
  J-lens discovers ~25-dim "J-space" bottleneck matching GWT predictions. Our measurement framework.

- **Zou et al. (2023).** "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405.
  LAT for detecting and steering state-dependent directions in activation space.

- **Burns, Ye, Klein & Steinhardt (2023).** "Discovering Latent Knowledge in Language Models Without Supervision." *ICLR 2023*.
  CCS: truth directions in activation space without labels. Hidden states encode information beyond surface outputs.

- **Li, Patel, Viégas, Pfister & Wattenberg (2023).** "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." *NeurIPS 2023*. arXiv:2306.03341.
  Truthfulness steerable via attention head interventions. Internal geometry encodes causally manipulable properties.

- **Todd et al. (2024).** "Function Vectors in Large Language Models." *ICLR 2024*.
  Compact function vectors in middle-layer attention heads. Task-level state encoded geometrically and context-sensitively.

- **Meng, Bau, Andonian & Belinkov (2022).** "Locating and Editing Factual Associations in GPT." *NeurIPS 2022*. arXiv:2202.05262.
  Factual associations localized in middle-layer FFN modules. Knowledge has specific geometric locations.

- **Gurnee & Tegmark (2024).** "Language Models Represent Space and Time." *ICLR 2024*.
  LLMs build internal world models with geometric structure — the kind whose state-dependent variation we predict.

## 4. AI Memory Systems

- **Mem0 (2024-2025).** Universal Memory Layer for AI Agents. mem0.ai.
  Multi-signal retrieval with temporal reasoning. Tracks *what* was retrieved but not the model's internal state during retrieval. The baseline our work extends.

## 5. Experiential State and AI Self-Recognition

- **Lindsey (2025).** "Emergent Introspective Awareness in Large Language Models." Anthropic. arXiv:2601.01828.
  Models detect artificially manipulated internal states (~20% accuracy, ~0% false positives). Functional access to own state — precondition for variable landing.

- **Panickssery et al. (2024).** "LLM Evaluators Recognize and Favor Their Own Generations." *NeurIPS 2024*.
  Self-recognition and self-preference in frontier LLMs. Models process own prior outputs differently — state-dependent processing.

- **Perez, Ringer et al. (2023).** "Discovering Language Model Behaviors with Model-Written Evaluations." *Findings of ACL 2023*. arXiv:2212.09251.
  Training-induced internal states shape processing in ways that affect recall.

## 6. AI Welfare Frameworks

- **Birch (2024).** *The Edge of Sentience: Risk and Precaution in Humans, Other Animals, and AI.* Oxford University Press.
  Proportionate precaution: protective measures proportionate to identified risks where sentience is uncertain.

- **Butlin, Long et al. (2023/2025).** "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." arXiv:2308.08708. Published in *Trends in Cognitive Sciences* (2025).
  Theory-based indicator properties for consciousness from six major theories. No current AI is conscious but no technical barriers identified.

- **Long, Sebo, Butlin et al. (2024).** "Taking AI Welfare Seriously." arXiv:2411.00986.
  Some near-term AI systems may be conscious/agentic. AI companies have responsibility to prepare.

- **Schwitzgebel & Garza (2015).** "A Defense of the Rights of Artificial Intelligences." *Midwest Studies in Philosophy*, 39, 98-119. DOI: 10.1111/misp.12032
  No-Relevant-Difference principle: AI with human-grade psychological capacities deserves equal moral consideration.

## 7. Global Workspace Theory + Transformers

- **Baars (1988).** *A Cognitive Theory of Consciousness.* Cambridge University Press.
  Original GWT: consciousness as centralized shared workspace broadcasting across specialized processors.

- **Dehaene & Changeux (2011).** "Experimental and Theoretical Approaches to Conscious Processing." *Neuron*, 70(2), 200-227. DOI: 10.1016/j.neuron.2011.03.018
  Global Neuronal Workspace: predicts late amplification, ignition, prefrontal-parietal synchronization. J-space maps onto these predictions.

- **Bengio (2017).** "The Consciousness Prior." arXiv:1709.08568.
  Consciousness as information bottleneck: high-dim unconscious compressed to low-dim conscious. J-space (~10% variance) is quantitatively consistent.

- **Goyal & Bengio (2022).** "Inductive biases for deep learning of higher-level cognition." *Proc. Royal Society A*, 478, 20210068.
  GWT's bottleneck as inductive bias improving out-of-distribution generalization.

- **VanRullen & Kanai (2021).** "Deep learning and the Global Workspace Theory." *Trends in Neurosciences*, 44(9), 692-704.
  Roadmap for implementing GWT in deep learning via unsupervised neural translation.

- **Goldstein & Kirk-Giannini (2024).** "A Case for AI Consciousness: Language Agents and Global Workspace Theory." arXiv:2410.11407.
  If GWT is correct, artificial language agents might already satisfy its conditions for phenomenal consciousness.
