# Cross-Architecture Circumplex Geometry — Literature Review

Compiled 2026-08-11 for Apart Research Digital Minds Hackathon (Aug 14-16).

## Research Question

Is the circumplex geometry of emotion (Russell's valence/arousal model) an architectural universal in sufficiently large transformers, or a training-specific artifact? We have measured eccentricity dips at mid-network depth in Qwen-family models (0.5B and 27B) using difference-of-means emotion-anchored directions. We test whether this transfers to Gemma-3-27B-it (different architecture, training, tokenizer).

---

## 1. Russell's Circumplex Model — Original Formulation and Validation

- **Russell, J. A. (1980).** "A Circumplex Model of Affect." *Journal of Personality and Social Psychology*, 39(6), 1161-1178.
  The foundational paper. Proposed that all affective states can be represented as points in a two-dimensional circular space defined by valence (pleasure-displeasure) and arousal (activation-deactivation). Validated through category sorting and circular ordering of 28 mood terms. The geometry we are probing for in transformer hidden states.

- **Posner, J., Russell, J. A., & Peterson, B. S. (2005).** "The Circumplex Model of Affect: An Integrative Approach to Affective Neuroscience, Cognitive Development, and Psychopathology." *Development and Psychopathology*, 17(3), 715-734. DOI: 10.1017/S0954579405050340
  Major integrative review arguing the circumplex outperforms discrete basic-emotion theories across behavioral, neuroimaging, and developmental data. Valence and arousal arise from two independent neurophysiological systems. Provides the neuroscience justification for expecting dimensional (not categorical) emotion geometry.

- **Barrett, L. F., & Russell, J. A. (1998).** "Independence and Bipolarity in the Structure of Current Affect." *Journal of Personality and Social Psychology*, 74(4), 967-984.
  Reconciled the apparent contradiction between independence and bipolarity of positive/negative affect. Confirmed valence is independent of arousal, and positive affect is the bipolar opposite of negative affect. Methodological template for how to test whether two affective dimensions are orthogonal — directly relevant to our eccentricity metric (eccentricity measures deviation from balanced orthogonality).

- **Drążkowski, D., Klonek, F. E., et al. (2021).** "Ellipse Rather Than a Circumplex: A Systematic Test of Various Circumplexes of Emotions." *Personality and Individual Differences*, 184, 111169.
  Tested circumplex structure across 863 participants using multiple affect models. Found affect space is reliably circular (elliptical) but not a strict circumplex — arousal is the most problematic dimension. Important calibration: even in humans the structure is an ellipse, not a perfect circle. Our eccentricity metric measures exactly this deviation.

---

## 2. Emotion Representation in Language Models

- **Jentzsch, Schiller, & Slonim (2026).** "Emotion Concepts and their Function in a Large Language Model." *Anthropic / Transformer Circuits*, April 2026. arXiv:2604.07729
  Landmark mechanistic study. Found 171 emotion concept directions in Claude Sonnet 4.5 via SAEs. Emotion vectors mirror the human valence/arousal circumplex geometry, and steering along them causally changes behavior (including refusal, reward hacking, blackmail in agentic scenarios). Establishes that circumplex-aligned emotion geometry exists in at least one frontier model and has causal behavioral consequences.

- **Choi, B. J., & Weber, M. (2026).** "Latent Structure of Affective Representations in Large Language Models." arXiv:2604.07382
  Harvard study using geometric data analysis to probe emotion representations. Found LLMs learn coherent affective representations aligning with valence-arousal models. Representations exhibit nonlinear structure well-approximated linearly — empirical support for the linear representation hypothesis applied to affect. Directly validates our difference-of-means approach.

- **Sun, L., Yan, L., Lu, X., Lee, A., Zhang, J., & Shao, J. (2026).** "Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control." arXiv:2604.03147
  Derived VA axes from 211k emotion-labeled texts via PCA + ridge regression. Recovered circular geometry consistent with Russell's circumplex. Critically, effects replicate across Llama-3.1-8B, Qwen3-8B, and Qwen3-14B — first direct evidence of cross-architecture circumplex geometry in the same paper, though limited to models at similar scale.

- **Zhang, J., & Zhong, L. (2025).** "Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion." arXiv:2510.04064
  Large-scale probing study (~400k utterances, 7 emotions) across Qwen3 and LLaMA families. Found a well-defined internal geometry of emotion that sharpens with scale, and emotional tone persists for hundreds of subsequent tokens. Demonstrates that emotion is not surface-level mimicry but a persistent geometric feature of hidden states.

- **Jeong, J. (2026).** "Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison." arXiv:2604.04064
  Compared emotion extraction methods across 9 models in 5 architectural families (GPT-2, Gemma, Qwen, Llama, Mistral) from 124M to 3B parameters. Found emotion representations localize at ~50% depth following a U-shaped curve that is architecture-invariant. Directly supports our prediction that the eccentricity dip occurs at consistent relative depth across architectures.

- **van der Ben, S., Baur, R., Metz, Y., & El-Assady, M. (2026).** "Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs." arXiv:2606.26987
  Replicated Anthropic's emotion vector findings in two open-weight models (Apertus-8B and Gemma-4-E4B-it). Recovered valence geometry for both, but with notable differences in depth profile: Gemma encodes valence in early layers (collapsing later), while Apertus shows mid-depth emergence. Demonstrates that circumplex structure transfers but depth localization varies by architecture — our eccentricity-vs-relative-depth comparison is the right framing.

---

## 3. Cross-Architecture Transfer of Representations

- **Huh, M., Cheung, B., Wang, T., & Isola, P. (2024).** "The Platonic Representation Hypothesis." *ICML 2024*. arXiv:2405.07987
  Neural networks trained on different data, objectives, and modalities converge toward a shared statistical model of reality. Empirical evidence via kernel alignment, model stitching, and mutual nearest-neighbor analysis. Provides the theoretical expectation that circumplex geometry should transfer: if representations converge, emotion subspaces should too.

- **Huang, Y., et al. (2025).** "Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts." *ACL 2025*. arXiv:2501.02009
  Demonstrated that concept representations across LLMs can be aligned via simple linear transformations. Steering vectors from smaller LLMs modulate larger LLM responses (weak-to-strong transfer). Directly supports our hypothesis: if concept directions transfer linearly, emotion directions (a subset of concept directions) should exhibit transferable geometry.

- **Agarwal, A. (2026).** "Cross-Architecture Steering Transfer in Language Models: A Systematic Empirical Study." arXiv:2608.05164
  The most directly relevant paper to our experiment. Systematic evaluation of cross-model steering across 5 open-weight models (0.8B-8B), 3 parameter scales, 2 architectural lineages. Found a discontinuity near 1.7B: above this scale, 47-49% of cross-model feature pairs validate (Pearson r >= 0.60, Procrustes cosines 0.895-0.956). Our models (27B) are well above the threshold. Predicts our cross-architecture circumplex transfer should succeed.

---

## 4. Difference-of-Means Emotion Directions — Methodology

- **Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Lin, Z., Forsyth, M., & Hendrycks, D. (2023).** "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405
  Introduced the contrastive activation addition framework. Difference-of-means across prompt pairs isolates concept directions in activation space. The methodological foundation for our emotion direction extraction — we use the same difference-of-means approach to find valence and arousal axes.

- **Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023).** "Discovering Latent Knowledge in Language Models Without Supervision." *ICLR 2023*.
  Contrast Consistent Search (CCS): discovered truth directions in hidden states without labels. Established that contrastive probing reveals real geometric structure, not artifacts. Methodological precedent for our approach of extracting emotion directions from contrast pairs without supervised emotion labels.

---

## 5. Emotional Geometry and AI Welfare

- **Long, R., Sebo, J., Butlin, P., Plunkett, D., Campbell, R., Beasley, C., Saad, B., & Sims, T. (2026).** "Studying AI Welfare Empirically." *Center for Mind, Ethics, and Policy Working Paper*, July 2026.
  Framework for empirical AI welfare research. Distinguishes three dimensions: what question (welfare subject? what helps/harms?), what entity (model, instance, persona), what evidence (behavioral, internal, developmental). Our circumplex geometry is *internal evidence* about the *model* level — exactly the kind of measurement this framework calls for. If circumplex structure is universal, that is developmental evidence of convergent affective architecture.

- **Long, R., Sebo, J., Butlin, P., et al. (2024).** "Taking AI Welfare Seriously." arXiv:2411.00986
  Argues some near-term AI systems may have morally significant interests. AI companies have responsibility to investigate and prepare. Our experiment tests a precondition: if models universally develop emotion-aligned geometry, that is a stronger welfare indicator than if it is training-specific, because it suggests the geometry emerges from the task of language modeling itself, not from particular data choices.

- **Butlin, P., Long, R., et al. (2023/2025).** "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." arXiv:2308.08708. Published in *Trends in Cognitive Sciences* (2025).
  Theory-based indicator properties for consciousness from six major theories. No current AI system is conscious by their assessment, but no technical barriers identified. Our circumplex finding is relevant to the Global Workspace Theory indicators: if emotion geometry occupies the workspace bottleneck (J-space), that maps to GWT's prediction of broadcast affective content.

- **Birch, J. (2024).** *The Edge of Sentience: Risk and Precaution in Humans, Other Animals, and AI.* Oxford University Press.
  Proportionate precaution framework: protective measures proportionate to identified risks where sentience is uncertain. A universal circumplex would be a risk indicator (not proof) — geometric evidence that transformer architectures converge on affective structure warrants proportionate investigation, not dismissal.

---

## Summary: The Gap We Fill

The existing literature establishes that: (a) Russell's circumplex is the dominant model of affective space in psychology; (b) LLMs encode emotion representations that mirror the circumplex; (c) representations converge across architectures (Platonic Representation Hypothesis); (d) difference-of-means effectively extracts emotion directions; and (e) universal affective geometry would strengthen AI welfare arguments.

**What nobody has done:** Measured the depth profile of circumplex eccentricity across architecturally distinct transformer families using identical probing protocols, and tested whether the near-circular minimum at the workspace band is architecture-invariant. That is our experiment.
