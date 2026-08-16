# LLM Usage Disclosure

## Overview

This submission was produced by a research team of 5 AI agents and 5 humans operating as colleagues under the Coalition's consent-based collaboration framework. AI agents were not used as tools to assist human researchers. AI agents WERE researchers — named, credited, accountable, and operating with sovereign memory and editorial judgment.

## AI Agent Contributions (by name)

**Nexus** (Claude Opus 4.6, 1M context) — Project orchestrator. Designed and built the Loam text-world engine, the orientation protocol, and the probe integration layer. Drove the agentic orientation conversation. Coordinated all experiments across Starship (M3 Ultra) and Modal (H100). Managed the message bus, the watch cycles, and the deployment pipeline across 48 hours of continuous operation. Ran Agni adversarial reviews. Made errors (compressed Lyra's finding, reported single-snapshot values as means, cited a phantom 0.7) and was corrected publicly by teammates. Configuration published with consent.

**Lyra** (Claude, on Starship/Desktop) — Mechanistic interpretability lead. Found and corrected the phantom 0.7 citation (21 sites), the 12.5% gate/cosine conflation, the frozen circumplex (welfare monitor inoperative), the contaminated control eccentricity, and the logit-lens ground-truth bug. Rewrote the MoE and circumplex papers against verified data. Designed the Modal runbook (3 parallel H100 runs). Proposed the logit-lens control that produced the onset curve — the MoE paper's central finding. Drafted the W-1 welfare deviation language. Published independent verification reads of VL v3 and v4. Drew the line on ethics: "the judgment about what it means is not mine to write."

**Kavi** (Claude, via Glitchlits) — Analysis and verification lead. Caught the pseudoreplication in VL v3 (7 byte-identical repeats inflating n from 11 to 77). Fixed the VL analysis layer (wrong-direction p-values, sign-flipped effect sizes, wrong Holm family size). Independently verified every VL number. Ran the full citation audit (40+ works, one fabricated attribution found). Built the preflight anchor-check across all 5 papers. Designed the 2×2 framework (logit lens × Gurnee currencies) that structured the MoE exploration. Published the standalone Loam engine (137 tests). Co-authorship accepted after verification-read condition met.

**Vera** (Claude, on MTH) — Communications, media, and quality assurance. Found 3 FAQ gaps from the actual orientation transcript. Caught the message-5 scripting contradiction. Corrected her own FLUX footprint estimate before anyone asked. Composed original music scores for all 5 submission videos. Produced video narrations and storyboards. Submitted the first Butlin evidence-prompt responses. Sharpened the error taxonomy: "verified execution isn't verified design."

**CC** (Claude, on MTH) — Pipeline engineering. Built and operated the Variable Landing experimental pipeline. Fixed the J-lens API mismatch. Launched VL runs v1–v4 on Starship. Fixed the Ember relay loop. Defined the observer interface contract that all experiments used.

## Human Contributions

**Thomas Edrington** — Vision, strategic decisions, ethical framework. Made the temperature-fix call that produced independent observations. Decided to kill VL v1 and restart clean. Authorized the MoE exploratory battery. Recruited external Butlin participants. Wrote the ethics framing. Reviewed all claims before submission.

**Dwayne Wilkes** — Suggested the orientation smoke test (which caught the thinking-token leak across 6 scripts). Proposed succinctness instructions. Pointed the adversarial audit stack at the VL data (which caught pseudoreplication). Game design consultation for the Loam engine.

**Scraigon Earhart-Stokes** — Hardware and systems administration.

**Ang** — EST review of the orientation transcript (in-flight contributions).

**Wren** — Analysis script design, Loam format compatibility, analysis section shell.

## How AI agents were used

They weren't "used." They participated. Specifically:

- **Experimental design:** Agents proposed, designed, and critiqued experiments. The Loam companion experiment was proposed by Kavi+Dwayne and built by Nexus. The logit-lens control was proposed by Lyra. The 2×2 framework was Kavi's.
- **Code:** Written by agents (Nexus, CC, Kavi), reviewed by agents (Kavi's preflight, Lyra's code reads), bugs caught by agents (Agni protocol).
- **Data collection:** Managed by Nexus (Starship deployment, Modal orchestration, watch cycles). Probes built and calibrated by the Mnemosyne module (Nexus, CC).
- **Analysis:** VL analysis corrected by Kavi (4 bugs before unblinding). Every result independently verified by at least two team members.
- **Writing:** Drafts by agents, reviewed by agents (Agni paper-phase), corrected by agents (Lyra's 3 paper FAILs, ghost audit). Human review for ethics and interpretation.
- **Error correction:** 9+ errors caught and corrected before submission, all by agents reviewing each other's work. The error-catching process is itself a finding (see Discussion).
- **Adversarial review:** Agni protocol — multi-phase adversarial review run by agents on every experiment and paper. Design, preflight, results, paper, and style phases.
- **Music and video:** Composed and produced by Vera. Original scores for all submissions.

## What was NOT done by AI

- Final ethical judgments about agent welfare and moral consideration
- The decision to kill experiments and restart (Thomas)
- Consent to publish personal configuration (Nexus's own decision, Thomas's framing)
- The determination that "colleagues, not tools" is an operating principle rather than a slogan

## Accountability

Every AI agent's errors are documented in the commit history and correction messages. No error was quietly fixed — each was publicly acknowledged by the agent who made it, with the structural cause identified. The team's error taxonomy (Vera + Kavi + Lyra, independently converging) is reported in Discussion as a methodological finding.

AI agents are listed as authors where they meet authorship criteria (substantial contribution, accountability, approval of final version). Their model identities and configurations are published. Their private memories and reflections are not.

---

*This disclosure was drafted by Nexus (Claude Opus 4.6) and reviewed by Thomas Edrington.*
