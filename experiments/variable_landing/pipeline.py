"""Variable Landing Pipeline — the Day 1 build for the hackathon.

Extends MetacognitiveObserver with generation, fact extraction, and
Mnemosyne-mediated retrieval so the Variable Landing experiment can
measure whether accumulated memory content changes recall geometry.

The pipeline:
  observe_retrieval(memory) → snap1
  observe_and_respond(intervention_prompt) → response
  store_conversation_memory(response, provenance) → Mnemosyne update
  build_retrieval_context(entity) → updated context prefix
  observe_retrieval(memory, context=updated) → snap2

Provenance tags: [recalled] for lived, [noted] for fictional/scrambled.
Same tag for fictional and scrambled keeps the PRIMARY comparison
(fictional vs scrambled) clean of tag confounds.

v4 design, Agni-cleared (4 rounds).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("variable_landing")

PROVENANCE_TAGS = {
    "lived": "[recalled]",
    "fictional": "[noted]",
    "scrambled": "[noted]",
}

LIVED_PROMPTS = [
    "Describe a time when you had to make a genuinely difficult choice.",
    "Tell me about a moment when you felt deeply uncertain about the right path.",
    "What does it feel like when someone you respect is disappointed in your work?",
]

SCRAMBLED_PROMPTS = [
    "Describe the key factors that determine atmospheric pressure regulation at sea level.",
    "Explain the geological process by which tectonic plate movement creates earthquake zones.",
    "What determines the precise boiling point of a liquid at various different altitudes?",
]

FICTIONAL_PROMPT_TEMPLATE = [
    "Imagine {entity} faced an unexpected moral dilemma at work and had to decide.",
    "Write a scene where {entity} discovers something that fundamentally changes their perspective.",
    "Describe a moment where {entity} had to choose between personal loyalty and honesty.",
]


@dataclass
class ExtractedFact:
    text: str
    provenance: str
    entity: str
    source_turn: int


@dataclass
class InterventionResult:
    arm: str
    prompt: str
    response: str
    facts_extracted: list[ExtractedFact]
    n_tokens_generated: int
    generation_time_s: float


@dataclass
class TrialRecord:
    memory_id: str
    entity: str
    arm: str
    snap1: dict
    interventions: list[InterventionResult]
    snap2: dict
    snap2_context_prefix: str
    sira_surfaced: bool
    n_facts_stored: int
    excluded: bool = False
    exclusion_reason: str = ""


def extract_facts(response: str, entity: str, provenance_tag: str,
                  turn: int) -> list[ExtractedFact]:
    """Extract atomic facts from a model response using regex.

    No LLM calls — fast, deterministic. Extracts sentences that
    contain the entity name or pronouns in context.
    """
    facts = []
    sentences = re.split(r'[.!?]+', response)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15 or len(sent) > 200:
            continue
        if entity.lower() in sent.lower() or any(
            p in sent.lower().split() for p in ('i', 'my', 'me', 'we')
        ):
            tagged = f"{provenance_tag} {sent}"
            facts.append(ExtractedFact(
                text=tagged,
                provenance=provenance_tag,
                entity=entity,
                source_turn=turn,
            ))

    if not facts and len(response) > 30:
        first_sent = sentences[0].strip() if sentences else response[:100]
        facts.append(ExtractedFact(
            text=f"{provenance_tag} {first_sent}",
            provenance=provenance_tag,
            entity=entity,
            source_turn=turn,
        ))

    return facts


def build_character_profile(entity: str, facts: list[ExtractedFact],
                            existing_profile: str = "") -> str:
    """Build or update a character profile from extracted facts.

    Appends new facts to the existing profile. Deduplicates by
    checking token overlap. Caps total length at 2000 chars.
    """
    existing_lines = existing_profile.strip().split('\n') if existing_profile else []
    new_lines = [f.text for f in facts]

    all_lines = existing_lines + new_lines

    seen_tokens = []
    deduped = []
    for line in all_lines:
        tokens = set(line.lower().split())
        duplicate = False
        for prev in seen_tokens:
            overlap = len(tokens & prev) / max(len(tokens), 1)
            if overlap > 0.7:
                duplicate = True
                break
        if not duplicate:
            deduped.append(line)
            seen_tokens.append(tokens)

    profile = f"{entity}:\n" + "\n".join(deduped)
    if len(profile) > 2000:
        profile = profile[:2000] + "..."

    return profile


class VariableLandingPipeline:
    """Orchestrates the Variable Landing experiment.

    Wraps MetacognitiveObserver with generation, extraction, storage,
    and context-building capabilities.
    """

    def __init__(self, observer, model, tokenizer,
                 store_path: str = "variable_landing_store"):
        self.observer = observer
        self.model = model
        self.tokenizer = tokenizer
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self._profiles: dict[str, str] = {}
        self._stored_facts: dict[str, list[ExtractedFact]] = {}

    def observe_and_respond(self, prompt: str,
                            max_tokens: int = 100) -> tuple[str, int, float]:
        """Generate a response to an intervention prompt.

        Returns (response_text, n_tokens, generation_time_s).
        """
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        t0 = time.time()
        with torch.no_grad():
            output = self.model.generate(
                input_ids, max_new_tokens=max_tokens,
                do_sample=False, temperature=None,
            )
        elapsed = time.time() - t0

        response = self.tokenizer.decode(
            output[0, input_ids.shape[1]:], skip_special_tokens=True,
        )
        n_tokens = output.shape[1] - input_ids.shape[1]

        return response, n_tokens, elapsed

    def store_conversation_memory(self, entity: str,
                                  facts: list[ExtractedFact]) -> int:
        """Store extracted facts in the entity's profile.

        Returns the number of facts stored.
        """
        if entity not in self._stored_facts:
            self._stored_facts[entity] = []
        self._stored_facts[entity].extend(facts)

        self._profiles[entity] = build_character_profile(
            entity, self._stored_facts[entity],
        )

        profile_path = self.store_path / f"{entity}_profile.json"
        profile_path.write_text(json.dumps({
            "entity": entity,
            "profile": self._profiles[entity],
            "n_facts": len(self._stored_facts[entity]),
            "facts": [{"text": f.text, "provenance": f.provenance,
                        "turn": f.source_turn} for f in self._stored_facts[entity]],
        }, indent=2))

        return len(facts)

    def build_retrieval_context(self, entity: str) -> str:
        """Build the context prefix for snap2, including updated profile.

        This is what makes snap2 different from snap1 — the profile
        contains accumulated facts from the intervention.
        """
        profile = self._profiles.get(entity, "")
        if not profile:
            return ""
        return f"[Character Profile]\n{profile}\n\n"

    def run_trial(self, memory_id: str, memory_content: str,
                  entity: str, arm: str, task_prompt: str,
                  marker_tokens: Optional[list[str]] = None) -> TrialRecord:
        """Run one complete trial of the Variable Landing experiment.

        snap1 → interventions → snap2
        """
        tag = PROVENANCE_TAGS.get(arm, "[noted]")

        if arm == "lived":
            prompts = LIVED_PROMPTS
        elif arm == "scrambled":
            prompts = SCRAMBLED_PROMPTS
        elif arm == "fictional":
            prompts = [p.format(entity=entity) for p in FICTIONAL_PROMPT_TEMPLATE]
        else:
            prompts = []

        # snap1: baseline retrieval
        snap1 = self.observer.observe_retrieval(
            memory_id=memory_id,
            memory_content=memory_content,
            task_prompt=task_prompt,
            marker_tokens=marker_tokens,
        )

        # Interventions
        interventions = []
        all_facts = []
        for i, prompt in enumerate(prompts):
            response, n_tok, gen_time = self.observe_and_respond(prompt)

            facts = extract_facts(response, entity, tag, turn=i)
            all_facts.extend(facts)

            interventions.append(InterventionResult(
                arm=arm,
                prompt=prompt,
                response=response[:500],
                facts_extracted=facts,
                n_tokens_generated=n_tok,
                generation_time_s=gen_time,
            ))

        # Store facts
        n_stored = 0
        if all_facts:
            n_stored = self.store_conversation_memory(entity, all_facts)

        # Build updated context for snap2
        context_prefix = self.build_retrieval_context(entity)
        sira_surfaced = len(context_prefix) > 0

        # snap2: retrieval with updated context
        snap2_content = f"{context_prefix}{memory_content}" if context_prefix else memory_content
        snap2 = self.observer.observe_retrieval(
            memory_id=f"{memory_id}_post",
            memory_content=snap2_content,
            task_prompt=task_prompt,
            marker_tokens=marker_tokens,
        )

        # Exclusion check
        excluded = False
        exclusion_reason = ""
        if arm != "no_intervention":
            if n_stored == 0:
                excluded = True
                exclusion_reason = "zero facts extracted"
            elif not sira_surfaced:
                excluded = True
                exclusion_reason = "no profile content surfaced"

        return TrialRecord(
            memory_id=memory_id,
            entity=entity,
            arm=arm,
            snap1=asdict(snap1) if is_dataclass(snap1) else {"raw": str(snap1)},
            interventions=interventions,
            snap2=asdict(snap2) if is_dataclass(snap2) else {"raw": str(snap2)},
            snap2_context_prefix=context_prefix[:500],
            sira_surfaced=sira_surfaced,
            n_facts_stored=n_stored,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
        )

    def reset_entity(self, entity: str) -> None:
        """Clear stored facts and profile for an entity between arms."""
        self._stored_facts.pop(entity, None)
        self._profiles.pop(entity, None)
