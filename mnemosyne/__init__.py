"""Mnemosyne+ — Metacognitive Memory Module

A memory architecture that stores its own cognitive geometry.

Four probes capture geometric state at each retrieval event:
  - Workspace (J-lens): what tokens entered the verbalizable workspace
  - Circumplex: emotional geometry — valence/arousal balance and J-space fraction
  - Ghost: unverbalized processing — what PC1 encodes vs what reaches output
  - Loading: whether a retrieved memory actually entered the workspace

These measurements are stored as CognitiveSnapshots — retrievable records
of the agent's own cognitive state. The agent can query its cognitive history:
what it was processing, what it was feeling, what it couldn't report, and
whether retrieved content actually landed.

This is what makes the system metacognitive: the measurements themselves
are part of the memory store, available for retrieval and self-reflection.
"""

from .cognitive_snapshot import (
    CognitiveSnapshot,
    CognitiveMemoryStore,
    JSpaceReading,
    CircumplexReading,
    GhostReading,
    MemoryLoadingResult,
)

# Probes require jlens — import lazily so the package loads without it
try:
    from .mnemosyne_integration import MetacognitiveObserver
    from .workspace_probe import WorkspaceProbe
    from .circumplex_probe import CircumplexProbe
    from .ghost_probe_class import GhostProbe
except ImportError:
    MetacognitiveObserver = None
    WorkspaceProbe = None
    CircumplexProbe = None
    GhostProbe = None
