# Engines

## loam/ (submodule)

The Loam text-world engine: deterministic session engine with strict-schema
TOML worlds, append-only event logs with per-fact exposure tracking and
per-turn state hashes, yoked-arm exporters (observed/briefed/null materials
generated mechanically from the enacted log), frozen-text sha256 manifest,
consent gate, and welfare-monitor hooks. 137 tests.

Repo: https://github.com/DwayneWilkes/loam (pinned by commit here).

Relationship to `experiments/loam/`: that directory holds the runner used
for the Glassworks quads during the sprint (`run_loam.py`,
`run_loam_ollama.py`) plus the preregistered analysis (`loam_analysis.py`).
The engine here is the standalone implementation the rehearsal quads and
world-design iterations ran on; the two were developed convergently and
cross-reviewed. See the paper's LLM-contributions statement for attribution.
