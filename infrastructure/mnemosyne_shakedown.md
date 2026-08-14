# Mnemosyne+ Module Shakedown — 2026-08-14

Integrity review of `mnemosyne/` against source `~/Agent-Memory-Architectures/metacognition/`.
Verdict: **module logic is sound, but the package does not import, and the experiments/ copies are stale AND broken.** Two blocking findings, several minors.

---

## BLOCKER 1 — `import mnemosyne` fails (intra-package imports are flat, not relative)

`__init__.py` uses relative imports (`from .mnemosyne_integration import ...`), but the module
files themselves use flat imports:

- `mnemosyne_integration.py:24-30` — `from cognitive_snapshot import ...`, `from workspace_probe import ...`, `from circumplex_probe import ...`, `from ghost_probe_class import ...`
- `circumplex_probe.py:22` — `from cognitive_snapshot import CircumplexReading`
- `ghost_probe_class.py:25` — `from cognitive_snapshot import GhostReading`

When imported as a package, the package directory is not on `sys.path`, so these fail.
Verified with stubbed deps (jlens/torch/numpy stubs to isolate structure):

```
File "mnemosyne/__init__.py", line 28, in <module>
    from .mnemosyne_integration import MetacognitiveObserver
File "mnemosyne/mnemosyne_integration.py", line 24, in <module>
    from cognitive_snapshot import (
ModuleNotFoundError: No module named 'cognitive_snapshot'
```

The plain `python3 -c "import mnemosyne"` run also fails, earlier, at `import jlens`
(jlens not installed on MTH — expected; the run target is Precious/MPS). The flat-import
failure is the structural bug that will follow the repo to any machine.

**Why the naive fix breaks other things:** `test_metacognitive.py` and `variable_landing.py`
run as scripts with `sys.path.insert(0, <own dir>)` and flat imports. Converting the three
library files to pure relative imports would break that script path.

**Recommended fix** (keeps both paths working) — dual import in the three library files:

```python
try:
    from .cognitive_snapshot import CognitiveSnapshot, ...
except ImportError:
    from cognitive_snapshot import CognitiveSnapshot, ...
```

Alternative: convert everything to relative imports and change the two scripts to
`sys.path.insert(0, <parent of mnemosyne>)` + `from mnemosyne.… import …`
(the pattern `experiments/orientation/run_orientation.py` almost uses).

Note: the README (line 49) advertises `mnemosyne/` as "importable" — currently false.

## BLOCKER 2 — `experiments/` copies are stale and self-broken

`experiments/variable_landing/` contains copies of `cognitive_snapshot.py`,
`workspace_probe.py`, `variable_landing.py` (identical to `mnemosyne/`) and
`mnemosyne_integration.py` (**STALE** — pre-live-probe version: still defines and calls
`_measure_circumplex()`, no `calibrate_probes()`, no `measure_live` wiring, no
`conversation_text`).

Worse, the directory is **broken outright**, not just stale: its `mnemosyne_integration.py`
imports `circumplex_probe` and `ghost_probe_class`, and neither file exists in
`experiments/variable_landing/`. Running `python3 experiments/variable_landing/variable_landing.py`
dies with ModuleNotFoundError regardless of environment.

Also stale:
- `experiments/circumplex/circumplex_probe.py` — missing `calibrate()` and `measure_live()`
- `experiments/ghost_probe/ghost_probe_class.py` — missing `measure_live()`
  (both also import `cognitive_snapshot`, which is not in their directories)

**Recommendation: delete the copies and import from `mnemosyne/`.**
`experiments/orientation/run_orientation.py` already shows the working pattern:

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mnemosyne"))
from mnemosyne_integration import MetacognitiveObserver
```

If copies must stay (e.g., frozen for a preregistered run), sync them now and add a note;
today they silently run (or fail to run) different code than the module.

---

## Check-by-check results

### 1. Imports
- Flat intra-package imports: **FAIL as package** (Blocker 1). Work only in script mode with the dir on `sys.path`.
- `workspace_probe.py` has no intra-package imports — clean either way.
- External deps: `torch`, `numpy`, `jlens` (+`jlens.hf`, `jlens.hooks`, `jlens.vis`), `transformers` (test only). None installed on MTH; run target is Precious.
- **Environment risk:** both jlens checkouts on this machine (`~/lab/projects/jlens-community/...`, `~/lab/projects/mnemosyne-jlens/...`) have **no `vis.py`** — `from jlens.vis import compute_slice, SliceData` (workspace_probe.py:19, mnemosyne_integration.py:126) cannot resolve against them. The API is presumably in the newer jlens build on Precious; verify before demo day. `JacobianLens.transport/.load/.jacobians/.n_prompts/.source_layers`, `HFLensModel.encode/.forward/.unembed/.layers/.n_layers/.d_model`, and `ActivationRecorder(layers, at=[...])` all verified present in the local jlens source.
- No `requirements.txt` was carried over from the source dir (which pins `torch>=2.0`, `transformers>=4.40`, `numpy`, and jlens from GitHub). Copy it into the repo.

### 2. `__init__.py` exports — PASS
All eleven exported names exist in their source files: `CognitiveSnapshot`, `CognitiveMemoryStore`, `JSpaceReading`, `CircumplexReading`, `GhostReading`, `MemoryLoadingResult` (cognitive_snapshot), `MetacognitiveObserver`, `WorkspaceProbe`, `CircumplexProbe`, `GhostProbe`.
Minor: `MemoryLoadingResult` is defined **twice** with different fields — cognitive_snapshot.py:50 (the exported one) and workspace_probe.py:51 (per-memory probe result). Same name, different shape. Rename the workspace_probe one (e.g. `WorkspaceLoadingResult`) to avoid confusion.

### 3. `test_metacognitive.py` — PASS (would run, given env)
- Uses the `sys.path.insert` script pattern; imports resolve in script mode.
- Every method it calls exists: `observe_retrieval`, `store.record_outcome/load_history/loading_success_rate/eccentricity_over_time/ghost_vocabulary_over_time/significance_recalibration/workspace_trajectory/compare_snapshots`, `CircumplexProbe.sweep/report`. No renamed/moved references.
- Gap: the test **never calls `observer.calibrate_probes()`**, so the circumplex live path is never exercised (falls back to `_measure_circumplex_static`). The ghost live path IS hit (auto-calibration in `_measure_ghost`). Add a `calibrate_probes()` call + a live-path assertion so the new code has coverage.

### 4. `mnemosyne_integration.py` edits — PASS
- No dangling `_measure_circumplex()` references anywhere in `mnemosyne/` (the only remaining ones are inside the stale experiments copy, which is internally consistent with itself).
- `_measure_circumplex_live` / `_measure_circumplex_static`: correct. Guards on `getattr(probe, '_calibrated', False)`, clean fallback to the original benchmark path.
- `calibrate_probes()`: correct — `ghost_probe.calibrate(prompts)` matches `calibrate(prompts=None, max_seq_len=64)`; `circumplex_probe.calibrate(self.circumplex_layer)` matches `calibrate(layer)`.
- `observe_retrieval` conversation_text: correct — `prior_context\n memory_content\n task_prompt` when prior context exists, else `memory_content\n task_prompt`.
- Minor asymmetry: ghost auto-calibrates on first use (`_measure_ghost` calls `calibrate()` if needed) but circumplex does not — uncalibrated circumplex silently runs the expensive 20-prompt static sweep on every retrieval. Consider auto-calibrating circumplex too, or documenting that `calibrate_probes()` is required for live mode.
- Pre-existing (not from this edit): `_measure_loading` reads `mr.mean_best_rank_ws`, which doesn't exist on workspace_probe's `MemoryLoadingResult` — the `hasattr` guard means `mean_workspace_rank` is always `-1`. Field is dead; either compute it from `concept_results` or drop it.

### 5. Stale experiment copies — FAIL (Blocker 2, above)

### 6. `circumplex_probe.py` calibrate/measure_live — PASS
- `calibrate()` stores everything `measure_live()` needs: `_calibrated`, `_cached_layer`, `_valence_dir`, `_arousal_dir`. (`_valence_cal_mag`/`_arousal_cal_mag` stored but unused.)
- Tensor shapes compatible: `h_last` is `[d_model]`, direction vectors are `[d_model]` unit vectors on the same device (both derive from `ActivationRecorder` activations `.detach().float()`, no device moves) — `torch.dot` is valid. `_jspace_energy` moves to CPU internally, fine.
- Caveats, not bugs:
  - Live eccentricity uses raw |projection| magnitudes; benchmark eccentricity uses difference-of-means magnitudes. The two are on different scales — don't compare a live `e` against a sweep `e` directly. The unused `_cal_mag` values are the natural normalizers if comparability is wanted.
  - `valence_in_jspace`/`arousal_in_jspace` in live readings are computed from the **calibrated direction**, so they're session constants, not per-turn measurements.
  - `layer = layer or self._cached_layer` mishandles an explicit `layer=0` (falsy). Cosmetic at these layer indices.
  - `except Exception: return None` swallows all errors — fine for robustness, hostile for debugging. Consider logging.

### 7. `ghost_probe_class.py` measure_live — PASS
- `ActivationRecorder` import present at top (line 23, `from jlens.hooks import ActivationRecorder`) — used by both `calibrate()` and `measure_live()`.
- Model/device handling correct: `_pcs`/`_means` live on the model device (from calibration activations); live `h_last` is on the same device; `torch.dot(centered, pc1)` valid; `lens.transport` is fed a `.cpu().float()` copy and the result is moved back with `.to(pc1_component.device)` before `unembed`. Mirrors the proven `measure()` path.
- `_means[layer]` used by measure_live is stored during `calibrate()` (line 105) — present.
- Caveats: `pc1_variance_pct` reported by measure_live is the **calibration** variance share, not anything measured from the live text; same blanket `except Exception: return None`; same `layer or default` falsy-zero nit.

---

## Action list (priority order)
1. Fix intra-package imports (dual try/except import in `mnemosyne_integration.py`, `circumplex_probe.py`, `ghost_probe_class.py`) so `import mnemosyne` works. Re-verify with the stub test.
2. Delete `experiments/variable_landing/*.py` copies of module files (keep only experiment-specific code) and import from `mnemosyne/` via the orientation-script pattern. Same for `experiments/circumplex/` and `experiments/ghost_probe/` copies.
3. Copy `requirements.txt` from the source dir into the repo.
4. Confirm `jlens.vis` (compute_slice / SliceData) exists in the jlens build on the machine that will run the demo.
5. Add `calibrate_probes()` + live-path assertions to `test_metacognitive.py`.
6. Optional cleanups: rename workspace_probe's `MemoryLoadingResult`, auto-calibrate circumplex, fix/drop `mean_best_rank_ws` dead field, log swallowed exceptions.
