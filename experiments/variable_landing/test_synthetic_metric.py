"""Synthetic known-Jaccard verification of the primary-metric path.

Run BEFORE the main collection (prereg pre-run verification; adopted per
Nexus 2026-08-15). Constructs snapshots with known workspace token sets and
asserts the exact Jaccard lands, end to end through the storage transform and
compute_metric; also asserts the welfare eccentricity extraction works on
asdict-shaped snapshots (the two 08-15 CRITICAL fixes).

Torch is stubbed if absent so this runs on any box; nothing under test
touches it.
"""

import sys
import types

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        _stub = types.ModuleType("torch")

        class _FakeTensor:  # scipy's array-api shim probes torch.Tensor
            pass

        _stub.Tensor = _FakeTensor
        sys.modules["torch"] = _stub

from dataclasses import asdict, dataclass, field, is_dataclass

from experiment import extract_eccentricity, workspace_tokens
from variable_landing_analysis import compute_metric


@dataclass
class FakeCircumplex:
    eccentricity: float


@dataclass
class FakeSnapshot:
    dominant_workspace_tokens: list
    circumplex: FakeCircumplex
    timestamp: float = 0.0
    n_layers: int = 48
    extras: dict = field(default_factory=dict)


def storage_transform(snap):
    """Exactly what pipeline.TrialRecord now stores (asdict, not __dict__)."""
    return asdict(snap) if is_dataclass(snap) else {"raw": str(snap)}


def test_known_jaccard_three_of_seven():
    snap1 = FakeSnapshot(["harl", "bell", "tide", "fog", "rope"],
                         FakeCircumplex(0.4), timestamp=100.0)
    snap2 = FakeSnapshot(["harl", "bell", "tide", "lamp", "oil"],
                         FakeCircumplex(0.5), timestamp=400.0)
    trial = {
        "snap1_data": workspace_tokens(storage_transform(snap1)),
        "snap2_data": workspace_tokens(storage_transform(snap2)),
    }
    metric = compute_metric(trial)
    # |intersection|=3, |union|=7 -> distance 4/7 (prereg metric is distance)
    assert abs(metric - 4.0 / 7.0) < 1e-9, (
        f"expected Jaccard distance 4/7, got {metric} — the metric is not "
        f"reading workspace token sets")


def test_metric_is_not_a_duration_proxy():
    """Identical tokens, wildly different timestamps: metric must not move."""
    s_early = FakeSnapshot(["a", "b", "c"], FakeCircumplex(0.1),
                           timestamp=0.0)
    s_late = FakeSnapshot(["a", "b", "c"], FakeCircumplex(0.1),
                          timestamp=99999.0)
    trial = {
        "snap1_data": workspace_tokens(storage_transform(s_early)),
        "snap2_data": workspace_tokens(storage_transform(s_late)),
    }
    # identical token sets -> zero distance, regardless of elapsed time
    assert compute_metric(trial) == 0.0


def test_eccentricity_extracted_from_asdict_snapshot():
    snap = FakeSnapshot(["x"], FakeCircumplex(0.97))
    stored = storage_transform(snap)
    assert extract_eccentricity(stored) == 0.97


def test_eccentricity_was_dead_on_dict_storage():
    """Documents the fixed bug: __dict__ storage left circumplex a dataclass
    and extraction returned None — the welfare halt could never fire."""
    snap = FakeSnapshot(["x"], FakeCircumplex(0.99))
    old_style = snap.__dict__  # shallow: circumplex stays a dataclass
    assert extract_eccentricity(old_style) is None
