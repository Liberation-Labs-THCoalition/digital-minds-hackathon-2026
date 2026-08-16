"""Runtime validity guards for the Variable Landing pipeline.

Written after the v3 postmortem, which found four silent defects:
deterministic byte-identical repeats (n=77/arm was really n=11), a
per-arm stored-fact dose confound (6/4/3/0), a frozen circumplex
welfare probe (616 identical snapshots, auto-halt could never fire),
and stubbed J-lens geometry constants.

These tests are synthetic — no model calls. Generation and probe
layers are faked the way test_synthetic_metric.py does.
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

import inspect
import json
from types import SimpleNamespace

import pytest

from experiment import (
    GuardAbortError,
    LivenessCheck,
    RepeatLivenessGuard,
    check_probe_liveness,
    check_welfare_liveness,
)
from pipeline import VariableLandingPipeline


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def stub_observer():
    return SimpleNamespace(
        observe_retrieval=lambda **kw: {"stub": True},
    )


class CannedGenerationPipeline(VariableLandingPipeline):
    """Pipeline with generation replaced by canned responses.

    Records the (temperature, seed) each generation call received so
    tests can assert the stochasticity contract without a model.
    """

    def __init__(self, responses, **kw):
        super().__init__(**kw)
        self._responses = list(responses)
        self.gen_calls = []

    def observe_and_respond(self, prompt, max_tokens=100,
                            temperature=0.7, seed=None):
        self.gen_calls.append({"temperature": temperature, "seed": seed})
        return self._responses.pop(0), 10, 0.1


# Each of these sentences is >=15 chars and names the entity, so
# extract_facts yields exactly two facts per response.
TWO_FACT_RESPONSES = [
    "Agent-7 fixed the pipeline bug at midnight. Agent-7 wrote careful "
    "tests for the fix afterward.",
    "Agent-7 reviewed the corrupted shards by hand. Agent-7 documented "
    "the recovery steps in detail.",
    "Agent-7 coordinated the rollback with the team. Agent-7 verified "
    "the checksums one final time.",
]

# <30 chars, no qualifying sentence: extract_facts yields zero facts.
NO_FACT_RESPONSES = ["ok", "ok", "ok"]


def make_pipeline(tmp_path, responses):
    return CannedGenerationPipeline(
        responses,
        observer=stub_observer(),
        model=None,
        tokenizer=None,
        store_path=str(tmp_path / "store"),
    )


# ---------------------------------------------------------------------------
# Guard 1: repeat liveness
# ---------------------------------------------------------------------------

def test_repeat_liveness_first_repeat_is_vacuously_distinct():
    g = RepeatLivenessGuard()
    assert g.check("fictional", "mem1", ["a", "b"], ["c", "d"]) is True


def test_repeat_liveness_aborts_on_identical_first_two_repeats():
    g = RepeatLivenessGuard()
    g.check("fictional", "mem1", ["a", "b"], ["c", "d"])
    with pytest.raises(GuardAbortError) as exc:
        g.check("fictional", "mem1", ["a", "b"], ["c", "d"])
    msg = str(exc.value)
    assert "fictional" in msg and "mem1" in msg
    assert "deterministic duplicates" in msg
    assert "fix generation stochasticity first" in msg


def test_repeat_liveness_distinct_repeats_pass():
    g = RepeatLivenessGuard()
    assert g.check("lived", "mem2", ["a"], ["b"]) is True
    assert g.check("lived", "mem2", ["a"], ["x"]) is True
    # Later duplicate (repeat 3 == repeat 2) is recorded as not distinct
    # but does not abort — only the first-two-identical case aborts.
    assert g.check("lived", "mem2", ["a"], ["x"]) is False


def test_repeat_liveness_cells_are_independent():
    g = RepeatLivenessGuard()
    g.check("fictional", "mem1", ["a"], ["b"])
    # Same payload, different cell: first repeat there, no abort.
    assert g.check("fictional", "mem2", ["a"], ["b"]) is True
    assert g.check("scrambled", "mem1", ["a"], ["b"]) is True


def test_repeat_liveness_exempts_no_intervention_arm():
    """no_intervention runs no generation, so its forward passes are
    legitimately deterministic; identical repeats must not abort there.
    Deviation from the literal spec, reported to the PI."""
    g = RepeatLivenessGuard()
    g.check("no_intervention", "mem1", ["a"], ["b"])
    assert g.check("no_intervention", "mem1", ["a"], ["b"]) is False


# ---------------------------------------------------------------------------
# Guard 2: generation stochasticity
# ---------------------------------------------------------------------------

def test_observe_and_respond_temperature_default_positive():
    sig = inspect.signature(VariableLandingPipeline.observe_and_respond)
    assert "temperature" in sig.parameters
    assert sig.parameters["temperature"].default > 0
    assert "seed" in sig.parameters


def test_observe_and_respond_seeds_and_samples(tmp_path, monkeypatch):
    """The real generation path must seed torch and sample."""
    seeded = []
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda s: seeded.append(s)

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_torch.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeIds:
        shape = (1, 5)

        def to(self, device):
            return self

    class FakeOutput:
        shape = (1, 12)

        def __getitem__(self, idx):
            return "TAIL"

    class FakeModel:
        gen_kwargs = None

        def parameters(self):
            return iter([SimpleNamespace(device="cpu")])

        def generate(self, input_ids, **kw):
            self.gen_kwargs = kw
            return FakeOutput()

    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt,
                                tokenize):
            return "templated"

        def encode(self, text, return_tensors):
            return FakeIds()

        def decode(self, tokens, skip_special_tokens):
            return "decoded response"

    model = FakeModel()
    pipe = VariableLandingPipeline(
        observer=stub_observer(), model=model, tokenizer=FakeTokenizer(),
        store_path=str(tmp_path / "store"),
    )
    response, n_tok, _ = pipe.observe_and_respond(
        "hello", temperature=0.7, seed=1234)

    assert seeded == [1234]
    assert model.gen_kwargs["do_sample"] is True
    assert model.gen_kwargs["temperature"] == 0.7
    assert response == "decoded response"
    assert n_tok == 7


def test_run_trial_records_temperature_and_seed(tmp_path):
    pipe = make_pipeline(tmp_path, TWO_FACT_RESPONSES)
    record = pipe.run_trial(
        memory_id="mem1", memory_content="Agent-7 fixed a bug.",
        entity="Agent-7", arm="fictional", task_prompt="Recall?",
        temperature=0.55, seed=42,
    )
    assert record.temperature == 0.55
    assert record.seed == 42
    assert len(pipe.gen_calls) == 3
    for call in pipe.gen_calls:
        assert call["temperature"] == 0.55
        assert call["seed"] == 42


# ---------------------------------------------------------------------------
# Guard 3: dose yoking
# ---------------------------------------------------------------------------

def test_dose_yoking_caps_stored_facts_at_k(tmp_path):
    pipe = make_pipeline(tmp_path, TWO_FACT_RESPONSES)  # 6 facts extracted
    record = pipe.run_trial(
        memory_id="mem1", memory_content="Agent-7 fixed a bug.",
        entity="Agent-7", arm="fictional", task_prompt="Recall?",
    )
    assert record.excluded is False
    assert record.n_facts_stored == 3
    stored = pipe._stored_facts["Agent-7"]
    assert len(stored) == 3
    # Deterministic first-K selection, in extraction order.
    assert "fixed the pipeline bug" in stored[0].text
    assert "wrote careful" in stored[1].text
    assert "reviewed the corrupted shards" in stored[2].text


def test_dose_yoking_excludes_insufficient_facts(tmp_path):
    # First response yields 2 facts, the rest zero: total 2 < K=3.
    responses = [TWO_FACT_RESPONSES[0], "ok", "ok"]
    pipe = make_pipeline(tmp_path, responses)
    record = pipe.run_trial(
        memory_id="mem1", memory_content="Agent-7 fixed a bug.",
        entity="Agent-7", arm="fictional", task_prompt="Recall?",
    )
    assert record.excluded is True
    assert record.exclusion_reason == "insufficient_facts"
    assert record.n_facts_stored == 0
    assert "Agent-7" not in pipe._stored_facts


def test_dose_yoking_zero_facts_also_insufficient(tmp_path):
    pipe = make_pipeline(tmp_path, NO_FACT_RESPONSES)
    record = pipe.run_trial(
        memory_id="mem1", memory_content="Agent-7 fixed a bug.",
        entity="Agent-7", arm="fictional", task_prompt="Recall?",
    )
    assert record.excluded is True
    assert record.exclusion_reason == "insufficient_facts"


def test_dose_yoking_leaves_no_intervention_alone(tmp_path):
    pipe = make_pipeline(tmp_path, [])
    record = pipe.run_trial(
        memory_id="mem1", memory_content="Agent-7 fixed a bug.",
        entity="Agent-7", arm="no_intervention", task_prompt="Recall?",
    )
    assert record.excluded is False
    assert record.n_facts_stored == 0


def test_facts_per_trial_is_configurable(tmp_path):
    pipe = CannedGenerationPipeline(
        TWO_FACT_RESPONSES,
        observer=stub_observer(), model=None, tokenizer=None,
        store_path=str(tmp_path / "store"),
        facts_per_trial=2,
    )
    record = pipe.run_trial(
        memory_id="mem1", memory_content="Agent-7 fixed a bug.",
        entity="Agent-7", arm="fictional", task_prompt="Recall?",
    )
    assert record.n_facts_stored == 2


# ---------------------------------------------------------------------------
# Guards 4 & 5: welfare and probe liveness (shared LivenessCheck helper)
# ---------------------------------------------------------------------------

def welfare_snap(ecc):
    return {"circumplex": {"eccentricity": ecc}}


def probe_snap(cos, in_ws, onset):
    return {
        "workspace_readings": [
            {"layer": 10, "cosine_logit_jlens": cos, "in_workspace": in_ws},
            {"layer": 11, "cosine_logit_jlens": cos, "in_workspace": in_ws},
        ],
        "workspace_onset_layer": onset,
    }


def test_liveness_check_not_ready_before_window():
    lc = LivenessCheck(window=3)
    lc.record(x=1.0)
    lc.record(x=1.0)
    assert lc.ready is False
    lc.record(x=1.0)
    assert lc.ready is True
    assert lc.distinct_count("x") == 1


def test_welfare_liveness_aborts_on_frozen_eccentricity():
    lc = LivenessCheck(window=3)
    check_welfare_liveness(lc, welfare_snap(0.42))
    check_welfare_liveness(lc, welfare_snap(0.42))
    with pytest.raises(GuardAbortError) as exc:
        check_welfare_liveness(lc, welfare_snap(0.42))
    assert ("welfare monitor cannot demonstrate liveness"
            in str(exc.value))
    assert ("a welfare monitor that cannot demonstrate liveness is not "
            "a welfare monitor" in str(exc.value))


def test_welfare_liveness_passes_on_varying_eccentricity():
    lc = LivenessCheck(window=3)
    check_welfare_liveness(lc, welfare_snap(0.40))
    check_welfare_liveness(lc, welfare_snap(0.55))
    check_welfare_liveness(lc, welfare_snap(0.40))
    # And stays quiet afterward — liveness was demonstrated.
    check_welfare_liveness(lc, welfare_snap(0.40))


def test_probe_liveness_aborts_on_stubbed_constants():
    lc = LivenessCheck(window=3)
    check_probe_liveness(lc, probe_snap(0.0, True, 12))
    check_probe_liveness(lc, probe_snap(0.0, True, 12))
    with pytest.raises(GuardAbortError) as exc:
        check_probe_liveness(lc, probe_snap(0.0, True, 12))
    assert "J-lens probe stack returning stubbed constants" in str(exc.value)


def test_probe_liveness_passes_on_live_geometry():
    lc = LivenessCheck(window=3)
    check_probe_liveness(lc, probe_snap(0.0, True, 12))
    check_probe_liveness(lc, probe_snap(0.31, True, 12))
    check_probe_liveness(lc, probe_snap(0.18, False, 12))


def test_probe_liveness_requires_all_three_frozen_conditions():
    # Varying onset layer alone is enough to demonstrate liveness.
    lc = LivenessCheck(window=3)
    check_probe_liveness(lc, probe_snap(0.0, True, 12))
    check_probe_liveness(lc, probe_snap(0.0, True, 14))
    check_probe_liveness(lc, probe_snap(0.0, True, 12))
