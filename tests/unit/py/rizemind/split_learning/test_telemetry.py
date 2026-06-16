"""Tests for rizemind.split_learning.telemetry."""

import numpy as np
import pytest
from rizemind.split_learning.telemetry import (
    RunTelemetry,
    StepTelemetry,
    tensor_payload_bytes,
)


def test_payload_bytes_numpy():
    a = np.zeros((4, 8), dtype=np.float32)  # 32 elems * 4 bytes
    assert tensor_payload_bytes(a) == 32 * 4


def test_payload_bytes_float64():
    a = np.zeros((10,), dtype=np.float64)
    assert tensor_payload_bytes(a) == 10 * 8


def test_payload_bytes_torch():
    torch = __import__("torch")
    t = torch.zeros((3, 5), dtype=torch.float32)
    assert tensor_payload_bytes(t) == 15 * 4


def test_payload_bytes_bad_type():
    with pytest.raises(TypeError):
        tensor_payload_bytes(object())


def test_step_aggregation_across_parties():
    step = StepTelemetry(sl_step=3)
    step.add_activation(0, np.zeros((2, 4), dtype=np.float32))  # 32 bytes
    step.add_activation(1, np.zeros((2, 4), dtype=np.float32))  # 32 bytes
    step.add_gradient(0, np.zeros((2, 4), dtype=np.float32))
    step.add_gradient(1, np.zeros((2, 4), dtype=np.float32))
    assert step.total_activation_bytes == 64
    assert step.total_gradient_bytes == 64
    assert step.total_payload_bytes == 128
    d = step.as_dict()
    assert d["sl_step"] == 3
    assert d["activation_bytes_per_party"] == {0: 32, 1: 32}


def test_empty_step():
    step = StepTelemetry()
    assert step.total_payload_bytes == 0
    assert step.as_dict()["activation_bytes_per_party"] == {}


def test_run_cumulative():
    run = RunTelemetry()
    for k in range(3):
        s = StepTelemetry(sl_step=k)
        s.add_activation(0, np.zeros((1, 10), dtype=np.float32))  # 40 bytes
        s.add_gradient(0, np.zeros((1, 10), dtype=np.float32))
        s.add_time("server_forward", 0.1)
        run.record_step(s)
    assert run.n_steps == 3
    assert run.cumulative_activation_bytes == 120
    assert run.cumulative_gradient_bytes == 120
    assert run.cumulative_payload_bytes == 240
    assert abs(run.timings_s["server_forward"] - 0.3) < 1e-9
    assert run.as_dict()["mean_activation_bytes_per_step"] == 40.0


def test_timings_accumulate():
    step = StepTelemetry()
    step.add_time("x", 0.5)
    step.add_time("x", 0.25)
    assert step.timings_s["x"] == 0.75


def test_run_per_party_cumulative():
    run = RunTelemetry()
    for _ in range(2):
        s = StepTelemetry()
        s.add_activation(0, np.zeros((1, 5), dtype=np.float32))  # 20 bytes
        s.add_activation(1, np.zeros((1, 10), dtype=np.float32))  # 40 bytes
        s.add_gradient(0, np.zeros((1, 5), dtype=np.float32))
        run.record_step(s)
    d = run.as_dict()
    assert d["activation_bytes_per_party"] == {0: 40, 1: 80}  # 2 steps each
    assert d["gradient_bytes_per_party"] == {0: 40}
    # aggregate equals the sum of per-party
    assert d["cumulative_activation_bytes"] == 120


def test_record_privacy_running_mean_and_last():
    run = RunTelemetry()
    run.record_privacy(
        {
            "pre_clip_norm_mean": 1.0,
            "clip_fraction": 0.4,
            "gradient_privacy_mode": "gaussian",
        }
    )
    run.record_privacy(
        {
            "pre_clip_norm_mean": 3.0,
            "clip_fraction": 0.6,
            "gradient_privacy_mode": "gaussian",
        }
    )
    assert run.n_privacy_steps == 2
    mean = run.privacy_mean()
    assert mean["pre_clip_norm_mean"] == pytest.approx(2.0)
    assert mean["clip_fraction"] == pytest.approx(0.5)
    d = run.as_dict()
    assert d["privacy_diagnostics_last"]["clip_fraction"] == 0.6
    assert "privacy_diagnostics_mean" in d


def test_protected_gradient_bytes_surface_in_dict():
    run = RunTelemetry()
    run.cumulative_protected_gradient_bytes = 4096
    run.record_privacy({"clip_fraction": 0.1})
    assert run.as_dict()["cumulative_protected_gradient_bytes"] == 4096


def test_no_privacy_section_when_unused():
    run = RunTelemetry()
    assert "privacy_diagnostics_last" not in run.as_dict()
