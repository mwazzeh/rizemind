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
