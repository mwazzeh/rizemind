"""Unit tests for LayerProfiler and TrainingProfiler."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_split.analysis import LayerProfiler, TrainingProfiler

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )


@pytest.fixture
def simple_input():
    return torch.randn(32, 16)


@pytest.fixture
def simple_stats(simple_model, simple_input):
    p = LayerProfiler(simple_model, simple_input, n_warmup=2, n_reps=5, skip_gpu=True)
    return p.profile()


@pytest.fixture
def simple_profiler(simple_model, simple_input):
    return LayerProfiler(simple_model, simple_input, n_warmup=2, n_reps=5, skip_gpu=True)


# ---------------------------------------------------------------------------
# Layer count
# ---------------------------------------------------------------------------


def test_layer_count_matches_leaf_modules(simple_model, simple_stats):
    leaf_count = sum(1 for _, m in simple_model.named_modules() if not list(m.children()))
    assert len(simple_stats) == leaf_count


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_first_layer_input_shape(simple_stats):
    assert simple_stats[0].in_shape == (32, 16)


def test_first_layer_output_shape(simple_stats):
    assert simple_stats[0].out_shape == (32, 8)


def test_relu_preserves_shape(simple_stats):
    relu_stats = [s for s in simple_stats if s.layer_type == "ReLU"]
    s = relu_stats[0]
    assert s.in_shape == s.out_shape


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_linear_param_count(simple_stats):
    linears = [s for s in simple_stats if s.layer_type == "Linear"]
    assert linears[0].n_params == 136   # 16*8 + 8
    assert linears[1].n_params == 36    # 8*4 + 4


def test_relu_has_no_params(simple_stats):
    relu_stats = [s for s in simple_stats if s.layer_type == "ReLU"]
    assert relu_stats[0].n_params == 0
    assert relu_stats[0].param_bytes == 0


def test_total_params(simple_stats):
    assert sum(s.n_params for s in simple_stats) == 172


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def test_linear_flops_positive(simple_stats):
    linears = [s for s in simple_stats if s.layer_type == "Linear"]
    assert all(s.flops > 0 for s in linears)


def test_linear_flops_value(simple_stats):
    # Linear(16, 8) with batch 32: FLOPs = 2 * 32 * 16 * 8 = 8192
    linears = [s for s in simple_stats if s.layer_type == "Linear"]
    assert linears[0].flops == 2 * 32 * 16 * 8


def test_macs_is_half_flops(simple_stats):
    for s in simple_stats:
        assert s.macs == s.flops // 2


def test_relu_flops_zero(simple_stats):
    relu_stats = [s for s in simple_stats if s.layer_type == "ReLU"]
    assert relu_stats[0].flops == 0


# ---------------------------------------------------------------------------
# Activation bytes
# ---------------------------------------------------------------------------


def test_activation_bytes_linear_first(simple_stats):
    assert simple_stats[0].activation_bytes == 32 * 8 * 4  # float32


def test_transfer_kb(simple_stats):
    for s in simple_stats:
        assert abs(s.transfer_kb - s.activation_bytes / 1024) < 1e-6


# ---------------------------------------------------------------------------
# Cumulative metrics
# ---------------------------------------------------------------------------


def test_cumulative_flops_increases(simple_stats):
    prev = -1
    for s in simple_stats:
        assert s.cumulative_flops >= prev
        prev = s.cumulative_flops


def test_last_layer_remaining_zero(simple_stats):
    assert simple_stats[-1].remaining_flops == 0


def test_cumulative_plus_remaining_equals_total(simple_stats):
    total = sum(s.flops for s in simple_stats)
    for s in simple_stats:
        assert s.cumulative_flops + s.remaining_flops == total


def test_cumulative_cpu_increases(simple_stats):
    prev = -1.0
    for s in simple_stats:
        assert s.cumulative_cpu_us >= prev
        prev = s.cumulative_cpu_us


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_recommendations_keys(simple_model, simple_input):
    p = LayerProfiler(simple_model, simple_input, n_warmup=1, n_reps=2, skip_gpu=True)
    stats = p.profile()
    recs = p.recommendations(stats)
    assert {"min_client_compute", "min_transfer", "best_balance", "min_client_memory"} == set(recs.keys())


def test_recommendations_valid_indices(simple_model, simple_input):
    p = LayerProfiler(simple_model, simple_input, n_warmup=1, n_reps=2, skip_gpu=True)
    stats = p.profile()
    recs = p.recommendations(stats)
    valid = {s.idx for s in stats}
    for key, idx in recs.items():
        assert idx in valid, f"Recommendation '{key}' returned invalid index {idx}"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_cpu_time_positive(simple_stats):
    for s in simple_stats:
        assert s.cpu_time_us >= 0.0


def test_gpu_time_zero_when_skipped(simple_model, simple_input):
    p = LayerProfiler(simple_model, simple_input, n_warmup=1, n_reps=2, skip_gpu=True)
    stats = p.profile()
    assert all(s.gpu_time_us == 0.0 for s in stats)
    assert not p.gpu_was_collected


# ---------------------------------------------------------------------------
# dtype propagation
# ---------------------------------------------------------------------------


def test_float16_activation_bytes(simple_model):
    sample = torch.randn(32, 16, dtype=torch.float16)
    model = simple_model.half()
    p = LayerProfiler(model, sample, n_warmup=1, n_reps=2, skip_gpu=True)
    stats = p.profile()
    # float16 = 2 bytes per element; Linear(16,8) output = 32*8*2 = 512
    assert stats[0].activation_bytes == 32 * 8 * 2


def test_float16_flops_same_as_float32():
    """FLOPs are dtype-independent (structural count)."""
    s32 = torch.randn(32, 16, dtype=torch.float32)
    s16 = torch.randn(32, 16, dtype=torch.float16)
    m32 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4)).float()
    m16 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4)).half()
    p32 = LayerProfiler(m32, s32, n_warmup=1, n_reps=2, skip_gpu=True)
    p16 = LayerProfiler(m16, s16, n_warmup=1, n_reps=2, skip_gpu=True)
    stats32 = p32.profile()
    stats16 = p16.profile()
    for a, b in zip(stats32, stats16):
        assert a.flops == b.flops


# ---------------------------------------------------------------------------
# GPU timing (skipped if no CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_timing_collected_when_device_cpu(simple_model, simple_input):
    """GPU timing runs automatically even with device=cpu."""
    p = LayerProfiler(simple_model, simple_input, device="cpu", n_warmup=2, n_reps=5)
    stats = p.profile()
    assert p.gpu_was_collected
    assert any(s.gpu_time_us > 0 for s in stats)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cumulative_gpu_increases(simple_model, simple_input):
    p = LayerProfiler(simple_model, simple_input, device="cpu", n_warmup=2, n_reps=5)
    stats = p.profile()
    prev = -1.0
    for s in stats:
        assert s.cumulative_gpu_us >= prev
        prev = s.cumulative_gpu_us


# ---------------------------------------------------------------------------
# TrainingProfiler
# ---------------------------------------------------------------------------


def test_training_layer_count(simple_model, simple_input, simple_stats):
    trainer = TrainingProfiler(
        simple_model, simple_input, simple_stats, n_reps=3, skip_gpu=True
    )
    train_stats = trainer.profile()
    assert len(train_stats) == len(simple_stats)


def test_training_backward_time_nonnegative(simple_model, simple_input, simple_stats):
    trainer = TrainingProfiler(
        simple_model, simple_input, simple_stats, n_reps=3, skip_gpu=True
    )
    train_stats = trainer.profile()
    for s in train_stats:
        assert s.bwd_cpu_time_us >= 0.0


def test_training_param_kb_linear(simple_model, simple_input, simple_stats):
    trainer = TrainingProfiler(
        simple_model, simple_input, simple_stats, n_reps=2, skip_gpu=True
    )
    train_stats = trainer.profile()
    linears = [s for s in train_stats if simple_stats[s.idx].layer_type == "Linear"]
    for s in linears:
        assert s.param_kb > 0.0


def test_training_relu_param_zero(simple_model, simple_input, simple_stats):
    trainer = TrainingProfiler(
        simple_model, simple_input, simple_stats, n_reps=2, skip_gpu=True
    )
    train_stats = trainer.profile()
    relus = [s for s in train_stats if simple_stats[s.idx].layer_type == "ReLU"]
    for s in relus:
        assert s.param_kb == 0.0
        assert s.grad_kb == 0.0


def test_training_adam_ge_sgd(simple_model, simple_input, simple_stats):
    """Adam memory >= SGD memory (Adam has 2x optimizer state)."""
    trainer = TrainingProfiler(
        simple_model, simple_input, simple_stats, n_reps=2, skip_gpu=True
    )
    train_stats = trainer.profile()
    for s in train_stats:
        assert s.total_train_mem_adam_kb >= s.total_train_mem_sgd_kb


def test_training_cumulative_bwd_increases(simple_model, simple_input, simple_stats):
    trainer = TrainingProfiler(
        simple_model, simple_input, simple_stats, n_reps=3, skip_gpu=True
    )
    train_stats = trainer.profile()
    prev = -1.0
    for s in train_stats:
        assert s.cumulative_bwd_cpu_us >= prev
        prev = s.cumulative_bwd_cpu_us


def test_training_names_match_inference(simple_model, simple_input, simple_stats):
    trainer = TrainingProfiler(
        simple_model, simple_input, simple_stats, n_reps=2, skip_gpu=True
    )
    train_stats = trainer.profile()
    for inf_s, tr_s in zip(simple_stats, train_stats):
        assert inf_s.name == tr_s.name
        assert inf_s.idx == tr_s.idx
