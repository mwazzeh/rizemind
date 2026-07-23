"""Unit tests for the clipped-Gaussian cut-gradient privacy mechanism.

These verify the exact per-sample clipping math, zero/below/above-bound cases,
invalid configs, zero-noise equivalence, Gaussian shape/dtype, research-seeded
reproducibility, different-seed divergence, secure-mode seedlessness, and the
aggregate (non-sensitive) diagnostics.
"""

import numpy as np
import pytest
from rizemind.split_learning.gradient_privacy import (
    GradientPrivacyConfig,
    clip_rows,
    make_rng,
    privatize_joint_gradient,
    row_l2_norms,
)


# --------------------------- config validation ----------------------------
def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        GradientPrivacyConfig(mode="bogus")


def test_invalid_clip_norm_rejected():
    with pytest.raises(ValueError):
        GradientPrivacyConfig(mode="clip", clip_norm=0.0)
    with pytest.raises(ValueError):
        GradientPrivacyConfig(mode="gaussian", clip_norm=-1.0)


def test_invalid_rng_mode_rejected():
    with pytest.raises(ValueError):
        GradientPrivacyConfig(mode="none", rng_mode="nope")


def test_negative_noise_multiplier_rejected():
    with pytest.raises(ValueError):
        GradientPrivacyConfig(mode="gaussian", clip_norm=1.0, noise_multiplier=-0.1)


def test_invalid_delta_rejected():
    with pytest.raises(ValueError):
        GradientPrivacyConfig(mode="clip", clip_norm=1.0, delta=1.5)


# --------------------------- exact clipping -------------------------------
def test_clip_exact_on_known_vectors():
    # row norms: 5, 13, 1
    g = np.array([[3.0, 4.0], [5.0, 12.0], [0.6, 0.8]], dtype=np.float32)
    clipped, pre = clip_rows(g, clip_norm=1.0)
    assert np.allclose(pre, [5.0, 13.0, 1.0])
    out_norms = row_l2_norms(clipped)
    # rows above bound clipped to exactly 1.0; the unit row stays at 1.0.
    assert np.allclose(out_norms, [1.0, 1.0, 1.0], atol=1e-6)
    # direction preserved for the first row: (3,4)->(0.6,0.8)
    assert np.allclose(clipped[0], [0.6, 0.8], atol=1e-6)


def test_clip_zero_vector_unchanged():
    g = np.zeros((2, 4), dtype=np.float32)
    clipped, pre = clip_rows(g, clip_norm=1.0)
    assert np.allclose(pre, 0.0)
    assert np.allclose(clipped, 0.0)  # no division by zero


def test_clip_below_bound_unchanged():
    g = np.array([[0.1, 0.0], [0.0, 0.2]], dtype=np.float32)
    clipped, _ = clip_rows(g, clip_norm=1.0)
    assert np.allclose(clipped, g)  # already within bound


def test_clip_above_bound_scaled_down():
    g = np.array([[10.0, 0.0]], dtype=np.float32)
    clipped, _ = clip_rows(g, clip_norm=2.0)
    assert np.allclose(clipped, [[2.0, 0.0]])


def test_clip_invalid_bound_raises():
    with pytest.raises(ValueError):
        clip_rows(np.ones((2, 2)), clip_norm=0.0)


# --------------------------- mechanism modes ------------------------------
def test_none_mode_is_passthrough():
    g = np.random.RandomState(0).randn(5, 8).astype(np.float32)
    cfg = GradientPrivacyConfig(mode="none")
    out, diag = privatize_joint_gradient(g, cfg)
    assert np.allclose(out, g)
    assert diag["clip_fraction"] == 0.0
    assert diag["noise_std"] == 0.0


def test_clip_mode_no_noise_matches_clip_rows():
    g = np.random.RandomState(1).randn(6, 10).astype(np.float32) * 3
    cfg = GradientPrivacyConfig(mode="clip", clip_norm=1.0)
    out, diag = privatize_joint_gradient(g, cfg)
    expected, _ = clip_rows(g, 1.0)
    assert np.allclose(out, expected)
    assert diag["noise_std"] == 0.0
    assert 0.0 <= diag["clip_fraction"] <= 1.0


def test_zero_noise_gaussian_equals_clip():
    g = np.random.RandomState(2).randn(6, 10).astype(np.float32) * 3
    clip_out, _ = privatize_joint_gradient(
        g, GradientPrivacyConfig(mode="clip", clip_norm=1.0)
    )
    gauss0_out, _ = privatize_joint_gradient(
        g,
        GradientPrivacyConfig(mode="gaussian", clip_norm=1.0, noise_multiplier=0.0),
        research_seed=0,
    )
    assert np.allclose(clip_out, gauss0_out)


def test_gaussian_noise_shape_and_dtype():
    g = np.random.RandomState(3).randn(7, 12).astype(np.float32)
    cfg = GradientPrivacyConfig(mode="gaussian", clip_norm=1.0, noise_multiplier=0.5)
    out, diag = privatize_joint_gradient(g, cfg, research_seed=1)
    assert out.shape == g.shape
    assert out.dtype == np.float32
    assert diag["noise_std"] == pytest.approx(0.5)
    assert diag["snr"] is not None


def test_research_seed_is_reproducible():
    g = np.random.RandomState(4).randn(8, 16).astype(np.float32)
    cfg = GradientPrivacyConfig(mode="gaussian", clip_norm=1.0, noise_multiplier=1.0)
    a, _ = privatize_joint_gradient(g, cfg, research_seed=42)
    b, _ = privatize_joint_gradient(g, cfg, research_seed=42)
    assert np.allclose(a, b)


def test_different_seeds_differ():
    g = np.random.RandomState(5).randn(8, 16).astype(np.float32)
    cfg = GradientPrivacyConfig(mode="gaussian", clip_norm=1.0, noise_multiplier=1.0)
    a, _ = privatize_joint_gradient(g, cfg, research_seed=1)
    b, _ = privatize_joint_gradient(g, cfg, research_seed=2)
    assert not np.allclose(a, b)


def test_secure_rng_has_no_reusable_seed():
    # secure mode draws fresh OS entropy each call -> two calls differ, and the
    # config exposes no seed to persist.
    cfg = GradientPrivacyConfig(
        mode="gaussian", clip_norm=1.0, noise_multiplier=1.0, rng_mode="secure"
    )
    g = np.ones((4, 8), dtype=np.float32) * 5
    a, _ = privatize_joint_gradient(g, cfg)
    b, _ = privatize_joint_gradient(g, cfg)
    assert not np.allclose(a, b)
    assert not hasattr(cfg, "seed")
    rng = make_rng(cfg, research_seed=None)  # must not require a seed
    assert isinstance(rng, np.random.Generator)


def test_research_rng_requires_seed():
    cfg = GradientPrivacyConfig(mode="gaussian", clip_norm=1.0, noise_multiplier=1.0)
    with pytest.raises(ValueError):
        make_rng(cfg, research_seed=None)


# --------------------------- diagnostics ----------------------------------
def test_diagnostics_are_aggregate_only():
    g = np.random.RandomState(6).randn(20, 32).astype(np.float32) * 2
    cfg = GradientPrivacyConfig(mode="gaussian", clip_norm=1.0, noise_multiplier=0.3)
    _out, diag = privatize_joint_gradient(g, cfg, research_seed=0)
    # Every diagnostic value is a scalar / None / str — never an array.
    for v in diag.values():
        assert v is None or np.isscalar(v) or isinstance(v, str)
    assert {
        "pre_clip_norm_mean",
        "post_clip_norm_mean",
        "clip_fraction",
        "noise_std",
        "snr",
    } <= set(diag)


def test_config_as_dict_no_formal_dp_claim():
    cfg = GradientPrivacyConfig(mode="gaussian", clip_norm=0.5, noise_multiplier=1.0)
    d = cfg.as_dict()
    assert d["formal_dp_claim"] is False
    assert d["epsilon"] is None
    assert d["gradient_noise_std"] == pytest.approx(0.5)


def test_2d_required():
    with pytest.raises(ValueError):
        privatize_joint_gradient(np.ones((4,)), GradientPrivacyConfig(mode="clip"))
