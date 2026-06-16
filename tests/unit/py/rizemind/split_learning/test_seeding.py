"""Tests for rizemind.split_learning.seeding."""

import random

import numpy as np

from rizemind.split_learning.seeding import seed_everything


def test_returns_seed():
    assert seed_everything(123) == 123


def test_same_seed_reproduces_python_and_numpy():
    seed_everything(42)
    a_py = [random.random() for _ in range(5)]
    a_np = np.random.rand(5)
    seed_everything(42)
    b_py = [random.random() for _ in range(5)]
    b_np = np.random.rand(5)
    assert a_py == b_py
    assert np.allclose(a_np, b_np)


def test_different_seed_changes_draws():
    seed_everything(1)
    a = np.random.rand(10)
    seed_everything(2)
    b = np.random.rand(10)
    assert not np.allclose(a, b)


def test_torch_init_reproducible_when_available():
    torch = __import__("torch")
    seed_everything(7)
    w1 = torch.nn.Linear(8, 4).weight.detach().clone()
    seed_everything(7)
    w2 = torch.nn.Linear(8, 4).weight.detach().clone()
    assert torch.equal(w1, w2)
    seed_everything(8)
    w3 = torch.nn.Linear(8, 4).weight.detach().clone()
    assert not torch.equal(w1, w3)


def test_deterministic_flag_does_not_raise():
    # Should be a no-op-safe call even on CPU.
    assert seed_everything(5, deterministic=True) == 5
