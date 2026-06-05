from types import SimpleNamespace

import numpy as np
import torch
from torch_split_vfl.client import (
    _SL_BOTTOM_KEY,
    _SL_INPUT_KEY,
    VerticalSplitClient,
)
from torch_split_vfl.task import (
    BottomMLP,
    VerticalPartition,
    get_weights,
)


def _make_ctx() -> SimpleNamespace:
    return SimpleNamespace(state={}, node_id="node-x", run_config={}, node_config={})


def _make_client(partition_id: int = 0) -> tuple[VerticalSplitClient, SimpleNamespace]:
    ctx = _make_ctx()
    # Seed init so the ReLU units start live: an unseeded model can land with
    # every hidden unit negative on this fixed input, zeroing the gradient and
    # making the "weights moved" check spuriously fail.
    torch.manual_seed(0)
    bottom = BottomMLP(in_features=4, hidden_dim=3)
    features = torch.arange(20, dtype=torch.float32).view(5, 4)
    partition = VerticalPartition(features=features, batch_size=2)
    client = VerticalSplitClient(
        partition_id=partition_id,
        bottom=bottom,
        train_partition=partition,
        learning_rate=0.1,
        ctx=ctx,
        demo=False,
    )
    return client, ctx


def test_forward_returns_activation_and_metrics_and_persists_state():
    client, ctx = _make_client(partition_id=3)
    out, n, metrics = client._forward({"sl_step": 0})

    assert isinstance(out, list) and len(out) == 1
    act = out[0]
    # batch_size=2, hidden_dim=3 → (2, 3)
    assert act.shape == (2, 3)
    assert n == 2
    assert metrics["partition_id"] == 3
    assert _SL_INPUT_KEY in ctx.state
    assert _SL_BOTTOM_KEY in ctx.state


def test_backward_applies_gradient_and_updates_weights():
    client, ctx = _make_client()
    # Run a forward to populate state.
    client._forward({"sl_step": 0})
    weights_before = get_weights(client.bottom)

    # Synthetic gradient at the cut matching activation shape (2, 3).
    grad = np.ones((2, 3), dtype=np.float32)
    updated, n, metrics = client._backward([grad])

    assert n == 2
    assert metrics["partition_id"] == 0
    assert isinstance(updated, list) and len(updated) == len(weights_before)
    # At least one parameter should have moved given a non-zero gradient.
    moved = any(not np.allclose(a, b) for a, b in zip(weights_before, updated))
    assert moved
    # Saved bottom weights match the returned ones (cached for next round).
    cached = ctx.state[_SL_BOTTOM_KEY].to_numpy_ndarrays()
    for u, c in zip(updated, cached):
        np.testing.assert_array_equal(u, c)
    # Input arrays were cleared.
    assert _SL_INPUT_KEY not in ctx.state


def test_backward_without_state_returns_current_weights():
    client, _ctx = _make_client()
    out, n, metrics = client._backward([np.zeros((2, 3), dtype=np.float32)])
    assert n == 0
    assert metrics == {"partition_id": 0}
    assert len(out) == len(get_weights(client.bottom))
