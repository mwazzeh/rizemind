from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from flwr.common import Code, Parameters, Status, ndarrays_to_parameters
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.serialization import (
    _SL_TENSOR_TYPE,
    parameters_to_tensor,
    tensor_to_parameters,
)
from rizemind.split_learning.vertical_strategy import (
    _BACKWARD_PHASE,
    _FORWARD_PHASE,
    VerticalSplitLearningStrategy,
)

_CONFIG = SplitLearningConfig(cut_layer=0)


def _client(cid: str) -> ClientProxy:
    cp = Mock(spec=ClientProxy)
    cp.cid = cid
    return cp


def _activation_res(
    partition_id: int | None,
    payload: bytes | None = b"\x01",
    *,
    activation: np.ndarray | None = None,
) -> FitRes:
    """Build a forward-round FitRes.

    Args:
        partition_id: When ``None`` the ``partition_id`` metric is omitted
            entirely (to exercise the missing-metric error path).
        payload: Single bytes payload if ``activation`` is not given.
        activation: When supplied, packed via ``ndarrays_to_parameters`` so the
            test can read it back as a tensor.
    """
    if activation is not None:
        params = ndarrays_to_parameters([activation])
        params = Parameters(tensors=params.tensors, tensor_type=_SL_TENSOR_TYPE)
    else:
        assert payload is not None
        params = Parameters(tensors=[payload], tensor_type=_SL_TENSOR_TYPE)
    metrics: dict = {}
    if partition_id is not None:
        metrics["partition_id"] = partition_id
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=1,
        metrics=metrics,
    )


def _weight_res(partition_id: int | None, weights: list[np.ndarray]) -> FitRes:
    metrics: dict = {}
    if partition_id is not None:
        metrics["partition_id"] = partition_id
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(weights),
        num_examples=1,
        metrics=metrics,
    )


def _on_train_step_stub(grad_store, loss):
    def fn(step_idx, ordered):
        fn.last_call = (step_idx, ordered)
        return grad_store, loss

    fn.last_call = None
    return fn


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_initial_phase_is_forward():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    assert s._phase == _FORWARD_PHASE
    assert s._step_idx == 0


def test_initialize_parameters_returns_empty():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    p = s.initialize_parameters(MagicMock())
    assert p is not None
    assert p.tensors == []


def test_num_clients_validation():
    with pytest.raises(ValueError):
        VerticalSplitLearningStrategy(
            config=_CONFIG, num_clients=0, on_train_step=_on_train_step_stub({}, 0.0)
        )


# ---------------------------------------------------------------------------
# configure_fit — forward
# ---------------------------------------------------------------------------


def test_configure_fit_forward_sends_sl_step_and_empty_params():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    cm = MagicMock()
    c0, c1 = _client("a"), _client("b")
    cm.sample.return_value = [c0, c1]

    out = s.configure_fit(
        server_round=1, parameters=Parameters([], ""), client_manager=cm
    )

    assert len(out) == 2
    for client, fit_ins in out:
        assert client in (c0, c1)
        assert fit_ins.parameters.tensors == []
        assert fit_ins.config["sl_step"] == 0


def test_configure_fit_forward_uses_on_forward_config_fn():
    extra = {"foo": 42.0}
    s = VerticalSplitLearningStrategy(
        config=_CONFIG,
        num_clients=1,
        on_train_step=_on_train_step_stub({}, 0.0),
        on_forward_config_fn=lambda sr, step: extra,
    )
    cm = MagicMock()
    cm.sample.return_value = [_client("a")]

    out = s.configure_fit(1, Parameters([], ""), cm)
    assert out[0][1].config["foo"] == 42.0


def test_configure_fit_forward_raises_when_sampler_returns_fewer_than_num_clients():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=4, on_train_step=_on_train_step_stub({}, 0.0)
    )
    cm = MagicMock()
    cm.sample.return_value = [_client("a"), _client("b")]
    with pytest.raises(RuntimeError, match="expected 4 client"):
        s.configure_fit(1, Parameters([], ""), cm)


# ---------------------------------------------------------------------------
# configure_fit — backward
# ---------------------------------------------------------------------------


def test_configure_fit_backward_dispatches_per_client_gradients():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    s._phase = _BACKWARD_PHASE
    s._gradient_store = {
        "a": Parameters(tensors=[b"\xaa"], tensor_type=_SL_TENSOR_TYPE),
        "b": Parameters(tensors=[b"\xbb"], tensor_type=_SL_TENSOR_TYPE),
    }
    cm = MagicMock()
    c0, c1 = _client("a"), _client("b")
    cm.all.return_value = {"a": c0, "b": c1}

    out = s.configure_fit(2, Parameters([], ""), cm)

    sent = {cli.cid: ins.parameters.tensors[0] for cli, ins in out}
    assert sent == {"a": b"\xaa", "b": b"\xbb"}


def test_configure_fit_backward_each_client_receives_only_its_own_gradient():
    """Cross-check: each cid receives exactly the gradient stored under that cid."""
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=4, on_train_step=_on_train_step_stub({}, 0.0)
    )
    s._phase = _BACKWARD_PHASE
    s._gradient_store = {
        cid: Parameters(tensors=[bytes([i])], tensor_type=_SL_TENSOR_TYPE)
        for i, cid in enumerate(["c0", "c1", "c2", "c3"])
    }
    cm = MagicMock()
    proxies = {cid: _client(cid) for cid in s._gradient_store}
    cm.all.return_value = proxies

    out = s.configure_fit(2, Parameters([], ""), cm)

    sent = {cli.cid: ins.parameters.tensors[0] for cli, ins in out}
    assert sent == {
        "c0": bytes([0]),
        "c1": bytes([1]),
        "c2": bytes([2]),
        "c3": bytes([3]),
    }


def test_configure_fit_backward_warns_on_unmatched_cid(caplog):
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    s._phase = _BACKWARD_PHASE
    s._gradient_store = {
        "a": Parameters(tensors=[b"\xaa"], tensor_type=_SL_TENSOR_TYPE),
        "ghost": Parameters(tensors=[b"\xff"], tensor_type=_SL_TENSOR_TYPE),
    }
    cm = MagicMock()
    cm.all.return_value = {"a": _client("a")}

    with caplog.at_level("WARNING"):
        out = s.configure_fit(2, Parameters([], ""), cm)

    assert len(out) == 1
    assert any("unknown cid" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# aggregate_fit — forward (activations)
# ---------------------------------------------------------------------------


def test_aggregate_fit_forward_orders_by_partition_id_then_invokes_train_step():
    on_train = _on_train_step_stub(
        grad_store={
            "alpha": Parameters([b"g0"], _SL_TENSOR_TYPE),
            "beta": Parameters([b"g1"], _SL_TENSOR_TYPE),
        },
        loss=0.5,
    )
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=on_train
    )

    # Submit out-of-order: client "beta" has partition_id=1, "alpha" has 0.
    results = [
        (_client("beta"), _activation_res(partition_id=1, payload=b"act1")),
        (_client("alpha"), _activation_res(partition_id=0, payload=b"act0")),
    ]
    params, metrics = s.aggregate_fit(1, results, [])

    assert s._phase == _BACKWARD_PHASE
    assert metrics["train_loss"] == 0.5
    assert metrics["sl_step"] == 0.0
    assert metrics["num_partitions"] == 2.0
    assert params is not None and params.tensors == []

    # Ordered by partition_id: alpha (0), then beta (1).
    step_idx, ordered = on_train.last_call
    assert step_idx == 0
    pids = [pid for pid, _, _ in ordered]
    assert pids == [0, 1]
    cids = [cid for _, cid, _ in ordered]
    assert cids == ["alpha", "beta"]


def test_aggregate_fit_forward_orders_4_clients_arriving_out_of_order():
    """K=4: arbitrary arrival order must be sorted by partition_id."""
    on_train = _on_train_step_stub(
        grad_store={
            cid: Parameters([b"g"], _SL_TENSOR_TYPE) for cid in ["d", "a", "c", "b"]
        },
        loss=0.1,
    )
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=4, on_train_step=on_train
    )

    # cid → partition_id mapping: shuffled arrival order.
    arrivals = [("c", 2), ("a", 0), ("d", 3), ("b", 1)]
    results = [
        (_client(cid), _activation_res(partition_id=pid)) for cid, pid in arrivals
    ]
    s.aggregate_fit(1, results, [])

    step_idx, ordered = on_train.last_call
    assert step_idx == 0
    assert [pid for pid, _, _ in ordered] == [0, 1, 2, 3]
    assert [cid for _, cid, _ in ordered] == ["a", "b", "c", "d"]


def test_aggregate_fit_forward_raises_on_duplicate_partition_id():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    results = [
        (_client("a"), _activation_res(partition_id=0)),
        (_client("b"), _activation_res(partition_id=0)),  # duplicate
    ]
    with pytest.raises(ValueError, match="duplicate partition_id=0"):
        s.aggregate_fit(1, results, [])


def test_aggregate_fit_forward_raises_on_missing_partition_id_metric():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    results = [
        (_client("a"), _activation_res(partition_id=0)),
        (_client("b"), _activation_res(partition_id=None)),  # missing
    ]
    with pytest.raises(ValueError, match="without 'partition_id'"):
        s.aggregate_fit(1, results, [])


def test_aggregate_fit_forward_raises_on_out_of_range_partition_id():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    results = [
        (_client("a"), _activation_res(partition_id=0)),
        (_client("b"), _activation_res(partition_id=5)),  # out of range
    ]
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        s.aggregate_fit(1, results, [])


def test_aggregate_fit_forward_raises_on_wrong_client_count():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=4, on_train_step=_on_train_step_stub({}, 0.0)
    )
    results = [
        (_client("a"), _activation_res(partition_id=0)),
        (_client("b"), _activation_res(partition_id=1)),
    ]
    with pytest.raises(ValueError, match="expected 4 forward result"):
        s.aggregate_fit(1, results, [])


def test_aggregate_fit_forward_raises_when_grad_store_missing_cid():
    """on_train_step must return one gradient per input cid."""
    on_train = _on_train_step_stub(
        grad_store={"a": Parameters([b"g"], _SL_TENSOR_TYPE)},  # missing "b"
        loss=0.5,
    )
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=on_train
    )
    results = [
        (_client("a"), _activation_res(partition_id=0)),
        (_client("b"), _activation_res(partition_id=1)),
    ]
    with pytest.raises(ValueError, match="missing=\\['b'\\]"):
        s.aggregate_fit(1, results, [])


def test_aggregate_fit_forward_raises_when_grad_store_has_extra_cid():
    on_train = _on_train_step_stub(
        grad_store={
            "a": Parameters([b"g"], _SL_TENSOR_TYPE),
            "b": Parameters([b"g"], _SL_TENSOR_TYPE),
            "ghost": Parameters([b"g"], _SL_TENSOR_TYPE),
        },
        loss=0.5,
    )
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=on_train
    )
    results = [
        (_client("a"), _activation_res(partition_id=0)),
        (_client("b"), _activation_res(partition_id=1)),
    ]
    with pytest.raises(ValueError, match="unexpected=\\['ghost'\\]"):
        s.aggregate_fit(1, results, [])


def test_aggregate_fit_with_empty_results_does_not_advance(caplog):
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    with caplog.at_level("WARNING"):
        params, _metrics = s.aggregate_fit(1, [], [])
    assert s._phase == _FORWARD_PHASE
    assert s._step_idx == 0
    assert params is not None and params.tensors == []
    assert any("no results received" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# aggregate_fit — backward (weight updates)
# ---------------------------------------------------------------------------


def test_aggregate_fit_backward_caches_bottom_weights_advances_step():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    s._phase = _BACKWARD_PHASE
    s._gradient_store = {"a": Parameters([b"x"], _SL_TENSOR_TYPE)}
    s._last_train_loss = 0.42

    w0 = [np.array([1.0, 2.0], dtype=np.float32)]
    w1 = [np.array([3.0, 4.0], dtype=np.float32)]
    results = [
        (_client("a"), _weight_res(partition_id=0, weights=w0)),
        (_client("b"), _weight_res(partition_id=1, weights=w1)),
    ]
    params, metrics = s.aggregate_fit(2, results, [])

    assert s._phase == _FORWARD_PHASE
    assert s._step_idx == 1
    assert s._gradient_store == {}
    assert params is not None and params.tensors == []
    assert metrics["train_loss"] == 0.42
    np.testing.assert_array_equal(s._bottom_weights_by_pid[0][0], w0[0])
    np.testing.assert_array_equal(s._bottom_weights_by_pid[1][0], w1[0])


def test_aggregate_fit_backward_does_not_average_per_client_bottoms():
    """Cached weights must remain distinct per partition — no FedAvg."""
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    s._phase = _BACKWARD_PHASE
    s._gradient_store = {"a": Parameters([b"x"], _SL_TENSOR_TYPE)}

    w0 = [np.array([10.0, 10.0], dtype=np.float32)]
    w1 = [np.array([-10.0, -10.0], dtype=np.float32)]
    results = [
        (_client("a"), _weight_res(partition_id=0, weights=w0)),
        (_client("b"), _weight_res(partition_id=1, weights=w1)),
    ]
    s.aggregate_fit(2, results, [])

    # Each partition keeps its own weights (no averaging towards zero).
    np.testing.assert_array_equal(s._bottom_weights_by_pid[0][0], w0[0])
    np.testing.assert_array_equal(s._bottom_weights_by_pid[1][0], w1[0])
    assert not np.allclose(
        s._bottom_weights_by_pid[0][0], s._bottom_weights_by_pid[1][0]
    )


def test_aggregate_fit_backward_raises_on_duplicate_partition_id():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    s._phase = _BACKWARD_PHASE
    s._gradient_store = {"a": Parameters([b"x"], _SL_TENSOR_TYPE)}

    w = [np.zeros(1, dtype=np.float32)]
    results = [
        (_client("a"), _weight_res(partition_id=0, weights=w)),
        (_client("b"), _weight_res(partition_id=0, weights=w)),
    ]
    with pytest.raises(ValueError, match="duplicate partition_id=0"):
        s.aggregate_fit(2, results, [])


def test_aggregate_fit_backward_warns_on_missing_partition_id_metric(caplog):
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=2, on_train_step=_on_train_step_stub({}, 0.0)
    )
    s._phase = _BACKWARD_PHASE
    s._gradient_store = {"a": Parameters([b"x"], _SL_TENSOR_TYPE)}

    w = [np.zeros(1, dtype=np.float32)]
    results = [
        (_client("a"), _weight_res(partition_id=0, weights=w)),
        (_client("b"), _weight_res(partition_id=None, weights=w)),
    ]
    with caplog.at_level("WARNING"):
        s.aggregate_fit(2, results, [])
    assert any("without partition_id" in rec.message for rec in caplog.records)
    assert 0 in s._bottom_weights_by_pid
    # The result with no partition_id metric was skipped.
    assert 1 not in s._bottom_weights_by_pid


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_none_until_all_clients_reported():
    on_eval = MagicMock(return_value=(0.1, {"acc": 0.9}))
    s = VerticalSplitLearningStrategy(
        config=_CONFIG,
        num_clients=2,
        on_train_step=_on_train_step_stub({}, 0.0),
        on_evaluate=on_eval,
    )
    # Nothing cached yet → skip.
    assert s.evaluate(1, Parameters([], "")) is None
    # Only one client reported → still skip.
    s._bottom_weights_by_pid[0] = [np.zeros(1, dtype=np.float32)]
    assert s.evaluate(2, Parameters([], "")) is None
    # Both reported → invoke.
    s._bottom_weights_by_pid[1] = [np.zeros(1, dtype=np.float32)]
    out = s.evaluate(3, Parameters([], ""))
    assert out == (0.1, {"acc": 0.9})
    on_eval.assert_called_once()


def test_evaluate_disabled_when_callback_missing():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=1, on_train_step=_on_train_step_stub({}, 0.0)
    )
    assert s.evaluate(1, Parameters([], "")) is None


# ---------------------------------------------------------------------------
# distributed eval — no-ops (centralized-only)
# ---------------------------------------------------------------------------


def test_configure_evaluate_is_noop():
    """No distributed eval phase — eval is centralized via evaluate()."""
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=1, on_train_step=_on_train_step_stub({}, 0.0)
    )
    assert s.configure_evaluate(1, Parameters([], ""), MagicMock()) == []


def test_aggregate_evaluate_is_noop():
    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=1, on_train_step=_on_train_step_stub({}, 0.0)
    )
    out = s.aggregate_evaluate(1, [], [])
    assert out == (None, {})


# ---------------------------------------------------------------------------
# End-to-end: real torch tail + per-client gradient slicing
# ---------------------------------------------------------------------------


def test_gradient_slicing_preserves_per_client_activation_shape():
    """Integration: server tail backprop produces a per-client gradient whose
    shape matches that client's activation. This exercises the path the example
    relies on for cross-checking strategy + on_train_step.
    """
    torch.manual_seed(0)
    num_clients = 3
    hidden = 4
    batch = 5

    # Build a tiny tail that takes (B, K*H) -> (B, num_classes).
    tail = nn.Linear(num_clients * hidden, 10)
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 10, (batch,))

    # Each client has a randomly shaped activation (B, hidden).
    acts_np = [
        np.random.randn(batch, hidden).astype(np.float32) for _ in range(num_clients)
    ]
    cids = [f"c{i}" for i in range(num_clients)]

    def on_train_step(sl_step, ordered):
        del sl_step
        tensors = [parameters_to_tensor(p, requires_grad=True) for _, _, p in ordered]
        joint = torch.cat(tensors, dim=1)
        logits = tail(joint)
        loss = criterion(logits, labels)
        loss.backward()
        out = {}
        for (_, cid, _), t in zip(ordered, tensors):
            assert t.grad is not None
            out[cid] = tensor_to_parameters(t.grad)
        return out, loss.item()

    s = VerticalSplitLearningStrategy(
        config=_CONFIG, num_clients=num_clients, on_train_step=on_train_step
    )
    results = [
        (_client(cid), _activation_res(partition_id=i, activation=acts_np[i]))
        for i, cid in enumerate(cids)
    ]
    s.aggregate_fit(1, results, [])

    # Each gradient parameter must round-trip to a tensor of the original
    # activation shape — proving the slicing was per-client.
    assert set(s._gradient_store) == set(cids)
    for cid, grad_params in s._gradient_store.items():
        grad = parameters_to_tensor(grad_params)
        idx = cids.index(cid)
        assert grad.shape == acts_np[idx].shape
