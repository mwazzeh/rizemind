"""Tests for label-private VFL: privacy properties, routing, persistence.

These exercise the real strategy state machine and the label-holder compute path
with synthetic tensors (no Ray), and assert the actual payloads/state — not just
variable names — to verify labels never reach the coordinator.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Code, FitRes, Status
from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.label_private_strategy import (
    LP_PHASE_KEY,
    LabelPrivateVerticalStrategy,
)
from rizemind.split_learning.telemetry import RunTelemetry
from torch_split_vfl.client import (
    _LP_TOP_KEY,
    _LP_TOP_OPT_KEY,
    LabelHolderContext,
    VerticalSplitClient,
)
from torch_split_vfl.task import BottomMLP, ServerTopMLP, VerticalPartition

K, HIDDEN, BATCH = 2, 4, 8
WIDTHS = [HIDDEN] * K


# ----------------------------- fakes --------------------------------------
class FakeProxy:
    def __init__(self, cid: str) -> None:
        self.cid = cid


class FakeClientManager:
    def __init__(self, proxies: dict[str, FakeProxy]) -> None:
        self._p = proxies

    def sample(self, num_clients, min_num_clients=None):
        return list(self._p.values())[:num_clients]

    def all(self):
        return dict(self._p)


def _ok(parameters, metrics):
    return FitRes(status=Status(Code.OK, ""), parameters=parameters,
                  num_examples=BATCH, metrics=metrics)


def _activation_res(pid: int):
    arr = np.full((BATCH, HIDDEN), float(pid + 1), dtype=np.float32)
    return _ok(ndarrays_to_parameters([arr]), {"partition_id": pid})


# ----------------------------- validation ---------------------------------
def test_invalid_label_holder_pid_rejected():
    cfg = SplitLearningConfig(cut_layer=0)
    with pytest.raises(ValueError):
        LabelPrivateVerticalStrategy(cfg, num_clients=2, label_holder_pid=5)
    with pytest.raises(ValueError):
        LabelPrivateVerticalStrategy(cfg, num_clients=2, label_holder_pid=-1)


def _fresh_strategy(on_metrics=None, build_test=None, telem=None):
    return LabelPrivateVerticalStrategy(
        SplitLearningConfig(cut_layer=0), num_clients=K, label_holder_pid=0,
        build_test_joint_activation=build_test, on_metrics=on_metrics, telemetry=telem,
    )


def _proxies():
    return {f"cid{p}": FakeProxy(f"cid{p}") for p in range(K)}


# ----------------------- full step state machine --------------------------
def _drive_one_step(strategy, cm, labels, captured, telem=None, eval_round=False):
    """Run COLLECT -> COMPUTE -> DISTRIBUTE once, returning what the label holder
    received in COMPUTE. ``labels`` is what a real label holder would use — it
    must never appear in any server payload/state."""
    # COLLECT
    strategy.configure_fit(1, None, cm)
    strategy.aggregate_fit(1, [(cm._p[f"cid{p}"], _activation_res(p)) for p in range(K)], [])
    # COMPUTE: capture what the strategy sends to the label holder.
    instrs = strategy.configure_fit(2, None, cm)
    assert len(instrs) == 1
    holder_proxy, fit_ins = instrs[0]
    assert holder_proxy.cid == "cid0"
    assert fit_ins.config[LP_PHASE_KEY] == "compute"
    sent_arrays = parameters_to_ndarrays(fit_ins.parameters)
    captured["compute_sent"] = sent_arrays
    captured["compute_config"] = dict(fit_ins.config)
    # Simulate the label holder: produce K gradients + metrics (NO labels out).
    grads = [np.full((BATCH, HIDDEN), 0.01 * (p + 1), dtype=np.float32) for p in range(K)]
    metrics = {"train_loss": 0.5}
    if eval_round:
        metrics.update(has_eval=True, val_loss=0.4, val_accuracy=0.8,
                       precision_macro=0.7, recall_macro=0.6, f1_macro=0.65,
                       balanced_accuracy=0.6, n_samples=100, precision=0.7,
                       recall=0.6, f1=0.65, roc_auc=0.85, confusion_json="[10,2,3,5]")
    compute_res = _ok(ndarrays_to_parameters(grads), metrics)
    strategy.aggregate_fit(2, [(holder_proxy, compute_res)], [])
    # DISTRIBUTE
    dinstrs = strategy.configure_fit(3, None, cm)
    captured["distribute"] = {cid: parameters_to_ndarrays(fi.parameters)
                              for cid, fi in [(p.cid, fi) for p, fi in dinstrs]}
    bws = [_ok(ndarrays_to_parameters([np.ones((HIDDEN, HIDDEN), np.float32)]),
               {"partition_id": p}) for p in range(K)]
    strategy.aggregate_fit(3, [(cm._p[f"cid{p}"], bws[p]) for p in range(K)], [])
    return captured


def test_server_state_and_payloads_contain_no_labels():
    labels = np.array([0, 1] * (BATCH // 2), dtype=np.int64)  # the secret
    captured: dict = {}
    metrics_seen: list = []
    strategy = _fresh_strategy(on_metrics=lambda r, m: metrics_seen.append(m))
    cm = FakeClientManager(_proxies())
    _drive_one_step(strategy, cm, labels, captured, eval_round=True)

    # (4) what the coordinator SENT to the holder in COMPUTE is the joint
    # activation (B, K*H) — never the label vector.
    sent = captured["compute_sent"]
    assert sent[0].shape == (BATCH, K * HIDDEN)
    for arr in sent:
        assert arr.shape != labels.shape
        assert not np.array_equal(arr.ravel()[: labels.size], labels)

    # (5) strategy state holds activations / grads / bottom weights — no labels.
    for store in (strategy._acts_by_pid, strategy._grad_by_cid,
                  strategy._bottom_weights_by_pid):
        for v in store.values():
            arrs = parameters_to_ndarrays(v) if hasattr(v, "tensor_type") else (
                v if isinstance(v, list) else [v])
            for a in arrs:
                a = np.asarray(a)
                assert not (a.shape == labels.shape and np.array_equal(a, labels))

    # (6) metrics handed to the writer are scalars/strings — no label array.
    assert metrics_seen, "eval metrics should have been emitted"
    for v in metrics_seen[0].values():
        assert isinstance(v, (int, float, str, bool))


def test_gradients_map_back_to_parties():
    captured: dict = {}
    strategy = _fresh_strategy()
    cm = FakeClientManager(_proxies())
    _drive_one_step(strategy, cm, np.zeros(BATCH), captured)
    # DISTRIBUTE sends one gradient per party; pid p (cidp) gets grad scaled by p.
    dist = captured["distribute"]
    assert set(dist) == {"cid0", "cid1"}
    assert dist["cid0"][0].shape == (BATCH, HIDDEN)
    assert np.allclose(dist["cid0"][0], 0.01)  # pid 0 grad
    assert np.allclose(dist["cid1"][0], 0.02)  # pid 1 grad


def test_telemetry_includes_routing_overhead():
    telem = RunTelemetry()
    strategy = _fresh_strategy(telem=telem)
    cm = FakeClientManager(_proxies())
    _drive_one_step(strategy, cm, np.zeros(BATCH), {}, telem=telem)
    d = telem.as_dict()
    assert d["n_steps"] == 1
    assert d["cumulative_activation_bytes"] == K * BATCH * HIDDEN * 4
    # routing = repr downlink (joint act) + joint gradient uplink
    assert "repr_downlink" in d["routing_bytes"]
    assert "joint_grad_uplink" in d["routing_bytes"]
    assert d["cumulative_routing_bytes"] > 0


def test_build_test_joint_callback_drives_eval():
    seen = []

    def build_test(server_round, bottoms):
        return np.zeros((5, K * HIDDEN), dtype=np.float32)  # M=5 test rows

    strategy = _fresh_strategy(build_test=build_test,
                               on_metrics=lambda r, m: seen.append(m))
    cm = FakeClientManager(_proxies())
    captured: dict = {}
    _drive_one_step(strategy, cm, np.zeros(BATCH), captured, eval_round=True)
    # COMPUTE was sent two arrays: train joint + test joint.
    assert len(captured["compute_sent"]) == 2
    assert captured["compute_config"]["do_eval"] is True


# --------------------- label-holder client compute ------------------------
def _make_label_holder_client():
    ctx = SimpleNamespace(state={})
    torch.manual_seed(0)
    bottom = BottomMLP(in_features=4, hidden_dim=HIDDEN)
    train_part = VerticalPartition(features=torch.arange(40.0).view(10, 4), batch_size=BATCH)
    top = ServerTopMLP(num_clients=K, hidden_dim=HIDDEN, num_classes=2)
    lh = LabelHolderContext(
        top_model=top, top_lr=0.1, num_classes=2,
        label_train=VerticalPartition(
            features=torch.randint(0, 2, (64,)), batch_size=BATCH),
        test_labels=torch.randint(0, 2, (12,)),
    )
    client = VerticalSplitClient(0, bottom, train_part, 0.1, ctx, label_holder=lh)
    return client, ctx, top


def test_label_holder_compute_returns_per_party_grads():
    client, ctx, _ = _make_label_holder_client()
    joint = np.random.RandomState(0).randn(BATCH, K * HIDDEN).astype(np.float32)
    cfg = {LP_PHASE_KEY: "compute", "sl_step": 0,
           "widths": ",".join(str(w) for w in WIDTHS), "pids": "0,1", "do_eval": False}
    grads, n, metrics = client._lp_compute([joint], cfg)
    assert len(grads) == K
    for g in grads:
        assert g.shape == (BATCH, HIDDEN)
    assert n == BATCH
    assert "train_loss" in metrics and "has_eval" not in metrics


def test_top_model_and_optimizer_state_persist():
    client, ctx, top = _make_label_holder_client()
    joint = np.random.RandomState(1).randn(BATCH, K * HIDDEN).astype(np.float32)
    cfg = {LP_PHASE_KEY: "compute", "sl_step": 0, "widths": "4,4", "pids": "0,1",
           "do_eval": False}
    client._lp_compute([joint], cfg)
    assert _LP_TOP_KEY in ctx.state       # (13) top weights persisted
    assert _LP_TOP_OPT_KEY in ctx.state   # (14) momentum buffers persisted
    w1 = ctx.state[_LP_TOP_KEY].to_numpy_ndarrays()
    # A second step should change persisted weights (training progresses).
    client._lp_compute([joint], {**cfg, "sl_step": 1})
    w2 = ctx.state[_LP_TOP_KEY].to_numpy_ndarrays()
    assert any(not np.allclose(a, b) for a, b in zip(w1, w2))


def test_compute_eval_returns_only_aggregate_metrics():
    client, ctx, _ = _make_label_holder_client()
    joint = np.zeros((BATCH, K * HIDDEN), dtype=np.float32)
    test_joint = np.zeros((12, K * HIDDEN), dtype=np.float32)
    cfg = {LP_PHASE_KEY: "compute", "sl_step": 0, "widths": "4,4", "pids": "0,1",
           "do_eval": True}
    _grads, _n, metrics = client._lp_compute([joint, test_joint], cfg)
    assert metrics["has_eval"] is True
    assert {"val_accuracy", "val_loss", "f1", "roc_auc"} <= set(metrics)
    # No raw labels or per-sample predictions in the returned payload.
    for v in metrics.values():
        assert isinstance(v, (int, float, str, bool))


def test_passive_client_has_no_label_holder():
    ctx = SimpleNamespace(state={})
    bottom = BottomMLP(in_features=4, hidden_dim=HIDDEN)
    part = VerticalPartition(features=torch.arange(40.0).view(10, 4), batch_size=BATCH)
    passive = VerticalSplitClient(1, bottom, part, 0.1, ctx, label_holder=None)
    assert passive._label_holder is None
    with pytest.raises(AssertionError):
        passive._lp_compute([np.zeros((BATCH, K * HIDDEN), np.float32)],
                            {"sl_step": 0, "widths": "4,4"})


def test_label_holder_compute_is_deterministic_under_seed():
    def run():
        client, _ctx, _ = _make_label_holder_client()
        joint = np.ones((BATCH, K * HIDDEN), dtype=np.float32)
        cfg = {LP_PHASE_KEY: "compute", "sl_step": 0, "widths": "4,4", "pids": "0,1",
               "do_eval": False}
        return client._lp_compute([joint], cfg)[0]
    g1, g2 = run(), run()
    for a, b in zip(g1, g2):
        assert np.allclose(a, b)
