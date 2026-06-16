"""Label-private vertical split-learning strategy.

Like :class:`~rizemind.split_learning.vertical_strategy.VerticalSplitLearningStrategy`,
all K parties hold different features for the same samples. The difference: the
**labels and the top model never reach the coordinator**. One configured
*label-holder* party owns the labels, the top model, the loss, and all
label-dependent metrics. The coordinator only routes tensors and aggregate
scalars.

Per training step uses **three** Flower rounds (vs two in the baseline):

- **COLLECT** (round 3k-2): every party runs its bottom and returns an
  activation. The strategy caches them ordered by ``partition_id``.
- **COMPUTE** (round 3k-1): the strategy concatenates the K activations into a
  *joint activation* and sends it to **only the label-holder client**. That
  client runs the top model against its **local labels**, backpropagates, and
  returns one gradient per party (in ``partition_id`` order) plus the train loss
  and — on eval rounds — aggregate metrics. **No labels or predictions are
  returned**, only tensors and scalars.
- **DISTRIBUTE** (round 3k): each party receives its gradient slice and updates
  its bottom model.

Evaluation stays label-free at the coordinator: a caller-supplied
``build_test_joint_activation`` callback turns the cached bottom weights + test
*features* (never labels) into a joint test activation, which is shipped to the
label holder inside the same COMPUTE round; the holder computes metrics against
its local test labels and returns aggregate scalars only.

The coordinator/strategy never loads, stores, or logs a label tensor. See the
phase-3 ``THREAT_MODEL.md`` for the precise guarantees and non-guarantees.
"""

from __future__ import annotations

from collections.abc import Callable
from logging import INFO, WARNING

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.serialization import _SL_TENSOR_TYPE
from rizemind.split_learning.telemetry import RunTelemetry, StepTelemetry

_PHASE_COLLECT = "collect"
_PHASE_COMPUTE = "compute"
_PHASE_DISTRIBUTE = "distribute"

_PARTITION_ID_KEY = "partition_id"
#: Config key telling the label-private client which phase/role it is in.
LP_PHASE_KEY = "lp_phase"

_EMPTY = Parameters(tensors=[], tensor_type="")

#: (server_round, bottom_weights_by_pid) -> joint test activation (M, sum_H) or
#: None to skip eval this round. Must NOT use labels.
BuildTestJointFn = Callable[[int, dict[int, list[np.ndarray]]], "np.ndarray | None"]
#: (server_round, aggregate_metrics) -> None. For result writing (no labels).
OnMetricsFn = Callable[[int, dict[str, float]], None]


class LabelPrivateVerticalStrategy(Strategy):
    """VFL strategy that keeps labels + the top model off the coordinator.

    Args:
        config: Split-learning config (``num_rounds_per_step`` is informational
            here; this strategy always uses three rounds per step).
        num_clients: Number of vertical participants (K).
        label_holder_pid: ``partition_id`` of the party that owns the labels and
            top model. Must be in ``[0, num_clients)``.
        build_test_joint_activation: Optional callback producing the joint test
            activation from cached bottom weights (no labels). ``None`` disables
            evaluation.
        on_metrics: Optional callback receiving aggregate metrics for result
            writing (no labels).
        telemetry: Optional :class:`RunTelemetry` accumulator.
    """

    def __init__(
        self,
        config: SplitLearningConfig,
        num_clients: int,
        label_holder_pid: int,
        build_test_joint_activation: BuildTestJointFn | None = None,
        on_metrics: OnMetricsFn | None = None,
        telemetry: RunTelemetry | None = None,
    ) -> None:
        super().__init__()
        if num_clients < 1:
            raise ValueError(f"num_clients must be >= 1, got {num_clients}")
        if not 0 <= label_holder_pid < num_clients:
            raise ValueError(
                f"label_holder_pid must be in [0, {num_clients}), got "
                f"{label_holder_pid}"
            )
        self.config = config
        self.num_clients = num_clients
        self.label_holder_pid = label_holder_pid
        self.build_test_joint_activation = build_test_joint_activation
        self.on_metrics = on_metrics
        self.telemetry = telemetry

        self._phase = _PHASE_COLLECT
        self._step_idx = 0
        # COLLECT outputs (cleared each step). NB: activations + cids only —
        # never labels.
        self._acts_by_pid: dict[int, np.ndarray] = {}
        self._cid_by_pid: dict[int, str] = {}
        # COMPUTE outputs.
        self._grad_by_cid: dict[str, Parameters] = {}
        self._bottom_weights_by_pid: dict[int, list[np.ndarray]] = {}
        self._last_train_loss = float("nan")
        self._pending_eval: tuple[float, dict[str, Scalar]] | None = None
        self._step_tel: StepTelemetry | None = None

    # ------------------------------------------------------------------
    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        del client_manager
        return _EMPTY

    # ------------------------------------------------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        del parameters
        if self._phase == _PHASE_COLLECT:
            sampled = client_manager.sample(self.num_clients, min_num_clients=self.num_clients)
            if len(sampled) != self.num_clients:
                raise RuntimeError(
                    f"label-private COLLECT: expected {self.num_clients} clients, "
                    f"got {len(sampled)}"
                )
            cfg: dict[str, Scalar] = {LP_PHASE_KEY: _PHASE_COLLECT, "sl_step": self._step_idx}
            self._step_tel = StepTelemetry(sl_step=self._step_idx)
            log(INFO, "configure_fit r=%d COLLECT sl_step=%d K=%d",
                server_round, self._step_idx, self.num_clients)
            return [(c, FitIns(_EMPTY, cfg)) for c in sampled]

        if self._phase == _PHASE_COMPUTE:
            holder_cid = self._cid_by_pid[self.label_holder_pid]
            holder = client_manager.all().get(holder_cid)
            if holder is None:
                raise RuntimeError(
                    f"label-private COMPUTE: label holder cid={holder_cid} "
                    f"(pid={self.label_holder_pid}) not in client_manager"
                )
            # Joint TRAIN activation in pid order; never includes labels.
            order = sorted(self._acts_by_pid)
            widths = [self._acts_by_pid[p].shape[1] for p in order]
            joint_train = np.concatenate([self._acts_by_pid[p] for p in order], axis=1)
            arrays = [joint_train]
            do_eval = False
            if self.build_test_joint_activation is not None:
                joint_test = self.build_test_joint_activation(
                    server_round, dict(self._bottom_weights_by_pid)
                )
                if joint_test is not None:
                    arrays.append(np.asarray(joint_test, dtype=np.float32))
                    do_eval = True
            params = ndarrays_to_parameters(arrays)
            cfg = {
                LP_PHASE_KEY: _PHASE_COMPUTE,
                "sl_step": self._step_idx,
                "widths": ",".join(str(w) for w in widths),
                "pids": ",".join(str(p) for p in order),
                "do_eval": do_eval,
            }
            if self._step_tel is not None:
                # coordinator -> label holder representation payload
                self._step_tel.add_time("_repr_bytes_marker", 0.0)
                self._repr_bytes = int(sum(a.nbytes for a in arrays))
            log(INFO, "configure_fit r=%d COMPUTE -> label holder pid=%d do_eval=%s",
                server_round, self.label_holder_pid, do_eval)
            return [(holder, FitIns(params, cfg))]

        # DISTRIBUTE
        available = client_manager.all()
        instructions = []
        for cid, grad in self._grad_by_cid.items():
            client = available.get(cid)
            if client is not None:
                instructions.append((client, FitIns(grad, {LP_PHASE_KEY: _PHASE_DISTRIBUTE})))
        log(INFO, "configure_fit r=%d DISTRIBUTE -> %d parties", server_round,
            len(instructions))
        return instructions

    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        del failures
        if not results:
            log(WARNING, "aggregate_fit r=%d: no results in phase=%s",
                server_round, self._phase)
            return _EMPTY, {"train_loss": self._last_train_loss}

        if self._phase == _PHASE_COLLECT:
            self._collect(results)
            self._phase = _PHASE_COMPUTE
            return _EMPTY, {"sl_step": float(self._step_idx)}

        if self._phase == _PHASE_COMPUTE:
            self._compute(results, server_round)
            self._phase = _PHASE_DISTRIBUTE
            return _EMPTY, {"train_loss": self._last_train_loss}

        # DISTRIBUTE
        self._distribute(results, server_round)
        self._grad_by_cid.clear()
        self._acts_by_pid.clear()
        self._cid_by_pid.clear()
        if self._step_tel is not None and self.telemetry is not None:
            self.telemetry.record_step(self._step_tel)
        self._step_idx += 1
        self._phase = _PHASE_COLLECT
        return _EMPTY, {"train_loss": self._last_train_loss}

    def _collect(self, results: list[tuple[ClientProxy, FitRes]]) -> None:
        if len(results) != self.num_clients:
            raise ValueError(
                f"label-private COLLECT expected {self.num_clients} activations, "
                f"got {len(results)}"
            )
        seen: dict[int, str] = {}
        for client, res in results:
            if _PARTITION_ID_KEY not in res.metrics:
                raise ValueError("label-private COLLECT: missing partition_id")
            pid = int(res.metrics[_PARTITION_ID_KEY])
            if not 0 <= pid < self.num_clients:
                raise ValueError(f"label-private COLLECT: bad partition_id={pid}")
            if pid in seen:
                raise ValueError(f"label-private COLLECT: duplicate partition_id={pid}")
            seen[pid] = client.cid
            arr = parameters_to_ndarrays(res.parameters)[0]
            self._acts_by_pid[pid] = arr
            self._cid_by_pid[pid] = client.cid
            if self._step_tel is not None:
                self._step_tel.add_activation(pid, arr)

    def _compute(self, results: list[tuple[ClientProxy, FitRes]], server_round: int) -> None:
        # Exactly one result: from the label holder.
        if len(results) != 1:
            raise ValueError(
                f"label-private COMPUTE expected 1 result from the label holder, "
                f"got {len(results)}"
            )
        _client, res = results[0]
        grads = parameters_to_ndarrays(res.parameters)
        order = sorted(self._acts_by_pid)
        if len(grads) != len(order):
            raise ValueError(
                f"label holder returned {len(grads)} gradients, expected {len(order)}"
            )
        self._grad_by_cid = {}
        for pid, g in zip(order, grads):
            cid = self._cid_by_pid[pid]
            gp = ndarrays_to_parameters([g])
            self._grad_by_cid[cid] = Parameters(tensors=gp.tensors, tensor_type=_SL_TENSOR_TYPE)
            if self._step_tel is not None:
                self._step_tel.add_gradient(pid, g)
        self._last_train_loss = float(res.metrics.get("train_loss", float("nan")))
        # Aggregate eval metrics (scalars only — no labels/predictions).
        if res.metrics.get("has_eval"):
            metrics = {
                k: float(v)
                for k, v in res.metrics.items()
                if k not in ("train_loss", "has_eval", _PARTITION_ID_KEY)
                and isinstance(v, (int, float))
            }
            loss = float(res.metrics.get("val_loss", float("nan")))
            self._pending_eval = (loss, metrics)
            if self.on_metrics is not None:
                self.on_metrics(server_round, metrics)

    def _distribute(self, results: list[tuple[ClientProxy, FitRes]], server_round: int) -> None:
        for client, res in results:
            if _PARTITION_ID_KEY not in res.metrics:
                log(WARNING, "label-private DISTRIBUTE: cid=%s missing partition_id",
                    client.cid)
                continue
            pid = int(res.metrics[_PARTITION_ID_KEY])
            if res.parameters.tensors:
                self._bottom_weights_by_pid[pid] = parameters_to_ndarrays(res.parameters)

    # ------------------------------------------------------------------
    def configure_evaluate(self, server_round, parameters, client_manager):
        del server_round, parameters, client_manager
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        del server_round, results, failures
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        """Surface the latest label-holder-computed aggregate eval to Flower."""
        del parameters
        if self._pending_eval is None:
            return None
        out = self._pending_eval
        self._pending_eval = None
        return out
