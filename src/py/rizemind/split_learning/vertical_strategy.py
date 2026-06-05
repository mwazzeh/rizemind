"""Strategy for vertical split federated learning.

In **vertical** split federated learning, all K participants hold *different
features* for the *same* samples (aligned by sample id). At each step every
client runs its local **bottom model** on its own feature slice and sends an
activation up to the server, which concatenates the K activations, runs the
**server tail** (with the server-held labels), and ships per-client gradients
back so each client can update its local bottom model.

Two Flower rounds compose one training step (same as horizontal split
learning), driven by a deterministic ``sl_step`` counter the strategy ships to
clients:

- **Round 2k-1** (forward): every client pulls its batch deterministically from
  ``sl_step=k`` and returns its activation. The strategy stores activations
  ordered by ``partition_id`` (sent in ``FitRes.metrics``) and invokes
  ``on_train_step``, which runs the tail forward/backward and returns
  per-client gradients keyed by ``cid``.
- **Round 2k** (backward): each client receives its gradient slice, finishes
  the backward pass, updates its bottom model, and returns the updated bottom
  weights. The strategy caches them by ``partition_id`` so it can run end-to-end
  evaluation centrally inside :meth:`evaluate`.

Unlike :class:`SplitLearningStrategy`, no cross-client weight averaging takes
place: each client's bottom model encodes a different feature partition and
is not semantically interchangeable with any other client's bottom. Bottom
weights stay local; the server only caches them by ``partition_id`` for
centralized evaluation.

Validation
----------
The strategy validates VFL invariants strictly. Sample alignment across the K
feature slices breaks silently if any of these are off, so we fail loudly:

- Every forward result must carry ``partition_id`` in ``FitRes.metrics``.
- ``partition_id`` values must be in ``[0, num_clients)``.
- Duplicate ``partition_id`` values are rejected.
- Exactly ``num_clients`` results must arrive in the forward round.
- ``on_train_step`` must return a gradient ``Parameters`` for every cid that
  contributed an activation, and no extras.
"""

from __future__ import annotations

from collections.abc import Callable
from logging import DEBUG, INFO, WARNING

import numpy as np
from flwr.common import parameters_to_ndarrays
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

_FORWARD_PHASE = "forward"
_BACKWARD_PHASE = "backward"

#: Metric key clients must include in ``FitRes.metrics`` to identify their
#: vertical partition.
_PARTITION_ID_KEY = "partition_id"

#: Empty placeholder ``Parameters`` shipped to clients during forward rounds.
#: Vertical clients pull their own data using ``sl_step``; no global weights
#: cross the wire.
_EMPTY_PARAMETERS = Parameters(tensors=[], tensor_type="")

OrderedActivations = list[tuple[int, str, Parameters]]
OnTrainStepFn = Callable[[int, OrderedActivations], tuple[dict[str, Parameters], float]]
OnEvaluateFn = Callable[
    [int, dict[int, list[np.ndarray]]],
    tuple[float, dict[str, Scalar]] | None,
]
OnForwardConfigFn = Callable[[int, int], dict[str, Scalar]]


class VerticalSplitLearningStrategy(Strategy):
    """Flower strategy for vertical split federated learning.

    Attributes:
        config: Split-learning configuration (consulted for
            ``num_rounds_per_step``; ``cut_layer`` is informational).
        num_clients: Number of vertical participants. All of them must
            participate in every step (no client sampling in VFL — sample ids
            must align across the K feature slices).
        on_train_step: Callable invoked at the end of each forward round.
            Receives ``(sl_step, ordered_activations)`` where
            ``ordered_activations`` is a list of
            ``(partition_id, cid, activation_params)`` sorted by ``partition_id``
            so the server concatenates in a deterministic order. Returns
            ``({cid: gradient_params}, train_loss)``. Must produce exactly one
            gradient per input cid.
        on_evaluate: Optional centralized eval callback. Receives
            ``(server_round, bottom_weights_by_pid)`` mapping each reported
            client to its latest cached bottom-model weights. Returns
            ``(loss, metrics)`` or ``None`` to skip the round.
        on_forward_config_fn: Optional extra config injector for the forward
            round. Receives ``(server_round, sl_step)`` and returns extra
            ``Scalar`` keys merged into ``FitIns.config``.
    """

    def __init__(
        self,
        config: SplitLearningConfig,
        num_clients: int,
        on_train_step: OnTrainStepFn,
        on_evaluate: OnEvaluateFn | None = None,
        on_forward_config_fn: OnForwardConfigFn | None = None,
    ) -> None:
        super().__init__()
        if num_clients < 1:
            raise ValueError(f"num_clients must be >= 1, got {num_clients}")
        self.config = config
        self.num_clients = num_clients
        self.on_train_step = on_train_step
        self.on_evaluate = on_evaluate
        self.on_forward_config_fn = on_forward_config_fn
        self._phase: str = _FORWARD_PHASE
        self._step_idx: int = 0
        self._gradient_store: dict[str, Parameters] = {}
        self._bottom_weights_by_pid: dict[int, list[np.ndarray]] = {}
        self._last_train_loss: float = float("nan")
        log(
            DEBUG,
            "VerticalSplitLearningStrategy: initialised (num_clients=%d, phase=%s)",
            num_clients,
            self._phase,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Return empty parameters — VFL has no global model on the server."""
        del client_manager
        return _EMPTY_PARAMETERS

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Dispatch forward (one per client) or backward (one gradient per client).

        Args:
            server_round: Current Flower round number.
            parameters: Ignored (no global parameters in VFL).
            client_manager: Flower client manager.

        Returns:
            List of ``(ClientProxy, FitIns)`` pairs.
        """
        del parameters

        if self._phase == _BACKWARD_PHASE:
            available = client_manager.all()
            instructions: list[tuple[ClientProxy, FitIns]] = []
            unmatched: list[str] = []
            for cid, grad_params in self._gradient_store.items():
                client = available.get(cid)
                if client is None:
                    unmatched.append(cid)
                    continue
                instructions.append((client, FitIns(grad_params, {})))
            if unmatched:
                log(
                    WARNING,
                    "configure_fit (round=%d): backward — %d gradient(s) targeted "
                    "unknown cid(s) (no longer in client_manager): %s",
                    server_round,
                    len(unmatched),
                    unmatched,
                )
            log(
                INFO,
                "configure_fit (round=%d): backward — dispatching %d gradient(s) "
                "to clients",
                server_round,
                len(instructions),
            )
            return instructions

        sampled = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        if len(sampled) != self.num_clients:
            raise RuntimeError(
                f"VerticalSplitLearningStrategy.configure_fit: expected "
                f"{self.num_clients} client(s) from client_manager.sample(), "
                f"got {len(sampled)}. Vertical SFL requires every participant "
                f"to be present in every round (sample ids must align)."
            )
        cfg: dict[str, Scalar] = {"sl_step": self._step_idx}
        if self.on_forward_config_fn is not None:
            cfg.update(self.on_forward_config_fn(server_round, self._step_idx))
        instructions = [(client, FitIns(_EMPTY_PARAMETERS, cfg)) for client in sampled]
        log(
            INFO,
            "configure_fit (round=%d): forward — sl_step=%d, dispatching to %d "
            "client(s)",
            server_round,
            self._step_idx,
            len(instructions),
        )
        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Drive the forward → backward phase transitions.

        Forward results carry activations (``tensor_type ==
        "split_learning.activation"``); ``on_train_step`` is invoked with the
        activations ordered by ``partition_id`` and its returned per-client
        gradients are queued for the next backward round.

        Backward results carry each client's updated bottom-model weights; the
        strategy caches them by ``partition_id`` for centralized evaluation
        and advances ``sl_step``.

        Args:
            server_round: Current Flower round number.
            results: ``(ClientProxy, FitRes)`` pairs returned by clients.
            failures: Failed results / exceptions (unused).

        Returns:
            ``(parameters, metrics)`` — ``parameters`` is always
            ``_EMPTY_PARAMETERS``; ``metrics`` carries the latest train loss.

        Raises:
            ValueError: When forward results violate a VFL invariant
                (missing/duplicate/out-of-range ``partition_id``, wrong count,
                or mismatched gradient ``cid`` set returned by
                ``on_train_step``).
        """
        del failures

        if not results:
            log(
                WARNING,
                "aggregate_fit (round=%d): no results received; staying in "
                "phase=%s, sl_step=%d",
                server_round,
                self._phase,
                self._step_idx,
            )
            return _EMPTY_PARAMETERS, {"train_loss": self._last_train_loss}

        if self._is_activation_results(results):
            ordered = self._collect_forward_activations(results)
            grad_store, train_loss = self.on_train_step(self._step_idx, ordered)
            self._validate_gradient_store(grad_store, ordered)
            self._gradient_store = dict(grad_store)
            self._last_train_loss = float(train_loss)
            self._phase = _BACKWARD_PHASE
            log(
                INFO,
                "aggregate_fit (round=%d): forward done — sl_step=%d K=%d "
                "train_loss=%.4f → backward",
                server_round,
                self._step_idx,
                self.num_clients,
                self._last_train_loss,
            )
            return _EMPTY_PARAMETERS, {
                "train_loss": self._last_train_loss,
                "sl_step": float(self._step_idx),
                "num_partitions": float(self.num_clients),
            }

        self._cache_backward_weights(results, server_round)
        self._gradient_store.clear()
        self._step_idx += 1
        self._phase = _FORWARD_PHASE
        log(
            INFO,
            "aggregate_fit (round=%d): backward done — advancing to sl_step=%d "
            "(cached %d/%d bottom(s))",
            server_round,
            self._step_idx,
            len(self._bottom_weights_by_pid),
            self.num_clients,
        )
        return _EMPTY_PARAMETERS, {"train_loss": self._last_train_loss}

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """No distributed evaluation phase — :meth:`evaluate` runs centrally."""
        del server_round, parameters, client_manager
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """No-op — distributed evaluation is disabled."""
        del server_round, results, failures
        return None, {}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> tuple[float, dict[str, Scalar]] | None:
        """Run centralized end-to-end evaluation via ``on_evaluate``.

        Skipped until ``on_evaluate`` is set and every participant has reported
        at least one set of bottom-model weights (i.e. after the first backward
        round completes). Eval is centralized: the server holds the labels and
        loads the cached per-client bottoms — no client averaging happens.

        Args:
            server_round: Current Flower round number.
            parameters: Ignored.

        Returns:
            Result of ``on_evaluate`` or ``None`` to skip the round.
        """
        del parameters
        if self.on_evaluate is None:
            return None
        if len(self._bottom_weights_by_pid) < self.num_clients:
            return None
        return self.on_evaluate(server_round, dict(self._bottom_weights_by_pid))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_forward_activations(
        self,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> OrderedActivations:
        """Validate and order forward-round activations by ``partition_id``.

        Args:
            results: Forward-round client results.

        Returns:
            ``[(partition_id, cid, params), ...]`` sorted ascending by
            ``partition_id``.

        Raises:
            ValueError: If any result is missing ``partition_id`` metadata,
                a ``partition_id`` is out of ``[0, num_clients)``, two results
                carry the same ``partition_id``, or the result count is not
                exactly ``num_clients``.
        """
        if len(results) != self.num_clients:
            raise ValueError(
                f"VerticalSplitLearningStrategy: expected {self.num_clients} "
                f"forward result(s), got {len(results)}. Vertical SFL requires "
                f"every participant to report each step so sample ids stay "
                f"aligned across feature slices."
            )

        ordered: OrderedActivations = []
        seen_pids: dict[int, str] = {}
        for client, res in results:
            if _PARTITION_ID_KEY not in res.metrics:
                raise ValueError(
                    f"VerticalSplitLearningStrategy: client cid={client.cid!r} "
                    f"returned a forward result without {_PARTITION_ID_KEY!r} "
                    f"in FitRes.metrics. The client must report its "
                    f"partition_id so the server can concatenate activations "
                    f"in a deterministic order."
                )
            pid = int(res.metrics[_PARTITION_ID_KEY])
            if pid < 0 or pid >= self.num_clients:
                raise ValueError(
                    f"VerticalSplitLearningStrategy: client cid={client.cid!r} "
                    f"reported partition_id={pid}, expected a value in "
                    f"[0, {self.num_clients})."
                )
            if pid in seen_pids:
                other = seen_pids[pid]
                raise ValueError(
                    f"VerticalSplitLearningStrategy: duplicate partition_id={pid} "
                    f"reported by both cid={other!r} and cid={client.cid!r}. "
                    f"Every vertical participant must own a unique partition_id."
                )
            seen_pids[pid] = client.cid
            ordered.append((pid, client.cid, res.parameters))

        ordered.sort(key=lambda item: item[0])
        return ordered

    def _validate_gradient_store(
        self,
        grad_store: dict[str, Parameters],
        ordered: OrderedActivations,
    ) -> None:
        """Ensure ``on_train_step`` returned exactly one gradient per input cid.

        Args:
            grad_store: ``{cid: gradient_params}`` returned by
                ``on_train_step``.
            ordered: Forward activations passed in.

        Raises:
            ValueError: When ``grad_store`` is missing cids or contains extra
                cids not present in ``ordered``.
        """
        expected = {cid for _, cid, _ in ordered}
        got = set(grad_store)
        missing = expected - got
        extra = got - expected
        if missing or extra:
            raise ValueError(
                f"VerticalSplitLearningStrategy: on_train_step returned a "
                f"gradient store with mismatched cids "
                f"(missing={sorted(missing)}, unexpected={sorted(extra)}). "
                f"Expected exactly one gradient per input activation cid."
            )

    def _cache_backward_weights(
        self,
        results: list[tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> None:
        """Cache each client's updated bottom-model weights by ``partition_id``.

        Skips results that are missing ``partition_id`` or carry empty
        parameters (logged as warnings). Raises on duplicate ``partition_id``
        because two clients claiming the same partition would silently
        overwrite each other in the cache.

        Args:
            results: Backward-round client results.
            server_round: Current Flower round (for logs).

        Raises:
            ValueError: When two backward results share the same
                ``partition_id``.
        """
        seen_pids: dict[int, str] = {}
        for client, res in results:
            if _PARTITION_ID_KEY not in res.metrics:
                log(
                    WARNING,
                    "aggregate_fit (round=%d) backward: client cid=%s returned "
                    "result without %s metric; not caching bottom weights.",
                    server_round,
                    client.cid,
                    _PARTITION_ID_KEY,
                )
                continue
            pid = int(res.metrics[_PARTITION_ID_KEY])
            if pid in seen_pids:
                other = seen_pids[pid]
                raise ValueError(
                    f"VerticalSplitLearningStrategy: duplicate partition_id="
                    f"{pid} in backward results (cid={other!r} and "
                    f"cid={client.cid!r})."
                )
            seen_pids[pid] = client.cid
            if not res.parameters.tensors:
                log(
                    WARNING,
                    "aggregate_fit (round=%d) backward: pid=%d returned empty "
                    "parameters; not caching bottom weights.",
                    server_round,
                    pid,
                )
                continue
            self._bottom_weights_by_pid[pid] = parameters_to_ndarrays(res.parameters)

    @staticmethod
    def _is_activation_results(results: list[tuple[ClientProxy, FitRes]]) -> bool:
        """Return True if the first result carries split-learning activations.

        Args:
            results: List of ``(ClientProxy, FitRes)`` pairs.

        Returns:
            ``True`` when the first ``FitRes`` has
            ``tensor_type == "split_learning.activation"``.
        """
        if not results:
            return False
        _, first = results[0]
        return first.parameters.tensor_type == _SL_TENSOR_TYPE
