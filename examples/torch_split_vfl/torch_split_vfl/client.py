"""Client-side vertical split-learning logic.

Each client owns the bottom MLP for ONE feature strip of every training image.
A training step takes two Flower rounds, dispatched by
:class:`~rizemind.split_learning.vertical_strategy.VerticalSplitLearningStrategy`:

  Round 2k-1 (forward)
      - Read ``sl_step`` from ``FitIns.config``.
      - Pull the deterministic strip batch from the local
        :class:`~torch_split_vfl.task.VerticalPartition`.
      - Persist the batch + the current bottom weights into ``context.state``
        so the backward round can reproduce an identical forward pass.
      - Run the bottom model, return the activation as a single ndarray
        alongside ``partition_id`` in ``metrics``.

  Round 2k (backward)
      - Restore the saved batch + bottom weights from ``context.state``.
      - Redo the forward pass (same weights → same grad_fn).
      - Apply the server-provided gradient at the cut.
      - Step the optimizer and return the updated bottom weights so the
        strategy can cache them for centralized evaluation.

Cross-round state (``context.state``)
-------------------------------------
``_SL_BOTTOM_KEY``  — bottom-model weights (``ArrayRecord``)
``_SL_INPUT_KEY``   — last forward-round input batch (``ArrayRecord``)
"""

from __future__ import annotations

from logging import INFO

import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log
from flwr.common.record import ArrayRecord
from rizemind.split_learning.mod import (
    SL_PHASE_BACKWARD,
    SL_PHASE_KEY,
    split_learning_mod,
)

from .task import (
    VerticalPartition,
    build_bottom_model,
    get_dataset_spec,
    get_weights,
    make_client_train_partition,
    set_weights,
)

_SL_BOTTOM_KEY = "vfl_bottom"
_SL_INPUT_KEY = "vfl_input_x"

_DEMO_TAG = "[VSL-DEMO]"


def _shape(t) -> str:
    dims = list(t.shape)
    inner = ", ".join(str(d) for d in dims)
    return f"({inner},)" if len(dims) == 1 else f"({inner})"


class VerticalSplitClient(NumPyClient):
    """NumPyClient owning one bottom MLP over a single feature strip.

    Attributes:
        partition_id: Zero-based vertical-client id (sets the strip + concat
            order on the server).
        bottom: Local bottom MLP.
        train_partition: Aligned vertical view of the train split.
        optimizer: SGD optimizer for ``bottom.parameters()``.
    """

    def __init__(
        self,
        partition_id: int,
        bottom: nn.Module,
        train_partition: VerticalPartition,
        learning_rate: float,
        ctx: Context,
        demo: bool = False,
    ) -> None:
        self.partition_id = partition_id
        self.bottom = bottom
        self.train_partition = train_partition
        # No momentum: a fresh client (and optimizer) is built each round and
        # momentum buffers are not persisted in context.state, so any momentum
        # would silently reset every step. Use plain SGD to avoid implying state
        # that isn't carried across rounds.
        self.optimizer = optim.SGD(bottom.parameters(), lr=learning_rate, momentum=0.0)
        self._ctx = ctx
        self._demo = demo

    def _demo_log(self, phase: str, msg: str) -> None:
        if self._demo:
            log(
                INFO,
                "%s CLIENT pid=%d  %s | %s",
                _DEMO_TAG,
                self.partition_id,
                phase,
                msg,
            )

    # ------------------------------------------------------------------
    # Flower dispatch
    # ------------------------------------------------------------------

    def fit(self, parameters, config):
        """Dispatch to forward or backward pass based on ``sl_phase``."""
        if config.get(SL_PHASE_KEY) != SL_PHASE_BACKWARD:
            return self._forward(config)
        return self._backward(parameters)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, config):
        sl_step = int(config["sl_step"])
        x = self.train_partition.get_batch_for_step(sl_step)

        self._ctx.state[_SL_INPUT_KEY] = ArrayRecord(numpy_ndarrays=[x.numpy()])
        self._ctx.state[_SL_BOTTOM_KEY] = ArrayRecord(
            numpy_ndarrays=get_weights(self.bottom)
        )

        self.optimizer.zero_grad()
        activation = self.bottom(x)

        self._demo_log(
            "FORWARD ",
            f"sl_step={sl_step}  {_shape(x)} -> activation {_shape(activation)}",
        )

        return (
            [activation.detach().numpy()],
            x.shape[0],
            {"partition_id": self.partition_id, "sl_step": float(sl_step)},
        )

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def _backward(self, parameters):
        if _SL_INPUT_KEY not in self._ctx.state or len(parameters) == 0:
            return get_weights(self.bottom), 0, {"partition_id": self.partition_id}

        x = torch.from_numpy(self._ctx.state[_SL_INPUT_KEY].to_numpy_ndarrays()[0])
        saved_weights = self._ctx.state[_SL_BOTTOM_KEY].to_numpy_ndarrays()
        del self._ctx.state[_SL_INPUT_KEY]

        set_weights(self.bottom, saved_weights)
        self.optimizer.zero_grad()
        activation = self.bottom(x)

        grad = torch.from_numpy(parameters[0])
        activation.backward(grad)
        self.optimizer.step()

        updated_weights = get_weights(self.bottom)
        self._ctx.state[_SL_BOTTOM_KEY] = ArrayRecord(numpy_ndarrays=updated_weights)

        self._demo_log(
            "BACKWARD",
            f"gradient {_shape(grad)} applied, bottom updated + cached",
        )

        return (
            updated_weights,
            x.shape[0],
            {"partition_id": self.partition_id},
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, parameters, config):
        """No distributed evaluation in v1 — server runs centralized eval."""
        del parameters, config
        return 0.0, 0, {}


# ---------------------------------------------------------------------------
# client_fn
# ---------------------------------------------------------------------------


def client_fn(context: Context):
    """Construct a :class:`VerticalSplitClient` from Flower context."""
    partition_id = int(context.node_config["partition-id"])
    num_clients = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    learning_rate = float(context.run_config["learning-rate"])
    hidden_dim = int(context.run_config["hidden-dim"])
    demo = bool(context.run_config.get("demo", False))
    dataset = str(context.run_config.get("dataset", "mnist"))
    max_train_samples = int(context.run_config.get("max-train-samples", 0))

    spec = get_dataset_spec(dataset)

    bottom = build_bottom_model(
        spec=spec,
        partition_id=partition_id,
        num_clients=num_clients,
        hidden_dim=hidden_dim,
    )
    if _SL_BOTTOM_KEY in context.state:
        set_weights(bottom, context.state[_SL_BOTTOM_KEY].to_numpy_ndarrays())

    train_partition = make_client_train_partition(
        spec=spec,
        partition_id=partition_id,
        num_clients=num_clients,
        batch_size=batch_size,
        max_train_samples=max_train_samples or None,
    )

    return VerticalSplitClient(
        partition_id=partition_id,
        bottom=bottom,
        train_partition=train_partition,
        learning_rate=learning_rate,
        ctx=context,
        demo=demo,
    ).to_client()


app = ClientApp(client_fn=client_fn, mods=[split_learning_mod])
