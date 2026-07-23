"""Client-side split-learning logic for the torch_split example.

The client owns the ClientHead (layers before the cut point).
Each SplitLearningStrategy training step spans two Flower rounds:

  Round 2k-1  (sl_phase="forward")
    - Fetch the next real MNIST mini-batch from the local partition.
    - Run the forward pass through the head.
    - Persist the input, labels, and head weights in context.state so the
      backward round can reconstruct an identical forward pass.
    - Return [activation, labels]; split_learning_mod re-tags tensor_type.

  Round 2k  (sl_phase="backward")
    - Restore input + head weights from context.state.
    - Redo the forward pass (same weights → same grad_fn).
    - Apply the server gradient and step the optimizer.
    - Persist updated head weights for the next forward round.
    - Return updated head weights.

Cross-round state (context.state keys)
---------------------------------------
_SL_HEAD_KEY     — head model weights (ArrayRecord)
_SL_INPUT_KEY    — last forward-round input batch (ArrayRecord)
_SL_LABELS_KEY   — last forward-round labels (ArrayRecord)
_SL_TRAIN_STATE  — training cursor: {batch_idx, epoch} (ConfigRecord)

Multi-batch training
--------------------
Each Flower round processes exactly one MNIST mini-batch from the client's
local partition.  The partition is materialized once per process and batches
are fetched via direct tensor indexing, so advancing ``batch_idx`` in
``context.state`` remains O(1) across Flower rounds.  Increasing
``num-server-rounds`` in pyproject.toml covers more of the local dataset per
federated round.  A full epoch requires ``2 * train_partition.num_batches``
Flower rounds.
"""

from __future__ import annotations

from logging import INFO

import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log
from flwr.common.record import ArrayRecord, ConfigRecord
from rizemind.split_learning.mod import SL_PHASE_BACKWARD, SL_PHASE_KEY
from rizemind.split_learning.seeding import seed_everything
from torch.utils.data import DataLoader

from .task import (
    CachedTrainPartition,
    build_split_models,
    deserialize_ndarrays_from_bytes,
    get_dataset_spec,
    get_weights,
    load_partition_data,
    partition_config_from_run_config,
    set_weights,
)

# context.state keys
_SL_HEAD_KEY = "sl_head"
_SL_INPUT_KEY = "sl_input_x"
_SL_LABELS_KEY = "sl_labels_y"
_SL_TRAIN_STATE = "sl_train_state"

_DEMO_TAG = "[SL-DEMO]"


def _shape(arr) -> str:
    dims = list(arr.shape)
    inner = ", ".join(str(d) for d in dims)
    return f"({inner},)" if len(dims) == 1 else f"({inner})"


class SplitFlowerClient(NumPyClient):
    """NumPyClient for split learning on MNIST.

    Uses real MNIST data loaded from a Flower Datasets partition.
    Tracks training progress (batch index, epoch) in ``context.state``
    so that consecutive Flower rounds advance through the dataset.
    """

    def __init__(
        self,
        head: nn.Module,
        tail: nn.Module,
        train_partition: CachedTrainPartition,
        valloader: DataLoader,
        learning_rate: float,
        ctx: Context,
        demo: bool = False,
    ) -> None:
        self.head = head
        self.tail = tail  # held for local evaluate(); weights from server
        self.train_partition = train_partition
        self.valloader = valloader
        # No momentum: a fresh client (and optimizer) is built each round and
        # momentum buffers are not persisted in context.state, so any momentum
        # would silently reset every step. Use plain SGD to avoid implying state
        # that isn't carried across rounds.
        self.optimizer = optim.SGD(head.parameters(), lr=learning_rate, momentum=0.0)
        self.criterion = nn.CrossEntropyLoss()
        self._ctx = ctx
        self._demo = demo
        self._node_short = str(ctx.node_id)[-4:]

        # Restore or initialise the training cursor
        if _SL_TRAIN_STATE not in ctx.state:
            ctx.state[_SL_TRAIN_STATE] = ConfigRecord({"batch_idx": 0, "epoch": 0})
        self._batch_idx: int = int(ctx.state[_SL_TRAIN_STATE]["batch_idx"])
        self._epoch: int = int(ctx.state[_SL_TRAIN_STATE]["epoch"])

    # ------------------------------------------------------------------
    # Demo logging
    # ------------------------------------------------------------------

    def _demo_log(self, phase: str, msg: str) -> None:
        if self._demo:
            log(
                INFO,
                "%s CLIENT node-%-4s  %s | %s",
                _DEMO_TAG,
                self._node_short,
                phase,
                msg,
            )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return the (x, y) batch at self._batch_idx, then advance cursor."""
        epoch = self._epoch
        batch_idx = self._batch_idx
        x, y = self.train_partition.get_batch(batch_idx=batch_idx, epoch=epoch)

        # Advance cursor
        self._batch_idx += 1
        if self._batch_idx >= self.train_partition.num_batches:
            self._batch_idx = 0
            self._epoch += 1
        self._ctx.state[_SL_TRAIN_STATE]["batch_idx"] = self._batch_idx
        self._ctx.state[_SL_TRAIN_STATE]["epoch"] = self._epoch

        return x, y, epoch, batch_idx

    # ------------------------------------------------------------------
    # Flower fit dispatch
    # ------------------------------------------------------------------

    def fit(self, parameters, config):
        """Dispatch to forward or backward pass based on sl_phase."""
        if config.get(SL_PHASE_KEY) != SL_PHASE_BACKWARD:
            return self._forward(parameters)
        return self._backward(parameters)

    # ------------------------------------------------------------------
    # Forward round
    # ------------------------------------------------------------------

    def _forward(self, _parameters):
        """Fetch the next real data batch, run forward, return activations+labels."""
        x, y, epoch, batch_idx = self._get_batch()

        # Persist batch for the backward round (client instance is recreated).
        self._ctx.state[_SL_INPUT_KEY] = ArrayRecord(numpy_ndarrays=[x.numpy()])
        self._ctx.state[_SL_LABELS_KEY] = ArrayRecord(numpy_ndarrays=[y.numpy()])
        self._ctx.state[_SL_HEAD_KEY] = ArrayRecord(
            numpy_ndarrays=get_weights(self.head)
        )

        self.optimizer.zero_grad()
        activation = self.head(x)

        self._demo_log(
            "FORWARD ",
            f"epoch={epoch} batch={batch_idx}  "
            f"{_shape(x)} -> activation {_shape(activation)}, labels {_shape(y)}",
        )
        self._demo_log("FORWARD ", "activation + labels sent to server")

        return (
            [activation.detach().numpy(), y.numpy()],
            x.shape[0],
            {
                "epoch": float(self._epoch),
                "batch_idx": float(self._batch_idx),
            },
        )

    # ------------------------------------------------------------------
    # Backward round
    # ------------------------------------------------------------------

    def _backward(self, parameters):
        """Restore saved state, redo forward, apply server gradient."""
        if _SL_INPUT_KEY not in self._ctx.state or len(parameters) == 0:
            return get_weights(self.head), 0, {}

        x = torch.from_numpy(self._ctx.state[_SL_INPUT_KEY].to_numpy_ndarrays()[0])
        saved_weights = self._ctx.state[_SL_HEAD_KEY].to_numpy_ndarrays()
        del self._ctx.state[_SL_INPUT_KEY]
        del self._ctx.state[_SL_LABELS_KEY]
        del self._ctx.state[_SL_HEAD_KEY]

        self._demo_log("BACKWARD", "state restored from context.state")

        set_weights(self.head, saved_weights)
        self.optimizer.zero_grad()
        activation = self.head(x)

        grad = torch.from_numpy(parameters[0])
        activation.backward(grad)
        self.optimizer.step()

        updated_weights = get_weights(self.head)
        self._ctx.state[_SL_HEAD_KEY] = ArrayRecord(numpy_ndarrays=updated_weights)

        self._demo_log(
            "BACKWARD", f"gradient {_shape(grad)} applied, head updated + saved"
        )

        return (
            updated_weights,
            x.shape[0],
            {
                "epoch": float(self._epoch),
                "batch_idx": float(self._batch_idx),
            },
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, parameters, config):
        """Evaluate with the current global head and server-provided tail."""
        tail_payload = config.get("sl_tail_weights")
        if not isinstance(tail_payload, bytes):
            return 0.0, 0, {}

        head_snapshot = get_weights(self.head)
        tail_snapshot = get_weights(self.tail)
        head_mode = self.head.training
        tail_mode = self.tail.training

        try:
            set_weights(self.head, parameters)
            set_weights(self.tail, deserialize_ndarrays_from_bytes(tail_payload))
            self.head.eval()
            self.tail.eval()

            total = correct = 0
            loss_sum = 0.0
            with torch.no_grad():
                for batch in self.valloader:
                    x = batch["image"]
                    y = batch["label"]
                    logits = self.tail(self.head(x))
                    loss_sum += self.criterion(logits, y).item() * len(y)
                    correct += (logits.argmax(1) == y).sum().item()
                    total += len(y)
        finally:
            set_weights(self.head, head_snapshot)
            set_weights(self.tail, tail_snapshot)
            self.head.train(head_mode)
            self.tail.train(tail_mode)

        avg_loss = loss_sum / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, total, {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# client_fn
# ---------------------------------------------------------------------------


def client_fn(context: Context):
    """Construct a SplitFlowerClient from Flower context."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    val_ratio = float(context.run_config.get("val-ratio", 0.1))
    learning_rate = float(context.run_config["learning-rate"])
    hidden_dim = int(context.run_config["hidden-dim"])
    demo = bool(context.run_config.get("demo", False))
    dataset = str(context.run_config.get("dataset", "mnist"))
    max_train_samples = int(context.run_config.get("max-train-samples", 0))
    seed = int(context.run_config.get("seed", 42))
    spec = get_dataset_spec(dataset)
    partition_config = partition_config_from_run_config(context.run_config)

    # First-class reproducibility: seed head init per client (seed + pid).
    seed_everything(seed + partition_id)

    head, tail = build_split_models(spec, hidden_dim=hidden_dim)

    # Restore persisted head weights from previous round.
    if _SL_HEAD_KEY in context.state:
        set_weights(head, context.state[_SL_HEAD_KEY].to_numpy_ndarrays())

    train_partition, valloader = load_partition_data(
        partition_id,
        num_partitions,
        batch_size,
        spec=spec,
        partition_config=partition_config,
        val_ratio=val_ratio,
        max_train_samples=max_train_samples,
    )

    return SplitFlowerClient(
        head, tail, train_partition, valloader, learning_rate, context, demo
    ).to_client()


from rizemind.split_learning.mod import split_learning_mod  # noqa: E402

app = ClientApp(client_fn, mods=[split_learning_mod])
