"""Server-side vertical split-learning logic.

The server owns the labels and the classifier tail. Its job each step is:

1. Receive ordered activations ``[(partition_id, cid, params), ...]`` from
   :class:`~rizemind.split_learning.vertical_strategy.VerticalSplitLearningStrategy`.
2. Concatenate the K activations (sorted by ``partition_id`` for determinism).
3. Pull the matching label batch from the shared deterministic permutation.
4. Run the tail forward, compute cross-entropy loss, backprop to obtain the
   per-client gradient slice at the cut, dispatch them back.

Centralized evaluation
----------------------
After each backward round the strategy caches every client's latest bottom
weights. The server's ``on_evaluate`` callback rebuilds clones of each
bottom model, loads the cached weights, and runs the full pipeline
(bottom_0, ..., bottom_{K-1}, tail) on the held-out MNIST test split.
"""

from __future__ import annotations

import json
from logging import INFO
from pathlib import Path

import torch
import torch.nn as nn
from flwr.common import Context, Parameters
from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.serialization import tensor_to_parameters
from rizemind.split_learning.vertical_strategy import (
    OrderedActivations,
    VerticalSplitLearningStrategy,
)

from .task import (
    DatasetSpec,
    VerticalPartition,
    build_bottom_model,
    build_server_top,
    extract_activation,
    get_dataset_spec,
    make_server_train_labels,
    party_test_features,
    set_weights,
    test_labels,
)

_DEMO_TAG = "[VSL-DEMO]"


def _shape(t) -> str:
    dims = list(t.shape)
    inner = ", ".join(str(d) for d in dims)
    return f"({inner},)" if len(dims) == 1 else f"({inner})"


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def make_on_train_step(
    tail: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    labels_partition: VerticalPartition,
    round_losses: list[float],
    demo: bool,
    log_shapes_steps: int = 1,
):
    """Return an ``on_train_step`` closure for the vertical strategy.

    The closure runs one tail forward+backward pass per call and returns
    per-client gradients keyed by ``cid``.

    Args:
        tail: Server tail model.
        optimizer: Optimizer for the tail's parameters.
        criterion: Loss function (cross-entropy).
        labels_partition: Server's label-only :class:`VerticalPartition`
            (same shuffle seed as the clients).
        round_losses: Shared list appended with each step's mean loss for
            metric reporting.
        demo: When True, log a one-line educational trace per step.
        log_shapes_steps: For the first N steps, always log per-partition
            activation / gradient shapes (independent of ``demo``). Keeps the
            run inspectable without spamming long runs.
    """

    def on_train_step(
        sl_step: int, ordered: OrderedActivations
    ) -> tuple[dict[str, Parameters], float]:
        labels = labels_partition.get_batch_for_step(sl_step).long()

        activations: list[torch.Tensor] = []
        cids: list[str] = []
        for pid, cid, params in ordered:
            act = extract_activation(params)
            activations.append(act)
            cids.append(cid)
            if demo:
                log(
                    INFO,
                    "%s SERVER  sl_step=%d  pid=%d cid=%s activation %s",
                    _DEMO_TAG,
                    sl_step,
                    pid,
                    cid[-4:],
                    _shape(act),
                )

        joint = torch.cat(activations, dim=1)
        optimizer.zero_grad()
        logits = tail(joint)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        grad_store: dict[str, Parameters] = {}
        for cid, act in zip(cids, activations):
            assert act.grad is not None
            grad_store[cid] = tensor_to_parameters(act.grad)

        if sl_step < log_shapes_steps:
            shape_summary = ", ".join(
                f"pid={pid}:{_shape(a)}" for (pid, _, _), a in zip(ordered, activations)
            )
            log(
                INFO,
                "on_train_step sl_step=%d K=%d  joint=%s  logits=%s  "
                "loss=%.4f  acts={%s}",
                sl_step,
                len(activations),
                _shape(joint),
                _shape(logits),
                loss.item(),
                shape_summary,
            )

        if demo:
            log(
                INFO,
                "%s SERVER  sl_step=%d  loss=%.4f  dispatched %d gradient(s)",
                _DEMO_TAG,
                sl_step,
                loss.item(),
                len(grad_store),
            )

        round_losses.append(loss.item())
        return grad_store, loss.item()

    return on_train_step


# ---------------------------------------------------------------------------
# Centralized evaluation
# ---------------------------------------------------------------------------


def make_on_evaluate(
    tail: nn.Module,
    spec: DatasetSpec,
    num_clients: int,
    hidden_dim: int,
    batch_size: int,
    round_losses: list[float],
    eval_every: int,
    eval_max_samples: int,
    num_rounds: int,
    sl_config: SplitLearningConfig,
    results_path: str,
):
    """Build the strategy's centralized ``on_evaluate`` callback.

    The callback runs the full vertical pipeline (each cached bottom model on
    its strip, then the server tail on the concatenation) over the MNIST test
    split.
    """
    cap = eval_max_samples if eval_max_samples and eval_max_samples > 0 else None
    labels = test_labels(spec, cap)

    # Pre-slice the test features per party once (image strip or tabular cols).
    strip_views: list[torch.Tensor] = [
        party_test_features(spec, pid, num_clients, cap) for pid in range(num_clients)
    ]

    eval_criterion = nn.CrossEntropyLoss()

    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text("")

    def should_evaluate_round(server_round: int) -> bool:
        if eval_every <= 1:
            return True
        return server_round % eval_every == 0 or server_round >= num_rounds

    def on_evaluate(
        server_round: int,
        bottom_weights_by_pid: dict,
    ) -> tuple[float, dict[str, Scalar]] | None:
        if not should_evaluate_round(server_round):
            return None

        bottoms = []
        for pid in range(num_clients):
            if pid not in bottom_weights_by_pid:
                return None
            m = build_bottom_model(spec, pid, num_clients, hidden_dim)
            set_weights(m, bottom_weights_by_pid[pid])
            m.eval()
            bottoms.append(m)
        tail.eval()

        total = correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                stop = min(start + batch_size, len(labels))
                acts = [b(strip_views[i][start:stop]) for i, b in enumerate(bottoms)]
                joint = torch.cat(acts, dim=1)
                logits = tail(joint)
                y = labels[start:stop]
                loss_sum += eval_criterion(logits, y).item() * (stop - start)
                correct += (logits.argmax(1) == y).sum().item()
                total += stop - start

        tail.train()

        accuracy = correct / max(total, 1)
        avg_loss = loss_sum / max(total, 1)
        latest_train_loss = round_losses[-1] if round_losses else float("nan")

        log(
            INFO,
            "evaluate  round=%d  val_loss=%.4f  val_acc=%.4f  train_loss=%.4f",
            server_round,
            avg_loss,
            accuracy,
            latest_train_loss,
        )

        if results_path:
            record = {
                "round": server_round,
                "step": server_round // sl_config.num_rounds_per_step,
                "val_loss": avg_loss,
                "val_accuracy": accuracy,
                "train_loss": latest_train_loss,
            }
            with Path(results_path).open("a") as fh:
                fh.write(json.dumps(record) + "\n")

        return avg_loss, {
            "val_accuracy": accuracy,
            "val_loss": avg_loss,
            "train_loss": latest_train_loss,
        }

    return on_evaluate


# ---------------------------------------------------------------------------
# server_fn
# ---------------------------------------------------------------------------


def server_fn(context: Context):
    """Build the ServerApp components for vertical split federated learning."""
    num_clients = int(context.run_config["min-available-clients"])
    batch_size = int(context.run_config["batch-size"])
    learning_rate = float(context.run_config["learning-rate"])
    hidden_dim = int(context.run_config["hidden-dim"])
    num_rounds = int(context.run_config["num-server-rounds"])
    dataset = str(context.run_config.get("dataset", "mnist"))
    max_train_samples = int(context.run_config.get("max-train-samples", 0))
    eval_every = max(1, int(context.run_config.get("eval-every", 1)))
    eval_max_samples = int(context.run_config.get("eval-max-samples", 0))
    results_path = str(context.run_config.get("results-path", "")).strip()
    demo = bool(context.run_config.get("demo", False))

    spec = get_dataset_spec(dataset)
    sl_config = SplitLearningConfig(cut_layer=0)

    tail = build_server_top(spec, num_clients=num_clients, hidden_dim=hidden_dim)
    optimizer = torch.optim.SGD(tail.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    labels_partition = make_server_train_labels(
        spec=spec,
        batch_size=batch_size,
        max_train_samples=max_train_samples or None,
    )

    round_losses: list[float] = []

    on_train_step = make_on_train_step(
        tail=tail,
        optimizer=optimizer,
        criterion=criterion,
        labels_partition=labels_partition,
        round_losses=round_losses,
        demo=demo,
    )

    on_evaluate = make_on_evaluate(
        tail=tail,
        spec=spec,
        num_clients=num_clients,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        round_losses=round_losses,
        eval_every=eval_every,
        eval_max_samples=eval_max_samples,
        num_rounds=num_rounds,
        sl_config=sl_config,
        results_path=results_path,
    )

    strategy = VerticalSplitLearningStrategy(
        config=sl_config,
        num_clients=num_clients,
        on_train_step=on_train_step,
        on_evaluate=on_evaluate,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
