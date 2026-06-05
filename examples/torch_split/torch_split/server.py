"""Server-side split-learning logic for the torch_split example.

The server owns the ServerTail (layers after the cut point).

server_backward_fn is wired into SplitLearningStrategy so that after every
forward round it:
  1. Unpacks each client's activation and real batch labels.
  2. Runs the server-side forward+backward pass.
  3. Accumulates per-round train loss for metric reporting.
  4. Returns per-client gradients at the cut point.

Evaluation
----------
FedAvg.evaluate_fn runs every round using the current global parameters
(= averaged client head weights published after each backward round) and the
server's tail model.  It evaluates accuracy and loss on the MNIST test set.

Distributed evaluation is also enabled: the server serializes the current tail
weights into ``EvaluateIns.config`` so clients can compute validation accuracy
on their own held-out partition without changing the training flow.
"""

from __future__ import annotations

import json
from logging import INFO
from pathlib import Path

import torch
import torch.nn as nn
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import Parameters, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.serialization import tensor_to_parameters
from rizemind.split_learning.strategy import SplitLearningStrategy

from .task import (
    build_split_models,
    extract_activation_and_labels,
    get_dataset_spec,
    get_train_batch_counts,
    get_weights,
    make_server_eval_loader,
    num_server_rounds_for_target_epochs,
    partition_config_from_run_config,
    serialize_ndarrays_to_bytes,
    set_weights,
)

_DEMO_TAG = "[SL-DEMO]"


def _shape(t) -> str:
    dims = list(t.shape)
    inner = ", ".join(str(d) for d in dims)
    return f"({inner},)" if len(dims) == 1 else f"({inner})"


def _parameters_match_model(parameters, model: nn.Module) -> bool:
    """Return True when ndarray parameter shapes match a torch model."""
    if isinstance(parameters, Parameters):
        parameters = parameters_to_ndarrays(parameters)
    expected = [tuple(param.shape) for param in model.parameters()]
    actual = [tuple(array.shape) for array in parameters]
    return expected == actual


class HeadShapeAwareFedAvg(FedAvg):
    """FedAvg that gates distributed evaluation.

    Client-side evaluation is skipped (1) until the global parameters match the
    client head shape, and (2) on rounds excluded by ``should_evaluate_round``
    (used to throttle evaluation frequency).
    """

    def __init__(self, should_evaluate, should_evaluate_round, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.should_evaluate = should_evaluate
        self.should_evaluate_round = should_evaluate_round

    def configure_evaluate(
        self,
        server_round: int,
        parameters,
        client_manager: ClientManager,
    ):
        if not self.should_evaluate(parameters):
            log(
                INFO,
                "configure_evaluate: current parameters do not match client head, "
                "skipping distributed evaluation",
            )
            return []
        if not self.should_evaluate_round(server_round):
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------


def weighted_avg(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Weighted average of a scalar metric across clients."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {
        key: sum(n * float(m[key]) for n, m in metrics if key in m) / total
        for key in metrics[0][1]
    }


# ---------------------------------------------------------------------------
# Server backward function factory
# ---------------------------------------------------------------------------


def make_server_backward_fn(
    tail: nn.Module,
    learning_rate: float,
    demo: bool,
    round_losses: list[float],
):
    """Return a server_backward_fn closure.

    The closure accumulates the mean per-round train loss into *round_losses*
    so that ``evaluate_fn`` can report it to Flower.

    Args:
        tail: Server-side model.
        learning_rate: Tail optimizer learning rate.
        demo: Emit educational log lines when True.
        round_losses: Shared mutable list; each call appends a mean loss value.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(tail.parameters(), lr=learning_rate, momentum=0.9)

    def server_backward_fn(
        activation_store: dict[str, Parameters],
    ) -> dict[str, Parameters]:
        gradients: dict[str, Parameters] = {}
        step_losses: list[float] = []

        for cid, act_params in activation_store.items():
            activation, labels = extract_activation_and_labels(act_params)
            short_cid = str(cid)[-4:]

            if demo:
                log(
                    INFO,
                    "%s SERVER         BACKWARD | client ...%-4s  activation %s + labels %s",
                    _DEMO_TAG,
                    short_cid,
                    _shape(activation),
                    _shape(labels),
                )

            optimizer.zero_grad()
            logits = tail(activation)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            step_losses.append(loss.item())
            assert activation.grad is not None

            if demo:
                log(
                    INFO,
                    "%s SERVER         BACKWARD | client ...%-4s  loss=%.4f  gradient %s ready",
                    _DEMO_TAG,
                    short_cid,
                    loss.item(),
                    _shape(activation.grad),
                )

            gradients[cid] = tensor_to_parameters(activation.grad)

        if step_losses:
            round_losses.append(sum(step_losses) / len(step_losses))

        return gradients

    return server_backward_fn


# ---------------------------------------------------------------------------
# Server function
# ---------------------------------------------------------------------------


def server_fn(context: Context):
    """Build the ServerApp components for split-learning training."""
    cut_layer = int(context.run_config["cut-layer"])
    hidden_dim = int(context.run_config["hidden-dim"])
    learning_rate = float(context.run_config["learning-rate"])
    min_clients = int(context.run_config["min-available-clients"])
    batch_size = int(context.run_config["batch-size"])
    val_ratio = float(context.run_config.get("val-ratio", 0.1))
    target_epochs = float(context.run_config.get("target-epochs", 0.0))
    demo = bool(context.run_config.get("demo", False))
    dataset = str(context.run_config.get("dataset", "mnist"))
    max_train_samples = int(context.run_config.get("max-train-samples", 0))
    results_path = str(context.run_config.get("results-path", "")).strip()
    eval_every = max(1, int(context.run_config.get("eval-every", 1)))
    eval_max_samples = int(context.run_config.get("eval-max-samples", 0))
    spec = get_dataset_spec(dataset)
    partition_config = partition_config_from_run_config(context.run_config)

    sl_config = SplitLearningConfig(cut_layer=cut_layer)
    num_rounds = int(context.run_config["num-server-rounds"])

    if target_epochs > 0.0:
        train_batch_counts = get_train_batch_counts(
            num_partitions=min_clients,
            batch_size=batch_size,
            spec=spec,
            partition_config=partition_config,
            val_ratio=val_ratio,
            max_train_samples=max_train_samples,
        )
        num_rounds = num_server_rounds_for_target_epochs(
            target_epochs=target_epochs,
            train_batch_counts=train_batch_counts,
            num_rounds_per_step=sl_config.num_rounds_per_step,
        )
        log(
            INFO,
            "server_fn: target_epochs=%.4f -> num_server_rounds=%d "
            "(max_train_batches=%d, partitioner=%s)",
            target_epochs,
            num_rounds,
            max(train_batch_counts),
            partition_config.kind,
        )

    eval_head, tail = build_split_models(spec, hidden_dim=hidden_dim)
    initial_parameters = ndarrays_to_parameters(get_weights(tail))

    # Shared list that server_backward_fn appends per-round loss to.
    round_losses: list[float] = []

    # Load server-side evaluation data once.
    eval_loader = make_server_eval_loader(batch_size, spec, max_samples=eval_max_samples)
    eval_criterion = nn.CrossEntropyLoss()

    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text("")

    def should_evaluate_round(server_round: int) -> bool:
        """Throttle evaluation: every ``eval_every`` rounds, plus the last one."""
        if eval_every <= 1:
            return True
        return server_round % eval_every == 0 or server_round >= num_rounds

    def evaluate_fn(
        server_round: int,
        parameters,
        config: dict[str, Scalar],
    ):
        """Evaluate averaged head + server tail on the MNIST test set.

        ``parameters`` = current averaged client head weights (published by
        FedAvg after each backward round).  In the first few rounds Flower
        passes the initial tail weights instead; those have the wrong shape
        for ClientHead so we skip evaluation until the shapes match.
        """
        # Guard: parameters may carry tail weights (wrong shape) in round 0.
        if not _parameters_match_model(parameters, eval_head):
            return None
        if not should_evaluate_round(server_round):
            return None

        set_weights(eval_head, parameters)
        eval_head.eval()
        tail.eval()

        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                x = batch["image"]
                y = batch["label"]
                activation = eval_head(x)
                logits = tail(activation)
                loss_sum += eval_criterion(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)

        eval_head.train()
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

    def on_evaluate_config_fn(server_round: int) -> dict[str, Scalar]:
        del server_round
        return {"sl_tail_weights": serialize_ndarrays_to_bytes(get_weights(tail))}

    # Use FedAvg as base strategy with server-side evaluate_fn.
    # evaluate_fn provides centralized val_loss + val_accuracy every round.
    # on_evaluate_config_fn additionally ships tail weights so clients can
    # report validation accuracy on their local held-out partition.
    base_strategy = HeadShapeAwareFedAvg(
        should_evaluate=lambda parameters: _parameters_match_model(parameters, eval_head),
        should_evaluate_round=should_evaluate_round,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        fit_metrics_aggregation_fn=weighted_avg,
        evaluate_metrics_aggregation_fn=weighted_avg,
    )

    strategy = SplitLearningStrategy(
        strategy=base_strategy,
        config=sl_config,
        server_backward_fn=make_server_backward_fn(
            tail, learning_rate, demo, round_losses
        ),
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
