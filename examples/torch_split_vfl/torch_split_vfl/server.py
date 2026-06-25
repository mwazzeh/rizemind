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
import time
from logging import INFO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flwr.common import Context, Parameters
from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.gradient_privacy import GradientPrivacyConfig
from rizemind.split_learning.label_private_strategy import LabelPrivateVerticalStrategy
from rizemind.split_learning.metrics import classification_metrics
from rizemind.split_learning.seeding import seed_everything
from rizemind.split_learning.serialization import tensor_to_parameters
from rizemind.split_learning.telemetry import RunTelemetry, StepTelemetry, timed
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
    num_configured_parties,
    parse_active_parties,
    party_feature_dim,
    party_test_features,
    resolve_active_groups,
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
    telemetry: RunTelemetry | None = None,
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
        step_tel = StepTelemetry(sl_step=sl_step)
        labels = labels_partition.get_batch_for_step(sl_step).long()

        activations: list[torch.Tensor] = []
        cids: list[str] = []
        pids: list[int] = []
        for pid, cid, params in ordered:
            act = extract_activation(params)
            activations.append(act)
            cids.append(cid)
            pids.append(pid)
            step_tel.add_activation(pid, act)  # activation bytes client->server
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

        with timed(step_tel.timings_s, "server_forward"):
            joint = torch.cat(activations, dim=1)
            optimizer.zero_grad()
            logits = tail(joint)
            loss = criterion(logits, labels)
        with timed(step_tel.timings_s, "server_backward"):
            loss.backward()
            optimizer.step()

        grad_store: dict[str, Parameters] = {}
        for pid, cid, act in zip(pids, cids, activations):
            assert act.grad is not None
            grad_store[cid] = tensor_to_parameters(act.grad)
            step_tel.add_gradient(pid, act.grad)  # gradient bytes server->client

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
        if telemetry is not None:
            telemetry.record_step(step_tel)
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
    active_parties: tuple[int, ...] | None = None,
    telemetry: RunTelemetry | None = None,
    schema_extra: dict | None = None,
):
    """Build the strategy's centralized ``on_evaluate`` callback.

    Runs the full vertical pipeline (each cached bottom model on its feature
    slice, then the server tail on the concatenation) over the held-out test
    split and computes richer classification metrics **from the actual VFL
    model's predictions** (accuracy, loss, precision/recall/F1, balanced
    accuracy, confusion counts, and ROC-AUC for binary tasks).

    On the final round it also writes a structured ``*.summary.json`` next to
    ``results_path`` containing the run schema, final/best metrics, and
    communication/timing telemetry.
    """
    cap = eval_max_samples if eval_max_samples and eval_max_samples > 0 else None
    labels = test_labels(spec, cap)

    # Pre-slice the test features per party once (image strip or tabular cols).
    strip_views: list[torch.Tensor] = [
        party_test_features(spec, pid, num_clients, cap, active_parties)
        for pid in range(num_clients)
    ]

    eval_criterion = nn.CrossEntropyLoss()
    eval_acc_history: list[float] = []
    best_holder = {"accuracy": float("-inf"), "loss": float("inf")}

    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text("")

    def should_evaluate_round(server_round: int) -> bool:
        if eval_every <= 1:
            return True
        return server_round % eval_every == 0 or server_round >= num_rounds

    def _write_summary(final_metrics: dict) -> None:
        if not results_path:
            return
        final5 = (
            sum(eval_acc_history[-5:]) / len(eval_acc_history[-5:])
            if eval_acc_history
            else float("nan")
        )
        summary = {
            "schema_version": 2,
            **(schema_extra or {}),
            "num_eval_points": len(eval_acc_history),
            "final_accuracy": eval_acc_history[-1]
            if eval_acc_history
            else float("nan"),
            "best_accuracy": best_holder["accuracy"],
            "final5_mean_accuracy": final5,
            "best_loss": best_holder["loss"],
            "final_metrics": final_metrics,
        }
        if "_run_start" in summary:
            summary["wall_clock_s"] = round(
                time.perf_counter() - summary.pop("_run_start"), 2
            )
        if telemetry is not None:
            summary["communication"] = telemetry.as_dict()
        Path(results_path).with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2)
        )

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
            m = build_bottom_model(spec, pid, num_clients, hidden_dim, active_parties)
            set_weights(m, bottom_weights_by_pid[pid])
            m.eval()
            bottoms.append(m)
        tail.eval()

        eval_start = time.perf_counter()
        total = 0
        loss_sum = 0.0
        preds: list[np.ndarray] = []
        pos_scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                stop = min(start + batch_size, len(labels))
                acts = [b(strip_views[i][start:stop]) for i, b in enumerate(bottoms)]
                joint = torch.cat(acts, dim=1)
                logits = tail(joint)
                y = labels[start:stop]
                loss_sum += eval_criterion(logits, y).item() * (stop - start)
                preds.append(logits.argmax(1).cpu().numpy())
                if spec.num_classes == 2:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    pos_scores.append(probs.cpu().numpy())
                total += stop - start

        tail.train()

        y_true = labels.cpu().numpy()
        y_pred = np.concatenate(preds) if preds else np.array([], dtype=int)
        y_score = np.concatenate(pos_scores) if pos_scores else None
        metrics = classification_metrics(
            y_true, y_pred, y_score, num_classes=spec.num_classes
        )
        accuracy = metrics["accuracy"]
        avg_loss = loss_sum / max(total, 1)
        latest_train_loss = round_losses[-1] if round_losses else float("nan")
        eval_time = time.perf_counter() - eval_start
        if telemetry is not None:
            telemetry.timings_s["eval"] = (
                telemetry.timings_s.get("eval", 0.0) + eval_time
            )

        eval_acc_history.append(accuracy)
        best_holder["accuracy"] = max(best_holder["accuracy"], accuracy)
        best_holder["loss"] = min(best_holder["loss"], avg_loss)

        log(
            INFO,
            "evaluate  round=%d  val_loss=%.4f  val_acc=%.4f  f1=%.4f  "
            "bal_acc=%.4f  train_loss=%.4f",
            server_round,
            avg_loss,
            accuracy,
            metrics.get("f1", metrics.get("f1_macro", float("nan"))),
            metrics["balanced_accuracy"],
            latest_train_loss,
        )

        if results_path:
            record = {
                "round": server_round,
                "step": server_round // sl_config.num_rounds_per_step,
                "val_loss": avg_loss,
                "val_accuracy": accuracy,
                "train_loss": latest_train_loss,
                # richer metrics from the actual VFL predictions:
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "n_samples": metrics["n_samples"],
            }
            if spec.num_classes == 2:
                record.update(
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1=metrics["f1"],
                    roc_auc=metrics["roc_auc"],
                )
            with Path(results_path).open("a") as fh:
                fh.write(json.dumps(record) + "\n")

        if server_round >= num_rounds:
            _write_summary(metrics)

        return avg_loss, {
            "val_accuracy": accuracy,
            "val_loss": avg_loss,
            "train_loss": latest_train_loss,
        }

    return on_evaluate


# ---------------------------------------------------------------------------
# server_fn
# ---------------------------------------------------------------------------


def make_label_private_eval(
    spec: DatasetSpec,
    num_clients: int,
    hidden_dim: int,
    active_parties: tuple[int, ...] | None,
    eval_every: int,
    eval_max_samples: int,
    num_rounds: int,
    rounds_per_step: int,
    results_path: str,
    telemetry: RunTelemetry,
    schema_extra: dict,
):
    """Build the coordinator-side eval-activation + metric-writer for LP mode.

    ``build_test_joint`` turns cached bottom weights + test **features** (never
    labels) into the joint test activation shipped to the label holder.
    ``on_metrics`` writes the aggregate scalar metrics the holder returns (no
    labels, no per-sample predictions).
    """
    cap = eval_max_samples if eval_max_samples and eval_max_samples > 0 else None
    strip_views = [
        party_test_features(spec, pid, num_clients, cap, active_parties)
        for pid in range(num_clients)
    ]
    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text("")
    history: list[float] = []
    best = {"accuracy": float("-inf"), "loss": float("inf")}
    num_steps = max(1, num_rounds // rounds_per_step)
    # eval_every is expressed in Flower rounds; convert to SL steps and align to
    # COMPUTE rounds (which occur at round = rounds_per_step*step + 2).
    eval_every_steps = max(1, eval_every // rounds_per_step)

    def _step_of(server_round: int) -> int:
        return (server_round - 2) // rounds_per_step

    def should_eval(server_round: int) -> bool:
        step = _step_of(server_round)
        return step % eval_every_steps == 0 or step >= num_steps - 1

    def is_last_eval(server_round: int) -> bool:
        return _step_of(server_round) >= num_steps - 1

    def build_test_joint(server_round, bottom_weights_by_pid):
        if not should_eval(server_round):
            return None
        acts = []
        for pid in range(num_clients):
            if pid not in bottom_weights_by_pid:
                return None
            m = build_bottom_model(spec, pid, num_clients, hidden_dim, active_parties)
            set_weights(m, bottom_weights_by_pid[pid])
            m.eval()
            with torch.no_grad():
                acts.append(m(strip_views[pid]).numpy())
        return np.concatenate(acts, axis=1)  # (M, K*hidden) — NO labels

    def write_summary(metrics: dict) -> None:
        if not results_path:
            return
        final5 = sum(history[-5:]) / len(history[-5:]) if history else float("nan")
        summary = {
            "schema_version": 2,
            **schema_extra,
            "num_eval_points": len(history),
            "final_accuracy": history[-1] if history else float("nan"),
            "best_accuracy": best["accuracy"],
            "final5_mean_accuracy": final5,
            "best_loss": best["loss"],
            "final_metrics": {
                k: v for k, v in metrics.items() if k != "confusion_json"
            },
            "final_confusion": json.loads(metrics.get("confusion_json", "[]")),
        }
        if "_run_start" in summary:
            summary["wall_clock_s"] = round(
                time.perf_counter() - summary.pop("_run_start"), 2
            )
        summary["communication"] = telemetry.as_dict()
        Path(results_path).with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2)
        )

    def on_metrics(server_round: int, metrics: dict) -> None:
        acc = metrics.get("val_accuracy")
        val_loss = metrics.get("val_loss")
        if acc is None:
            return
        history.append(float(acc))
        best["accuracy"] = max(best["accuracy"], float(acc))
        if val_loss is not None:
            best["loss"] = min(best["loss"], float(val_loss))
        log(
            INFO,
            "evaluate(label-private) round=%d val_acc=%.4f val_loss=%.4f f1=%.4f",
            server_round,
            float(acc),
            float(val_loss),
            float(metrics.get("f1", metrics.get("f1_macro", float("nan")))),
        )
        if results_path:
            record = {
                "round": server_round,
                "step": server_round // rounds_per_step,
                "val_loss": val_loss,
                "val_accuracy": acc,
                "train_loss": metrics.get("train_loss"),
                "precision_macro": metrics.get("precision_macro"),
                "recall_macro": metrics.get("recall_macro"),
                "f1_macro": metrics.get("f1_macro"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "n_samples": metrics.get("n_samples"),
            }
            if spec.num_classes == 2:
                record.update(
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    f1=metrics.get("f1"),
                    roc_auc=metrics.get("roc_auc"),
                )
            with Path(results_path).open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        if is_last_eval(server_round):
            write_summary(metrics)

    return build_test_joint, on_metrics


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
    seed = int(context.run_config.get("seed", 42))
    active_parties = parse_active_parties(context.run_config.get("active-parties", ""))

    label_private = bool(context.run_config.get("label-private", False))
    label_holder_pid = int(context.run_config.get("label-holder-party", 0))

    spec = get_dataset_spec(dataset)
    # Validate the active-party selection up front (clear failure on bad input).
    active_groups = resolve_active_groups(spec, num_clients, active_parties)
    sl_config = SplitLearningConfig(cut_layer=0)

    # First-class reproducibility: seed the server's tail initialisation.
    seed_everything(seed)

    # ------------------------------------------------------------------
    # Label-private mode: labels + top model live ONLY at the label holder.
    # The coordinator never loads labels here.
    # ------------------------------------------------------------------
    if label_private:
        if not 0 <= label_holder_pid < num_clients:
            raise ValueError(
                f"label-holder-party must be an active party in "
                f"[0, {num_clients}), got {label_holder_pid}"
            )
        telemetry = RunTelemetry()
        rounds_per_step = 3  # collect / compute / distribute
        party_dims = [
            party_feature_dim(spec, pid, num_clients, active_parties)
            for pid in range(num_clients)
        ]
        # Mirror the holder's gradient-privacy config into the schema (the server
        # never applies it — the holder does — but it records the parameters).
        gp_cfg = GradientPrivacyConfig(
            mode=str(context.run_config.get("gradient-privacy-mode", "none"))
            .strip()
            .lower(),
            clip_norm=float(context.run_config.get("gradient-clip-norm", 1.0)),
            noise_multiplier=float(
                context.run_config.get("gradient-noise-multiplier", 0.0)
            ),
            delta=float(context.run_config.get("privacy-delta", 1e-5)),
            rng_mode=str(context.run_config.get("privacy-rng-mode", "research-seeded"))
            .strip()
            .lower(),
        )
        num_steps_lp = num_rounds // rounds_per_step
        attack_enabled = bool(context.run_config.get("attack-enabled", True)) and (
            spec.num_classes == 2
        )
        schema_extra = {
            "schema_version": 3,
            "dataset": dataset,
            "mode": "vfl-label-private",
            "label_private": True,
            "label_holder_party": label_holder_pid,
            "top_model_location": f"label_holder (pid={label_holder_pid})",
            "labels_at_server": False,
            "seed": seed,
            "configured_parties": num_configured_parties(spec, num_clients),
            "active_party_count": num_clients,
            "active_parties": list(active_groups),
            "party_feature_dims": party_dims,
            "hidden_dim": hidden_dim,
            "total_cut_width": num_clients * hidden_dim,
            "num_rounds": num_rounds,
            "num_steps": num_steps_lp,
            "effective_steps": num_steps_lp,
            "sample_rate": (batch_size / max_train_samples)
            if max_train_samples
            else None,
            "learning_rate": learning_rate,
            "optimizer": "SGD(momentum=0.9 top@holder / 0.0 bottom)",
            "batch_size": batch_size,
            "max_train_samples": max_train_samples,
            "rounds_per_step": rounds_per_step,
            **gp_cfg.as_dict(),
            "attack_enabled": attack_enabled,
            "attack_seed": int(context.run_config.get("attack-seed", 0)),
            "attack_shadow_fraction": float(
                context.run_config.get("attack-shadow-fraction", 0.5)
            ),
            "_run_start": time.perf_counter(),
        }
        build_test_joint, on_metrics = make_label_private_eval(
            spec=spec,
            num_clients=num_clients,
            hidden_dim=hidden_dim,
            active_parties=active_parties,
            eval_every=eval_every,
            eval_max_samples=eval_max_samples,
            num_rounds=num_rounds,
            rounds_per_step=rounds_per_step,
            results_path=results_path,
            telemetry=telemetry,
            schema_extra=schema_extra,
        )
        strategy = LabelPrivateVerticalStrategy(
            config=sl_config,
            num_clients=num_clients,
            label_holder_pid=label_holder_pid,
            build_test_joint_activation=build_test_joint,
            on_metrics=on_metrics,
            telemetry=telemetry,
        )
        log(
            INFO,
            "LABEL-PRIVATE mode: labels+top model at holder pid=%d; coordinator "
            "loads NO labels. K=%d hidden=%d",
            label_holder_pid,
            num_clients,
            hidden_dim,
        )
        return ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=num_rounds)
        )

    tail = build_server_top(spec, num_clients=num_clients, hidden_dim=hidden_dim)
    optimizer = torch.optim.SGD(tail.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    labels_partition = make_server_train_labels(
        spec=spec,
        batch_size=batch_size,
        max_train_samples=max_train_samples or None,
        shuffle_seed=seed,
    )

    round_losses: list[float] = []
    telemetry = RunTelemetry()

    party_dims = [
        party_feature_dim(spec, pid, num_clients, active_parties)
        for pid in range(num_clients)
    ]
    schema_extra = {
        "dataset": dataset,
        "mode": "vfl",
        "seed": seed,
        "configured_parties": num_configured_parties(spec, num_clients),
        "active_party_count": num_clients,
        "active_parties": list(active_groups),
        "party_feature_dims": party_dims,
        "hidden_dim": hidden_dim,
        "total_cut_width": num_clients * hidden_dim,
        "num_rounds": num_rounds,
        "num_steps": num_rounds // sl_config.num_rounds_per_step,
        "learning_rate": learning_rate,
        "optimizer": "SGD(momentum=0.9 tail / 0.0 bottom)",
        "batch_size": batch_size,
        "max_train_samples": max_train_samples,
        "_run_start": time.perf_counter(),
    }

    on_train_step = make_on_train_step(
        tail=tail,
        optimizer=optimizer,
        criterion=criterion,
        labels_partition=labels_partition,
        round_losses=round_losses,
        demo=demo,
        telemetry=telemetry,
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
        active_parties=active_parties,
        telemetry=telemetry,
        schema_extra=schema_extra,
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
