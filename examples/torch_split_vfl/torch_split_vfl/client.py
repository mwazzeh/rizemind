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

import json
import time
from dataclasses import dataclass, field
from logging import INFO

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log
from flwr.common.record import ArrayRecord
from rizemind.split_learning.gradient_privacy import (
    GradientPrivacyConfig,
    privatize_joint_gradient,
)
from rizemind.split_learning.label_inference_attack import evaluate_label_inference
from rizemind.split_learning.label_private_strategy import LP_PHASE_KEY
from rizemind.split_learning.metrics import classification_metrics
from rizemind.split_learning.mod import (
    SL_PHASE_BACKWARD,
    SL_PHASE_KEY,
    split_learning_mod,
)
from rizemind.split_learning.seeding import seed_everything

from .task import (
    VerticalPartition,
    build_bottom_model,
    build_server_top,
    get_dataset_spec,
    get_weights,
    make_client_train_partition,
    make_server_train_labels,
    parse_active_parties,
    set_weights,
    test_labels,
)

_SL_BOTTOM_KEY = "vfl_bottom"
_SL_INPUT_KEY = "vfl_input_x"
# Label-holder cross-round persistence (label-private mode only).
_LP_TOP_KEY = "lp_top_weights"
_LP_TOP_OPT_KEY = "lp_top_momentum"


@dataclass
class LabelHolderContext:
    """State owned ONLY by the label-holder client in label-private mode.

    Holds the top model, the (local) labels, and eval data. None of this is ever
    sent to the coordinator — only aggregate scalar metrics and per-party
    gradients leave the label holder.
    """

    top_model: nn.Module
    top_lr: float
    num_classes: int
    label_train: VerticalPartition  # local train labels (aligned by shuffle seed)
    test_labels: torch.Tensor  # local test labels for eval
    # Phase-4 cut-gradient privacy (applied at the holder before release).
    privacy: GradientPrivacyConfig = field(default_factory=GradientPrivacyConfig)
    seed: int = 42
    train_batch_size: int = 64
    # Label-inference attack benchmark (binary tasks only).
    attack_enabled: bool = True
    attack_seed: int = 0
    attack_shadow_fraction: float = 0.5


_DEMO_TAG = "[VSL-DEMO]"


def _shape(t) -> str:
    dims = list(t.shape)
    inner = ", ".join(str(d) for d in dims)
    return f"({inner},)" if len(dims) == 1 else f"({inner})"


def _nan(value) -> float:
    """Coerce ``None`` to NaN so a metric stays a float-typed scalar."""
    return float("nan") if value is None else float(value)


def build_gradient_privacy_config(run_config) -> GradientPrivacyConfig:
    """Construct a :class:`GradientPrivacyConfig` from Flower run-config keys.

    Recognised keys (all optional, defaults reproduce the legacy ``none`` path):
    ``gradient-privacy-mode`` (none|clip|gaussian), ``gradient-clip-norm``,
    ``gradient-noise-multiplier``, ``privacy-delta``, ``privacy-rng-mode``
    (research-seeded|secure).
    """
    return GradientPrivacyConfig(
        mode=str(run_config.get("gradient-privacy-mode", "none")).strip().lower(),
        clip_norm=float(run_config.get("gradient-clip-norm", 1.0)),
        noise_multiplier=float(run_config.get("gradient-noise-multiplier", 0.0)),
        delta=float(run_config.get("privacy-delta", 1e-5)),
        rng_mode=str(run_config.get("privacy-rng-mode", "research-seeded"))
        .strip()
        .lower(),
    )


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
        label_holder: LabelHolderContext | None = None,
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
        # Set only on the label-holder client in label-private mode.
        self._label_holder = label_holder

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
        """Dispatch on phase.

        Label-private mode uses an explicit three-phase ``lp_phase`` (collect /
        compute / distribute); ``collect`` and ``distribute`` reuse the baseline
        forward/backward bottom logic, and only the label holder runs
        ``compute`` (top model + loss + metrics). Baseline mode falls back to the
        two-phase ``sl_phase``.
        """
        lp_phase = config.get(LP_PHASE_KEY)
        if lp_phase == "collect":
            return self._forward(config)
        if lp_phase == "compute":
            return self._lp_compute(parameters, config)
        if lp_phase == "distribute":
            return self._backward(parameters)
        # Baseline (non-label-private) path.
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
    # Label-private COMPUTE (label holder only): top model + loss + metrics
    # ------------------------------------------------------------------

    def _restore_top(self, top: nn.Module, opt: optim.Optimizer) -> None:
        """Load persisted top-model weights + SGD momentum from context.state."""
        if _LP_TOP_KEY in self._ctx.state:
            set_weights(top, self._ctx.state[_LP_TOP_KEY].to_numpy_ndarrays())
        if _LP_TOP_OPT_KEY in self._ctx.state:
            bufs = self._ctx.state[_LP_TOP_OPT_KEY].to_numpy_ndarrays()
            params = [p for g in opt.param_groups for p in g["params"]]
            for p, buf in zip(params, bufs):
                if buf.size > 0:
                    opt.state[p]["momentum_buffer"] = torch.from_numpy(buf.copy())

    def _save_top(self, top: nn.Module, opt: optim.Optimizer) -> None:
        """Persist top-model weights + SGD momentum buffers to context.state."""
        self._ctx.state[_LP_TOP_KEY] = ArrayRecord(numpy_ndarrays=get_weights(top))
        params = [p for g in opt.param_groups for p in g["params"]]
        bufs = []
        for p in params:
            st = opt.state.get(p, {})
            mb = st.get("momentum_buffer")
            bufs.append(
                mb.detach().cpu().numpy()
                if mb is not None
                else np.zeros(0, dtype=np.float32)
            )
        self._ctx.state[_LP_TOP_OPT_KEY] = ArrayRecord(numpy_ndarrays=bufs)

    def _lp_compute(self, parameters, config):
        """Label-holder step: run top model on the joint activation, return
        per-party gradients + aggregate scalar metrics. Labels never leave here.

        Phase-4: the joint cut gradient is clipped + (optionally) Gaussian-noised
        at the holder *before* it is split into party slices and serialized, so
        no clean gradient copy ever leaves the holder. Aggregate clip/noise
        diagnostics (never raw vectors) ride along in the scalar metrics.
        """
        lh = self._label_holder
        assert lh is not None, "compute phase reached a non-label-holder client"
        t_compute0 = time.perf_counter()
        sl_step = int(config["sl_step"])
        widths = [int(w) for w in str(config["widths"]).split(",")]

        top = lh.top_model
        top_opt = optim.SGD(top.parameters(), lr=lh.top_lr, momentum=0.9)
        self._restore_top(top, top_opt)
        top.train()

        joint = torch.from_numpy(parameters[0]).requires_grad_(True)
        labels = lh.label_train.get_batch_for_step(sl_step).long()

        top_opt.zero_grad()
        logits = top(joint)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        top_opt.step()
        self._save_top(top, top_opt)

        # --- Cut-gradient privacy: clip + noise the JOINT gradient before any
        # split/serialization. The clean `joint.grad` never leaves this method. ---
        clean_joint_grad = joint.grad.detach().numpy()
        t_priv0 = time.perf_counter()
        protected_joint, priv_diag = privatize_joint_gradient(
            clean_joint_grad,
            lh.privacy,
            # Per-step reproducible noise (research mode); secure mode ignores it.
            research_seed=lh.seed * 1_000_003 + sl_step,
        )
        priv_time = time.perf_counter() - t_priv0

        # Split the PROTECTED joint gradient back into per-party slices (pid order).
        grads, off = [], 0
        for w in widths:
            grads.append(np.ascontiguousarray(protected_joint[:, off : off + w]))
            off += w

        metrics: dict = {"train_loss": float(loss.item())}
        metrics.update(self._privacy_metrics(priv_diag, priv_time))
        if bool(config.get("do_eval")) and len(parameters) > 1:
            top.eval()
            with torch.no_grad():
                joint_test = torch.from_numpy(parameters[1])
                logits_te = top(joint_test)
                y = lh.test_labels[: logits_te.shape[0]]
                val_loss = float(
                    nn.functional.cross_entropy(logits_te, y.long()).item()
                )
                preds = logits_te.argmax(1).cpu().numpy()
                score = (
                    torch.softmax(logits_te, dim=1)[:, 1].cpu().numpy()
                    if lh.num_classes == 2
                    else None
                )
            m = classification_metrics(
                y.cpu().numpy(), preds, score, num_classes=lh.num_classes
            )
            metrics.update(
                has_eval=True,
                val_loss=val_loss,
                val_accuracy=m["accuracy"],
                precision_macro=m["precision_macro"],
                recall_macro=m["recall_macro"],
                f1_macro=m["f1_macro"],
                balanced_accuracy=m["balanced_accuracy"],
                n_samples=m["n_samples"],
                confusion_json=json.dumps(m["confusion"]),
            )
            if lh.num_classes == 2:
                metrics.update(
                    precision=m["precision"],
                    recall=m["recall"],
                    f1=m["f1"],
                    roc_auc=(
                        m["roc_auc"] if m["roc_auc"] is not None else float("nan")
                    ),
                )
                if lh.attack_enabled:
                    metrics.update(self._run_attack(top, parameters[1]))
            self._demo_log("COMPUTE", f"eval round: val_acc={m['accuracy']:.4f}")

        metrics["lp_compute_time_s"] = float(time.perf_counter() - t_compute0)
        return grads, int(joint.shape[0]), metrics

    # ------------------------------------------------------------------
    # Phase-4 helpers: privacy diagnostics + label-inference attack
    # ------------------------------------------------------------------

    @staticmethod
    def _privacy_metrics(priv_diag: dict, priv_time: float) -> dict:
        """Pack aggregate (non-sensitive) clip/noise diagnostics into scalars."""
        out: dict = {"lp_privacy_time_s": float(priv_time)}
        for k, v in priv_diag.items():
            if v is None:
                continue
            if isinstance(v, bool):
                out[f"gp_{k}"] = v
            elif isinstance(v, (int, float)):
                out[f"gp_{k}"] = float(v)
            elif isinstance(v, str):
                out[f"gp_{k}"] = v
        return out

    def _run_attack(self, top: nn.Module, joint_test_np: np.ndarray) -> dict:
        """Run the label-inference benchmark on coordinator-visible gradients.

        Reconstructs the per-sample cut gradient the coordinator would observe for
        the probe rows (∂loss_i/∂a_i scaled by the training batch size to match
        the deployed gradient scale), applies the SAME privacy mechanism, then
        measures Attack A / Attack B success. Also runs an activation-only control.
        Returns aggregate scalar metrics only (no labels, no gradient vectors).
        """
        lh = self._label_holder
        assert lh is not None
        t0 = time.perf_counter()
        y = lh.test_labels.long().cpu().numpy()
        joint_test = torch.from_numpy(joint_test_np).clone().requires_grad_(True)
        top.eval()
        logits = top(joint_test)
        # Sum reduction => per-sample grad rows ∂loss_i/∂a_i with no averaging;
        # scale by 1/B_train to match the coordinator-visible training scale.
        loss = nn.functional.cross_entropy(
            logits, torch.from_numpy(y).long(), reduction="sum"
        )
        loss.backward()
        probe_grad = (
            joint_test.grad.detach().numpy() / max(1, lh.train_batch_size)
        ).astype(np.float32)
        feat_time = time.perf_counter() - t0
        # Apply the deployed protection to the probe gradients before the attack.
        protected_probe, _ = privatize_joint_gradient(
            probe_grad,
            lh.privacy,
            research_seed=lh.seed * 7919 + 1,
        )
        res = evaluate_label_inference(
            protected_probe,
            y,
            seed=lh.attack_seed,
            shadow_fraction=lh.attack_shadow_fraction,
        )
        # Activation-only diagnostic control (does the activation leak labels?).
        act_ctrl = evaluate_label_inference(
            joint_test_np.astype(np.float32),
            y,
            seed=lh.attack_seed,
            shadow_fraction=lh.attack_shadow_fraction,
            include_attack_a=False,
        )
        out = {
            "attack_a_json": json.dumps(res["attack_a"]),
            "attack_b_json": json.dumps(res["attack_b"]),
            "attack_refs_json": json.dumps(res["references"]),
            "attack_n_eval": res["n_eval"],
            "attack_n_shadow": res["n_shadow"],
            "attack_split_seed": res["split_seed"],
            "attack_a_auc": _nan(res["attack_a"].get("roc_auc")),
            "attack_a_bal_acc": float(res["attack_a"]["balanced_accuracy"]),
            "attack_b_auc": _nan(res["attack_b"].get("roc_auc")),
            "attack_b_bal_acc": float(res["attack_b"]["balanced_accuracy"]),
            "attack_activation_b_auc": _nan(act_ctrl["attack_b"].get("roc_auc")),
            "lp_attack_feature_time_s": float(feat_time),
            "lp_attack_total_time_s": float(time.perf_counter() - t0),
        }
        top.train()
        return out

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
    seed = int(context.run_config.get("seed", 42))
    active_parties = parse_active_parties(context.run_config.get("active-parties", ""))

    spec = get_dataset_spec(dataset)

    # First-class reproducibility: seed bottom-model init per party (seed + pid)
    # so each party's bottom starts from a distinct but reproducible state.
    seed_everything(seed + partition_id)

    bottom = build_bottom_model(
        spec=spec,
        partition_id=partition_id,
        num_clients=num_clients,
        hidden_dim=hidden_dim,
        active_parties=active_parties,
    )
    if _SL_BOTTOM_KEY in context.state:
        set_weights(bottom, context.state[_SL_BOTTOM_KEY].to_numpy_ndarrays())

    train_partition = make_client_train_partition(
        spec=spec,
        partition_id=partition_id,
        num_clients=num_clients,
        batch_size=batch_size,
        max_train_samples=max_train_samples or None,
        active_parties=active_parties,
        shuffle_seed=seed,
    )

    # Label-private mode: only the configured label holder owns the top model +
    # labels. Passive parties never construct a LabelHolderContext, so they never
    # load labels.
    label_private = bool(context.run_config.get("label-private", False))
    label_holder_pid = int(context.run_config.get("label-holder-party", 0))
    label_holder = None
    if label_private and partition_id == label_holder_pid:
        eval_max_samples = int(context.run_config.get("eval-max-samples", 0))
        # Seed the top model init with `seed` (matches the baseline server tail).
        seed_everything(seed)
        top = build_server_top(spec, num_clients=num_clients, hidden_dim=hidden_dim)
        privacy = build_gradient_privacy_config(context.run_config)
        # Attack benchmark: binary tasks only (Adult). Default on for binary.
        attack_enabled = bool(context.run_config.get("attack-enabled", True)) and (
            spec.num_classes == 2
        )
        label_holder = LabelHolderContext(
            top_model=top,
            top_lr=learning_rate,
            num_classes=spec.num_classes,
            label_train=make_server_train_labels(
                spec=spec,
                batch_size=batch_size,
                max_train_samples=max_train_samples or None,
                shuffle_seed=seed,
            ),
            test_labels=test_labels(spec, eval_max_samples or None),
            privacy=privacy,
            seed=seed,
            train_batch_size=batch_size,
            attack_enabled=attack_enabled,
            attack_seed=int(context.run_config.get("attack-seed", 0)),
            attack_shadow_fraction=float(
                context.run_config.get("attack-shadow-fraction", 0.5)
            ),
        )

    return VerticalSplitClient(
        partition_id=partition_id,
        bottom=bottom,
        train_partition=train_partition,
        learning_rate=learning_rate,
        ctx=context,
        demo=demo,
        label_holder=label_holder,
    ).to_client()


app = ClientApp(client_fn=client_fn, mods=[split_learning_mod])
