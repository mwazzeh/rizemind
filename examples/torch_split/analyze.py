"""Split-point analysis tool for split learning.

Profiles a PyTorch model layer-by-layer and produces tables + plots to help
choose where to split the model between client and server.

Quick start
-----------
::

    # Default: DemoMLP, batch=32, float32, cpu, 10 warmup + 30 reps
    uv run python analyze.py

    # torch_split example model
    uv run python analyze.py --model torch_split

    # CNN, float16, batch 64, with backward-pass estimates
    uv run python analyze.py --model conv --dtype float16 --batch-size 64 --training

    # Compare CPU vs GPU timing (collected automatically if CUDA is available)
    uv run python analyze.py --model mlp

    # Skip GPU timing explicitly
    uv run python analyze.py --no-gpu-timing

    # Tables only, no PNG
    uv run python analyze.py --no-plot

Output
------
- Reproducibility header (model, input shape, dtype, device, timing config)
- Per-layer inference table (shapes, params, FLOPs, activation KB, cpu_us, gpu_us)
- Cut-point table (client%, transfer KB, balance score, cumulative timing)
- Training table (backward time, param/grad/optimizer/activation-cache memory)  [--training]
- Compact recommendation summary
- PNG plot: 4 or 5 subplots saved to --output
"""

from __future__ import annotations

import argparse

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_split.analysis import (
    LayerProfiler,
    LayerStats,
    TrainingLayerStats,
    TrainingProfiler,
    print_cutpoint_table,
    print_layer_table,
    print_recommendations,
    print_run_header,
    print_training_table,
)

# ---------------------------------------------------------------------------
# Built-in demo models
# ---------------------------------------------------------------------------


class TorchSplitNet(nn.Module):
    """Full torch_split architecture as a single profiling-friendly module.

    Matches the example's split: ``head_fc`` + ``head_relu`` run on the client,
    ``tail_fc`` runs on the server.
    """

    def __init__(
        self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10
    ):
        super().__init__()
        self.head_fc = nn.Linear(input_dim, hidden_dim)
        self.head_relu = nn.ReLU()
        self.tail_fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail_fc(self.head_relu(self.head_fc(x)))


class TorchSplitCNN(nn.Module):
    """The example's CIFAR-10 split architecture as one profiling module.

    Client head: ``conv1..pool2`` (activation ``(64, 8, 8)`` for 3x32x32 input).
    Server tail: ``flatten..fc2``.
    """

    def __init__(
        self, in_channels: int = 3, hidden_dim: int = 128, num_classes: int = 10
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc2(self.relu3(self.fc1(x)))


class DemoMLP(nn.Module):
    """5-layer MLP with decreasing width (784→512→256→128→64→10)."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        return self.fc5(x)


class DemoConvNet(nn.Module):
    """CNN (Conv→Pool→Conv→Pool→Flatten→FC→FC) on 1x28x28 images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc2(self.relu3(self.fc1(x)))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_MODELS = {
    "torch_split": {
        "cls": TorchSplitNet,
        "kwargs": {"input_dim": 784, "hidden_dim": 128, "num_classes": 10},
        "input_fn": lambda bs, dt: torch.randn(bs, 784, dtype=dt),
    },
    "torch_split_cnn": {
        "cls": TorchSplitCNN,
        "kwargs": {"in_channels": 3, "hidden_dim": 128, "num_classes": 10},
        "input_fn": lambda bs, dt: torch.randn(bs, 3, 32, 32, dtype=dt),
    },
    "mlp": {
        "cls": DemoMLP,
        "kwargs": {"input_dim": 784, "num_classes": 10},
        "input_fn": lambda bs, dt: torch.randn(bs, 784, dtype=dt),
    },
    "conv": {
        "cls": DemoConvNet,
        "kwargs": {"num_classes": 10},
        "input_fn": lambda bs, dt: torch.randn(bs, 1, 28, 28, dtype=dt),
    },
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _shape_str(s: tuple) -> str:
    return "(" + ", ".join(str(d) for d in s) + ")"


def plot_analysis(
    stats: list[LayerStats],
    train_stats: list[TrainingLayerStats] | None,
    model_name: str,
    run_info: str,
    output_path: str,
    gpu_available: bool,
) -> None:
    """Save a 4- or 5-subplot analysis figure to *output_path*."""
    names = [s.name for s in stats]
    xs = list(range(len(stats)))
    n_rows = 3 if train_stats else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))

    title = f"Split-Learning Layer Analysis: {model_name}\n{run_info}"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    _COLORS = {
        "Linear": "#4e79a7",
        "Conv2d": "#f28e2b",
        "ReLU": "#76b7b2",
        "MaxPool2d": "#59a14f",
        "Flatten": "#edc948",
        "BatchNorm2d": "#b07aa1",
    }
    bar_colors = [_COLORS.get(s.layer_type, "#aaa") for s in stats]

    # ── (0,0) Per-layer FLOPs ─────────────────────────────────────────────
    ax = axes[0, 0]
    flops_vals = [s.flops for s in stats]
    bars = ax.bar(xs, flops_vals, color=bar_colors, edgecolor="white", linewidth=0.4)
    ax.set_title("FLOPs per layer (symlog scale)", fontweight="bold")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("FLOPs")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, flops_vals):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{val / 1e6:.1f}M"
                if val >= 1e6
                else f"{val / 1e3:.0f}K"
                if val >= 1e3
                else str(val),
                ha="center",
                va="bottom",
                fontsize=6,
            )

    # ── (0,1) Activation transfer size ────────────────────────────────────
    ax = axes[0, 1]
    transfer_vals = [s.transfer_kb for s in stats]
    ax.bar(
        xs,
        transfer_vals,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
    )
    ax.set_title("Activation transfer size at each cut point", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"cut@{i}\n{s.name}" for i, s in enumerate(stats)],
        rotation=40,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Transfer size (KB)")
    ax.grid(axis="y", alpha=0.3)
    min_t_idx = min(range(len(transfer_vals[:-1])), key=lambda i: transfer_vals[i])
    ax.annotate(
        "min",
        xy=(min_t_idx, transfer_vals[min_t_idx]),
        xytext=(min_t_idx, transfer_vals[min_t_idx] + max(transfer_vals) * 0.1),
        ha="center",
        fontsize=8,
        color="red",
        arrowprops={"arrowstyle": "->", "color": "red"},
    )

    # ── (1,0) Cumulative client/server FLOPs ──────────────────────────────
    ax = axes[1, 0]
    total_flops = sum(s.flops for s in stats) or 1
    client_pcts = [s.cumulative_flops / total_flops * 100 for s in stats]
    server_pcts = [s.remaining_flops / total_flops * 100 for s in stats]
    ax.stackplot(
        xs,
        client_pcts,
        server_pcts,
        labels=["Client FLOPs %", "Server FLOPs %"],
        colors=["#4e79a7", "#f28e2b"],
        alpha=0.75,
    )
    balance_scores = [s.balance_score for s in stats]
    best_b = max(range(len(balance_scores[:-1])), key=lambda i: balance_scores[i])
    ax.axvline(best_b, color="red", linestyle="--", linewidth=1.5, label="best balance")
    ax.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Cumulative client/server FLOPs by cut point", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("% of total FLOPs")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(axis="y", alpha=0.3)

    # ── (1,1) CPU (+ GPU) timing per layer ────────────────────────────────
    ax = axes[1, 1]
    cpu_vals = [s.cpu_time_us for s in stats]
    width = 0.4 if gpu_available else 0.7
    ax.bar(
        [x - width / 2 for x in xs] if gpu_available else xs,
        cpu_vals,
        width=width,
        color="#4e79a7",
        label="CPU",
        edgecolor="white",
        linewidth=0.4,
    )
    if gpu_available:
        gpu_vals = [s.gpu_time_us for s in stats]
        ax.bar(
            [x + width / 2 for x in xs],
            gpu_vals,
            width=width,
            color="#f28e2b",
            label="GPU (cuda:0)",
            edgecolor="white",
            linewidth=0.4,
        )
        ax.legend(fontsize=8)
    ax.set_title("Forward execution time per layer (min over reps)", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Time (μs)")
    ax.grid(axis="y", alpha=0.3)

    # ── (2,0)/(2,1) Training memory + backward time [optional] ────────────
    if train_stats:
        cuts = train_stats[:-1]
        cut_xs = list(range(len(cuts)))

        ax = axes[2, 0]
        sgd_vals = [s.total_train_mem_sgd_kb / 1024 for s in cuts]
        adam_vals = [s.total_train_mem_adam_kb / 1024 for s in cuts]
        ax.plot(cut_xs, sgd_vals, "o-", color="#4e79a7", label="SGD (w/ momentum)")
        ax.plot(cut_xs, adam_vals, "s--", color="#e15759", label="Adam")
        ax.set_title("Training memory estimate (client side)", fontweight="bold")
        ax.set_xticks(cut_xs)
        ax.set_xticklabels(
            [f"cut@{s.idx}\n{s.name}" for s in cuts],
            rotation=40,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("Memory (MB)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2, 1]
        bwd_cpu_vals = [s.cumulative_bwd_cpu_us for s in cuts]
        ax.bar(
            cut_xs,
            bwd_cpu_vals,
            color="#76b7b2",
            label="CPU backward (cumul.)",
            edgecolor="white",
            linewidth=0.4,
        )
        if gpu_available:
            bwd_gpu_vals = [s.cumulative_bwd_gpu_us for s in cuts]
            ax.bar(
                cut_xs,
                bwd_gpu_vals,
                color="#f28e2b",
                alpha=0.6,
                label="GPU backward (cumul.)",
                edgecolor="white",
                linewidth=0.4,
            )
            ax.legend(fontsize=8)
        ax.set_title("Backward pass time through client layers", fontweight="bold")
        ax.set_xticks(cut_xs)
        ax.set_xticklabels(
            [f"cut@{s.idx}\n{s.name}" for s in cuts],
            rotation=40,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("Time (μs, cumulative)")
        ax.grid(axis="y", alpha=0.3)

    # Shared layer-type legend
    seen_types = sorted({s.layer_type for s in stats})
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=_COLORS.get(t, "#aaa"), label=t)
        for t in seen_types
    ]
    fig.legend(
        handles=patches,
        title="Layer type",
        loc="lower center",
        ncol=min(len(seen_types), 7),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Layer-by-layer split-point analysis for split learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python analyze.py\n"
            "  uv run python analyze.py --model conv --dtype float16 --training\n"
            "  uv run python analyze.py --model mlp --device cuda --n-reps 50\n"
            "  uv run python analyze.py --no-gpu-timing --no-plot\n"
        ),
    )
    p.add_argument(
        "--model",
        choices=list(_MODELS.keys()),
        default="mlp",
        help="Model to analyse (default: mlp).",
    )
    p.add_argument("--batch-size", type=int, default=32, metavar="N")
    p.add_argument(
        "--dtype",
        choices=list(_DTYPE_MAP.keys()),
        default="float32",
        help="Input and model dtype (default: float32).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help='Primary profiling device: "cpu", "cuda", "cuda:0", etc. (default: cpu).',
    )
    p.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        metavar="N",
        help="Forward passes before timing begins (default: 10).",
    )
    p.add_argument(
        "--n-reps",
        type=int,
        default=30,
        metavar="N",
        help="Timed forward passes; minimum is reported (default: 30).",
    )
    p.add_argument(
        "--training",
        action="store_true",
        help="Add backward-pass profiling and training memory estimates.",
    )
    p.add_argument(
        "--no-gpu-timing",
        action="store_true",
        help="Skip GPU timing even if CUDA is available.",
    )
    p.add_argument(
        "--output",
        default="split_analysis.png",
        help="Output PNG path (default: split_analysis.png).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation; print tables only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    cfg = _MODELS[args.model]

    model = cfg["cls"](**cfg["kwargs"])
    # Cast model to requested dtype
    model = model.to(dtype=dtype)
    sample = cfg["input_fn"](args.batch_size, dtype)
    model_name = type(model).__name__

    # ── Inference profiling ────────────────────────────────────────────────
    profiler = LayerProfiler(
        model,
        sample,
        device=args.device,
        n_warmup=args.n_warmup,
        n_reps=args.n_reps,
        skip_gpu=args.no_gpu_timing,
    )
    stats = profiler.profile()

    # ── Run header ─────────────────────────────────────────────────────────
    print_run_header(profiler, model_name, args.dtype, args.training)

    # ── Tables ─────────────────────────────────────────────────────────────
    gpu_ok = profiler.gpu_was_collected
    print_layer_table(stats, model_name, tuple(sample.shape), gpu_available=gpu_ok)
    print_cutpoint_table(stats, gpu_available=gpu_ok)

    # ── Training profiling (optional) ─────────────────────────────────────
    train_stats: list[TrainingLayerStats] | None = None
    if args.training:
        trainer = TrainingProfiler(
            model,
            sample,
            stats,
            device=args.device,
            n_reps=max(args.n_reps // 3, 5),
            skip_gpu=args.no_gpu_timing,
        )
        train_stats = trainer.profile()
        print_training_table(train_stats, gpu_available=gpu_ok)

    # ── Recommendations ────────────────────────────────────────────────────
    recs = profiler.recommendations(stats)
    print_recommendations(stats, recs, gpu_available=gpu_ok)

    # ── Plot ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        run_info = (
            f"input={_shape_str(tuple(sample.shape))}  dtype={args.dtype}"
            f"  device={args.device}" + ("  +GPU(cuda:0)" if gpu_ok else "")
        )
        plot_analysis(stats, train_stats, model_name, run_info, args.output, gpu_ok)

    print("  Done.\n")


if __name__ == "__main__":
    main()
