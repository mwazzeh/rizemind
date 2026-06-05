"""Generate plots from the torch_split_vfl experiment sweep.

Reads ``experiments/results/manifest.json`` and the per-run ``*.jsonl`` metric
logs, then writes comparison figures to ``experiments/plots/``.

Usage::

    uv run python experiments/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR = HERE / "plots"

# Consistent colors per run within a figure.
_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
]

# Stable color per comparison group for the summary bar chart.
_GROUP_COLORS = {
    "clients": "#4e79a7",
    "clients_ftw": "#76b7b2",
    "lr": "#f28e2b",
    "hidden": "#59a14f",
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs = [r["step"] for r in records]
    ys = [r[key] for r in records]
    return xs, ys


def _label_for(manifest: dict, run_id: str) -> str:
    for r in manifest["runs"]:
        if r["id"] == run_id:
            return r["label"]
    return run_id


def _plot_group(
    manifest: dict,
    run_ids: list[str],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> bool:
    """Plot one metric vs SL step for several runs. Returns True if drawn."""
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = False
    for i, rid in enumerate(run_ids):
        records = _load_jsonl(RESULTS_DIR / f"{rid}.jsonl")
        if not records:
            continue
        xs, ys = _series(records, metric)
        ax.plot(
            xs, ys, marker="o", markersize=3, linewidth=1.6,
            color=_PALETTE[i % len(_PALETTE)], label=_label_for(manifest, rid),
        )
        drawn = True

    if not drawn:
        plt.close(fig)
        return False

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Split-learning step (= 2 Flower rounds)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")
    return True


def _plot_final_accuracy_bar(manifest: dict, out_path: Path) -> None:
    runs = [r for r in manifest["runs"] if r["status"] == "ok"]
    if not runs:
        return
    runs.sort(key=lambda r: (r["group"], -r["final_val_accuracy_smooth"]))
    labels = [r["label"] for r in runs]
    accs = [r["final_val_accuracy_smooth"] for r in runs]
    colors = [_GROUP_COLORS.get(r["group"], "#999999") for r in runs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(runs)), accs, color=colors, edgecolor="white")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Final validation accuracy (mean of last 5 evals)")
    ax.set_xlim(0, 1)
    ax.set_title("VFL final accuracy by configuration", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, label=group)
        for group, color in _GROUP_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", title="swept factor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# Comparison groups: (filename stem, title, run ids)
# The two party-count charts are kept SEPARATE on purpose — merging them would
# conflate "more parties" with "wider server input".
_COMPARISONS = [
    ("k_sweep_growing_width",
     "VFL — party count, FIXED per-party hidden=64\n"
     "(total cut width = K*64 GROWS with K — capacity-confounded)",
     ["mnist_k2", "mnist_k4", "mnist_k7"]),
    ("k_sweep_fixed_total_width",
     "VFL — party count at ~FIXED total cut width (~128)\n"
     "(hidden_dim shrunk as K grows — isolates party count)",
     ["mnist_k2", "mnist_ftw_k4", "mnist_ftw_k7"]),
    ("lr_sweep", "VFL — effect of learning rate",
     ["mnist_lr0.02", "mnist_k2", "mnist_lr0.1"]),
    ("hidden_sweep", "VFL — effect of cut width (hidden-dim)",
     ["mnist_h32", "mnist_k2", "mnist_h128"]),
]


def main() -> None:
    manifest_path = RESULTS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path}; run run_sweep.py first")
    manifest = json.loads(manifest_path.read_text())
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating comparison plots...")
    for stem, title, ids in _COMPARISONS:
        _plot_group(manifest, ids, "val_accuracy", "Validation accuracy",
                    f"{title}\n(validation accuracy)", PLOTS_DIR / f"{stem}_acc.png")
        _plot_group(manifest, ids, "val_loss", "Validation loss",
                    f"{title}\n(validation loss)", PLOTS_DIR / f"{stem}_loss.png")

    print("Generating summary bar chart...")
    _plot_final_accuracy_bar(manifest, PLOTS_DIR / "final_accuracy_summary.png")

    print(f"\nAll plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
