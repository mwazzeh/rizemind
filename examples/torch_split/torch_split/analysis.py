"""Per-layer profiler for split-learning cut-point analysis.

Usage
-----
From Python::

    from torch_split.analysis import LayerProfiler, TrainingProfiler
    import torch, torch.nn as nn

    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    sample = torch.randn(32, 16)

    profiler = LayerProfiler(model, sample)
    stats = profiler.profile()  # inference metrics

    trainer = TrainingProfiler(model, sample, stats)
    train_stats = trainer.profile()  # backward + memory metrics

See ``analyze.py`` in the torch_split example root for the full CLI.

Metrics glossary
----------------
Inference (``LayerStats``):
  flops          — FLOPs counted by ``FlopCounterMode`` (linear/conv ops only;
                   elementwise ops like ReLU are zero).
  macs           — flops / 2 (standard convention: 1 MAC = 2 FLOPs).
  activation_bytes — exact output-tensor size in bytes; this is the wire cost
                     when the layer is the cut point.
  cpu_time_us    — minimum wall-clock forward time over n_reps (perf_counter).
  gpu_time_us    — minimum CUDA-event forward time over n_reps; 0 when GPU
                   timing was not collected.

Training (``TrainingLayerStats``):
  bwd_cpu_time_us — cumulative backward time through client layers 0..idx
                    (min over n_reps; approximation — see caveats below).
  bwd_gpu_time_us — same, on CUDA.
  param_kb        — parameter storage for client layers 0..idx (float32 → 4B).
  grad_kb         — gradient buffer (= param_kb for float32).
  opt_sgd_kb      — SGD-with-momentum optimizer state (1x params).
  opt_adam_kb     — Adam optimizer state (2x params for m1 + m2).
  act_cache_kb    — activations cached during forward for backprop (sum 0..idx).
  total_train_mem_adam_kb — param + grad + adam_opt + act_cache (client side).

Caveats / approximations
------------------------
- FLOPs: elementwise ops (ReLU, MaxPool, Flatten) are not counted.
- CPU timing: includes Python dispatch overhead; use as relative comparison only.
- GPU timing: measured via ``torch.cuda.Event``; includes kernel launch overhead.
- Backward timing: single full-model backward; not split-model backward.
  The cumulative column assumes additive backward costs, which is an
  approximation for fused kernels.
- Activation cache: estimates one float32 activation tensor per layer. Some
  layers cache more (e.g. attention), others less (in-place ReLU).
- Optimizer memory: assumes all parameters are in the optimizer. In practice,
  you may freeze some layers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Inference-time statistics for one leaf layer."""

    idx: int
    name: str
    layer_type: str
    in_shape: tuple
    out_shape: tuple
    n_params: int
    param_bytes: int
    flops: int
    macs: int
    activation_bytes: int
    cpu_time_us: float
    gpu_time_us: float  # 0.0 if GPU timing not collected

    # Derived — filled by _compute_cumulative
    cumulative_flops: int = 0
    cumulative_params: int = 0
    cumulative_cpu_us: float = 0.0
    cumulative_gpu_us: float = 0.0
    remaining_flops: int = 0
    remaining_params: int = 0
    transfer_kb: float = 0.0
    client_flop_pct: float = 0.0
    server_flop_pct: float = 0.0
    balance_score: float = 0.0
    memory_inference_kb: float = 0.0


@dataclass
class TrainingLayerStats:
    """Training-time estimates for client layers up to (and including) cut index.

    All ``cumulative_*`` fields represent the cost if this layer is the cut point,
    i.e. the client owns layers 0..idx.
    """

    idx: int
    name: str

    # Per-layer backward pass timing (not cumulative)
    bwd_cpu_time_us: float
    bwd_gpu_time_us: float

    # Cumulative backward pass timing (client side up to cut here)
    cumulative_bwd_cpu_us: float = 0.0
    cumulative_bwd_gpu_us: float = 0.0

    # Per-layer memory
    param_kb: float = 0.0  # parameter storage
    grad_kb: float = 0.0  # gradient buffer (= param_kb for float32)
    opt_sgd_kb: float = 0.0  # SGD-with-momentum state (1x params)
    opt_adam_kb: float = 0.0  # Adam state (2x params)
    act_cache_kb: float = 0.0  # activation cached for backprop

    # Cumulative memory if cut here (client owns 0..idx)
    total_param_kb: float = 0.0
    total_grad_kb: float = 0.0
    total_opt_adam_kb: float = 0.0
    total_act_cache_kb: float = 0.0
    total_train_mem_sgd_kb: float = 0.0  # param + grad + sgd_opt + act_cache
    total_train_mem_adam_kb: float = 0.0  # param + grad + adam_opt + act_cache


# ---------------------------------------------------------------------------
# Inference profiler
# ---------------------------------------------------------------------------


class LayerProfiler:
    """Profile a PyTorch model layer-by-layer for split-point analysis.

    Given a model and a representative sample input, collects per-layer:

    - Input/output shapes
    - Parameter count and storage bytes
    - FLOPs (via ``torch.utils.flop_counter.FlopCounterMode``)
    - Activation size (bytes) — the wire cost when this layer is the cut point
    - CPU wall-clock forward time (minimum over ``n_reps``)
    - GPU CUDA-event forward time (if CUDA available and ``skip_gpu=False``)
    - Cumulative and remaining cost for every candidate split point

    When ``device="cpu"`` and CUDA is available, a separate GPU timing pass
    runs automatically so both CPU and GPU columns are always populated.

    Args:
        model: PyTorch model to profile.
        sample_input: Representative input tensor (batch dimension included).
        device: ``"cpu"`` or ``"cuda[:<id>]"`` — primary profiling device.
        n_warmup: Forward passes before timing starts.
        n_reps: Timed passes; minimum latency is reported.
        skip_gpu: If ``True``, skip GPU timing even if CUDA is available.
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        device: str = "cpu",
        n_warmup: int = 10,
        n_reps: int = 30,
        skip_gpu: bool = False,
    ) -> None:
        self.model = model.eval().to(device)
        self.sample_input = sample_input.to(device)
        self.device = device
        self.n_warmup = n_warmup
        self.n_reps = n_reps
        self.skip_gpu = skip_gpu
        self._primary_on_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self._run_gpu = (not skip_gpu) and torch.cuda.is_available()
        # Populated after profile()
        self.gpu_device_name: str = ""
        self.gpu_was_collected: bool = False
        if self._run_gpu:
            self.gpu_device_name = torch.cuda.get_device_name(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self) -> list[LayerStats]:
        """Run the full profiling pipeline and return ordered layer stats."""
        shape_cpu_info = self._collect_shapes_and_cpu_timing()
        gpu_times = self._collect_gpu_timing(shape_cpu_info) if self._run_gpu else {}
        flop_map = self._collect_flops()

        self.gpu_was_collected = bool(gpu_times)

        stats: list[LayerStats] = []
        for i, (name, info) in enumerate(shape_cpu_info.items()):
            layer_flops = flop_map.get(name, 0)
            params = info["params"]

            stats.append(
                LayerStats(
                    idx=i,
                    name=name,
                    layer_type=info["type"],
                    in_shape=info["in_shape"],
                    out_shape=info["out_shape"],
                    n_params=params,
                    param_bytes=params * 4,
                    flops=layer_flops,
                    macs=layer_flops // 2,
                    activation_bytes=info["activation_bytes"],
                    cpu_time_us=info["cpu_time_us"],
                    gpu_time_us=gpu_times.get(name, 0.0),
                )
            )

        self._compute_cumulative(stats)
        return stats

    def recommendations(self, stats: list[LayerStats]) -> dict[str, int]:
        """Return recommended cut-layer index for several objectives.

        Keys in the returned dict:
        - ``"min_client_compute"`` — cut with the cheapest client-side FLOPs prefix
        - ``"min_transfer"``       — cut where activation tensor is smallest
        - ``"best_balance"``       — cut closest to a 50/50 compute split
        - ``"min_client_memory"``  — cut where client stores fewest parameters
        """
        if not stats:
            return {}

        candidates = stats[:-1]  # last layer has no server side

        def pick_min(lst, key):
            return min(lst, key=key).idx if lst else stats[-1].idx

        return {
            "min_client_compute": pick_min(
                candidates, key=lambda s: s.cumulative_flops
            ),
            "min_transfer": pick_min(candidates, key=lambda s: s.transfer_kb),
            "best_balance": pick_min(
                candidates, key=lambda s: abs(s.client_flop_pct - 50.0)
            ),
            "min_client_memory": pick_min(
                candidates, key=lambda s: s.cumulative_params
            ),
        }

    # ------------------------------------------------------------------
    # Internal: shape + CPU timing
    # ------------------------------------------------------------------

    def _collect_shapes_and_cpu_timing(self) -> dict[str, dict[str, Any]]:
        """Collect shapes on first pass, then CPU timing over n_reps."""
        exec_order: list[str] = []
        shape_info: dict[str, dict[str, Any]] = {}

        def shape_hook(name: str):
            def hook(mod: nn.Module, inp, out):
                if name not in shape_info:
                    exec_order.append(name)
                    in_t = inp[0] if isinstance(inp, (list, tuple)) else inp
                    out_t = out if isinstance(out, torch.Tensor) else out[0]
                    shape_info[name] = {
                        "type": type(mod).__name__,
                        "in_shape": tuple(in_t.shape),
                        "out_shape": tuple(out_t.shape),
                        "params": sum(p.numel() for p in mod.parameters()),
                        "activation_bytes": int(out_t.numel() * out_t.element_size()),
                        "cpu_time_us": 0.0,
                    }

            return hook

        handles = []
        for name, mod in self.model.named_modules():
            if not any(True for _ in mod.children()):
                handles.append(mod.register_forward_hook(shape_hook(name)))

        with torch.no_grad():
            self.model(self.sample_input)

        for h in handles:
            h.remove()

        if not shape_info:
            return {}

        # CPU timing: pre/post hooks, minimum over n_reps
        cpu_best: dict[str, float] = {n: float("inf") for n in exec_order}
        pre_cpu: dict[str, float] = {}

        def pre_hook(name: str):
            def hook(mod: nn.Module, inp):
                pre_cpu[name] = time.perf_counter()

            return hook

        def post_hook(name: str):
            def hook(mod: nn.Module, inp, out):
                elapsed = (time.perf_counter() - pre_cpu.get(name, 0)) * 1e6
                if elapsed < cpu_best[name]:
                    cpu_best[name] = elapsed

            return hook

        handles = []
        for name in exec_order:
            for n, mod in self.model.named_modules():
                if n == name:
                    handles.append(mod.register_forward_pre_hook(pre_hook(name)))
                    handles.append(mod.register_forward_hook(post_hook(name)))
                    break

        with torch.no_grad():
            for _ in range(self.n_warmup):
                self.model(self.sample_input)
            for _ in range(self.n_reps):
                self.model(self.sample_input)

        for h in handles:
            h.remove()

        for name in exec_order:
            shape_info[name]["cpu_time_us"] = cpu_best.get(name, 0.0)

        return {name: shape_info[name] for name in exec_order}

    # ------------------------------------------------------------------
    # Internal: GPU timing (always on cuda:0 for comparability)
    # ------------------------------------------------------------------

    def _collect_gpu_timing(
        self, shape_info: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Run a separate GPU timing pass on cuda:0.

        If the primary device is already CUDA, reuse the existing model/input.
        Otherwise, clone both to cuda:0 temporarily.
        """
        if self._primary_on_cuda:
            gpu_model = self.model
            gpu_input = self.sample_input
        else:
            gpu_model = type(self.model).__new__(type(self.model))
            # Deepcopy via state_dict transfer
            gpu_model = self._clone_model_to_gpu()
            gpu_input = self.sample_input.cuda()

        exec_order = list(shape_info.keys())
        gpu_best: dict[str, float] = {n: float("inf") for n in exec_order}
        cuda_start: dict[str, torch.cuda.Event] = {}
        cuda_end: dict[str, torch.cuda.Event] = {}

        def gpu_pre(name: str):
            def hook(mod, inp):
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                cuda_start[name] = start

            return hook

        def gpu_post(name: str):
            def hook(mod, inp, out):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                cuda_end[name] = end

            return hook

        handles = []
        for name in exec_order:
            for n, mod in gpu_model.named_modules():
                if n == name:
                    handles.append(mod.register_forward_pre_hook(gpu_pre(name)))
                    handles.append(mod.register_forward_hook(gpu_post(name)))
                    break

        with torch.no_grad():
            for _ in range(self.n_warmup):
                gpu_model(gpu_input)
                torch.cuda.synchronize()

            for _ in range(self.n_reps):
                cuda_start.clear()
                cuda_end.clear()
                gpu_model(gpu_input)
                torch.cuda.synchronize()
                for name in exec_order:
                    if name in cuda_start and name in cuda_end:
                        t_us = cuda_start[name].elapsed_time(cuda_end[name]) * 1e3
                        if t_us < gpu_best[name]:
                            gpu_best[name] = t_us

        for h in handles:
            h.remove()

        return {n: v for n, v in gpu_best.items() if v < float("inf")}

    def _clone_model_to_gpu(self) -> nn.Module:
        """Return a GPU copy of self.model with identical weights."""
        import copy

        clone = copy.deepcopy(self.model).cuda().eval()
        return clone

    # ------------------------------------------------------------------
    # Internal: FLOPs
    # ------------------------------------------------------------------

    def _collect_flops(self) -> dict[str, int]:
        """Return per-leaf-layer FLOPs using FlopCounterMode."""
        leaf_paths = {
            name
            for name, mod in self.model.named_modules()
            if not any(True for _ in mod.children())
        }
        model_cls = type(self.model).__name__

        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                self.model(self.sample_input)

        per_layer: dict[str, int] = {}
        for key, op_dict in fc.get_flop_counts().items():
            if key in ("Global", model_cls):
                continue
            layer_path = (
                key[len(f"{model_cls}.") :] if key.startswith(f"{model_cls}.") else key
            )
            if layer_path in leaf_paths:
                per_layer[layer_path] = int(sum(op_dict.values()))

        return per_layer

    # ------------------------------------------------------------------
    # Internal: cumulative metrics
    # ------------------------------------------------------------------

    def _compute_cumulative(self, stats: list[LayerStats]) -> None:
        """Fill cumulative and split-point derived fields in-place."""
        total_flops = sum(s.flops for s in stats)
        total_params = sum(s.n_params for s in stats)

        cum_flops = cum_params = 0
        cum_cpu = cum_gpu = 0.0
        for s in stats:
            cum_flops += s.flops
            cum_params += s.n_params
            cum_cpu += s.cpu_time_us
            cum_gpu += s.gpu_time_us

            s.cumulative_flops = cum_flops
            s.cumulative_params = cum_params
            s.cumulative_cpu_us = cum_cpu
            s.cumulative_gpu_us = cum_gpu
            s.remaining_flops = total_flops - cum_flops
            s.remaining_params = total_params - cum_params
            s.transfer_kb = s.activation_bytes / 1024.0
            s.client_flop_pct = (
                (cum_flops / total_flops * 100.0) if total_flops else 0.0
            )
            s.server_flop_pct = 100.0 - s.client_flop_pct
            s.balance_score = 1.0 - abs(s.client_flop_pct - 50.0) / 50.0
            s.memory_inference_kb = (s.param_bytes + s.activation_bytes) / 1024.0


# ---------------------------------------------------------------------------
# Training profiler
# ---------------------------------------------------------------------------


class TrainingProfiler:
    """Estimate per-layer backward-pass cost and training memory for cut-point selection.

    Uses ``register_full_backward_pre_hook`` and ``register_full_backward_hook``
    to time the backward pass through each leaf layer.  Memory estimates are
    derived analytically from parameter counts and activation sizes.

    Args:
        model: Same model used for inference profiling.
        sample_input: Same sample input used for inference profiling.
        inference_stats: Output of ``LayerProfiler.profile()`` — provides
            activation sizes for the activation-cache estimate.
        device: Device for backward timing (same as ``LayerProfiler.device``).
        n_reps: Number of forward+backward passes; minimum is reported.
        skip_gpu: Skip GPU backward timing even if CUDA is available.
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        inference_stats: list[LayerStats],
        device: str = "cpu",
        n_reps: int = 10,
        skip_gpu: bool = False,
    ) -> None:
        self.model = model.train().to(device)
        self.sample_input = sample_input.to(device)
        self.inference_stats = inference_stats
        self.device = device
        self.n_reps = n_reps
        self.skip_gpu = skip_gpu
        self._run_gpu = (not skip_gpu) and torch.cuda.is_available()

    def profile(self) -> list[TrainingLayerStats]:
        """Run backward timing and compute memory estimates."""
        bwd_cpu = self._collect_backward_timing_cpu()
        bwd_gpu = self._collect_backward_timing_gpu() if self._run_gpu else {}

        stats: list[TrainingLayerStats] = []
        for inf_s in self.inference_stats:
            name = inf_s.name
            param_kb = inf_s.param_bytes / 1024.0
            grad_kb = param_kb  # float32 grad = same as param
            opt_sgd_kb = param_kb  # SGD momentum: 1x params
            opt_adam_kb = 2.0 * param_kb  # Adam m1 + m2: 2x params
            act_cache_kb = inf_s.activation_bytes / 1024.0

            stats.append(
                TrainingLayerStats(
                    idx=inf_s.idx,
                    name=name,
                    bwd_cpu_time_us=bwd_cpu.get(name, 0.0),
                    bwd_gpu_time_us=bwd_gpu.get(name, 0.0),
                    param_kb=param_kb,
                    grad_kb=grad_kb,
                    opt_sgd_kb=opt_sgd_kb,
                    opt_adam_kb=opt_adam_kb,
                    act_cache_kb=act_cache_kb,
                )
            )

        self._compute_cumulative(stats)
        return stats

    # ------------------------------------------------------------------
    # Internal: backward timing
    # ------------------------------------------------------------------

    def _run_bwd_pass(
        self,
        model: nn.Module,
        x: torch.Tensor,
        pre_times: dict,
        post_times: dict,
        use_cuda_events: bool,
    ) -> None:
        """Run one forward+backward pass with backward hooks installed."""
        x_var = x.detach().requires_grad_(True)
        output = model(x_var)
        loss = output.sum()
        loss.backward()

    def _collect_backward_timing_cpu(self) -> dict[str, float]:
        """Time each layer's backward pass on CPU (minimum over n_reps)."""
        exec_order = [s.name for s in self.inference_stats]
        bwd_best: dict[str, float] = {n: float("inf") for n in exec_order}

        for _ in range(self.n_reps):
            pre_t: dict[str, float] = {}
            post_t: dict[str, float] = {}

            def make_bwd_pre(n: str):
                def h(mod, grad_output):
                    pre_t[n] = time.perf_counter()

                return h

            def make_bwd_post(n: str):
                def h(mod, grad_input, grad_output):
                    if n in pre_t:
                        post_t[n] = time.perf_counter()

                return h

            handles = []
            for name in exec_order:
                for n, mod in self.model.named_modules():
                    if n == name:
                        handles.append(
                            mod.register_full_backward_pre_hook(make_bwd_pre(name))
                        )
                        handles.append(
                            mod.register_full_backward_hook(make_bwd_post(name))
                        )
                        break

            x_var = self.sample_input.detach().requires_grad_(True)
            out = self.model(x_var)
            out.sum().backward()

            for h in handles:
                h.remove()

            for name in exec_order:
                if name in pre_t and name in post_t:
                    elapsed = (post_t[name] - pre_t[name]) * 1e6
                    if elapsed < bwd_best[name]:
                        bwd_best[name] = elapsed

        return {n: v for n, v in bwd_best.items() if v < float("inf")}

    def _collect_backward_timing_gpu(self) -> dict[str, float]:
        """Time each layer's backward pass on cuda:0 (minimum over n_reps)."""
        import copy

        if self.device.startswith("cuda"):
            gpu_model = self.model
            gpu_input = self.sample_input
        else:
            gpu_model = copy.deepcopy(self.model).cuda().train()
            gpu_input = self.sample_input.cuda()

        exec_order = [s.name for s in self.inference_stats]
        bwd_best: dict[str, float] = {n: float("inf") for n in exec_order}

        for _ in range(self.n_reps):
            cuda_pre: dict[str, torch.cuda.Event] = {}
            cuda_post: dict[str, torch.cuda.Event] = {}

            def make_gpu_pre(n: str):
                def h(mod, grad_output):
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    cuda_pre[n] = ev

                return h

            def make_gpu_post(n: str):
                def h(mod, grad_input, grad_output):
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    cuda_post[n] = ev

                return h

            handles = []
            for name in exec_order:
                for n, mod in gpu_model.named_modules():
                    if n == name:
                        handles.append(
                            mod.register_full_backward_pre_hook(make_gpu_pre(name))
                        )
                        handles.append(
                            mod.register_full_backward_hook(make_gpu_post(name))
                        )
                        break

            x_var = gpu_input.detach().requires_grad_(True)
            out = gpu_model(x_var)
            out.sum().backward()
            torch.cuda.synchronize()

            for h in handles:
                h.remove()

            for name in exec_order:
                if name in cuda_pre and name in cuda_post:
                    t_us = cuda_pre[name].elapsed_time(cuda_post[name]) * 1e3
                    if t_us < bwd_best[name]:
                        bwd_best[name] = t_us

        return {n: v for n, v in bwd_best.items() if v < float("inf")}

    # ------------------------------------------------------------------
    # Internal: cumulative training memory
    # ------------------------------------------------------------------

    def _compute_cumulative(self, stats: list[TrainingLayerStats]) -> None:
        """Fill cumulative backward time and memory fields in-place."""
        cum_bwd_cpu = cum_bwd_gpu = 0.0
        cum_param = cum_grad = cum_opt_adam = cum_act = 0.0

        for s in stats:
            cum_bwd_cpu += s.bwd_cpu_time_us
            cum_bwd_gpu += s.bwd_gpu_time_us
            cum_param += s.param_kb
            cum_grad += s.grad_kb
            cum_opt_adam += s.opt_adam_kb
            cum_act += s.act_cache_kb

            s.cumulative_bwd_cpu_us = cum_bwd_cpu
            s.cumulative_bwd_gpu_us = cum_bwd_gpu
            s.total_param_kb = cum_param
            s.total_grad_kb = cum_grad
            s.total_opt_adam_kb = cum_opt_adam
            s.total_act_cache_kb = cum_act
            s.total_train_mem_sgd_kb = (
                cum_param
                + cum_grad
                + (cum_param)  # SGD momentum = 1x
                + cum_act
            )
            s.total_train_mem_adam_kb = cum_param + cum_grad + cum_opt_adam + cum_act


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _fmt(n: int | float, width: int = 8) -> str:
    """Format a number with K/M/G suffix."""
    if isinstance(n, float):
        if abs(n) >= 1_000_000_000:
            return f"{n / 1e9:{width}.2f}G"
        if abs(n) >= 1_000_000:
            return f"{n / 1e6:{width}.2f}M"
        if abs(n) >= 1_000:
            return f"{n / 1e3:{width}.1f}K"
        return f"{n:{width}.1f}"
    if n >= 1_000_000_000:
        return f"{n / 1e9:{width}.2f}G"
    if n >= 1_000_000:
        return f"{n / 1e6:{width}.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:{width}.1f}K"
    return f"{n:{width}d}"


def _ss(s: tuple) -> str:
    return "(" + ", ".join(str(d) for d in s) + ")"


def print_run_header(
    profiler: LayerProfiler,
    model_name: str,
    dtype_name: str,
    training: bool,
) -> None:
    """Print a reproducibility header with all run parameters."""
    W = 85
    sep = "─" * W
    sample = profiler.sample_input
    batch = sample.shape[0]
    print(f"\n{sep}")
    print("  Split-Learning Layer Profiler")
    print(sep)
    print(f"  Model:       {model_name}")
    print(
        f"  Input:       shape={_ss(tuple(sample.shape))}  dtype={dtype_name}  batch={batch}"
    )
    print(f"  Run device:  {profiler.device}")
    print(
        f"  CPU timing:  {profiler.n_warmup} warmup + {profiler.n_reps} reps"
        "  (minimum over reps)"
    )
    if profiler.gpu_was_collected:
        print(
            f"  GPU timing:  {profiler.n_warmup} warmup + {profiler.n_reps} reps"
            f"  device=cuda:0  {profiler.gpu_device_name}"
        )
    else:
        reason = (
            "skipped (--no-gpu-timing)" if profiler.skip_gpu else "CUDA not available"
        )
        print(f"  GPU timing:  {reason}")
    if training:
        print(
            "  Training:    backward-pass profiling + memory estimates  (SGD and Adam)"
        )
    print(sep)


def print_layer_table(
    stats: list[LayerStats],
    model_name: str,
    batch_shape: tuple,
    gpu_available: bool = False,
) -> None:
    """Print per-layer inference statistics."""
    W = 120 if gpu_available else 108
    sep = "─" * W
    print(f"\n{sep}")
    label = (
        "inference only"
        if not gpu_available
        else "inference  (cpu_us and gpu_us = minimum over reps)"
    )
    print(f"  Layer Statistics  [{label}]")
    print(sep)
    gpu_col = f" {'gpu_us':>8}" if gpu_available else ""
    hdr = (
        f"  {'idx':>3}  {'name':<18} {'type':<12} {'in_shape':<18} {'out_shape':<18}"
        f" {'params':>8} {'FLOPs':>9} {'act_KB':>7} {'cpu_us':>8}{gpu_col}"
    )
    print(hdr)
    print(sep)
    for s in stats:
        gpu_val = f" {s.gpu_time_us:>8.1f}" if gpu_available else ""
        print(
            f"  {s.idx:>3}  {s.name:<18} {s.layer_type:<12}"
            f" {_ss(s.in_shape):<18} {_ss(s.out_shape):<18}"
            f" {_fmt(s.n_params):>8} {_fmt(s.flops):>9}"
            f" {s.transfer_kb:>7.2f} {s.cpu_time_us:>8.1f}{gpu_val}"
        )
    print(sep)
    total_p = sum(s.n_params for s in stats)
    total_f = sum(s.flops for s in stats)
    print(f"  {'TOTAL':<55} {_fmt(total_p):>8} {_fmt(total_f):>9}")
    note = "  NOTE: FLOPs counts linear/conv only; elementwise ops (ReLU/Pool) = 0."
    print(note)
    print(sep)


def print_cutpoint_table(
    stats: list[LayerStats],
    gpu_available: bool = False,
) -> None:
    """Print cut-point analysis with cumulative client/server costs."""
    W = 118 if gpu_available else 100
    sep = "─" * W
    print(f"\n{sep}")
    print("  Cut-Point Analysis  [client = layers 0..cut  |  server = rest]")
    print(sep)
    gpu_cols = (
        f" {'cpu_us*':>9} {'gpu_us*':>9}" if gpu_available else f" {'cpu_us*':>9}"
    )
    hdr = (
        f"  {'cut':>3}  {'after':<18} {'client_FLOPs':>13} {'server_FLOPs':>13}"
        f" {'client%':>8} {'xfer_KB':>9} {'balance':>8}{gpu_cols}"
    )
    print(hdr)
    print(sep)
    for s in stats[:-1]:
        gpu_val = f" {s.cumulative_gpu_us:>9.1f}" if gpu_available else ""
        print(
            f"  {s.idx:>3}  {s.name:<18}"
            f" {_fmt(s.cumulative_flops):>13} {_fmt(s.remaining_flops):>13}"
            f" {s.client_flop_pct:>7.1f}% {s.transfer_kb:>9.2f} {s.balance_score:>8.3f}"
            f" {s.cumulative_cpu_us:>9.1f}{gpu_val}"
        )
    print(sep)
    print("  * cpu_us / gpu_us: cumulative forward time for client layers 0..cut")
    print(sep)


def print_training_table(
    train_stats: list[TrainingLayerStats],
    gpu_available: bool = False,
) -> None:
    """Print training-cost estimates per candidate cut point."""
    W = 118
    sep = "─" * W
    print(f"\n{sep}")
    print("  Training Cost Estimates  [cumulative * = client layers 0..cut]")
    print(sep)
    gpu_col = f" {'bwd_gpu*':>9}" if gpu_available else ""
    hdr = (
        f"  {'cut':>3}  {'after':<18} {'bwd_cpu*':>9}{gpu_col}"
        f" {'param_KB':>9} {'grad_KB':>8} {'act_cache_KB':>13}"
        f" {'mem_SGD_KB':>11} {'mem_Adam_KB':>11}"
    )
    print(hdr)
    print(sep)
    # Skip last layer (no server side)
    for s in train_stats[:-1]:
        gpu_val = f" {s.cumulative_bwd_gpu_us:>9.1f}" if gpu_available else ""
        print(
            f"  {s.idx:>3}  {s.name:<18}"
            f" {s.cumulative_bwd_cpu_us:>9.1f}{gpu_val}"
            f" {s.total_param_kb:>9.2f} {s.total_grad_kb:>8.2f}"
            f" {s.total_act_cache_kb:>13.2f}"
            f" {s.total_train_mem_sgd_kb:>11.2f} {s.total_train_mem_adam_kb:>11.2f}"
        )
    print(sep)
    print("  * bwd_cpu/gpu: backward time through client layers (approximation).")
    print("  * param_KB: float32 parameter storage for client side.")
    print("  * act_cache_KB: activations held in memory during forward for backprop.")
    print("  * mem_SGD_KB: param + grad + SGD-momentum (1x param) + act_cache.")
    print("  * mem_Adam_KB: param + grad + Adam-state (2x param) + act_cache.")
    print("  INFERENCE-ONLY metrics missing from this table: FLOPs, transfer cost.")
    print(sep)


def print_recommendations(
    stats: list[LayerStats],
    recs: dict[str, int],
    gpu_available: bool = False,
) -> None:
    """Print a compact recommendation table."""
    idx_map = {s.idx: s for s in stats}
    W = 100
    sep = "─" * W
    print(f"\n{sep}")
    print("  RECOMMENDED CUT POINTS")
    print(sep)
    gpu_hdr = f" {'gpu_us':>8}" if gpu_available else ""
    hdr = (
        f"  {'Objective':<32} {'Cut layer':<20}"
        f" {'Client%':>8} {'Xfer_KB':>9} {'cpu_us':>8}{gpu_hdr} {'balance':>8}"
    )
    print(hdr)
    print(sep)
    objectives = [
        ("min_client_compute", "Min client compute"),
        ("min_transfer", "Min activation transfer"),
        ("best_balance", "Best compute balance"),
        ("min_client_memory", "Min client param memory"),
    ]
    for key, label in objectives:
        if key not in recs:
            continue
        s = idx_map.get(recs[key])
        if s is None:
            continue
        layer_label = f"{s.name} [{s.idx}]"
        gpu_val = f" {s.cumulative_gpu_us:>8.1f}" if gpu_available else ""
        print(
            f"  {label:<32} {layer_label:<20}"
            f" {s.client_flop_pct:>7.1f}% {s.transfer_kb:>9.2f}"
            f" {s.cumulative_cpu_us:>8.1f}{gpu_val} {s.balance_score:>8.3f}"
        )
    print(sep)
    print()
