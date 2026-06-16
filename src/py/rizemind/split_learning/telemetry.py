"""Communication & timing telemetry for the split-learning protocol.

Reports **tensor payload estimates** (element count x element size), not
transport-level network bytes — the activations/gradients that cross the wire,
plus wall-clock timings. NumPy-only; torch tensors are handled by duck-typing
so importing this module never requires torch.

No activation or gradient *contents* are ever recorded — only sizes and times.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["tensor_payload_bytes", "StepTelemetry", "RunTelemetry", "timed"]


def tensor_payload_bytes(tensor) -> int:
    """Estimate the serialised byte size of one tensor (element count x dtype size).

    Accepts a NumPy ndarray, a PyTorch tensor, or any object exposing either
    ``nbytes`` or both ``element_size()`` and ``nelement()``. This is a payload
    *estimate*, not measured transport bytes.

    Args:
        tensor: The tensor-like object to size.

    Returns:
        Estimated payload size in bytes.

    Raises:
        TypeError: If the object exposes no recognised size interface.
    """
    if isinstance(tensor, np.ndarray):
        return int(tensor.nbytes)
    # torch.Tensor: element_size() * nelement()
    if hasattr(tensor, "element_size") and hasattr(tensor, "nelement"):
        return int(tensor.element_size() * tensor.nelement())
    if hasattr(tensor, "nbytes"):
        return int(tensor.nbytes)
    raise TypeError(
        f"cannot estimate payload bytes for object of type {type(tensor)!r}"
    )


@dataclass
class StepTelemetry:
    """Per-step communication + timing accumulator for one SL step.

    Communication is tracked per party (``partition_id``) for activations
    (client -> server) and gradients (server -> client). Timings are summed
    over whatever phases the caller records.
    """

    sl_step: int = 0
    activation_bytes: dict[int, int] = field(default_factory=dict)
    gradient_bytes: dict[int, int] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)

    def add_activation(self, partition_id: int, tensor) -> int:
        b = tensor_payload_bytes(tensor)
        self.activation_bytes[partition_id] = self.activation_bytes.get(partition_id, 0) + b
        return b

    def add_gradient(self, partition_id: int, tensor) -> int:
        b = tensor_payload_bytes(tensor)
        self.gradient_bytes[partition_id] = self.gradient_bytes.get(partition_id, 0) + b
        return b

    def add_time(self, phase: str, seconds: float) -> None:
        self.timings_s[phase] = self.timings_s.get(phase, 0.0) + float(seconds)

    @property
    def total_activation_bytes(self) -> int:
        return int(sum(self.activation_bytes.values()))

    @property
    def total_gradient_bytes(self) -> int:
        return int(sum(self.gradient_bytes.values()))

    @property
    def total_payload_bytes(self) -> int:
        return self.total_activation_bytes + self.total_gradient_bytes

    def as_dict(self) -> dict:
        return {
            "sl_step": self.sl_step,
            "activation_bytes_per_party": dict(sorted(self.activation_bytes.items())),
            "gradient_bytes_per_party": dict(sorted(self.gradient_bytes.items())),
            "total_activation_bytes": self.total_activation_bytes,
            "total_gradient_bytes": self.total_gradient_bytes,
            "total_payload_bytes": self.total_payload_bytes,
            "timings_s": dict(self.timings_s),
        }


@dataclass
class RunTelemetry:
    """Cumulative telemetry across all SL steps of a run."""

    cumulative_activation_bytes: int = 0
    cumulative_gradient_bytes: int = 0
    n_steps: int = 0
    timings_s: dict[str, float] = field(default_factory=dict)

    def record_step(self, step: StepTelemetry) -> None:
        self.cumulative_activation_bytes += step.total_activation_bytes
        self.cumulative_gradient_bytes += step.total_gradient_bytes
        self.n_steps += 1
        for k, v in step.timings_s.items():
            self.timings_s[k] = self.timings_s.get(k, 0.0) + v

    @property
    def cumulative_payload_bytes(self) -> int:
        return self.cumulative_activation_bytes + self.cumulative_gradient_bytes

    def as_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "cumulative_activation_bytes": self.cumulative_activation_bytes,
            "cumulative_gradient_bytes": self.cumulative_gradient_bytes,
            "cumulative_payload_bytes": self.cumulative_payload_bytes,
            "mean_activation_bytes_per_step": (
                self.cumulative_activation_bytes / self.n_steps if self.n_steps else 0.0
            ),
            "timings_s": dict(self.timings_s),
        }


class timed:
    """Context manager that adds elapsed wall-clock seconds to a sink.

    Example:
        ``with timed(step.timings_s, "server_forward"): ...``
    """

    def __init__(self, sink: dict[str, float], key: str) -> None:
        self._sink = sink
        self._key = key
        self._start = 0.0

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        elapsed = time.perf_counter() - self._start
        self._sink[self._key] = self._sink.get(self._key, 0.0) + elapsed
