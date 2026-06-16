"""Reproducible seeding for split-learning runs.

A single entry point, :func:`seed_everything`, seeds Python, NumPy and (when
available) PyTorch CPU/CUDA RNGs. Importing this module never requires PyTorch:
torch is imported lazily inside the function so the base ``rizemind`` install
(which does not depend on torch) keeps working.

This replaces the previous external ``sitecustomize`` shim: callers seed
explicitly at the start of ``client_fn`` / ``server_fn`` instead of relying on
``PYTHONPATH`` injection.
"""

from __future__ import annotations

import random

import numpy as np

#: NumPy's legacy global seed must fit in ``[0, 2**32)``.
_NUMPY_SEED_MOD = 2**32


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and (if installed) PyTorch CPU/CUDA RNGs.

    Args:
        seed: Base seed. The same seed reproduces the same model initialisation
            and any RNG draws that follow, on the same backend.
        deterministic: When True, also request deterministic algorithms from
            PyTorch (``use_deterministic_algorithms(warn_only=True)`` and cuDNN
            deterministic mode). Off by default because it can disable some
            CUDA kernels or slow them down; remaining nondeterminism is then a
            property of the execution backend, not the seeding.

    Returns:
        The seed that was applied (handy for logging into result files).
    """
    random.seed(seed)
    np.random.seed(seed % _NUMPY_SEED_MOD)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed
