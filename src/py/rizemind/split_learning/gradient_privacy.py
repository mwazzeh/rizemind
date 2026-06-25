"""Clipped-Gaussian perturbation of cut gradients at the label holder.

In label-private VFL the label holder computes the joint cut gradient
``G in R^{BxD}`` (``D = K*H``) and returns it to the coordinator, which forwards
each party its slice. The sign/direction/magnitude of these rows is
label-correlated, so an honest-but-curious coordinator can infer labels from
them (see ``THREAT_MODEL.md``). This module bounds and masks that signal **before
any coordinator-bound serialization**.

Mechanism (exact order), applied to the joint per-sample gradient matrix ``G``:

1. compute each sample row's L2 norm ``||G_i||2``;
2. **clip** each row to the bound ``C``: ``G_i <- G_i * min(1, C/||G_i||2)``;
3. **add Gaussian noise** to each released per-sample row independently:
   ``G̃_i = clip(G_i) + N(0, sigma^2 I_D)`` with ``sigma = noise_multiplier * C``;
4. the (still per-sample) protected matrix ``G̃`` is split back into party slices
   by the caller and serialized.

Noise semantics (unambiguous): noise is added **independently to each released
per-sample row**, with per-coordinate standard deviation ``sigma = z * C`` where
``z`` is the noise multiplier. It is *not* added to a sum or to an average — the
protocol requires a usable per-sample gradient for every party, so a
sum-then-noise (DP-SGD) construction would not fit without changing the protocol.
Consequently a formal per-step DP statement would require a *per-release*
analysis that is **not** claimed here; see :func:`GradientPrivacyConfig` and
``THREAT_MODEL.md`` (``formal_dp_claim = false`` by default).

This module is NumPy-only and records aggregate, non-sensitive diagnostics
(norm statistics, clip fraction, sigma, SNR). It never logs or returns an individual
gradient vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GradientPrivacyConfig",
    "make_rng",
    "clip_rows",
    "row_l2_norms",
    "privatize_joint_gradient",
]

#: Allowed values for ``GradientPrivacyConfig.mode``.
MODE_NONE = "none"
MODE_CLIP = "clip"
MODE_GAUSSIAN = "gaussian"
_MODES = (MODE_NONE, MODE_CLIP, MODE_GAUSSIAN)

#: Allowed RNG modes.
RNG_RESEARCH = "research-seeded"
RNG_SECURE = "secure"
_RNG_MODES = (RNG_RESEARCH, RNG_SECURE)


@dataclass(frozen=True)
class GradientPrivacyConfig:
    """Configuration for cut-gradient clipping + Gaussian perturbation.

    Attributes:
        mode: ``"none"`` (pass-through), ``"clip"`` (per-row L2 clip only), or
            ``"gaussian"`` (clip + per-row Gaussian noise).
        clip_norm: Per-sample L2 clipping bound ``C`` (> 0). Ignored in
            ``"none"`` mode.
        noise_multiplier: ``z`` such that noise sigma = ``z * clip_norm``. Used only
            in ``"gaussian"`` mode. ``0.0`` makes ``"gaussian"`` equivalent to
            ``"clip"``.
        delta: Target delta recorded for *future* privacy accounting. Not used to
            compute any epsilon here.
        rng_mode: ``"research-seeded"`` (reproducible, deterministic — NOT for a
            real privacy deployment) or ``"secure"`` (OS-backed entropy, no
            reusable seed persisted).
        formal_dp_claim: Always ``False`` in this phase — no audited accountant
            backs an epsilon under the actual sampling/composition. Kept as an explicit
            field so result readers never have to infer it.
    """

    mode: str = MODE_NONE
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    delta: float = 1e-5
    rng_mode: str = RNG_RESEARCH
    formal_dp_claim: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if self.rng_mode not in _RNG_MODES:
            raise ValueError(
                f"rng_mode must be one of {_RNG_MODES}, got {self.rng_mode!r}"
            )
        if self.mode != MODE_NONE and not (self.clip_norm > 0):
            raise ValueError(
                f"clip_norm must be > 0 in mode {self.mode!r}, got {self.clip_norm}"
            )
        if self.noise_multiplier < 0:
            raise ValueError(
                f"noise_multiplier must be >= 0, got {self.noise_multiplier}"
            )
        if not (0.0 < self.delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")

    @property
    def noise_std(self) -> float:
        """Per-coordinate Gaussian sigma actually applied (0 unless gaussian mode)."""
        if self.mode != MODE_GAUSSIAN:
            return 0.0
        return float(self.noise_multiplier) * float(self.clip_norm)

    @property
    def is_active(self) -> bool:
        """True when the mechanism changes the gradient (clip and/or noise)."""
        return self.mode != MODE_NONE

    def as_dict(self) -> dict:
        return {
            "gradient_privacy_mode": self.mode,
            "gradient_clip_norm": float(self.clip_norm) if self.is_active else None,
            "gradient_noise_multiplier": (
                float(self.noise_multiplier) if self.mode == MODE_GAUSSIAN else 0.0
            ),
            "gradient_noise_std": self.noise_std,
            "privacy_delta": float(self.delta),
            "privacy_rng_mode": self.rng_mode,
            "formal_dp_claim": bool(self.formal_dp_claim),
            "epsilon": None,
        }


def make_rng(
    config: GradientPrivacyConfig, *, research_seed: int | None
) -> np.random.Generator:
    """Build the noise RNG for a release.

    Research-seeded mode returns a deterministic generator from ``research_seed``
    (reproducible experiments). Secure mode returns a generator seeded from OS
    entropy via :func:`numpy.random.default_rng` with no argument; the entropy is
    not exposed or persisted, so protected runs cannot be replayed bit-for-bit.

    Args:
        config: The gradient-privacy configuration.
        research_seed: Deterministic seed for ``"research-seeded"`` mode. Ignored
            (and may be ``None``) in ``"secure"`` mode.

    Returns:
        A NumPy ``Generator``.

    Raises:
        ValueError: If research mode is requested without a seed.
    """
    if config.rng_mode == RNG_SECURE:
        # OS entropy; the seed is never returned or stored.
        return np.random.default_rng()
    if research_seed is None:
        raise ValueError("research-seeded RNG requires a research_seed")
    return np.random.default_rng(int(research_seed))


def row_l2_norms(grad: np.ndarray) -> np.ndarray:
    """Return the per-row (per-sample) L2 norms of a ``(B, D)`` gradient matrix."""
    g = np.asarray(grad, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"expected a 2-D (B, D) gradient, got shape {g.shape}")
    return np.sqrt(np.sum(g * g, axis=1))


def clip_rows(grad: np.ndarray, clip_norm: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip each row of ``grad`` to L2 norm ``clip_norm``.

    Row ``i`` is scaled by ``min(1, clip_norm / ||row_i||2)``. Zero rows are left
    unchanged (no division by zero).

    Args:
        grad: ``(B, D)`` gradient matrix.
        clip_norm: Positive L2 bound ``C``.

    Returns:
        ``(clipped, pre_clip_norms)`` where ``clipped`` has the same shape/dtype
        family as the input (float) and ``pre_clip_norms`` is the ``(B,)`` vector
        of original row norms.

    Raises:
        ValueError: If ``clip_norm <= 0`` or ``grad`` is not 2-D.
    """
    if not (clip_norm > 0):
        raise ValueError(f"clip_norm must be > 0, got {clip_norm}")
    g = np.asarray(grad, dtype=np.float64)
    norms = row_l2_norms(g)
    # Avoid divide-by-zero: factor is 1.0 wherever the row norm is 0.
    safe = np.where(norms > 0, norms, 1.0)
    factors = np.minimum(1.0, clip_norm / safe)
    clipped = g * factors[:, None]
    return clipped.astype(np.float32, copy=False), norms


def privatize_joint_gradient(
    grad: np.ndarray,
    config: GradientPrivacyConfig,
    *,
    rng: np.random.Generator | None = None,
    research_seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply the clip + Gaussian mechanism to a joint cut-gradient matrix.

    Processing order (see module docstring): per-row L2 norm → clip each row to
    ``C`` → add per-row Gaussian noise ``N(0, sigma^2I)`` with ``sigma = z*C`` (gaussian
    mode only). In ``"none"`` mode the input is returned unchanged.

    Args:
        grad: Joint per-sample gradient ``(B, D)`` produced at the label holder.
        config: Gradient-privacy configuration.
        rng: Optional pre-built NumPy generator (used for noise). If omitted, one
            is built via :func:`make_rng`.
        research_seed: Deterministic seed used to build the RNG when ``rng`` is
            ``None`` and ``rng_mode == "research-seeded"``.

    Returns:
        ``(protected, diagnostics)``. ``protected`` is a float32 ``(B, D)``
        matrix; in ``"none"`` mode it is the input cast to float32. ``diagnostics``
        is a dict of aggregate, non-sensitive statistics — never any individual
        gradient vector.
    """
    g = np.asarray(grad, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"expected a 2-D (B, D) gradient, got shape {g.shape}")
    b, d = g.shape

    diag: dict = {
        "gradient_privacy_mode": config.mode,
        "batch_size": int(b),
        "grad_width": int(d),
    }

    if config.mode == MODE_NONE:
        pre = row_l2_norms(g)
        diag.update(_norm_stats(pre, prefix="pre_clip"))
        diag.update(
            clip_fraction=0.0,
            clip_norm=None,
            noise_std=0.0,
            snr=None,
        )
        diag.update(_norm_stats(pre, prefix="post_clip"))
        return g.astype(np.float32, copy=False), diag

    clipped, pre = clip_rows(g, config.clip_norm)
    clip_fraction = float(np.mean(pre > config.clip_norm)) if b else 0.0
    post = row_l2_norms(clipped)
    diag.update(_norm_stats(pre, prefix="pre_clip"))
    diag.update(_norm_stats(post, prefix="post_clip"))
    diag.update(clip_fraction=clip_fraction, clip_norm=float(config.clip_norm))

    sigma = config.noise_std
    if config.mode == MODE_GAUSSIAN and sigma > 0:
        if rng is None:
            rng = make_rng(config, research_seed=research_seed)
        noise = rng.normal(loc=0.0, scale=sigma, size=(b, d))
        protected = clipped.astype(np.float64) + noise
        # Signal-to-noise: mean post-clip signal energy per row vs noise energy.
        # noise row energy ~ d * sigma^2; signal row energy ~ mean(post^2).
        signal_energy = float(np.mean(post**2)) if b else 0.0
        noise_energy = float(d) * sigma * sigma
        snr = signal_energy / noise_energy if noise_energy > 0 else None
        diag.update(noise_std=float(sigma), snr=snr)
        return protected.astype(np.float32, copy=False), diag

    # clip-only (or gaussian with z == 0): no noise.
    diag.update(noise_std=0.0, snr=None)
    return clipped, diag


def _norm_stats(norms: np.ndarray, *, prefix: str) -> dict:
    """Aggregate, non-sensitive per-row-norm statistics (never raw vectors)."""
    if norms.size == 0:
        return {
            f"{prefix}_norm_mean": 0.0,
            f"{prefix}_norm_median": 0.0,
            f"{prefix}_norm_p90": 0.0,
            f"{prefix}_norm_p99": 0.0,
            f"{prefix}_norm_max": 0.0,
        }
    return {
        f"{prefix}_norm_mean": float(np.mean(norms)),
        f"{prefix}_norm_median": float(np.median(norms)),
        f"{prefix}_norm_p90": float(np.percentile(norms, 90)),
        f"{prefix}_norm_p99": float(np.percentile(norms, 99)),
        f"{prefix}_norm_max": float(np.max(norms)),
    }
