"""
Monte Carlo ground truth for T(theta) = E_X[max_i M_theta(X)_i] without hiding
reference noise.

Two independent reference streams A and B (independent input seeds) give the
cross-fidelity error estimator

    err_cross = (T_hat - T_ref_A)(T_hat - T_ref_B),
    E[err_cross | theta] = (T_hat - T(theta))^2,

whose average over network seeds estimates the deterministic MSE with no
additive target-noise bias. err_cross may legitimately be negative; it is
never clipped here.

Backends:
- "gaussian": X ~ N(0, I_n), value = max_i M(X)_i.
- "spherical" (Rao-Blackwellized): for bias-free positively homogeneous
  networks (ReLU, zero biases), M(cX) = c M(X) for c > 0, so with X = R * u,
  u uniform on S^{n-1} independent of R = |X|:

      T = E[R] * E_u[max_i M(u)_i],   E[R] = sqrt(2) Gamma((n+1)/2)/Gamma(n/2).

  Implemented as value = E[R] * max_i M(X)_i / |X| for Gaussian X, which
  integrates out the radius exactly and reduces variance. Automatically
  refused for networks with biases.

Streaming float64 accumulation, adaptive stopping on the standard error, and
OOM-aware batch backoff.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from mlp_kprop.mlp import MLP

logger = logging.getLogger(__name__)

__all__ = ["ReferenceResult", "expected_chi_norm", "reference_estimate", "mc_flops_per_sample"]


@dataclass
class ReferenceResult:
    mean: float
    se: float
    var: float          # per-sample variance of the backend's values
    num_samples: int
    stopping_reason: str
    backend: str
    seed: int
    batch_size: int     # final (possibly backed-off) batch size


def expected_chi_norm(n: int) -> float:
    """E|X| for X ~ N(0, I_n): sqrt(2) * Gamma((n+1)/2) / Gamma(n/2)."""
    return math.sqrt(2.0) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))


@torch.no_grad()
def _batch_values(mlp: MLP, x: Tensor, backend: str, e_r: float) -> Tensor:
    """Per-sample reference values (float64) for one batch of Gaussian inputs."""
    out = mlp(x).out
    m = out.max(dim=1).values
    if backend == "spherical":
        m = e_r * m / x.norm(dim=1)
    return m.to(torch.float64)


@torch.no_grad()
def reference_estimate(
    mlp: MLP,
    seed: int,
    backend: str = "spherical",
    target_se: float = 1e-4,
    min_samples: int = 100_000,
    max_samples: int = 200_000_000,
    batch_size: int = 262_144,
    device: torch.device | None = None,
) -> ReferenceResult:
    """Streaming MC estimate of E_X[max_i M(X)_i] with adaptive stopping.

    Stops when the standard error of the mean falls below target_se (after at
    least min_samples), or at max_samples (stopping_reason records which).
    """
    if backend == "spherical" and mlp.has_bias():
        raise ValueError(
            "Spherical (Rao-Blackwellized) reference requires a bias-free network; "
            "positive homogeneity fails with biases."
        )
    if device is None:
        device = next(mlp.parameters()).device
    n = mlp.input_dim
    e_r = expected_chi_norm(n)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    count = 0
    total = torch.zeros((), dtype=torch.float64, device=device)
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    stopping = "max_samples"
    bs = batch_size
    while count < max_samples:
        try:
            x = torch.randn(min(bs, max_samples - count), n, device=device, generator=gen)
            vals = _batch_values(mlp, x, backend, e_r)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(bs // 2, 1024)
            logger.warning(f"OOM in reference batch; backing off to batch_size={bs}")
            continue
        count += vals.numel()
        total += vals.sum()
        total_sq += vals.square().sum()
        if count >= min_samples:
            mean = (total / count).item()
            var = max((total_sq / count).item() - mean * mean, 0.0) * count / max(count - 1, 1)
            se = math.sqrt(var / count)
            if se <= target_se:
                stopping = "target_se"
                break
    mean = (total / count).item()
    var = max((total_sq / count).item() - mean * mean, 0.0) * count / max(count - 1, 1)
    se = math.sqrt(var / count)
    return ReferenceResult(
        mean=mean,
        se=se,
        var=var,
        num_samples=count,
        stopping_reason=stopping,
        backend=backend,
        seed=seed,
        batch_size=bs,
    )


@torch.no_grad()
def mc_flops_per_sample(mlp: MLP, backend: str = "gaussian") -> float:
    """Modeled per-sample FLOPs of the MC baseline, matching ARC's convention.

    ARC measures one MLP forward per sample with its instrumented counter
    (random-number generation excluded); we add the max reduction (n-1 ops)
    and, for the spherical backend, the norm (2n+1) and scale (2) ops.
    """
    from mlp_kprop.flop_utils import NamedFlopCounter

    device = next(mlp.parameters()).device
    n = mlp.input_dim
    batch = 128
    x = torch.randn(batch, n, device=device)
    with NamedFlopCounter() as counter:
        mlp(x)
    forward = counter.total() / batch
    out_dim = mlp.output_dim
    extra = out_dim - 1  # max reduction
    if backend == "spherical":
        extra += 2 * n + 1 + 2  # squared norm + sqrt + divide/scale
    return float(forward + extra)
