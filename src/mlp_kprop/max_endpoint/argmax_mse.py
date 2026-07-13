"""
Unbiased Brier-MSE estimation for expected-argmax predictions from winner
counts (spec sections 12-13).

For a fixed network theta with true winner distribution q and a deterministic
prediction q_hat, draw m i.i.d. Gaussian inputs, let c_i be the number of
samples whose argmax coordinate is i (tie policy: torch.argmax first index;
ties otherwise ignored). Unbiased estimators:

    q_hat . q   <-  sum_i q_hat_i c_i / m
    ||q||^2     <-  sum_i c_i (c_i - 1) / (m (m - 1))     (collision U-statistic)

so

    mse_unbiased(q_hat, c)
      = ||q_hat||^2 - 2 sum_i q_hat_i c_i / m + sum_i c_i(c_i-1)/(m(m-1))

satisfies E[mse_unbiased | theta] = ||q_hat - q||_2^2. It may legitimately be
negative for finite m and is never clipped. All arithmetic is float64.

Monte Carlo baseline: the m-sample empirical winner frequency q_hat_MC has

    E||q_hat_MC - q||_2^2 = (1 - ||q||^2) / m,

with ||q||^2 estimated by the same collision U-statistic, giving a smooth MC
MSE curve over sample budgets without repeated MC runs.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mlp_kprop.mlp import MLP

__all__ = [
    "winner_counts",
    "mse_unbiased",
    "mse_unbiased_two_sample",
    "collision_probability",
    "mc_mse_predicted",
    "argmax_flops_per_sample",
]


@torch.no_grad()
def winner_counts(
    mlp: MLP,
    num_samples: int,
    seed: int,
    device: torch.device | None = None,
    batch_size: int = 262_144,
) -> Tensor:
    """Winner counts c_i over num_samples Gaussian inputs (int64, length n).

    Tie policy: torch.argmax (first maximizing index); ties otherwise ignored.
    """
    if device is None:
        device = next(mlp.parameters()).device
    n_in = mlp.input_dim
    n_out = mlp.output_dim
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    counts = torch.zeros(n_out, dtype=torch.int64, device=device)
    done = 0
    bs = batch_size
    while done < num_samples:
        try:
            x = torch.randn(min(bs, num_samples - done), n_in, device=device, generator=gen)
            idx = mlp(x).out.argmax(dim=1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(bs // 2, 1024)
            continue
        counts += torch.bincount(idx, minlength=n_out)
        done += x.shape[0]
    return counts.cpu()


def mse_unbiased(q_hat: Tensor, counts: Tensor) -> float:
    """Unbiased estimate of ||q_hat - q||_2^2 from one block of winner counts."""
    q = q_hat.detach().to(torch.float64).cpu()
    c = counts.detach().to(torch.float64).cpu()
    m = float(c.sum())
    assert m >= 2, "need at least two samples for the collision estimator"
    dot = float((q * c).sum()) / m
    coll = float((c * (c - 1.0)).sum()) / (m * (m - 1.0))
    return float((q * q).sum()) - 2.0 * dot + coll


def mse_unbiased_two_sample(q_hat: Tensor, i: int, j: int) -> float:
    """Two-observation estimator (validation, spec section 12):

        ||q_hat||^2 - q_hat[I] - q_hat[J] + 1{I = J},

    unbiased for ||q_hat - q||_2^2 when I, J are independent argmax draws.
    """
    q = q_hat.detach().to(torch.float64).cpu()
    return float((q * q).sum()) - float(q[i]) - float(q[j]) + (1.0 if i == j else 0.0)


def collision_probability(counts: Tensor) -> float:
    """Unbiased collision U-statistic for ||q||^2: sum_i c_i(c_i-1)/(m(m-1))."""
    c = counts.detach().to(torch.float64).cpu()
    m = float(c.sum())
    assert m >= 2
    return float((c * (c - 1.0)).sum()) / (m * (m - 1.0))


def mc_mse_predicted(q_norm_sq: float, m: int | float) -> float:
    """E||q_hat_MC - q||_2^2 for m-sample Monte Carlo: (1 - ||q||^2)/m."""
    return (1.0 - q_norm_sq) / float(m)


@torch.no_grad()
def argmax_flops_per_sample(mlp: MLP) -> float:
    """Modeled per-sample FLOPs of the MC argmax baseline.

    One instrumented MLP forward (ARC convention, RNG excluded) plus the
    argmax reduction (n-1 comparisons). Input radius does not affect argmax
    for zero-bias positively homogeneous ReLU nets, so there is no separate
    Rao-Blackwellized spherical baseline here.
    """
    from mlp_kprop.flop_utils import NamedFlopCounter

    device = next(mlp.parameters()).device
    batch = 128
    x = torch.randn(batch, mlp.input_dim, device=device)
    with NamedFlopCounter() as counter:
        mlp(x)
    forward = counter.total() / batch
    return float(forward + (mlp.output_dim - 1))
