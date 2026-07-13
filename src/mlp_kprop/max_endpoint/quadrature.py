"""
Numerically stable quadrature for the product-Gaussian maximum

    Psi(mu, sigma) = E_Z[max_i (mu_i + sigma_i Z_i)],   Z_i iid N(0,1).

With a_i(t) = (t - mu_i)/sigma_i and F(t) = prod_i Phi(a_i(t)),

    Psi = int_0^inf [1 - F(t)] dt - int_{-inf}^0 F(t) dt
        = hi - int_lo^hi F(t) dt + O(tail),

for any lo <= hi with F(lo) ~ 0 and F(hi) ~ 1 (the second form holds for any
placement of 0 relative to [lo, hi]; it follows from
E[M] = b + int_b^inf (1-F) - int_{-inf}^b F with b = hi).

F is always evaluated through log F(t) = sum_i log_ndtr(a_i(t)); CDF values are
never multiplied directly in floating point.

Two backends:
- "gl": fixed-node Gauss-Legendre on [lo, hi] (vectorized, GPU-safe, production).
  Endpoints are found by bisection on log F. Convergence is verified by
  comparing Q and 2Q nodes.
- "adaptive": scipy.integrate.quad on the two half-line integrals (slow CPU
  reference used by tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "QuadratureCfg",
    "gauss_legendre_nodes",
    "log_F",
    "find_endpoints",
    "product_gaussian_max",
    "product_gaussian_max_reference",
]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass
class QuadratureCfg:
    """Configuration for the production Gauss-Legendre backend.

    Attributes:
        num_nodes: number of Gauss-Legendre nodes Q.
        convergence_factor: Psi is recomputed at convergence_factor * Q nodes;
            the difference is reported as the quadrature error estimate.
        tail_log_eps: bisection targets log F(lo) = tail_log_eps and
            log(1 - F(hi)) = tail_log_eps  (default corresponds to ~1e-25).
        bisect_iters: number of bisection iterations for each endpoint.
        dtype: floating dtype for all quadrature computations.
    """

    num_nodes: int = 256
    convergence_factor: int = 2
    tail_log_eps: float = math.log(1e-25)
    bisect_iters: int = 80
    dtype: torch.dtype = field(default=torch.float64)


def gauss_legendre_nodes(
    q: int, lo: float, hi: float, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Gauss-Legendre nodes/weights mapped from [-1, 1] to [lo, hi]."""
    x, w = np.polynomial.legendre.leggauss(q)
    t = torch.as_tensor(0.5 * (hi - lo) * (x + 1.0) + lo, device=device, dtype=dtype)
    w = torch.as_tensor(0.5 * (hi - lo) * w, device=device, dtype=dtype)
    return t, w


def log_F(t: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """log F(t) = sum_i log Phi((t - mu_i)/sigma_i), stably via log_ndtr.

    Args:
        t: (...,) evaluation points.
        mu, sigma: (n,) coordinate means and standard deviations.
    Returns:
        (...,) tensor of log F values.
    """
    a = (t.unsqueeze(-1) - mu) / sigma
    return torch.special.log_ndtr(a).sum(dim=-1)


def _bisect(f, lo: float, hi: float, iters: int) -> float:
    """Find t with f(t) = 0 for increasing f, f(lo) <= 0 <= f(hi), by bisection."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def find_endpoints(
    mu: Tensor, sigma: Tensor, tail_log_eps: float, bisect_iters: int = 80
) -> tuple[float, float]:
    """Find finite endpoints [lo, hi] with log F(lo) <= tail_log_eps and
    log(1 - F(hi)) <= tail_log_eps, by bisection on the monotone function log F.

    The initial bracket [mu_i - c sigma_i, mu_i + c sigma_i] with c = 14 already
    guarantees both tail conditions for n up to ~1e9; bisection then tightens it.
    """
    c = 14.0
    lo0 = float((mu - c * sigma).min())
    hi0 = float((mu + c * sigma).max())

    def logf(t: float) -> float:
        return float(log_F(torch.tensor(t, device=mu.device, dtype=mu.dtype), mu, sigma))

    # log F is increasing in t. lo: log F(lo) = tail_log_eps.
    lo = _bisect(lambda t: logf(t) - tail_log_eps, lo0, hi0, bisect_iters)
    # hi: log(1 - F(hi)) = tail_log_eps, i.e. log(-expm1(log F(hi))) = tail_log_eps.
    # log(-expm1(x)) is decreasing in x (x<0), and x -> 0- as t -> inf.
    def neg_tail(t: float) -> float:
        x = logf(t)
        # -expm1(x) = 1 - F; guard against x == 0 exactly.
        one_minus_f = max(-math.expm1(min(x, -1e-300)), 1e-300)
        return -(math.log(one_minus_f) - tail_log_eps)

    hi = _bisect(neg_tail, lo0, hi0, bisect_iters)
    assert lo < hi, f"Endpoint bisection failed: lo={lo} >= hi={hi}"
    return lo, hi


def _psi_gl(mu: Tensor, sigma: Tensor, q: int, lo: float, hi: float) -> tuple[Tensor, int]:
    """Psi ~= hi - int_lo^hi F(t) dt with Q-node Gauss-Legendre.

    Returns (psi, flops). FLOP model: per node per coordinate ~4 ops for a_i and
    log_ndtr, plus the sum reduction, exp and weighted sum: 5*Q*n + 3*Q.
    """
    t, w = gauss_legendre_nodes(q, lo, hi, mu.device, mu.dtype)
    f = torch.exp(log_F(t, mu, sigma))
    psi = hi - (w * f).sum()
    flops = 5 * q * mu.numel() + 3 * q
    return psi, flops


def product_gaussian_max(
    mu: Tensor, sigma: Tensor, cfg: QuadratureCfg | None = None
) -> tuple[float, float, dict]:
    """Production estimate of Psi(mu, sigma) = E[max_i(mu_i + sigma_i Z_i)].

    Returns:
        (psi, error_estimate, info) where error_estimate = |psi_Q - psi_2Q|
        (psi is the higher-resolution value) and info records endpoints,
        node counts and modeled FLOPs.
    """
    if cfg is None:
        cfg = QuadratureCfg()
    mu = mu.to(dtype=cfg.dtype)
    sigma = sigma.to(dtype=cfg.dtype)
    assert (sigma > 0).all(), "product_gaussian_max requires strictly positive sigma"
    lo, hi = find_endpoints(mu, sigma, cfg.tail_log_eps, cfg.bisect_iters)
    q = cfg.num_nodes
    psi_q, flops_q = _psi_gl(mu, sigma, q, lo, hi)
    q2 = cfg.convergence_factor * q
    psi_q2, flops_q2 = _psi_gl(mu, sigma, q2, lo, hi)
    err = float(abs(psi_q2 - psi_q))
    info = {
        "lo": lo,
        "hi": hi,
        "num_nodes": q,
        "num_nodes_check": q2,
        "flops": flops_q + flops_q2 + 2 * cfg.bisect_iters * (4 * mu.numel() + 1),
    }
    return float(psi_q2), err, info


def product_gaussian_max_reference(
    mu: np.ndarray | Tensor, sigma: np.ndarray | Tensor, epsabs: float = 1e-13
) -> float:
    """Slow adaptive high-accuracy CPU reference for Psi(mu, sigma).

    Uses scipy.integrate.quad on
        Psi = int_0^inf [1 - F(t)] dt - int_{-inf}^0 F(t) dt
    with F evaluated through scipy.special.log_ndtr.
    """
    from scipy import integrate, special

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    def logF(t: float) -> float:
        return float(np.sum(special.log_ndtr((t - mu) / sigma)))

    def one_minus_F(t: float) -> float:
        return -math.expm1(logF(t))

    def F(t: float) -> float:
        return math.exp(logF(t))

    hi_pt = float((mu + 14 * sigma).max())
    lo_pt = float((mu - 14 * sigma).min())
    upper, _ = integrate.quad(one_minus_F, 0.0, max(hi_pt, 1.0), epsabs=epsabs, limit=500)
    lower, _ = integrate.quad(F, min(lo_pt, -1.0), 0.0, epsabs=epsabs, limit=500)
    return upper - lower
