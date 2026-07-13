"""
Endpoint derivatives of the product-Gaussian maximum.

Let a_i(t) = (t - mu_i)/sigma_i, B(t) = prod_j Phi(a_j(t)) and let He_k denote
the probabilists' Hermite polynomials

    He_0(x) = 1,  He_1(x) = x,  He_2(x) = x^2 - 1,  He_3(x) = x^3 - 3x.

For a multi-index beta with support S = supp(beta), p = |S|:

    d^beta/dmu^beta Psi(mu, sigma)
      = (-1)^(p-1) * int_R B(t) * prod_{i in S} u_{i, beta_i}(t) dt,

with unary endpoint weights

    u_{i,k}(t) = phi(a_i(t)) He_{k-1}(a_i(t)) / (sigma_i^k Phi(a_i(t))).

The hazard-type ratio phi/Phi is computed as exp(log phi - log_ndtr), which is
stable for arbitrarily negative arguments; B is computed as exp(sum log_ndtr),
which underflows gracefully to 0 exactly where the integrand is negligible.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from mlp_kprop.max_endpoint.quadrature import (
    QuadratureCfg,
    find_endpoints,
    gauss_legendre_nodes,
)

__all__ = ["hermite_he", "EndpointWorkspace", "endpoint_derivative_reference"]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def hermite_he(k: int, x: Tensor) -> Tensor:
    """Probabilists' Hermite polynomial He_k(x) for k = 0..3 (closed form),
    with a recurrence fallback He_{k+1} = x He_k - k He_{k-1} for k > 3."""
    if k == 0:
        return torch.ones_like(x)
    if k == 1:
        return x
    if k == 2:
        return x * x - 1.0
    if k == 3:
        return x * (x * x - 3.0)
    h_prev, h = hermite_he(2, x), hermite_he(3, x)
    for j in range(3, k):
        h_prev, h = h, x * h - j * h_prev
    return h


class EndpointWorkspace:
    """Per-quadrature-node quantities shared by all endpoint diagrams.

    Precomputes, on a fixed node set t_q (q = 1..Q):
        a[q, i]      = (t_q - mu_i)/sigma_i
        logB[q]      = sum_i log Phi(a[q, i])
        B[q]         = exp(logB)
        ratio[q, i]  = phi(a)/Phi(a) = exp(log phi(a) - log_ndtr(a))
        u_k[q, i]    = ratio * He_{k-1}(a) / sigma_i^k     (lazily, k = 1..4)

    All tensors use the dtype/device of mu. The analytic FLOP count of the
    setup and of each u_k is tracked in self.flops.
    """

    def __init__(self, mu: Tensor, sigma: Tensor, t: Tensor, w: Tensor):
        assert mu.ndim == 1 and sigma.ndim == 1 and t.ndim == 1
        self.mu = mu
        self.sigma = sigma
        self.t = t
        self.w = w
        self.n = mu.shape[0]
        self.q = t.shape[0]
        a = (t[:, None] - mu[None, :]) / sigma[None, :]
        self.a = a
        log_phi = -0.5 * a * a - _LOG_SQRT_2PI
        log_ndtr = torch.special.log_ndtr(a)
        self.logB = log_ndtr.sum(dim=1)
        self.B = torch.exp(self.logB)
        self.wB = w * self.B  # quadrature weight folded into B
        self.ratio = torch.exp(log_phi - log_ndtr)
        self._u: dict[int, Tensor] = {}
        # a: 2 ops; log_phi: 3; log_ndtr: 1; sum: 1; exp(B): 1/Q..; ratio: 2.
        self.flops = 9 * self.q * self.n + 3 * self.q

    def u(self, k: int) -> Tensor:
        """Unary endpoint weight u_{., k} of shape (Q, n)."""
        assert k >= 1, "Endpoint weights are defined for derivative order k >= 1."
        if k not in self._u:
            he = hermite_he(k - 1, self.a)
            self._u[k] = self.ratio * he / self.sigma[None, :] ** k
            # He eval (<=5 ops) + ratio mult + sigma^k division.
            self.flops += 7 * self.q * self.n
        return self._u[k]

    def vertex_weight(self, orders: tuple[int, ...]) -> Tensor:
        """Product of unary weights prod_j u_{., orders_j}, shape (Q, n).

        A quotient vertex formed by merging derivative blocks of sizes
        (k_1, ..., k_m) carries the product u_{i,k_1} * ... * u_{i,k_m}.
        """
        out = self.u(orders[0])
        for k in orders[1:]:
            out = out * self.u(k)
            self.flops += self.q * self.n
        return out

    def integrate(self, g: Tensor) -> Tensor:
        """int B(t) g(t) dt ~= sum_q w_q B(t_q) g[q] for g of shape (Q,)."""
        self.flops += 2 * self.q
        return (self.wB * g).sum()


def endpoint_derivative_reference(
    mu: Tensor, sigma: Tensor, beta: tuple[int, ...], quad_cfg: QuadratureCfg | None = None
) -> float:
    """Direct evaluation of d^beta Psi / dmu^beta via the endpoint integral.

    Intended for small n in tests; uses a dense Gauss-Legendre grid with the
    same endpoint selection as the production quadrature.
    """
    if quad_cfg is None:
        quad_cfg = QuadratureCfg(num_nodes=2048)
    mu = mu.to(torch.float64)
    sigma = sigma.to(torch.float64)
    lo, hi = find_endpoints(mu, sigma, quad_cfg.tail_log_eps, quad_cfg.bisect_iters)
    t, w = gauss_legendre_nodes(quad_cfg.num_nodes, lo, hi, mu.device, torch.float64)
    ws = EndpointWorkspace(mu, sigma, t, w)
    support = [i for i, b in enumerate(beta) if b > 0]
    p = len(support)
    assert p >= 1, "beta must have nonempty support"
    g = torch.ones_like(t)
    for i in support:
        g = g * ws.u(beta[i])[:, i]
    return float((-1.0) ** (p - 1) * ws.integrate(g))
