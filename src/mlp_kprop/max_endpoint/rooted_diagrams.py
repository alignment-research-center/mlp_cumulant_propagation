"""
Explicit rooted-diagram contractions for validating the argmax gradient
(small n, dense tensors; tests only).

Differentiating one operator term applied to Psi with respect to mu_r roots
the contraction at coordinate r. At the level of unrestricted slot labelings
this is the multi-index coefficient shift (spec section 5):

    d/dmu_r [ c * sum_l T[l] * d^{beta(l)} Psi ]
      = c * sum_l T[l] * d^{beta(l) + e_r} Psi,

where beta(l) is the multiplicity vector of the labeling l. The root either
adds a new endpoint-only coordinate (r not in l: support grows by one, an
extra u_{r,1} factor and a sign flip) or increments the Hermite multiplicity
of an existing vertex (r in l: u_{r,k} -> u_{r,k+1}) — both cases are handled
uniformly by d^{beta+e_r} Psi. The root introduces no new tensor edge, so the
primal graph (and treewidth) is unchanged.

This module evaluates that identity *directly* (dense loops over labelings,
endpoint derivatives from a shared quadrature workspace), independently of
torch autograd, so it can serve as the reference for the production gradient.
It also provides a DenseTermBridge + production-style autograd gradient of the
compiled quotient diagrams for the same term, so tests can compare:

    gradient of scalar diagram contraction  (production path, autograd)
        vs
    explicit rooted contraction             (shift identity, this module).
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
from torch import Tensor

from mlp_kprop.max_endpoint.arc_bridge import ArcBridge
from mlp_kprop.max_endpoint.diagrams import TermSpec, compile_term
from mlp_kprop.max_endpoint.hermite_endpoint import EndpointWorkspace
from mlp_kprop.max_endpoint.quadrature import (
    QuadratureCfg,
    find_endpoints,
    gauss_legendre_nodes,
)
from mlp_kprop.max_endpoint.treewidth import VEFactor, contract_factors, find_elimination_order

__all__ = [
    "endpoint_workspace_on_grid",
    "derivative_from_workspace",
    "scalar_term_reference",
    "rooted_term_reference",
    "DenseTermBridge",
    "term_argmax_gradient_autograd",
]


def endpoint_workspace_on_grid(
    mu: Tensor, sigma: Tensor, num_nodes: int = 2048
) -> EndpointWorkspace:
    """Shared endpoint workspace on the production-style fixed grid."""
    mu = mu.detach().to(torch.float64)
    sigma = sigma.detach().to(torch.float64)
    cfg = QuadratureCfg(num_nodes=num_nodes)
    lo, hi = find_endpoints(mu, sigma, cfg.tail_log_eps, cfg.bisect_iters)
    t, w = gauss_legendre_nodes(num_nodes, lo, hi, mu.device, torch.float64)
    return EndpointWorkspace(mu, sigma, t, w)


def derivative_from_workspace(ws: EndpointWorkspace, beta: dict[int, int]) -> float:
    """d^beta Psi / dmu^beta from cached unary weights: beta maps i -> order."""
    assert beta, "beta must have nonempty support"
    g = torch.ones_like(ws.t)
    for i, k in beta.items():
        g = g * ws.u(k)[:, i]
    p = len(beta)
    return float((-1.0) ** (p - 1) * ws.integrate(g))


def _labelings(term: TermSpec, n: int):
    """Yield (labeling dict slot->coordinate, tensor product value) pairs."""
    slots = term.slots
    for labels in itertools.product(range(n), repeat=len(slots)):
        lab = dict(zip(slots, labels))
        ok = True
        for pair in term.distinct_pairs:
            a, b = tuple(pair)
            if lab[a] == lab[b]:
                ok = False
                break
        if ok:
            yield lab


def _tensor_value(term: TermSpec, tensors: dict[str, Tensor], lab: dict[int, int]) -> float:
    val = 1.0
    for f in term.factors:
        val *= float(tensors[f.kind][tuple(lab[s] for s in f.legs)])
    return val


def scalar_term_reference(
    term: TermSpec, tensors: dict[str, Tensor], ws: EndpointWorkspace
) -> float:
    """Direct (term applied to Psi): c * sum_l T[l] d^{beta(l)} Psi."""
    total = 0.0
    for lab in _labelings(term, ws.n):
        beta: dict[int, int] = {}
        for s in term.slots:
            beta[lab[s]] = beta.get(lab[s], 0) + 1
        total += _tensor_value(term, tensors, lab) * derivative_from_workspace(ws, beta)
    return float(term.coef) * total


def rooted_term_reference(
    term: TermSpec, tensors: dict[str, Tensor], ws: EndpointWorkspace
) -> Tensor:
    """Explicit rooted contraction grad_mu (term Psi) via the shift identity."""
    n = ws.n
    q = torch.zeros(n, dtype=torch.float64)
    for lab in _labelings(term, ws.n):
        beta: dict[int, int] = {}
        for s in term.slots:
            beta[lab[s]] = beta.get(lab[s], 0) + 1
        tv = _tensor_value(term, tensors, lab)
        for r in range(n):
            rooted = dict(beta)
            rooted[r] = rooted.get(r, 0) + 1  # beta + e_r
            q[r] += tv * derivative_from_workspace(ws, rooted)
    return float(term.coef) * q


class DenseTermBridge:
    """Test-only bridge resolving diagram factor kinds from dense tensors."""

    def __init__(self, tensors: dict[str, Tensor], mu: Tensor, sigma: Tensor):
        self.tensors = {k: v.to(torch.float64) for k, v in tensors.items()}
        self.mu = mu.detach().to(torch.float64)
        self.sigma = sigma.detach().to(torch.float64)
        self.n = self.mu.shape[0]
        self.dtype = torch.float64
        self.k3_repr = "dense_dict"
        self.status: list[str] = []

    def build(
        self, kind: str, legs: tuple[int, ...], next_var: int, domains: dict[int, int]
    ) -> tuple[list[VEFactor], int]:
        vars_, t = ArcBridge._dense_merged(self.tensors[kind], legs)
        return [VEFactor(vars=vars_, tensor=t)], next_var


def term_argmax_gradient_autograd(
    term: TermSpec,
    tensors: dict[str, Tensor],
    mu: Tensor,
    sigma: Tensor,
    num_nodes: int = 2048,
) -> tuple[Tensor, float]:
    """Production-style gradient of the compiled scalar contraction of `term`.

    Compiles the term to quotient diagrams, contracts them by variable
    elimination on a fixed grid with mu as an autograd leaf, and returns
    (grad_mu of the summed correction, scalar correction value). One backward
    for the whole term; never one per coordinate.
    """
    bridge = DenseTermBridge(tensors, mu, sigma)
    diagrams = compile_term(term)
    cfg = QuadratureCfg(num_nodes=num_nodes)
    lo, hi = find_endpoints(bridge.mu, bridge.sigma, cfg.tail_log_eps, cfg.bisect_iters)
    mu_leaf = bridge.mu.clone().requires_grad_(True)
    with torch.enable_grad():
        t, w = gauss_legendre_nodes(num_nodes, lo, hi, bridge.mu.device, torch.float64)
        ws = EndpointWorkspace(mu_leaf, bridge.sigma, t, w)
        total = torch.zeros((), dtype=torch.float64)
        for d in diagrams:
            domains: dict[int, int] = {}
            factors = []
            for v, orders in enumerate(d.vertex_orders):
                domains[v] = bridge.n
                factors.append(VEFactor(vars=(v,), tensor=ws.vertex_weight(orders), batched=True))
            next_var = d.num_vertices
            for kind, legs in d.factors:
                fs, next_var = bridge.build(kind, legs, next_var, domains)
                factors.extend(fs)
            info = find_elimination_order(sorted(domains.keys()), [f.vars for f in factors])
            vals = contract_factors(factors, domains, info.order, batch=ws.q)
            total = total + d.coef * ws.integrate(vals)
        grad = torch.autograd.grad(total, mu_leaf)[0]
    return grad.detach(), float(total.detach())
