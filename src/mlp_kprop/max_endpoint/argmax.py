"""
Expected one-hot argmax vector of a random MLP output, by differentiating the
scalar expected-max endpoint with respect to an additive output-mean shift.

Identity: let H(y) = max_j y_j and T(a) = E_X[H(Y + a)]. Ignoring ties,

    q_i = P(i = argmax_j Y_j) = dT/da_i |_{a=0}.

The additive shift enters every scalar endpoint estimator only through the
mean vector mu; sigma, the off-diagonal covariance, and all higher cumulants
are held fixed (they are *detached*: the ArcBridge detaches every cumulant
tensor, and sigma is re-detached here). Hence

    q_hat[E] = grad_mu E(mu, sigma, Sigma_od, kappa3, kappa4_trace)

for each nested scalar estimator E in {E0, E1_cov1, E2_cov2, E2_k3, E2_full}.

Implementation: the exact scalar pipeline (same compiled quotient diagrams,
same variable-elimination contractions, same Gauss-Legendre nodes) is rebuilt
with mu as an autograd leaf. The integration endpoints [lo, hi] are computed
from detached parameters by the scalar bisection and then held constant, so no
data-dependent control flow is differentiated (the neglected d(hi)/dmu term is
O(1 - F(hi)) ~ exp(tail_log_eps) ~ 1e-25). One reverse pass per *diagram* (a
small, width-independent count) produces all n coordinates simultaneously;
computing q never evaluates the scalar estimator n times. Because the root
derivative only adds unary (single-variable) endpoint factors, the primal
graph and hence the treewidth of every contraction is unchanged from the
scalar case; autograd's VJPs through the einsum joins have the same table
dimensions as the forward joins.

Endpoint Hermite degree: differentiating a merged degree-4 vertex weight
u_{i,4} ~ He_3 produces He_4-level terms (derivative order 5, via
du_{i,k}/dmu_i = u_{i,k+1} + u_{i,1} u_{i,k}); autograd applies this shift
automatically, so K=3 estimators exercise endpoint degree 2K-1 = 5.

FLOP convention (documented model, see docs/argmax_endpoint_status.md):
reverse-mode differentiation of this multilinear contraction pipeline costs a
small constant times the forward count (each VE join of k tables needs k VJP
einsums over the same joint index space; templates have k <= 4, dominant joins
k = 2). We record

    flops_endpoint_backward = BACKWARD_FLOP_FACTOR * flops_endpoint_forward

with BACKWARD_FLOP_FACTOR = 2: the *width exponent* of the backward pass is
identical to the forward one by construction. Wall-clock time and peak memory
are recorded separately as empirical checks.

Tie policy: torch_argmax_first_index; ties otherwise ignored (the maximizing
coordinate is almost surely unique; no jitter or tie machinery anywhere).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from mlp_kprop.max_endpoint.arc_bridge import ArcBridge
from mlp_kprop.max_endpoint.estimator import (
    _contract_diagram,
    _nested_estimates,
    compile_terms_for_bridge,
)
from mlp_kprop.max_endpoint.hermite_endpoint import EndpointWorkspace
from mlp_kprop.max_endpoint.quadrature import (
    QuadratureCfg,
    find_endpoints,
    gauss_legendre_nodes,
)
from mlp_kprop.max_endpoint.treewidth import FlopTally

__all__ = [
    "ArgmaxEndpointResult",
    "argmax_endpoint_estimate",
    "product_gaussian_argmax",
    "product_gaussian_argmax_reference",
    "project_to_simplex",
    "simplex_diagnostics",
    "TIE_POLICY",
    "BACKWARD_FLOP_FACTOR",
]

TIE_POLICY = "torch_argmax_first_index; ties otherwise ignored"

# Documented constant-factor model for reverse-mode contraction cost.
BACKWARD_FLOP_FACTOR = 2


def project_to_simplex(q: Tensor) -> Tensor:
    """Euclidean projection of a vector onto the probability simplex.

    Standard sort-based algorithm. Secondary estimator only: the raw
    differentiated expansion is the primary one and is never silently
    renormalized.
    """
    assert q.ndim == 1
    n = q.shape[0]
    s, _ = torch.sort(q, descending=True)
    cssv = torch.cumsum(s, dim=0) - 1.0
    ks = torch.arange(1, n + 1, device=q.device, dtype=q.dtype)
    rho = int((s - cssv / ks > 0).nonzero().max())
    tau = cssv[rho] / (rho + 1)
    return (q - tau).clamp(min=0.0)


def simplex_diagnostics(q: Tensor) -> dict[str, float]:
    """Raw-estimate simplex diagnostics (spec sections 11/18)."""
    qs = float(q.sum())
    return {
        "q_sum": qs,
        "q_min": float(q.min()),
        "q_max": float(q.max()),
        "q_l1": float(q.abs().sum()),
        "q_l2_sq": float((q * q).sum()),
        "simplex_residual": abs(qs - 1.0),
        "num_negative_coordinates": int((q < 0).sum()),
    }


@dataclass
class ArgmaxEndpointResult:
    """All nested argmax estimates (raw + projected) plus diagnostics."""

    q_raw: dict[str, Tensor]            # estimator name -> (n,) float64 raw q_hat
    q_projected: dict[str, Tensor]      # estimator name -> (n,) projected q_hat
    scalar_estimates: dict[str, float]  # scalar T_hat byproduct (cross-check)
    corrections: dict[str, float]
    psi: float
    quadrature_scalar_error: dict[str, float]       # |T_2Q - T_Q|
    quadrature_argmax_linf_error: dict[str, float]  # max_i |q_2Q - q_Q|_i
    quadrature_argmax_l2_error: dict[str, float]
    simplex: dict[str, dict[str, float]]            # per-estimator raw diagnostics
    status: list[str]
    equivalences: dict[str, str]
    max_treewidth: int
    treewidth_exact: bool
    num_diagrams: int
    max_table_numel: int
    flops_endpoint_forward: int
    flops_endpoint_backward: int
    flops_endpoint_total: int
    wall_seconds: float
    num_clamped_var: int
    k3_repr: str
    k4_sector: str
    tie_policy: str = TIE_POLICY
    info: dict[str, Any] = field(default_factory=dict)


def _argmax_at_nodes(
    bridge: ArcBridge,
    terms: dict[str, list],
    lo: float,
    hi: float,
    num_nodes: int,
) -> tuple[float, dict[str, float], Tensor, dict[str, Tensor], dict[str, Any]]:
    """Psi, corrections, and their mu-gradients on a Q-node grid.

    Returns (psi, corrections, grad_psi, grad_corrections, diagnostics).
    Gradients are taken w.r.t. a fresh leaf copy of mu; sigma and every
    cumulant tensor remain constants of the graph. One backward per diagram;
    dropping the per-diagram output after its grad call frees that diagram's
    saved intermediates while retain_graph keeps the shared O(Qn) workspace.
    """
    mu_leaf = bridge.mu.detach().clone().requires_grad_(True)
    sigma = bridge.sigma.detach()
    with torch.enable_grad():
        t, w = gauss_legendre_nodes(num_nodes, lo, hi, bridge.mu.device, bridge.dtype)
        ws = EndpointWorkspace(mu_leaf, sigma, t, w)
        # Psi = hi - int F dt (hi held constant; F = B on this grid).
        psi_t = hi - (ws.w * ws.B).sum()
        grad_psi = torch.autograd.grad(psi_t, mu_leaf, retain_graph=True)[0]

        tally = FlopTally()
        corrections: dict[str, float] = {}
        grad_corr: dict[str, Tensor] = {}
        max_tw, tw_exact, n_diag = 0, True, 0
        for name, diagrams in terms.items():
            corr_val = 0.0
            g = torch.zeros_like(mu_leaf)
            for d in diagrams:
                vals, width, exact = _contract_diagram(d, bridge, ws, tally)
                corr_d = ws.integrate(vals)
                g = g + torch.autograd.grad(corr_d, mu_leaf, retain_graph=True)[0]
                corr_val += float(corr_d.detach())
                max_tw = max(max_tw, width)
                tw_exact = tw_exact and exact
                n_diag += 1
            corrections[name] = corr_val
            grad_corr[name] = g.detach()
    diag = {
        "max_treewidth": max_tw,
        "treewidth_exact": tw_exact,
        "num_diagrams": n_diag,
        "max_table_numel": tally.max_table,
        "flops_forward": tally.flops + ws.flops + 2 * num_nodes,
    }
    return float(psi_t), corrections, grad_psi.detach(), grad_corr, diag


def argmax_endpoint_estimate(
    K: dict[int, Any],
    quad_cfg: QuadratureCfg | None = None,
    device: torch.device | None = None,
    dense_max_n: int = 16,
) -> ArgmaxEndpointResult:
    """All nested argmax estimators q_hat[E] = grad_mu E for one kprop tower.

    The full pipeline runs at both Q and convergence_factor * Q nodes; the
    higher-resolution q is reported and the difference recorded as the
    quadrature error (scalar, and linf/l2 on q).
    """
    if quad_cfg is None:
        quad_cfg = QuadratureCfg()
    t0 = time.time()
    bridge = ArcBridge(K, dtype=quad_cfg.dtype, device=device, dense_max_n=dense_max_n)
    terms = compile_terms_for_bridge(bridge)
    lo, hi = find_endpoints(bridge.mu, bridge.sigma, quad_cfg.tail_log_eps, quad_cfg.bisect_iters)

    q1 = quad_cfg.num_nodes
    q2 = quad_cfg.convergence_factor * q1
    psi1, corr1, gpsi1, gcorr1, diag1 = _argmax_at_nodes(bridge, terms, lo, hi, q1)
    psi2, corr2, gpsi2, gcorr2, diag2 = _argmax_at_nodes(bridge, terms, lo, hi, q2)

    # Scalar partial sums (byproduct; identical logic to the scalar estimator).
    est1, _, _ = _nested_estimates(psi1, corr1, bridge)
    est2, equivalences, status = _nested_estimates(psi2, corr2, bridge)
    # Vector partial sums: q_hat[E] via the same nesting on the gradients.
    qv1, _, _ = _nested_estimates(gpsi1, gcorr1, bridge)
    qv2, _, _ = _nested_estimates(gpsi2, gcorr2, bridge)

    quad_scalar = {k: abs(est2[k] - est1[k]) for k in est2}
    quad_linf = {k: float((qv2[k] - qv1[k]).abs().max()) for k in qv2}
    quad_l2 = {k: float((qv2[k] - qv1[k]).norm()) for k in qv2}

    q_raw = {k: v.detach().clone() for k, v in qv2.items()}
    q_projected = {k: project_to_simplex(v) for k, v in q_raw.items()}
    simplex = {k: simplex_diagnostics(v) for k, v in q_raw.items()}

    fwd = diag1["flops_forward"] + diag2["flops_forward"]
    bwd = BACKWARD_FLOP_FACTOR * fwd
    return ArgmaxEndpointResult(
        q_raw=q_raw,
        q_projected=q_projected,
        scalar_estimates=est2,
        corrections=corr2,
        psi=psi2,
        quadrature_scalar_error=quad_scalar,
        quadrature_argmax_linf_error=quad_linf,
        quadrature_argmax_l2_error=quad_l2,
        simplex=simplex,
        status=bridge.status + status,
        equivalences=equivalences,
        max_treewidth=diag2["max_treewidth"],
        treewidth_exact=diag2["treewidth_exact"],
        num_diagrams=diag2["num_diagrams"],
        max_table_numel=diag2["max_table_numel"],
        flops_endpoint_forward=fwd,
        flops_endpoint_backward=bwd,
        flops_endpoint_total=fwd + bwd,
        wall_seconds=time.time() - t0,
        num_clamped_var=bridge.params.num_clamped_var,
        k3_repr=bridge.k3_repr,
        k4_sector=bridge.k4_sector,
        info={"lo": lo, "hi": hi, "num_nodes": (q1, q2)},
    )


def product_gaussian_argmax(
    mu: Tensor, sigma: Tensor, quad_cfg: QuadratureCfg | None = None
) -> tuple[Tensor, float, dict]:
    """Production winner probabilities of the independent Gaussian reference.

        q_i^(0) = dPsi/dmu_i,  Psi(mu, sigma) = E[max_i(mu_i + sigma_i Z_i)],

    obtained by differentiating the scalar product-Gaussian max quadrature
    (fixed endpoints from detached parameters; evaluated at both Q and 2Q
    nodes, the 2Q value is returned, the difference is the error estimate).
    """
    if quad_cfg is None:
        quad_cfg = QuadratureCfg()
    mu = mu.detach().to(dtype=quad_cfg.dtype)
    sigma = sigma.detach().to(dtype=quad_cfg.dtype)
    assert (sigma > 0).all()
    lo, hi = find_endpoints(mu, sigma, quad_cfg.tail_log_eps, quad_cfg.bisect_iters)

    def q_at(num_nodes: int) -> Tensor:
        mu_leaf = mu.clone().requires_grad_(True)
        with torch.enable_grad():
            t, w = gauss_legendre_nodes(num_nodes, lo, hi, mu.device, mu.dtype)
            a = (t[:, None] - mu_leaf[None, :]) / sigma[None, :]
            log_f = torch.special.log_ndtr(a).sum(dim=1)
            psi = hi - (w * torch.exp(log_f)).sum()
        return torch.autograd.grad(psi, mu_leaf)[0].detach()

    g1 = q_at(quad_cfg.num_nodes)
    g2 = q_at(quad_cfg.convergence_factor * quad_cfg.num_nodes)
    err = float((g2 - g1).abs().max())
    info = {"lo": lo, "hi": hi, "num_nodes": quad_cfg.num_nodes}
    return g2, err, info


def product_gaussian_argmax_reference(
    mu: Tensor, sigma: Tensor, num_nodes: int = 4096
) -> Tensor:
    """Explicit one-dimensional winner-probability integral (test reference):

        q_i^(0) = int phi(a_i(t))/sigma_i * prod_{j != i} Phi(a_j(t)) dt
                = int B(t) u_{i,1}(t) dt,

    with B = prod_j Phi(a_j) and the hazard weight u_{i,1} = phi/(sigma_i Phi)
    of hermite_endpoint (the i-th CDF factor of B cancels against u's Phi).
    """
    mu = mu.detach().to(torch.float64)
    sigma = sigma.detach().to(torch.float64)
    cfg = QuadratureCfg(num_nodes=num_nodes)
    lo, hi = find_endpoints(mu, sigma, cfg.tail_log_eps, cfg.bisect_iters)
    t, w = gauss_legendre_nodes(num_nodes, lo, hi, mu.device, torch.float64)
    ws = EndpointWorkspace(mu, sigma, t, w)
    return (ws.wB[:, None] * ws.u(1)).sum(dim=0)
