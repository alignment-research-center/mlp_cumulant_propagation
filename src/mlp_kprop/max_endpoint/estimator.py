"""
Nested endpoint estimators of T(theta) = E_X[max_i M_theta(X)_i] from a kprop
cumulant tower at the network output.

With Psi(mu, sigma) = E[max_i(mu_i + sigma_i Z_i)] (product-Gaussian max) and
the cumulant differential operators

    C2       = (1/2) sum_{i != j} Sigma_ij d_i d_j
    Cr       = (1/r!) sum kappa_r[i_1..i_r] d_{i_1}..d_{i_r}     (r = 3, 4),

the nested estimators are the partial sums of exp(C2 + C3 + C4) Psi (mu):

    E0      = Psi
    E1      = Psi + C2 Psi
    E2_cov  = E1 + (1/2) C2^2 Psi
    E2_k3   = E2_cov + C3 Psi
    E2_full = E2_k3 + C4_trace Psi

where C4_trace uses only the harmonic trace sector of kappa4 that ARC's kprop
retains at the given (k_max, kind); see arc_bridge for the exact sector.

Each correction is an integral int B(t) * (diagram contraction)(t) dt over a
shared Gauss-Legendre grid; diagrams are compiled offline (diagrams.py) and
contracted by variable elimination (treewidth.py) directly against ARC's
factorized representations (arc_bridge.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from mlp_kprop.max_endpoint.arc_bridge import ArcBridge
from mlp_kprop.max_endpoint.diagrams import (
    CompiledDiagram,
    TermSpec,
    compile_term,
    term_c2,
    term_c2_squared_half,
    term_c3_generic,
    term_c3_trace,
    term_c4_core_metric,
    term_c4_dense,
)
from mlp_kprop.max_endpoint.hermite_endpoint import EndpointWorkspace
from mlp_kprop.max_endpoint.quadrature import (
    QuadratureCfg,
    find_endpoints,
    gauss_legendre_nodes,
)
from mlp_kprop.max_endpoint.treewidth import (
    FlopTally,
    VEFactor,
    contract_factors,
    find_elimination_order,
)

__all__ = ["MaxEndpointResult", "max_endpoint_estimate", "compile_terms_for_bridge"]

_COMPILE_CACHE: dict[str, list[CompiledDiagram]] = {}
_ELIM_CACHE: dict[Any, Any] = {}


@dataclass
class MaxEndpointResult:
    """All nested estimates plus diagnostics for one kprop tower."""

    estimates: dict[str, float]          # estimator name -> value
    corrections: dict[str, float]        # term name -> integrated correction
    psi: float
    quadrature_error: dict[str, float]   # estimator name -> |value_2Q - value_Q|
    status: list[str]
    equivalences: dict[str, str]         # estimator -> estimator it equals exactly
    max_treewidth: int
    treewidth_exact: bool
    num_diagrams: int
    max_table_numel: int
    flops_endpoint: int
    wall_seconds: float
    num_clamped_var: int
    k3_repr: str
    k4_sector: str
    info: dict[str, Any] = field(default_factory=dict)


def compile_terms_for_bridge(bridge: ArcBridge) -> dict[str, list[CompiledDiagram]]:
    """Compile (with caching) the operator terms available for this tower."""

    def cached(term: TermSpec) -> list[CompiledDiagram]:
        if term.name not in _COMPILE_CACHE:
            _COMPILE_CACHE[term.name] = compile_term(term)
        return _COMPILE_CACHE[term.name]

    terms: dict[str, list[CompiledDiagram]] = {}
    if bridge.has("cov_od"):
        terms["C2"] = cached(term_c2())
        terms["C2sq_half"] = cached(term_c2_squared_half())
    if bridge.has("k3"):
        if bridge.k3_repr == "trace":
            terms["C3"] = cached(term_c3_trace())
        else:
            terms["C3"] = cached(term_c3_generic("k3"))
    if bridge.has("k4"):
        if bridge.k4_sector == "dense":
            terms["C4_trace"] = cached(term_c4_dense())
        else:
            terms["C4_trace"] = cached(term_c4_core_metric())
    return terms


def _contract_diagram(
    diagram: CompiledDiagram,
    bridge: ArcBridge,
    ws: EndpointWorkspace,
    tally: FlopTally,
) -> tuple[Tensor, int, bool]:
    """Evaluate one diagram at all quadrature nodes; returns (values (Q,), width, exact)."""
    domains: dict[int, int] = {}
    factors = []
    for v, orders in enumerate(diagram.vertex_orders):
        domains[v] = bridge.n
        factors.append(VEFactor(vars=(v,), tensor=ws.vertex_weight(orders), batched=True))
    next_var = diagram.num_vertices
    for kind, legs in diagram.factors:
        fs, next_var = bridge.build(kind, legs, next_var, domains)
        factors.extend(fs)

    elim_key = (
        diagram.vertex_orders,
        diagram.factors,
        bridge.k3_repr,
        tuple(sorted(domains.items())),
    )
    if elim_key not in _ELIM_CACHE:
        all_vars = sorted(domains.keys())
        _ELIM_CACHE[elim_key] = find_elimination_order(all_vars, [f.vars for f in factors])
    info = _ELIM_CACHE[elim_key]
    values = contract_factors(factors, domains, info.order, batch=ws.q, tally=tally)
    return diagram.coef * values, info.width, info.exact


def _evaluate_at_nodes(
    bridge: ArcBridge,
    terms: dict[str, list[CompiledDiagram]],
    lo: float,
    hi: float,
    num_nodes: int,
) -> tuple[float, dict[str, float], dict[str, Any]]:
    """Psi and all corrections on a Q-node Gauss-Legendre grid on [lo, hi]."""
    t, w = gauss_legendre_nodes(num_nodes, lo, hi, bridge.mu.device, bridge.dtype)
    ws = EndpointWorkspace(bridge.mu, bridge.sigma, t, w)
    # Psi = hi - int F dt; F = B on this grid.
    psi = float(hi - (ws.w * ws.B).sum())
    tally = FlopTally()
    corrections: dict[str, float] = {}
    max_tw, tw_exact, n_diag = 0, True, 0
    for name, diagrams in terms.items():
        total = torch.zeros_like(t)
        for d in diagrams:
            vals, width, exact = _contract_diagram(d, bridge, ws, tally)
            total = total + vals
            max_tw = max(max_tw, width)
            tw_exact = tw_exact and exact
            n_diag += 1
        corrections[name] = float(ws.integrate(total))
    diag = {
        "max_treewidth": max_tw,
        "treewidth_exact": tw_exact,
        "num_diagrams": n_diag,
        "max_table_numel": tally.max_table,
        "flops": tally.flops + ws.flops + 2 * num_nodes,
    }
    return psi, corrections, diag


def _nested_estimates(
    psi: float, corrections: dict[str, float], bridge: ArcBridge
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    """Partial sums E0 .. E2_full over available sectors."""
    estimates = {"E0_product_gaussian": psi}
    equivalences: dict[str, str] = {}
    status: list[str] = []
    # NOTE: `running = running + x` (not +=) so this also works elementwise on
    # tensors without in-place aliasing (reused by max_endpoint.argmax on the
    # per-term mu-gradients).
    running = psi
    if "C2" in corrections:
        running = running + corrections["C2"]
        estimates["E1_cov1"] = running
        running = running + corrections["C2sq_half"]
        estimates["E2_cov2"] = running
    else:
        # k_max=1: off-diagonal covariance identically zero in the tracked
        # representation, so E1/E2_cov coincide with E0.
        equivalences["E1_cov1"] = "E0_product_gaussian"
        equivalences["E2_cov2"] = "E0_product_gaussian"
        status.append("cov_offdiag_unavailable")
    if "C3" in corrections and "C2" in corrections:
        running = running + corrections["C3"]
        estimates["E2_k3"] = running
    elif "C3" not in corrections:
        status.append("k3_unavailable")
    if "C4_trace" in corrections and "C3" in corrections and "C2" in corrections:
        running = running + corrections["C4_trace"]
        estimates["E2_full"] = running
    elif "C4_trace" not in corrections:
        status.append("k4_trace_unavailable")
    return estimates, equivalences, status


def max_endpoint_estimate(
    K: dict[int, Any],
    quad_cfg: QuadratureCfg | None = None,
    device: torch.device | None = None,
    dense_max_n: int = 16,
) -> MaxEndpointResult:
    """Compute all nested endpoint estimators for one kprop output tower.

    Args:
        K: kprop tower at the output layer (HTensor / FactoredTensor values).
        quad_cfg: quadrature configuration; the full pipeline is evaluated at
            both Q and convergence_factor * Q nodes, the latter is reported and
            the difference recorded as the quadrature error estimate.
        device: device for endpoint math (default: device of K[1]).
    """
    if quad_cfg is None:
        quad_cfg = QuadratureCfg()
    t0 = time.time()
    bridge = ArcBridge(K, dtype=quad_cfg.dtype, device=device, dense_max_n=dense_max_n)
    terms = compile_terms_for_bridge(bridge)
    lo, hi = find_endpoints(bridge.mu, bridge.sigma, quad_cfg.tail_log_eps, quad_cfg.bisect_iters)

    q1 = quad_cfg.num_nodes
    q2 = quad_cfg.convergence_factor * q1
    psi1, corr1, diag1 = _evaluate_at_nodes(bridge, terms, lo, hi, q1)
    psi2, corr2, diag2 = _evaluate_at_nodes(bridge, terms, lo, hi, q2)

    est1, _, _ = _nested_estimates(psi1, corr1, bridge)
    est2, equivalences, status = _nested_estimates(psi2, corr2, bridge)
    quad_err = {k: abs(est2[k] - est1[k]) for k in est2}
    status = bridge.status + status

    return MaxEndpointResult(
        estimates=est2,
        corrections=corr2,
        psi=psi2,
        quadrature_error=quad_err,
        status=status,
        equivalences=equivalences,
        max_treewidth=diag2["max_treewidth"],
        treewidth_exact=diag2["treewidth_exact"],
        num_diagrams=diag2["num_diagrams"],
        max_table_numel=diag2["max_table_numel"],
        flops_endpoint=diag1["flops"] + diag2["flops"],
        wall_seconds=time.time() - t0,
        num_clamped_var=bridge.params.num_clamped_var,
        k3_repr=bridge.k3_repr,
        k4_sector=bridge.k4_sector,
        info={"lo": lo, "hi": hi, "num_nodes": (q1, q2)},
    )
