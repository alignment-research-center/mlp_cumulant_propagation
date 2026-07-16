"""
Direct dense Edgeworth endpoint estimator for T(theta) = E_X[max_i M(X)_i].

This is the deliberately simple, correctness-first counterpart of the
diagram/treewidth pipeline in `estimator.py`. It shares the same mathematics
(quadrature.py endpoints, hermite_endpoint.py unary weights u_{i,k}, the same
cumulant extraction conventions as arc_bridge.py) but evaluates every operator
term by

  1. materializing the *integrated* dense endpoint derivative tensor

         D_m[i_1..i_m] = d^m Psi / dmu_{i_1} .. dmu_{i_m},   m in {2, 3, 4},

     built by explicit equality-pattern (set-partition) enumeration: for a
     set partition P = {B_1..B_p} of the m derivative slots, entries whose
     index equality pattern is exactly P equal

         (-1)^(p-1) int B(t) prod_v u_{i_v, |B_v|}(t) dt      (labels distinct);

  2. contracting D_m against *dense* cumulant tensors with plain torch.einsum.

Integration over quadrature nodes and contraction commute, so materializing
the integrated D_m (memory n^m, node-independent) is equivalent to per-node
contraction; nodes are processed in small chunks while accumulating D_m to
bound intermediate memory at chunk * n^(m-1).

No treewidth, variable elimination, Moebius inversion, or factorized
contraction is used anywhere. Dense allocations are guarded by
`max_dense_bytes` and refused with a clear diagnostic.

Cumulant sectors (same conventions as arc_bridge.py):
- K[2] r=0: full covariance -> off-diagonal part C. r=1 (k_max=1): variances
  only, off-diagonal sector unavailable (E1/E2_* collapse to E0).
- K[3]: densified via .to_tensor() (FactoredTensor -> Sym(sum_r a x b x c);
  HTensor r=1 -> Sym(v x M) trace sector; r=0 dense core).
- K[4] HTensor r=1 (k_max=3 AUGMENT): kappa4 ~ Sym(C x M); r=2 (k_max=3
  SIMPLE): kappa4 ~ c Sym(M x M). Contracted as the surrogate
  (1/24) sum C_ij M_kl D4_ijkl, exact because D4 is fully symmetric (Sym may
  be dropped against a symmetric tensor). This is a *trace sector*, not the
  full fourth cumulant; the estimator is named E2_k3_k4trace accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from mlp_kprop.max_endpoint.arc_bridge import extract_gaussian_params
from mlp_kprop.max_endpoint.hermite_endpoint import EndpointWorkspace
from mlp_kprop.max_endpoint.quadrature import (
    QuadratureCfg,
    find_endpoints,
    gauss_legendre_nodes,
)
from mlp_kprop.partitions import set_partitions

__all__ = [
    "DenseMemoryError",
    "DirectDenseCfg",
    "DirectDenseResult",
    "dense_derivative_tensor",
    "direct_dense_estimate",
]


class DenseMemoryError(RuntimeError):
    """Raised instead of attempting an unsafe dense allocation."""


@dataclass
class DirectDenseCfg:
    quad: QuadratureCfg = field(default_factory=QuadratureCfg)
    max_dense_bytes: int = 40 * 10**9  # refuse any single dense tensor above this
    node_chunk: int = 16               # quadrature nodes per accumulation chunk
    # Which derivative orders may be materialized. Order 2 enables C2 (E1);
    # order 4 enables C2^2/2 and C4_trace; order 3 enables C3.
    dense_orders: tuple[int, ...] = (2, 3, 4)


@dataclass
class DirectDenseResult:
    estimates: dict[str, float]          # estimator name -> value (2Q pass)
    corrections: dict[str, float]        # term name -> integrated correction
    psi: float
    quadrature_error: dict[str, float]   # estimator name -> |value_2Q - value_Q|
    status: list[str]
    equivalences: dict[str, str]
    largest_dense_tensor_order: int
    largest_dense_tensor_shape: tuple[int, ...]
    estimated_dense_bytes: int           # max bytes of any dense tensor built
    flops_endpoint: int
    flops_by_part: dict[str, int]
    wall_seconds: float
    num_clamped_var: int
    k3_repr: str
    k4_sector: str
    info: dict[str, Any] = field(default_factory=dict)


def dense_bytes(n: int, order: int, dtype: torch.dtype = torch.float64) -> int:
    """Bytes of a dense (n,)*order tensor."""
    return n**order * torch.tensor([], dtype=dtype).element_size()


def _guard_dense(n: int, order: int, node_chunk: int, max_bytes: int, what: str) -> int:
    """Estimate peak bytes for building a dense order-`order` tensor and refuse
    the allocation if it exceeds max_bytes. Returns the estimate."""
    itemsize = 8  # float64
    result = n**order * itemsize
    # Largest chunked einsum intermediate while accumulating: (chunk, n^(order-1)).
    intermediate = node_chunk * n ** max(order - 1, 1) * itemsize
    est = result + intermediate
    if est > max_bytes:
        raise DenseMemoryError(
            f"dense_refused: {what} at n={n} order={order} needs ~{est/1e9:.2f} GB "
            f"(tensor {result/1e9:.2f} GB + chunk intermediate {intermediate/1e9:.2f} GB) "
            f"> max_dense_bytes={max_bytes/1e9:.2f} GB"
        )
    return est


class _Tally:
    """Analytic FLOP tally with named parts."""

    def __init__(self) -> None:
        self.by_part: dict[str, int] = {}

    def add(self, part: str, flops: int) -> None:
        self.by_part[part] = self.by_part.get(part, 0) + int(flops)

    @property
    def total(self) -> int:
        return sum(self.by_part.values())


def _pattern_tensor(
    ws: EndpointWorkspace, orders: tuple[int, ...], node_chunk: int, tally: _Tally
) -> Tensor:
    """V[i_1..i_p] = sum_q wB[q] prod_v u_{i_v, orders_v}[q] over ALL label
    tuples (repeated labels included; caller masks them out). p = len(orders).
    Nodes are accumulated in chunks of node_chunk."""
    p = len(orders)
    us = [ws.u(k) for k in orders]
    n, q_total = ws.n, ws.q
    out: Tensor | None = None
    for start in range(0, q_total, node_chunk):
        sl = slice(start, min(start + node_chunk, q_total))
        first = ws.wB[sl, None] * us[0][sl]  # (q, n)
        if p == 1:
            chunk = first.sum(dim=0)
        elif p == 2:
            chunk = torch.einsum("qa,qb->ab", first, us[1][sl])
        elif p == 3:
            chunk = torch.einsum("qa,qb,qc->abc", first, us[1][sl], us[2][sl])
        elif p == 4:
            chunk = torch.einsum(
                "qa,qb,qc,qd->abcd", first, us[1][sl], us[2][sl], us[3][sl]
            )
        else:
            raise NotImplementedError(f"pattern tensors only implemented for p <= 4 (got {p})")
        out = chunk if out is None else out + chunk
    # FLOP model: fold wB (Q n); left-to-right pairwise einsum builds (q, n^j)
    # for j = 2..p (Q n^j mults each) and the final q-contraction is
    # 2 Q n^p multiply-adds; plus n^p adds per chunk accumulation.
    flops = q_total * n
    for j in range(2, p + 1):
        flops += q_total * n**j
    flops += 2 * q_total * n**p + (q_total // max(node_chunk, 1)) * n**p
    tally.add(f"D{sum(orders)}_patterns", flops)
    return out


def dense_derivative_tensor(
    ws: EndpointWorkspace,
    order: int,
    node_chunk: int = 16,
    max_dense_bytes: int = 40 * 10**9,
    tally: _Tally | None = None,
) -> Tensor:
    """Integrated dense endpoint derivative tensor D_order, shape (n,)*order.

    D[i_1..i_m] = d^m Psi / dmu_{i_1}..dmu_{i_m}, assembled by exact
    equality-pattern enumeration over set partitions of the m slots.
    """
    assert 1 <= order <= 4, "direct dense derivative tensors implemented for order <= 4"
    if tally is None:
        tally = _Tally()
    n = ws.n
    _guard_dense(n, order, node_chunk, max_dense_bytes, f"D{order}")
    D = torch.zeros((n,) * order, dtype=ws.mu.dtype, device=ws.mu.device)
    for partition in set_partitions(order):
        blocks = [tuple(sorted(b)) for b in partition]
        p = len(blocks)
        orders = tuple(len(b) for b in blocks)
        V = _pattern_tensor(ws, orders, node_chunk, tally)
        if p > 1:
            V = V * float((-1) ** (p - 1))
            # Exact-pattern semantics: zero every label tuple with a collision.
            for a in range(p):
                for b in range(a + 1, p):
                    V.diagonal(dim1=a, dim2=b).zero_()
        # Embed the p-label tensor onto the partition diagonal of D:
        # slot s takes the label of the block containing s.
        strides = tuple(sum(D.stride(s) for s in blk) for blk in blocks)
        D.as_strided((n,) * p, strides).add_(V)
        tally.add(f"D{order}_patterns", n**p)
    return D


# ---------------------------------------------------------------------------
# Dense cumulant extraction
# ---------------------------------------------------------------------------


class _DenseCumulants:
    """Dense (mu, sigma, off-diagonal Sigma, kappa3, kappa4-sector) for one
    kprop output tower. Mirrors arc_bridge.ArcBridge sector logic, but always
    densifies kappa3 (guarded)."""

    def __init__(
        self,
        K: dict[int, Any],
        dtype: torch.dtype,
        device: torch.device | None,
        max_dense_bytes: int,
        tally: _Tally,
    ):
        from mlp_kprop.factor_k3 import FactoredTensor
        from mlp_kprop.harmonic import HTensor

        self.params = extract_gaussian_params(K, dtype=dtype)
        if device is None:
            device = self.params.mu.device
        self.device, self.dtype = device, dtype
        self.n = n = self.params.mu.shape[0]
        self.status = list(self.params.status)

        def prep(t: Tensor) -> Tensor:
            return t.detach().to(device=device, dtype=dtype)

        self.mu = prep(self.params.mu)
        self.sigma = prep(self.params.sigma)

        K2 = K[2]
        if K2.r == 0:
            sig = prep(K2.core)
            self.cov_od: Tensor | None = sig - torch.diag(sig.diagonal())
        else:
            self.cov_od = None
            self.status.append("cov_offdiag_unavailable")

        self.k3_dense: Tensor | None = None
        self.k3_repr = "none"
        self.refused: dict[str, str] = {}
        if 3 in K:
            K3 = K[3]
            try:
                _guard_dense(n, 3, 1, max_dense_bytes, "kappa3_densify")
            except DenseMemoryError as e:
                self.refused["kappa3"] = str(e)
                self.status.append("dense_refused_kappa3")
                K3 = None
            if K3 is None:
                pass
            elif isinstance(K3, FactoredTensor):
                # to_tensor() = Sym(sum_r a_r x b_r x c_r); densification cost
                # ~ 2 n^3 R for the unfactored product plus 6 n^3 for Sym.
                rank = K3.factors[0].shape[1]
                self.k3_dense = prep(K3.to_tensor())
                self.k3_repr = "factored_densified"
                tally.add("densify_k3", 2 * n**3 * rank + 6 * n**3)
            elif isinstance(K3, HTensor):
                self.k3_dense = prep(K3.to_tensor())
                self.k3_repr = "dense" if K3.r == 0 else "trace_densified"
                tally.add("densify_k3", 6 * n**3 * max(K3.r, 1))
            else:
                raise ValueError(f"Unsupported K[3] representation: {K3!r}")

        # kappa4: keep the (core, metric) surrogate pair; do NOT materialize n^4.
        self.k4_core: Tensor | None = None
        self.k4_metric: Tensor | None = None
        self.k4_dense: Tensor | None = None
        self.k4_sector = "none"
        if 4 in K:
            K4 = K[4]
            assert isinstance(K4, HTensor), f"Unsupported K[4] type {type(K4)!r}"
            if K4.r == 1:
                self.k4_core = prep(K4.core)
                metric = K4.metric
                self.k4_metric = prep(metric if metric.ndim == 2 else torch.diag(metric))
                self.k4_sector = "r1_traceful"
            elif K4.r == 2:
                metric = K4.metric
                m = prep(metric if metric.ndim == 2 else torch.diag(metric))
                self.k4_core = prep(K4.core) * m
                self.k4_metric = m
                self.k4_sector = "r2_double_trace"
            elif K4.r == 0:
                _guard_dense(n, 4, 1, max_dense_bytes, "kappa4_densify")
                self.k4_dense = prep(K4.core)
                self.k4_sector = "dense"
            else:
                raise ValueError(f"Unsupported K[4] radial index {K4.r}")


# ---------------------------------------------------------------------------
# Main estimate
# ---------------------------------------------------------------------------


def _corrections_at_nodes(
    cum: _DenseCumulants,
    lo: float,
    hi: float,
    num_nodes: int,
    cfg: DirectDenseCfg,
    tally: _Tally,
    dense_track: dict,
) -> tuple[float, dict[str, float]]:
    """Psi and all direct-dense corrections on a Q-node grid on [lo, hi]."""
    t, w = gauss_legendre_nodes(num_nodes, lo, hi, cum.mu.device, cum.dtype)
    ws = EndpointWorkspace(cum.mu, cum.sigma, t, w)
    psi = float(hi - (ws.w * ws.B).sum())
    tally.add("quadrature", 2 * num_nodes)
    n = cum.n
    corrections: dict[str, float] = {}

    def track(order: int) -> None:
        est = _guard_dense(n, order, cfg.node_chunk, cfg.max_dense_bytes, f"D{order}")
        if order > dense_track["order"]:
            dense_track["order"] = order
            dense_track["shape"] = (n,) * order
        dense_track["bytes"] = max(dense_track["bytes"], est)

    want = set(dense_track["effective_orders"])
    if cum.cov_od is not None and 2 in want:
        track(2)
        D2 = dense_derivative_tensor(ws, 2, cfg.node_chunk, cfg.max_dense_bytes, tally)
        corrections["C2"] = float(0.5 * (cum.cov_od * D2).sum())
        tally.add("C2_contraction", 2 * n**2)
        del D2
    if cum.k3_dense is not None and 3 in want:
        track(3)
        dense_track["bytes"] = max(dense_track["bytes"], dense_bytes(n, 3, cum.dtype))
        D3 = dense_derivative_tensor(ws, 3, cfg.node_chunk, cfg.max_dense_bytes, tally)
        corrections["C3"] = float((cum.k3_dense * D3).sum() / 6.0)
        tally.add("C3_contraction", 2 * n**3)
        del D3
    need_d4 = (cum.cov_od is not None or cum.k4_sector != "none") and 4 in want
    if need_d4:
        track(4)
        D4 = dense_derivative_tensor(ws, 4, cfg.node_chunk, cfg.max_dense_bytes, tally)
        if cum.cov_od is not None:
            # (1/2) C2^2 Psi = (1/8) sum_{i!=j,k!=l} C_ij C_kl D4_ijkl.
            # The i!=j / k!=l restrictions are automatic: cov_od has zero diag.
            tmp = torch.einsum("ijkl,kl->ij", D4, cum.cov_od)
            corrections["C2sq_half"] = float(0.125 * (tmp * cum.cov_od).sum())
            tally.add("C2sq_contraction", 2 * n**4 + 2 * n**2)
            del tmp
        if cum.k4_dense is not None:
            corrections["C4_trace"] = float((cum.k4_dense * D4).sum() / 24.0)
            tally.add("C4_contraction", 2 * n**4)
        elif cum.k4_sector != "none":
            # Surrogate contraction (exact against the symmetric D4):
            # (1/24) sum_ijkl core_ij metric_kl D4_ijkl.
            tmp = torch.einsum("ijkl,kl->ij", D4, cum.k4_metric)
            corrections["C4_trace"] = float((tmp * cum.k4_core).sum() / 24.0)
            tally.add("C4_contraction", 2 * n**4 + 2 * n**2)
            del tmp
        del D4
    tally.add("workspace", ws.flops)
    return psi, corrections


def _nested_estimates(
    psi: float, corrections: dict[str, float]
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    """Partial sums E0 .. E2_k3_k4trace over available terms (spec section 9)."""
    estimates = {"E0_product_gaussian": psi}
    equivalences: dict[str, str] = {}
    status: list[str] = []
    running = psi
    if "C2" in corrections:
        running += corrections["C2"]
        estimates["E1_cov1"] = running
        if "C2sq_half" in corrections:
            running += corrections["C2sq_half"]
            estimates["E2_cov"] = running
        else:
            status.append("c2sq_not_computed")
    else:
        equivalences["E1_cov1"] = "E0_product_gaussian"
        equivalences["E2_cov"] = "E0_product_gaussian"
        status.append("cov_offdiag_unavailable")
    if "C3" in corrections and "C2sq_half" in corrections:
        running += corrections["C3"]
        estimates["E2_k3"] = running
    elif "C3" not in corrections:
        status.append("k3_unavailable")
    if "C4_trace" in corrections and "E2_k3" in estimates:
        running += corrections["C4_trace"]
        estimates["E2_k3_k4trace"] = running
    elif "C4_trace" not in corrections:
        status.append("k4_trace_unavailable")
    return estimates, equivalences, status


def direct_dense_estimate(
    K: dict[int, Any],
    cfg: DirectDenseCfg | None = None,
    device: torch.device | None = None,
) -> DirectDenseResult:
    """All nested direct-dense Edgeworth estimators for one kprop output tower.

    The full pipeline runs at Q and at convergence_factor * Q quadrature nodes;
    the 2Q values are reported, |2Q - Q| per estimator is the quadrature error.
    """
    if cfg is None:
        cfg = DirectDenseCfg()
    t0 = time.time()
    tally = _Tally()
    dense_track = {"order": 0, "shape": (), "bytes": 0}
    cum = _DenseCumulants(K, cfg.quad.dtype, device, cfg.max_dense_bytes, tally)
    # Refuse unaffordable derivative orders up-front (graceful degradation):
    # a refused D4 loses E2_cov/E2_k3/E2_k3_k4trace but keeps E0/E1.
    refused = dict(cum.refused)
    effective_orders = []
    for order in sorted(set(cfg.dense_orders)):
        try:
            _guard_dense(cum.n, order, cfg.node_chunk, cfg.max_dense_bytes, f"D{order}")
            effective_orders.append(order)
        except DenseMemoryError as e:
            refused[f"D{order}"] = str(e)
            cum.status.append(f"dense_refused_D{order}")
    dense_track["effective_orders"] = tuple(effective_orders)
    lo, hi = find_endpoints(cum.mu, cum.sigma, cfg.quad.tail_log_eps, cfg.quad.bisect_iters)
    q1 = cfg.quad.num_nodes
    q2 = cfg.quad.convergence_factor * q1
    psi1, corr1 = _corrections_at_nodes(cum, lo, hi, q1, cfg, tally, dense_track)
    psi2, corr2 = _corrections_at_nodes(cum, lo, hi, q2, cfg, tally, dense_track)
    est1, _, _ = _nested_estimates(psi1, corr1)
    est2, equivalences, status = _nested_estimates(psi2, corr2)
    quad_err = {k: abs(est2[k] - est1[k]) for k in est2}
    return DirectDenseResult(
        estimates=est2,
        corrections=corr2,
        psi=psi2,
        quadrature_error=quad_err,
        status=cum.status + status,
        equivalences=equivalences,
        largest_dense_tensor_order=dense_track["order"],
        largest_dense_tensor_shape=tuple(dense_track["shape"]),
        estimated_dense_bytes=int(dense_track["bytes"]),
        flops_endpoint=tally.total,
        flops_by_part=dict(tally.by_part),
        wall_seconds=time.time() - t0,
        num_clamped_var=cum.params.num_clamped_var,
        k3_repr=cum.k3_repr,
        k4_sector=cum.k4_sector,
        info={"lo": lo, "hi": hi, "num_nodes": (q1, q2), "dense_refused": refused},
    )
