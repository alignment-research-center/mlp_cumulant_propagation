"""
Adapters contracting endpoint diagrams directly against ARC's cumulant
representations, without materializing dense order-3/4 tensors.

Supported sector representations at the network output (final preactivation):

- K[1]: HTensor r=0, core = mean vector mu.
- K[2]: HTensor r=0, core = covariance Sigma (full);  or r=1 (k_max=1), core
  scalar c with a *vector* metric m, i.e. Sigma = c * diag(m): variances only,
  the off-diagonal covariance sector is unavailable.
- K[3]:
    * FactoredTensor (k_max=3, factor=True):
        kappa3 = Sym(sum_r A[:, r] (x) B[:, r] (x) C[:, r]).
      The rank index r becomes an internal variable-elimination variable, so
      diagrams contract in O(Q n R) without materializing kappa3.
    * HTensor r=0: dense core (reference path, small n only).
    * HTensor d=3, r=1 (k_max=2 AUGMENT): kappa3 = Sym(v (x) M), the harmonic
      trace sector R^1 H_1; contracted as the surrogate v_i M_jk.
- K[4]:
    * HTensor d=4, r=1 (k_max=3 AUGMENT): kappa4 = Sym(C (x) M) with core
      matrix C and metric M = W_L W_L^T; this is the projection P_{>=1} kappa4,
      i.e. the harmonic sectors R^1 H_2 (+) R^2 H_0, with the traceless H_4
      part dropped by kprop.
    * HTensor d=4, r=2 (k_max=3 SIMPLE / k_max=2 AUGMENT): kappa4 =
      c * Sym(M (x) M), the pure double-trace sector P_{>=2} kappa4 = R^2 H_0.
    * HTensor r=0: dense core (reference path, small n only).

Because every diagram contraction is against a fully symmetric derivative
tensor, Sym(...) may be replaced by the plain tensor product before
equality-pattern decomposition; the bridge therefore only ever provides
factorized surrogates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from mlp_kprop.max_endpoint.treewidth import VEFactor

logger = logging.getLogger(__name__)

__all__ = ["GaussianParams", "ArcBridge", "extract_gaussian_params"]

VAR_CLAMP_EPS = 1e-10


@dataclass
class GaussianParams:
    mu: Tensor
    var: Tensor
    sigma: Tensor
    num_clamped_var: int
    status: list[str] = field(default_factory=list)


def _htensor_metric_matrix(metric: Tensor) -> Tensor:
    """Coerce an HTensor metric (vector or matrix) to a full matrix."""
    if metric.ndim == 1:
        return torch.diag(metric)
    assert metric.ndim == 2
    return metric


def extract_gaussian_params(K: dict[int, Any], dtype: torch.dtype = torch.float64) -> GaussianParams:
    """Extract (mu, sigma) of the independent Gaussian reference from a kprop tower.

    Negative variances (which the truncated cumulant expansion can produce at
    small width) are clamped to VAR_CLAMP_EPS and *flagged*, never silently.
    """
    status: list[str] = []
    K1 = K[1]
    assert K1.r == 0, "K[1] must have radial index 0"
    mu = K1.core.detach().to(dtype)
    if not torch.isfinite(mu).all():
        raise ValueError(
            "kprop_nonfinite: kprop mean contains non-finite entries "
            "(the truncated cumulant expansion diverges at this width/depth)."
        )
    K2 = K[2]
    if K2.r == 0:
        var = K2.core.detach().to(dtype).diagonal().clone()
    elif K2.r == 1:
        metric = K2.metric.detach().to(dtype)
        mvec = metric if metric.ndim == 1 else metric.diagonal()
        var = (K2.core.detach().to(dtype) * mvec).clone()
    else:
        raise ValueError(f"Unsupported K[2] radial index {K2.r}")
    if not torch.isfinite(var).all():
        raise ValueError(
            "kprop_nonfinite: kprop variances contain non-finite entries "
            "(the truncated cumulant expansion diverges at this width/depth)."
        )
    bad = int((var <= 0).sum())
    if bad > 0:
        status.append(f"negative_variance_clamped:{bad}")
        logger.warning(f"Clamping {bad} nonpositive variances to {VAR_CLAMP_EPS}")
        var = var.clamp(min=VAR_CLAMP_EPS)
    return GaussianParams(mu=mu, var=var, sigma=var.sqrt(), num_clamped_var=bad, status=status)


class ArcBridge:
    """Resolves diagram factor kinds to concrete VE factors for one kprop tower.

    Factor kinds:
        "cov_od":   off-diagonal covariance Sigma - diag(Sigma) (2 legs).
        "k3":       third cumulant 3-edge (dense or rank-factorized).
        "k3_core_vec"/"k3_metric": Sym(v (x) M) trace surrogate of kappa3.
        "k4_core"/"k4_metric":     Sym(C (x) M) trace surrogate of kappa4
                                   (SIMPLE: C = c * M).
    """

    def __init__(
        self,
        K: dict[int, Any],
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
        dense_max_n: int = 16,
    ):
        from mlp_kprop.factor_k3 import FactoredTensor
        from mlp_kprop.harmonic import HTensor

        self.params = extract_gaussian_params(K, dtype=dtype)
        if device is None:
            device = self.params.mu.device
        self.device = device
        self.dtype = dtype
        self.n = self.params.mu.shape[0]
        self.status = list(self.params.status)

        def prep(t: Tensor) -> Tensor:
            return t.detach().to(device=device, dtype=dtype)

        self.mu = prep(self.params.mu)
        self.var = prep(self.params.var)
        self.sigma = prep(self.params.sigma)

        # --- covariance ---
        K2 = K[2]
        if K2.r == 0:
            sig = prep(K2.core)
            self.cov_od: Tensor | None = sig - torch.diag(sig.diagonal())
        else:
            # k_max=1: only variances tracked; off-diagonal sector unavailable.
            self.cov_od = None
            self.status.append("cov_offdiag_unavailable")

        # --- third cumulant ---
        self.k3_repr: str = "none"
        self.k3_factors: tuple[Tensor, Tensor, Tensor] | None = None
        self.k3_dense: Tensor | None = None
        self.k3_vec: Tensor | None = None
        self.k3_metric: Tensor | None = None
        if 3 in K:
            K3 = K[3]
            if isinstance(K3, FactoredTensor):
                a, b, c = (prep(f) for f in K3.factors)
                self.k3_factors = (a, b, c)
                self.k3_repr = "factored"
                self.k3_rank = a.shape[1]
            elif isinstance(K3, HTensor) and K3.r == 0:
                assert self.n <= dense_max_n, (
                    f"Dense K3 contraction only permitted for n <= {dense_max_n} (got n={self.n})."
                )
                self.k3_dense = prep(K3.core)
                self.k3_repr = "dense"
            elif isinstance(K3, HTensor) and K3.r == 1:
                self.k3_vec = prep(K3.core)
                self.k3_metric = prep(_htensor_metric_matrix(K3.metric))
                self.k3_repr = "trace"
            else:
                raise ValueError(f"Unsupported K[3] representation: {K3!r}")

        # --- fourth cumulant trace sector ---
        self.k4_sector: str = "none"
        self.k4_core: Tensor | None = None
        self.k4_metric: Tensor | None = None
        if 4 in K:
            K4 = K[4]
            assert isinstance(K4, HTensor), f"Unsupported K[4] type {type(K4)!r}"
            metric = prep(_htensor_metric_matrix(K4.metric))
            if K4.r == 1:
                self.k4_core = prep(K4.core)
                self.k4_metric = metric
                self.k4_sector = "r1_traceful"  # P_{>=1} kappa4 = R H_2 + R^2 H_0
            elif K4.r == 2:
                c = prep(K4.core)
                assert c.ndim == 0, "r=2 K[4] must have scalar core"
                self.k4_core = c * metric
                self.k4_metric = metric
                self.k4_sector = "r2_double_trace"  # P_{>=2} kappa4 = R^2 H_0
            elif K4.r == 0:
                assert self.n <= dense_max_n, (
                    f"Dense K4 contraction only permitted for n <= {dense_max_n} (got n={self.n})."
                )
                self.k4_core = prep(K4.core)
                self.k4_metric = None
                self.k4_sector = "dense"
            else:
                raise ValueError(f"Unsupported K[4] radial index {K4.r}")

    # ------------------------------------------------------------------
    def has(self, kind: str) -> bool:
        if kind == "cov_od":
            return self.cov_od is not None
        if kind == "k3":
            return self.k3_repr != "none"
        if kind == "k4":
            return self.k4_sector != "none"
        raise ValueError(f"Unknown sector {kind}")

    # ------------------------------------------------------------------
    @staticmethod
    def _dense_merged(t: Tensor, legs: tuple[int, ...]) -> tuple[tuple[int, ...], Tensor]:
        """Restrict a dense tensor to merged legs via diagonal extraction.

        Returns (vars, tensor) with distinct vars, tensor dims aligned to vars.
        """
        legs_l = list(legs)
        while True:
            dup = None
            for i in range(len(legs_l)):
                for j in range(i + 1, len(legs_l)):
                    if legs_l[i] == legs_l[j]:
                        dup = (i, j)
                        break
                if dup:
                    break
            if dup is None:
                break
            i, j = dup
            t = t.diagonal(dim1=i, dim2=j)  # diagonal dim appended last
            leg = legs_l[i]
            del legs_l[j]
            del legs_l[i]
            legs_l.append(leg)
        return tuple(legs_l), t

    def _matrix_factor(self, m: Tensor, legs: tuple[int, ...]) -> VEFactor:
        if legs[0] == legs[1]:
            return VEFactor(vars=(legs[0],), tensor=m.diagonal())
        return VEFactor(vars=legs, tensor=m)

    def build(
        self, kind: str, legs: tuple[int, ...], next_var: int, domains: dict[int, int]
    ) -> tuple[list[VEFactor], int]:
        """Return VE factors realizing diagram factor `kind` on vertex vars `legs`.

        May allocate internal variables (rank indices) starting at next_var;
        returns the updated next free variable id.
        """
        if kind == "cov_od":
            assert self.cov_od is not None, "cov_od sector unavailable"
            assert legs[0] != legs[1], "cov_od legs are never merged (zero diagonal)"
            return [VEFactor(vars=legs, tensor=self.cov_od)], next_var
        if kind == "k3":
            if self.k3_repr == "factored":
                a, b, c = self.k3_factors  # type: ignore[misc]
                rho = next_var
                domains[rho] = a.shape[1]
                return (
                    [
                        VEFactor(vars=(legs[0], rho), tensor=a),
                        VEFactor(vars=(legs[1], rho), tensor=b),
                        VEFactor(vars=(legs[2], rho), tensor=c),
                    ],
                    next_var + 1,
                )
            if self.k3_repr == "dense":
                vars_, t = self._dense_merged(self.k3_dense, legs)  # type: ignore[arg-type]
                return [VEFactor(vars=vars_, tensor=t)], next_var
            raise ValueError(f"k3 factor requested but representation is {self.k3_repr}")
        if kind == "k3_core_vec":
            assert self.k3_repr == "trace"
            return [VEFactor(vars=(legs[0],), tensor=self.k3_vec)], next_var
        if kind == "k3_metric":
            assert self.k3_repr == "trace"
            return [self._matrix_factor(self.k3_metric, legs)], next_var
        if kind == "k4_core":
            if self.k4_sector == "dense":
                raise ValueError("dense K4 uses the 'k4_dense' factor kind")
            return [self._matrix_factor(self.k4_core, legs)], next_var
        if kind == "k4_metric":
            if self.k4_sector == "dense":
                raise ValueError("dense K4 uses the 'k4_dense' factor kind")
            return [self._matrix_factor(self.k4_metric, legs)], next_var
        if kind == "k4_dense":
            vars_, t = self._dense_merged(self.k4_core, legs)  # type: ignore[arg-type]
            return [VEFactor(vars=vars_, tensor=t)], next_var
        raise ValueError(f"Unknown factor kind {kind!r}")
