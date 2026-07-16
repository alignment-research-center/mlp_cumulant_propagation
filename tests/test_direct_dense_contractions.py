"""Dense einsum contractions and operator coefficients (spec 11.3/11.4/11.5).

- Vectorized contractions vs explicit Python nested loops at n = 5.
- Operator coefficients 1/2 (C2), 1/8 (C2^2/2), 1/6 (C3) verified numerically:
  * C2, C2^2/2 against the closed-form bivariate Gaussian max E[max(X1,X2)]
    (Taylor residual in the off-diagonal covariance must be O(c^3));
  * C3 against a synthetic non-Gaussian vector with known cumulants (the
    Edgeworth ladder E0/E1/E2_cov/E2_k3 must successively reduce the error).
"""

import itertools
import math

import pytest
import torch

from mlp_kprop.max_endpoint.direct_dense import dense_derivative_tensor
from mlp_kprop.max_endpoint.hermite_endpoint import EndpointWorkspace
from mlp_kprop.max_endpoint.quadrature import (
    QuadratureCfg,
    find_endpoints,
    gauss_legendre_nodes,
)

torch.set_default_dtype(torch.float64)


import pytest as _pytest


@_pytest.fixture(autouse=True)
def _float64_default():
    """Other test modules flip the global default dtype."""
    import torch as _torch
    prev = _torch.get_default_dtype()
    _torch.set_default_dtype(_torch.float64)
    yield
    _torch.set_default_dtype(prev)


def make_ws(mu, sigma, q=1024):
    cfg = QuadratureCfg()
    lo, hi = find_endpoints(mu, sigma, cfg.tail_log_eps, cfg.bisect_iters)
    t, w = gauss_legendre_nodes(q, lo, hi, mu.device, torch.float64)
    return EndpointWorkspace(mu, sigma, t, w), lo, hi


def corrections_direct(mu, sigma, cov_od, k3=None, k4_core=None, k4_metric=None, q=1024):
    """The exact contraction pattern used by direct_dense._corrections_at_nodes."""
    ws, lo, hi = make_ws(mu, sigma, q)
    psi = float(hi - (ws.w * ws.B).sum())
    D2 = dense_derivative_tensor(ws, 2)
    D4 = dense_derivative_tensor(ws, 4)
    out = {
        "psi": psi,
        "C2": float(0.5 * (cov_od * D2).sum()),
        "C2sq_half": float(
            0.125 * (torch.einsum("ijkl,kl->ij", D4, cov_od) * cov_od).sum()
        ),
    }
    if k3 is not None:
        D3 = dense_derivative_tensor(ws, 3)
        out["C3"] = float((k3 * D3).sum() / 6.0)
    if k4_core is not None:
        out["C4_trace"] = float(
            (torch.einsum("ijkl,kl->ij", D4, k4_metric) * k4_core).sum() / 24.0
        )
    return out


def test_contractions_vs_nested_loops():
    n = 5
    g = torch.Generator().manual_seed(1)
    mu = torch.randn(n, generator=g) * 0.3
    sigma = 0.6 + torch.rand(n, generator=g)
    ws, _, _ = make_ws(mu, sigma, q=512)
    D2 = dense_derivative_tensor(ws, 2)
    D3 = dense_derivative_tensor(ws, 3)
    D4 = dense_derivative_tensor(ws, 4)
    c = torch.randn(n, n, generator=g) * 0.05
    cov_od = (c + c.T) - torch.diag((c + c.T).diagonal())
    k3 = torch.randn(n, n, n, generator=g) * 0.02
    k3 = (
        k3
        + k3.permute(0, 2, 1)
        + k3.permute(1, 0, 2)
        + k3.permute(1, 2, 0)
        + k3.permute(2, 0, 1)
        + k3.permute(2, 1, 0)
    ) / 6
    core = torch.randn(n, n, generator=g) * 0.01
    core = core + core.T
    metric = torch.randn(n, n, generator=g) * 0.01
    metric = metric + metric.T

    # Nested loops.
    c2_loop = 0.5 * sum(
        float(cov_od[i, j] * D2[i, j]) for i in range(n) for j in range(n) if i != j
    )
    c2sq_loop = 0.125 * sum(
        float(cov_od[i, j] * cov_od[k, l] * D4[i, j, k, l])
        for i, j, k, l in itertools.product(range(n), repeat=4)
        if i != j and k != l
    )
    c3_loop = (1 / 6) * sum(
        float(k3[i, j, k] * D3[i, j, k]) for i, j, k in itertools.product(range(n), repeat=3)
    )
    c4_loop = (1 / 24) * sum(
        float(core[i, j] * metric[k, l] * D4[i, j, k, l])
        for i, j, k, l in itertools.product(range(n), repeat=4)
    )

    got = corrections_direct(mu, sigma, cov_od, k3, core, metric, q=512)
    assert got["C2"] == pytest.approx(c2_loop, rel=1e-10)
    assert got["C2sq_half"] == pytest.approx(c2sq_loop, rel=1e-10)
    assert got["C3"] == pytest.approx(c3_loop, rel=1e-10)
    assert got["C4_trace"] == pytest.approx(c4_loop, rel=1e-10)


def bivariate_max_closed_form(mu1, mu2, s1, s2, c):
    """E[max(X1, X2)] for jointly Gaussian X with Cov = [[s1^2, c], [c, s2^2]]."""
    theta = math.sqrt(s1 * s1 + s2 * s2 - 2 * c)
    d = (mu1 - mu2) / theta
    phi = math.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + math.erf(d / math.sqrt(2)))
    return mu1 * Phi + mu2 * (1 - Phi) + theta * phi


def test_c2_and_c2sq_coefficients_bivariate():
    """Psi + C2 Psi + (1/2) C2^2 Psi must match the exact correlated-Gaussian
    max to O(c^3): the residual after the c^2 term shrinks ~8x when c halves."""
    mu = torch.tensor([0.1, -0.2])
    sigma = torch.tensor([1.0, 1.3])
    residuals = []
    for c in (0.12, 0.06, 0.03):
        cov_od = torch.tensor([[0.0, c], [c, 0.0]])
        got = corrections_direct(mu, sigma, cov_od, q=2048)
        exact = bivariate_max_closed_form(0.1, -0.2, 1.0, 1.3, c)
        e1 = got["psi"] + got["C2"]
        e2 = e1 + got["C2sq_half"]
        # Nesting must improve the truncation error at small c.
        assert abs(e2 - exact) < abs(e1 - exact) < abs(got["psi"] - exact)
        residuals.append(abs(e2 - exact))
    # O(c^3) residual: halving c shrinks it ~8x (allow 5x..12x).
    for r_big, r_small in zip(residuals, residuals[1:]):
        assert 5.0 < r_big / r_small < 12.0, residuals


def test_c3_coefficient_synthetic_non_gaussian():
    """Synthetic vector with *independent* skewed coordinates (spec 11.5):
    S_j = Z_j + b_j (Z_j^2 - 1), so Sigma is exactly diagonal (all C2 terms
    vanish identically), the mean is 0, Var = 1 + 2 b^2, kappa3 = 6b + 8b^3.
    The C3 term with coefficient 1/6 must capture most of the gap between the
    product-Gaussian baseline and a high-accuracy MC expected max; a wrong
    coefficient (1/3) must not."""
    n = 4
    g = torch.Generator().manual_seed(3)
    b = 0.04 + 0.04 * torch.rand(n, generator=g)  # skewness knobs

    var_s = 1 + 2 * b**2
    k3_s = 6 * b + 8 * b**3

    mu = torch.zeros(n)
    sigma = var_s.sqrt()
    cov_od = torch.zeros(n, n)
    k3 = torch.zeros(n, n, n)
    for i in range(n):
        k3[i, i, i] = k3_s[i]
    got = corrections_direct(mu, sigma, cov_od, k3, q=2048)
    assert got["C2"] == 0.0 and got["C2sq_half"] == 0.0

    # High-accuracy MC of E[max_i S_i].
    gen = torch.Generator().manual_seed(4)
    total, count = 0.0, 0
    for _ in range(80):
        z = torch.randn(500_000, n, generator=gen)
        s = z + b[None, :] * (z * z - 1)
        total += float(s.max(dim=1).values.double().sum())
        count += s.shape[0]
    target = total / count  # SE ~ 1.5e-4

    gap = target - got["psi"]
    # C3 Psi (coefficient 1/6) explains the gap up to the omitted kappa4 term
    # (relative size ~ 8b ~ 0.5) and MC noise.
    assert abs(got["C3"] - gap) < 0.5 * abs(gap), (got["C3"], gap)
    # Wrong coefficient (1/3 instead of 1/6) is far off.
    assert abs(2 * got["C3"] - gap) > abs(got["C3"] - gap)
    # The ladder improves: |E2_k3 - target| < |E0 - target|.
    assert abs(got["psi"] + got["C3"] - target) < 0.5 * abs(got["psi"] - target)


def test_wrong_coefficient_would_fail_bivariate():
    """Guard the 1/8 factor: using 1/4 for the quadratic term breaks the
    O(c^3) Taylor match."""
    mu = torch.tensor([0.0, 0.0])
    sigma = torch.tensor([1.0, 1.0])
    c = 0.1
    cov_od = torch.tensor([[0.0, c], [c, 0.0]])
    got = corrections_direct(mu, sigma, cov_od, q=2048)
    exact = bivariate_max_closed_form(0.0, 0.0, 1.0, 1.0, c)
    right = got["psi"] + got["C2"] + got["C2sq_half"]
    wrong = got["psi"] + got["C2"] + 2 * got["C2sq_half"]
    assert abs(right - exact) < 0.1 * abs(wrong - exact)
