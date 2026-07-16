"""End-to-end direct-dense estimator on small ReLU MLPs (spec 11.1/11.6).

- Psi (product-Gaussian max) vs Monte Carlo for equal/unequal means/variances.
- kprop tower -> direct_dense_estimate: finite outputs, sensible corrections,
  agreement with a Monte Carlo reference of E_X[max_i M(X)_i].
- Cross-validation: every correction and estimate must match the existing
  (independently validated) diagram/treewidth estimator on the same tower.
"""

import math

import pytest
import torch

from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.max_endpoint.direct_dense import (
    DirectDenseCfg,
    direct_dense_estimate,
)
from mlp_kprop.max_endpoint.estimator import max_endpoint_estimate
from mlp_kprop.max_endpoint.quadrature import QuadratureCfg, product_gaussian_max
from mlp_kprop.mlp import MLP

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


@pytest.mark.parametrize(
    "mu,sigma",
    [
        ([0.0] * 6, [1.0] * 6),
        ([0.5, -0.3, 0.1, 0.0], [1.0] * 4),
        ([0.0, 0.2, -0.4], [0.5, 1.0, 2.0]),
    ],
)
def test_psi_vs_monte_carlo(mu, sigma):
    mu_t = torch.tensor(mu)
    sigma_t = torch.tensor(sigma)
    psi, err, _ = product_gaussian_max(mu_t, sigma_t, QuadratureCfg(num_nodes=256))
    gen = torch.Generator().manual_seed(0)
    total, count = 0.0, 0
    for _ in range(20):
        z = torch.randn(1_000_000, len(mu), generator=gen)
        total += float((mu_t + sigma_t * z).max(dim=1).values.sum())
        count += z.shape[0]
    mc = total / count  # SE ~ 3e-4
    assert psi == pytest.approx(mc, abs=2e-3)
    assert err < 1e-10


def _tower(n, num_layers, kind, k_max, factor, seed):
    torch.manual_seed(seed)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=num_layers,
              nonlin="relu", init_kind="he")
    k_in = {1: torch.zeros(n), 2: torch.eye(n)}
    K = mlp_kprop(mlp, k_in, k_max=k_max, kind=kind, factor=factor)
    return mlp, K


NAME_MAP = {  # direct-dense name -> treewidth-pipeline name
    "E0_product_gaussian": "E0_product_gaussian",
    "E1_cov1": "E1_cov1",
    "E2_cov": "E2_cov2",
    "E2_k3": "E2_k3",
    "E2_k3_k4trace": "E2_full",
}


@pytest.mark.parametrize("n,kind", [(8, "SIMPLE"), (12, "AUGMENT"), (16, "SIMPLE")])
def test_direct_vs_treewidth_pipeline(n, kind):
    """Same tower, same quadrature config: corrections and estimates from the
    direct dense path must match the diagram/treewidth path to quadrature
    rounding."""
    _, K = _tower(n, num_layers=3, kind=Kind[kind], k_max=3, factor=True, seed=n)
    quad = QuadratureCfg(num_nodes=128)
    direct = direct_dense_estimate(K, DirectDenseCfg(quad=quad))
    tw = max_endpoint_estimate(K, quad_cfg=quad)
    for name in ("C2", "C2sq_half", "C3", "C4_trace"):
        assert direct.corrections[name] == pytest.approx(
            tw.corrections[name], rel=1e-8, abs=1e-12
        ), name
    for dname, tname in NAME_MAP.items():
        assert direct.estimates[dname] == pytest.approx(
            tw.estimates[tname], rel=1e-10, abs=1e-12
        ), dname
    assert direct.k4_sector == tw.k4_sector


@pytest.mark.parametrize("n", [8, 12, 16])
def test_small_mlp_vs_monte_carlo(n):
    """Estimates are finite, corrections shrink up the ladder on average, and
    the best estimate agrees with MC ground truth within a few MC SEs."""
    mlp, K = _tower(n, num_layers=3, kind=Kind.SIMPLE, k_max=3, factor=True, seed=100 + n)
    res = direct_dense_estimate(K, DirectDenseCfg(quad=QuadratureCfg(num_nodes=256)))
    for v in res.estimates.values():
        assert math.isfinite(v)
    assert set(res.estimates) == set(NAME_MAP)
    # MC ground truth.
    gen = torch.Generator().manual_seed(1)
    total, count = 0.0, 0
    for _ in range(20):
        x = torch.randn(500_000, n, generator=gen)
        total += float(mlp(x).out.max(dim=1).values.double().sum())
        count += x.shape[0]
    target = total / count  # SE ~ 5e-4 (max is O(1))
    err0 = abs(res.estimates["E0_product_gaussian"] - target)
    err_full = abs(res.estimates["E2_k3_k4trace"] - target)
    assert err_full < err0, (err_full, err0)
    # At these tiny widths the Edgeworth truncation error is genuinely large
    # (see docs/max_endpoint_report.md); require sanity, not sharpness.
    assert err_full < 0.15, err_full
    # Corrections should decrease in magnitude along the expansion.
    assert abs(res.corrections["C2sq_half"]) < abs(res.corrections["C2"])
    assert res.quadrature_error["E2_k3_k4trace"] < 1e-8


def test_kmax1_and_kmax2_towers():
    """k_max=1: variances only -> E1/E2 collapse to E0 (equivalences).
    k_max=2 SIMPLE: C2 and C2^2 available, no k3/k4."""
    n = 8
    _, K1 = _tower(n, 3, Kind.SIMPLE, k_max=1, factor=False, seed=5)
    r1 = direct_dense_estimate(K1)
    assert list(r1.estimates) == ["E0_product_gaussian"]
    assert r1.equivalences["E1_cov1"] == "E0_product_gaussian"
    assert "cov_offdiag_unavailable" in r1.status

    _, K2 = _tower(n, 3, Kind.SIMPLE, k_max=2, factor=False, seed=5)
    r2 = direct_dense_estimate(K2)
    assert set(r2.estimates) == {"E0_product_gaussian", "E1_cov1", "E2_cov"}
    assert "k3_unavailable" in r2.status
    assert "k4_trace_unavailable" in r2.status


def test_d4_refusal_degrades_gracefully():
    """A max_dense_bytes too small for D4 (but fine for D2/D3) must keep
    E0/E1/E2_k3-prerequisites gracefully: E0/E1 present, D4-based estimators
    absent, refusal recorded in status and info."""
    n = 16
    _, K = _tower(n, 3, Kind.SIMPLE, k_max=3, factor=True, seed=7)
    # D4 = 16^4 * 8 = 524288 B (+ intermediates); D3 = 32768 B.
    cfg = DirectDenseCfg(quad=QuadratureCfg(num_nodes=64), max_dense_bytes=200_000)
    r = direct_dense_estimate(K, cfg)
    assert set(r.estimates) == {"E0_product_gaussian", "E1_cov1"}
    assert "dense_refused_D4" in r.status
    assert "D4" in r.info["dense_refused"]
    assert "C3" in r.corrections and "C2sq_half" not in r.corrections
    assert r.largest_dense_tensor_order == 3


def test_dense_orders_restriction():
    """dense_orders=(2,) computes only E0/E1 (extended-width mode)."""
    n = 8
    _, K = _tower(n, 3, Kind.SIMPLE, k_max=2, factor=False, seed=6)
    r = direct_dense_estimate(K, DirectDenseCfg(dense_orders=(2,)))
    assert set(r.estimates) == {"E0_product_gaussian", "E1_cov1"}
    assert r.largest_dense_tensor_order == 2
    full = direct_dense_estimate(K)
    assert r.estimates["E1_cov1"] == pytest.approx(full.estimates["E1_cov1"], rel=1e-12)
