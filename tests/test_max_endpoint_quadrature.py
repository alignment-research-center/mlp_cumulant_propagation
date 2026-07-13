"""Tests for the product-Gaussian max quadrature (spec test L1)."""

import math

import pytest
import torch

from mlp_kprop.max_endpoint.quadrature import (
    QuadratureCfg,
    find_endpoints,
    log_F,
    product_gaussian_max,
    product_gaussian_max_reference,
)

torch.manual_seed(0)


CASES = [
    (torch.tensor([0.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64)),
    (torch.tensor([0.0, 0.0], dtype=torch.float64), torch.tensor([1.0, 1.0], dtype=torch.float64)),
    (
        torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64),
        torch.tensor([0.5, 2.0, 1.0], dtype=torch.float64),
    ),
    (
        torch.tensor([10.0, -10.0], dtype=torch.float64),
        torch.tensor([0.1, 3.0], dtype=torch.float64),
    ),
    (torch.rand(6, dtype=torch.float64) * 4 - 2, torch.rand(6, dtype=torch.float64) * 2 + 0.1),
]


def test_single_gaussian_exact():
    """n=1: Psi = mu."""
    psi, err, _ = product_gaussian_max(
        torch.tensor([1.7], dtype=torch.float64), torch.tensor([2.3], dtype=torch.float64)
    )
    assert abs(psi - 1.7) < 1e-10
    assert err < 1e-10


def test_two_equal_gaussians_exact():
    """n=2, iid N(mu, s^2): E[max] = mu + s/sqrt(pi)."""
    mu, s = 0.3, 1.5
    psi, _, _ = product_gaussian_max(
        torch.tensor([mu, mu], dtype=torch.float64), torch.tensor([s, s], dtype=torch.float64)
    )
    assert abs(psi - (mu + s / math.sqrt(math.pi))) < 1e-10


@pytest.mark.parametrize("case", range(len(CASES)))
def test_gl_matches_adaptive_reference(case):
    mu, sigma = CASES[case]
    psi, err, _ = product_gaussian_max(mu.double(), sigma.double())
    ref = product_gaussian_max_reference(mu.numpy(), sigma.numpy())
    assert abs(psi - ref) < 1e-9, f"GL={psi} ref={ref}"
    assert err < 1e-9


@pytest.mark.parametrize("case", range(len(CASES)))
def test_gl_matches_monte_carlo(case):
    torch.manual_seed(1234 + case)
    mu, sigma = CASES[case]
    z = torch.randn(4_000_000, mu.shape[0], dtype=torch.float64)
    samples = (mu.double() + sigma.double() * z).max(dim=1).values
    mc = samples.mean().item()
    se = samples.std().item() / math.sqrt(samples.numel())
    psi, _, _ = product_gaussian_max(mu.double(), sigma.double())
    assert abs(psi - mc) < 5 * se, f"GL={psi} MC={mc}+-{se}"


def test_quadrature_convergence_with_nodes():
    mu, sigma = CASES[2]
    psi_lo, err_lo, _ = product_gaussian_max(mu, sigma, QuadratureCfg(num_nodes=64))
    psi_hi, err_hi, _ = product_gaussian_max(mu, sigma, QuadratureCfg(num_nodes=256))
    ref = product_gaussian_max_reference(mu.numpy(), sigma.numpy())
    assert abs(psi_hi - ref) <= abs(psi_lo - ref) + 1e-12
    assert err_hi <= err_lo + 1e-12


def test_endpoints_bracket_distribution():
    mu = torch.tensor([0.0, 5.0, -3.0], dtype=torch.float64)
    sigma = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)
    lo, hi = find_endpoints(mu, sigma, math.log(1e-25))
    # The mu=5, sigma=0.5 coordinate dominates the max: F only transitions
    # near t ~ 5, so lo need not lie below the smaller means -- it must only
    # satisfy the F-tail condition (checked below), and hi must cover the top.
    assert lo < hi and hi > 5.0
    lf_lo = float(log_F(torch.tensor(lo, dtype=torch.float64), mu, sigma))
    lf_hi = float(log_F(torch.tensor(hi, dtype=torch.float64), mu, sigma))
    assert abs(lf_lo - math.log(1e-25)) < 1e-6
    assert abs(-math.expm1(lf_hi) - 1e-25) < 1e-27


def test_large_n_no_underflow():
    """Hundreds of coordinates: log-domain CDF product must not underflow."""
    torch.manual_seed(7)
    n = 512
    mu = torch.randn(n, dtype=torch.float64)
    sigma = 0.5 + torch.rand(n, dtype=torch.float64)
    psi, err, _ = product_gaussian_max(mu, sigma)
    assert math.isfinite(psi)
    assert err < 1e-8
    # Sanity: expected max of ~512 unit-scale Gaussians is around 3.
    assert 2.0 < psi < 6.0
