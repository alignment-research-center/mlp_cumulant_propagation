"""End-to-end validation (spec tests L7, L8) plus ground-truth backends."""

import math

import torch

from mlp_kprop.harmonic import HTensor
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.max_endpoint.estimator import max_endpoint_estimate
from mlp_kprop.max_endpoint.ground_truth import (
    expected_chi_norm,
    mc_flops_per_sample,
    reference_estimate,
)
from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)


def _tower_from_dense(mu, sigma2, k3=None, k4=None):
    """Build a kprop-style tower from dense (float64) cumulant tensors."""
    K = {
        1: HTensor(mu, r=0),
        2: HTensor(torch.diag(sigma2) if sigma2.ndim == 1 else sigma2, r=0),
    }
    if k3 is not None:
        K[3] = HTensor(k3, r=0)
    if k4 is not None:
        K[4] = HTensor(k4, r=0)
    return K


def test_bivariate_gaussian_exact():
    """L7a: correlated bivariate Gaussian has a closed-form E[max]:

        E[max(Y1, Y2)] = mu1 Phi(d/th) + mu2 Phi(-d/th) + th phi(d/th),
        d = mu1 - mu2, th^2 = s1^2 + s2^2 - 2 s12.

    E2_cov must beat E0 by a large factor; here kappa3 = kappa4 = 0 so
    E2_k3 = E2_cov exactly.
    """
    mu = torch.tensor([0.2, -0.1], dtype=torch.float64)
    s1, s2, rho = 1.0, 0.7, 0.6
    s12 = rho * s1 * s2
    cov = torch.tensor([[s1**2, s12], [s12, s2**2]], dtype=torch.float64)
    d = float(mu[0] - mu[1])
    th = math.sqrt(s1**2 + s2**2 - 2 * s12)
    ndist = torch.distributions.Normal(0.0, 1.0)
    exact = (
        float(mu[0]) * float(ndist.cdf(torch.tensor(d / th)))
        + float(mu[1]) * float(ndist.cdf(torch.tensor(-d / th)))
        + th * math.exp(-0.5 * (d / th) ** 2) / math.sqrt(2 * math.pi)
    )
    zeros3 = torch.zeros(2, 2, 2, dtype=torch.float64)
    zeros4 = torch.zeros(2, 2, 2, 2, dtype=torch.float64)
    res = max_endpoint_estimate(_tower_from_dense(mu, cov, zeros3, zeros4))
    err0 = abs(res.estimates["E0_product_gaussian"] - exact)
    err2 = abs(res.estimates["E2_cov2"] - exact)
    assert err2 < 0.15 * err0, f"E0 err {err0}, E2_cov err {err2}"
    assert abs(res.estimates["E2_k3"] - res.estimates["E2_cov2"]) < 1e-12
    assert abs(res.estimates["E2_full"] - res.estimates["E2_cov2"]) < 1e-12


def test_synthetic_skewed_independent():
    """L7b: independent coordinates Y_i = a Z + c (Z^2 - 1), Z ~ N(0,1), with
    exact low-order cumulants

        var = a^2 + 2 c^2,  kappa3 = 6 a^2 c + 8 c^3,  kappa4 = 48 a^2 c^2 + 48 c^4.

    Adding C3 (and C4) must improve on the covariance-only estimator, in order.
    """
    torch.manual_seed(0)
    n = 4
    a = torch.tensor([1.0, 0.8, 1.2, 0.9], dtype=torch.float64)
    c = 0.5 * torch.tensor([0.15, 0.2, 0.1, 0.18], dtype=torch.float64)
    mu = torch.zeros(n, dtype=torch.float64)
    var = a**2 + 2 * c**2
    k3d = 6 * a**2 * c + 8 * c**3
    k4d = 48 * a**2 * c**2 + 48 * c**4
    k3 = torch.zeros(n, n, n, dtype=torch.float64)
    k4 = torch.zeros(n, n, n, n, dtype=torch.float64)
    for i in range(n):
        k3[i, i, i] = k3d[i]
        k4[i, i, i, i] = k4d[i]

    # Ground truth by large Monte Carlo.
    z = torch.randn(30_000_000, n, dtype=torch.float64)
    y = a * z + c * (z**2 - 1)
    m = y.max(dim=1).values
    truth = m.mean().item()
    se = m.std().item() / math.sqrt(m.numel())

    res = max_endpoint_estimate(_tower_from_dense(mu, torch.diag(var), k3, k4))
    err_cov = abs(res.estimates["E2_cov2"] - truth)
    err_k3 = abs(res.estimates["E2_k3"] - truth)
    err_full = abs(res.estimates["E2_full"] - truth)
    # Independent coordinates: C2 corrections vanish, E2_cov == E0.
    assert abs(res.estimates["E2_cov2"] - res.estimates["E0_product_gaussian"]) < 1e-10
    assert err_k3 < 0.3 * err_cov + 3 * se, f"cov={err_cov} k3={err_k3} (se={se})"
    assert err_full < 0.5 * err_k3 + 3 * se, f"k3={err_k3} full={err_full} (se={se})"


def test_small_mlp_estimators_vs_reference():
    """L8: n=16, shallow ReLU MLP; all estimator variants vs a high-precision
    Monte Carlo reference. Higher-order estimators must land within a few
    reference-sigmas of the truth and beat the product-Gaussian baseline."""
    torch.manual_seed(3)
    n = 16
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=4)
    ref = reference_estimate(
        mlp,
        seed=1000,
        backend="gaussian",
        target_se=3e-4,
        min_samples=2_000_000,
        max_samples=30_000_000,
        batch_size=1_000_000,
        device=torch.device("cpu"),
    )
    K = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.AUGMENT, factor=True)
    res = max_endpoint_estimate(K)
    err0 = abs(res.estimates["E0_product_gaussian"] - ref.mean)
    err_full = abs(res.estimates["E2_full"] - ref.mean)
    assert err_full < 0.5 * err0 + 5 * ref.se, (
        f"E0 err {err0}, E2_full err {err_full}, ref se {ref.se}"
    )


def test_spherical_reference_matches_gaussian():
    """F1 validation: Rao-Blackwellized spherical reference agrees with the
    Gaussian-input reference for a bias-free ReLU network, at lower variance."""
    torch.manual_seed(5)
    n = 12
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=3)
    assert not mlp.has_bias()
    kw = dict(min_samples=2_000_000, max_samples=2_000_000, batch_size=500_000,
              target_se=0.0, device=torch.device("cpu"))
    g = reference_estimate(mlp, seed=1, backend="gaussian", **kw)
    s = reference_estimate(mlp, seed=2, backend="spherical", **kw)
    tol = 4 * math.sqrt(g.se**2 + s.se**2)
    assert abs(g.mean - s.mean) < tol, f"gaussian={g.mean} spherical={s.mean} tol={tol}"
    assert s.var < g.var, "spherical reference should have lower per-sample variance"


def test_spherical_requires_bias_free():
    mlp = MLP(hidden_dim=8, num_layers=3, b_var=0.1)
    assert mlp.has_bias()
    try:
        reference_estimate(mlp, seed=0, backend="spherical", min_samples=10, max_samples=10)
        raise AssertionError("expected ValueError for biased network")
    except ValueError:
        pass


def test_positive_homogeneity():
    """max_i M(cX)_i = c max_i M(X)_i for bias-free ReLU nets (basis of F1)."""
    torch.manual_seed(0)
    mlp = MLP(hidden_dim=8, num_layers=4)
    x = torch.randn(100, 8)
    m1 = mlp(x).out.max(dim=1).values
    m3 = mlp(3.0 * x).out.max(dim=1).values
    assert torch.allclose(3.0 * m1, m3, rtol=1e-5)


def test_expected_chi_norm():
    for n in (1, 2, 10, 500):
        x = torch.randn(2_000_000, n)
        emp = x.norm(dim=1).mean().item()
        assert abs(expected_chi_norm(n) - emp) < 5e-3 * expected_chi_norm(n)


def test_mc_flops_per_sample_convention():
    """MC FLOPs ~ sum_l 2 n^2 per forward + max reduction, per sample."""
    n, depth = 16, 4
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth)
    f = mc_flops_per_sample(mlp, backend="gaussian")
    matmul = 2 * n * n * depth
    assert matmul <= f <= 1.5 * matmul + n, f"per-sample flops {f} vs matmul {matmul}"
    f_sph = mc_flops_per_sample(mlp, backend="spherical")
    assert f_sph > f
