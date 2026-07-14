"""Argmax endpoint basics: product-Gaussian winner probabilities (spec 16.1),
Hermite coefficient shift (16.3), simplex/symmetry properties (16.7)."""

import itertools
import math

import torch

from mlp_kprop.harmonic import HTensor
from mlp_kprop.max_endpoint.argmax import (
    argmax_endpoint_estimate,
    product_gaussian_argmax,
    product_gaussian_argmax_reference,
    project_to_simplex,
)
from mlp_kprop.max_endpoint.hermite_endpoint import hermite_he

torch.set_grad_enabled(False)  # argmax module must locally re-enable autograd

MU = torch.tensor([0.3, -0.2, 0.1, 0.0], dtype=torch.float64)
SIGMA = torch.tensor([1.0, 0.8, 1.2, 0.9], dtype=torch.float64)


def _dense_tower(mu, cov, k3=None, k4=None):
    K = {1: HTensor(mu, r=0), 2: HTensor(cov, r=0)}
    if k3 is not None:
        K[3] = HTensor(k3, r=0)
    if k4 is not None:
        K[4] = HTensor(k4, r=0)
    return K


def _rand_sym(shape, gen):
    t = torch.randn(*shape, dtype=torch.float64, generator=gen)
    out = torch.zeros_like(t)
    for perm in itertools.permutations(range(t.ndim)):
        out += t.permute(perm)
    return out / math.factorial(t.ndim)


# ---------------------------------------------------------------------------
# 16.1 product-Gaussian argmax
# ---------------------------------------------------------------------------

def test_product_gaussian_gradient_vs_explicit():
    q, err, _ = product_gaussian_argmax(MU, SIGMA)
    q_ref = product_gaussian_argmax_reference(MU, SIGMA)
    assert float((q - q_ref).abs().max()) < 1e-10
    assert err < 1e-12
    assert abs(float(q.sum()) - 1.0) < 1e-12


def test_product_gaussian_vs_monte_carlo():
    gen = torch.Generator().manual_seed(0)
    m = 4_000_000
    z = MU + SIGMA * torch.randn(m, 4, dtype=torch.float64, generator=gen)
    q_mc = torch.bincount(z.argmax(dim=1), minlength=4).double() / m
    q, _, _ = product_gaussian_argmax(MU, SIGMA)
    # MC se per coordinate ~ sqrt(q(1-q)/m) <= 2.5e-4
    assert float((q - q_mc).abs().max()) < 5 * 2.5e-4


def test_equal_means_and_variances_give_uniform():
    for n in (2, 5, 32):
        q, _, _ = product_gaussian_argmax(
            torch.zeros(n, dtype=torch.float64), torch.ones(n, dtype=torch.float64)
        )
        assert float((q - 1.0 / n).abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# 16.3 Hermite coefficient shift
# ---------------------------------------------------------------------------

def _he_multi(beta, g):
    """Unnormalized multi-index probabilists' Hermite prod_j He_{beta_j}(G_j)."""
    out = torch.ones(g.shape[0], dtype=torch.float64)
    for j, b in enumerate(beta):
        if b > 0:
            out = out * hermite_he(b, g[:, j])
    return out


def test_hermite_coefficient_shift():
    """E[1{i=argmax} He_beta(G)] = E[max_j G_j * He_{beta+e_i}(G)] (G std normal),
    for the repo's unnormalized He convention, and the orthonormal corollary
    coeff(argmax_i, psi_beta) = sqrt(beta_i + 1) * coeff(max, psi_{beta+e_i})."""
    n = 3
    gen = torch.Generator().manual_seed(42)
    m = 6_000_000
    g = torch.randn(m, n, dtype=torch.float64, generator=gen)
    mx = g.max(dim=1).values
    am = g.argmax(dim=1)
    for beta in [(0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 1, 0)]:
        he_beta = _he_multi(beta, g)
        for i in range(n):
            shifted = list(beta)
            shifted[i] += 1
            lhs_samples = (am == i).double() * he_beta
            rhs_samples = mx * _he_multi(tuple(shifted), g)
            lhs, rhs = float(lhs_samples.mean()), float(rhs_samples.mean())
            se = float((lhs_samples - rhs_samples).std()) / math.sqrt(m)
            assert abs(lhs - rhs) < max(6 * se, 5e-3), (
                f"beta={beta} i={i}: argmax coeff {lhs} vs shifted max coeff {rhs} (se={se})"
            )
            # Orthonormal Hermites psi_beta = He_beta / sqrt(beta!): the shift
            # introduces exactly a sqrt(beta_i + 1) factor.
            fac_beta = math.prod(math.factorial(b) for b in beta)
            fac_shift = math.prod(math.factorial(b) for b in shifted)
            coeff_argmax_on = lhs / math.sqrt(fac_beta)
            coeff_max_on = rhs / math.sqrt(fac_shift)
            assert abs(coeff_argmax_on - math.sqrt(beta[i] + 1) * coeff_max_on) < max(
                6 * se, 5e-3
            )


# ---------------------------------------------------------------------------
# 16.7 simplex and symmetry
# ---------------------------------------------------------------------------

def _full_tower(gen):
    n = 4
    cov = torch.diag(SIGMA**2).clone()
    cov[0, 1] = cov[1, 0] = 0.3
    cov[2, 3] = cov[3, 2] = -0.2
    k3 = 0.1 * _rand_sym((n, n, n), gen)
    k4 = 0.05 * _rand_sym((n, n, n, n), gen)
    return _dense_tower(MU.clone(), cov, k3, k4)


def test_all_truncations_sum_to_one():
    gen = torch.Generator().manual_seed(1)
    res = argmax_endpoint_estimate(_full_tower(gen))
    assert set(res.q_raw) == {"E0_product_gaussian", "E1_cov1", "E2_cov2", "E2_k3", "E2_full"}
    for name, q in res.q_raw.items():
        assert abs(float(q.sum()) - 1.0) < 1e-10, f"{name}: sum={float(q.sum())}"
        assert res.simplex[name]["simplex_residual"] < 1e-10


def test_permutation_equivariance():
    gen = torch.Generator().manual_seed(2)
    K = _full_tower(gen)
    res = argmax_endpoint_estimate(K)
    perm = torch.tensor([2, 0, 3, 1])
    Kp = {
        1: HTensor(K[1].core[perm], r=0),
        2: HTensor(K[2].core[perm][:, perm], r=0),
        3: HTensor(K[3].core[perm][:, perm][:, :, perm], r=0),
        4: HTensor(K[4].core[perm][:, perm][:, :, perm][:, :, :, perm], r=0),
    }
    res_p = argmax_endpoint_estimate(Kp)
    for name in res.q_raw:
        assert float((res_p.q_raw[name] - res.q_raw[name][perm]).abs().max()) < 1e-11, name


def test_simplex_projection_nonfinite_input():
    """Deep narrow towers can produce non-finite raw estimates; the projection
    must propagate NaN (flagged downstream) instead of crashing (regression:
    depth-6 pilot, w16, 15 clamped variances -> NaN q -> nonzero().max() on
    an empty tensor)."""
    q = torch.tensor([0.5, float("nan"), 0.3], dtype=torch.float64)
    p = project_to_simplex(q)
    assert p.shape == q.shape
    assert torch.isnan(p).all()
    q_inf = torch.tensor([0.5, float("inf"), 0.3], dtype=torch.float64)
    assert torch.isnan(project_to_simplex(q_inf)).all()


def test_simplex_projection():
    q = torch.tensor([0.5, -0.1, 0.4, 0.2], dtype=torch.float64)
    p = project_to_simplex(q)
    assert abs(float(p.sum()) - 1.0) < 1e-12
    assert float(p.min()) >= 0.0
    # Projection of a point already in the simplex is the identity.
    q2 = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    assert float((project_to_simplex(q2) - q2).abs().max()) < 1e-12
    # Projection must be the Euclidean-nearest simplex point (spot check).
    for cand in [torch.tensor([0.6, 0.0, 0.3, 0.1], dtype=torch.float64)]:
        assert float(((p - q) ** 2).sum()) <= float(((cand - q) ** 2).sum()) + 1e-12
