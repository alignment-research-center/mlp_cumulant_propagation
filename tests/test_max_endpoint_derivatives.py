"""Endpoint derivative formula vs finite differences (spec test L2).

The degree-1 formula is validated against central finite differences of the
adaptive-quadrature Psi; each higher degree is validated against central
finite differences of the (already validated) formula one degree lower:

    d^(beta + e_i) Psi ~= [d^beta Psi(mu + h e_i) - d^beta Psi(mu - h e_i)] / 2h.
"""

import itertools

import pytest
import torch

from mlp_kprop.max_endpoint.hermite_endpoint import endpoint_derivative_reference
from mlp_kprop.max_endpoint.quadrature import product_gaussian_max_reference

torch.manual_seed(0)

N = 4
MU = torch.tensor([0.3, -0.7, 1.1, 0.0], dtype=torch.float64)
SIGMA = torch.tensor([1.0, 0.6, 1.5, 0.9], dtype=torch.float64)
H = 1e-4


def _shift(mu, i, h):
    out = mu.clone()
    out[i] += h
    return out


@pytest.mark.parametrize("i", range(N))
def test_first_derivative_vs_fd_of_psi(i):
    beta = tuple(1 if j == i else 0 for j in range(N))
    d = endpoint_derivative_reference(MU, SIGMA, beta)
    fd = (
        product_gaussian_max_reference(_shift(MU, i, H).numpy(), SIGMA.numpy())
        - product_gaussian_max_reference(_shift(MU, i, -H).numpy(), SIGMA.numpy())
    ) / (2 * H)
    assert abs(d - fd) < 1e-6, f"beta={beta}: formula={d} fd={fd}"


def _betas_of_degree(total: int):
    """All multi-indices over N coordinates with |beta| = total, beta_i <= 4."""
    out = []
    for combo in itertools.combinations_with_replacement(range(N), total):
        beta = [0] * N
        for c in combo:
            beta[c] += 1
        if max(beta) <= 4:
            out.append(tuple(beta))
    return sorted(set(out))


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_higher_derivatives_recursively(degree):
    """d^(beta) vs FD_i of d^(beta - e_i), for a representative subset."""
    betas = _betas_of_degree(degree)
    # Keep runtime modest: subsample deterministic representative patterns,
    # always including the fully-repeated index (max Hermite order).
    betas = betas[:: max(1, len(betas) // 6)] + [
        b for b in betas if max(b) == min(degree, 4)
    ]
    for beta in sorted(set(betas)):
        i = next(j for j in range(N) if beta[j] > 0)
        lower = tuple(b - 1 if j == i else b for j, b in enumerate(beta))
        d = endpoint_derivative_reference(MU, SIGMA, beta)
        if sum(lower) == 0:
            continue
        fd = (
            endpoint_derivative_reference(_shift(MU, i, H), SIGMA, lower)
            - endpoint_derivative_reference(_shift(MU, i, -H), SIGMA, lower)
        ) / (2 * H)
        tol = 1e-5 * max(1.0, abs(fd))
        assert abs(d - fd) < tol, f"beta={beta}: formula={d} fd={fd}"


def test_sum_of_first_derivatives_is_one():
    """Translation invariance: sum_i dPsi/dmu_i = 1 (shifting all mu by c shifts Psi by c)."""
    total = sum(
        endpoint_derivative_reference(MU, SIGMA, tuple(1 if j == i else 0 for j in range(N)))
        for i in range(N)
    )
    assert abs(total - 1.0) < 1e-9
