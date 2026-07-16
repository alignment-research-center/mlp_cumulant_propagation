"""Correctness of the dense endpoint derivative tensors D2/D3/D4 built by
equality-pattern (set-partition) assembly in max_endpoint.direct_dense.

Checks:
- every equality pattern of D2/D3/D4 against endpoint_derivative_reference
  (the per-multi-index integral formula validated elsewhere vs autograd/FD);
- D2/D3/D4 against *independent* nested autograd of a differentiable
  product-Gaussian-max quadrature (all entries, n=3);
- finite-difference recursion: FD of D3 in mu reproduces D4.
"""

import math

import pytest
import torch

from mlp_kprop.max_endpoint.direct_dense import (
    DenseMemoryError,
    _Tally,
    dense_derivative_tensor,
)
from mlp_kprop.max_endpoint.hermite_endpoint import (
    EndpointWorkspace,
    endpoint_derivative_reference,
)
from mlp_kprop.max_endpoint.quadrature import (
    QuadratureCfg,
    find_endpoints,
    gauss_legendre_nodes,
)

torch.set_default_dtype(torch.float64)

N = 3
Q = 1024


@pytest.fixture(autouse=True)
def _float64_grad_enabled():
    """Other test modules flip the global default dtype / disable grad."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    with torch.enable_grad():
        yield
    torch.set_default_dtype(prev)


def make_ws(mu, sigma, q=Q):
    cfg = QuadratureCfg()
    lo, hi = find_endpoints(mu, sigma, cfg.tail_log_eps, cfg.bisect_iters)
    t, w = gauss_legendre_nodes(q, lo, hi, mu.device, torch.float64)
    return EndpointWorkspace(mu, sigma, t, w), lo, hi


@pytest.fixture(scope="module")
def params():
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(N, generator=g) * 0.5
    sigma = 0.5 + torch.rand(N, generator=g)
    return mu, sigma


def multi_index(idx, n):
    beta = [0] * n
    for i in idx:
        beta[i] += 1
    return tuple(beta)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_dense_vs_reference_all_patterns(params, order):
    """Every entry (hence every equality pattern) matches the per-index
    reference integral, on matched node counts."""
    mu, sigma = params
    ws, _, _ = make_ws(mu, sigma, q=2048)
    D = dense_derivative_tensor(ws, order, node_chunk=64)
    import itertools

    for idx in itertools.product(range(N), repeat=order):
        ref = endpoint_derivative_reference(
            mu, sigma, multi_index(idx, N), QuadratureCfg(num_nodes=2048)
        )
        got = float(D[idx])
        assert got == pytest.approx(ref, rel=1e-9, abs=1e-12), f"idx={idx}"


def psi_autograd(mu, sigma, lo, hi, q=Q):
    """Differentiable Psi(mu) = hi - int F dt on a fixed GL grid."""
    t, w = gauss_legendre_nodes(q, lo, hi, mu.device, torch.float64)
    a = (t[:, None] - mu[None, :]) / sigma[None, :]
    f = torch.exp(torch.special.log_ndtr(a).sum(dim=1))
    return hi - (w * f).sum()


def autograd_tensor(mu, sigma, lo, hi, order):
    """Order-`order` derivative tensor of Psi wrt mu by nested autograd."""
    mu = mu.clone().requires_grad_(True)
    out = torch.zeros((N,) * order)

    def rec(scalar, prefix):
        (g,) = torch.autograd.grad(
            scalar, mu, create_graph=len(prefix) + 1 < order, retain_graph=True
        )
        if len(prefix) + 1 == order:
            out[prefix] = g.detach()
        else:
            for i in range(N):
                rec(g[i], prefix + (i,))

    rec(psi_autograd(mu, sigma, lo, hi), ())
    return out


@pytest.mark.parametrize("order", [2, 3, 4])
def test_dense_vs_autograd(params, order):
    """Independent check: nested autograd of the scalar quadrature."""
    mu, sigma = params
    ws, lo, hi = make_ws(mu, sigma)
    D = dense_derivative_tensor(ws, order, node_chunk=128)
    A = autograd_tensor(mu, sigma, lo, hi, order)
    assert torch.allclose(D, A, rtol=1e-7, atol=1e-10), (D - A).abs().max()


def test_symmetry(params):
    mu, sigma = params
    ws, _, _ = make_ws(mu, sigma)
    D3 = dense_derivative_tensor(ws, 3)
    D4 = dense_derivative_tensor(ws, 4)
    assert torch.allclose(D3, D3.permute(1, 0, 2))
    assert torch.allclose(D3, D3.permute(2, 1, 0))
    assert torch.allclose(D4, D4.permute(1, 0, 2, 3))
    assert torch.allclose(D4, D4.permute(3, 1, 2, 0))
    assert torch.allclose(D4, D4.permute(0, 2, 1, 3))


def test_fd_recursion_d3_to_d4(params):
    """(D3(mu + h e_l) - D3(mu - h e_l)) / 2h ~= D4[..., l]."""
    mu, sigma = params
    ws, _, _ = make_ws(mu, sigma)
    D4 = dense_derivative_tensor(ws, 4)
    h = 1e-4
    for l in range(N):
        e = torch.zeros(N)
        e[l] = h
        wsp, _, _ = make_ws(mu + e, sigma)
        wsm, _, _ = make_ws(mu - e, sigma)
        fd = (dense_derivative_tensor(wsp, 3) - dense_derivative_tensor(wsm, 3)) / (2 * h)
        assert torch.allclose(fd, D4[..., l], rtol=5e-4, atol=1e-6), l


def test_first_derivative_sums_to_one(params):
    """sum_i dPsi/dmu_i = 1 (Psi(mu + c 1) = Psi(mu) + c)."""
    mu, sigma = params
    ws, _, _ = make_ws(mu, sigma)
    D1 = dense_derivative_tensor(ws, 1)
    assert float(D1.sum()) == pytest.approx(1.0, abs=1e-10)


def test_memory_guard():
    mu = torch.zeros(4096)
    sigma = torch.ones(4096)
    cfg = QuadratureCfg(num_nodes=8)
    lo, hi = find_endpoints(mu, sigma, cfg.tail_log_eps, cfg.bisect_iters)
    t, w = gauss_legendre_nodes(8, lo, hi, mu.device, torch.float64)
    big_ws = EndpointWorkspace(mu, sigma, t, w)
    with pytest.raises(DenseMemoryError, match="dense_refused"):
        dense_derivative_tensor(big_ws, 4, max_dense_bytes=10**9)


def test_flop_tally_positive(params):
    mu, sigma = params
    ws, _, _ = make_ws(mu, sigma)
    tally = _Tally()
    dense_derivative_tensor(ws, 4, tally=tally)
    assert tally.total > 0 and "D4_patterns" in tally.by_part
