import math

import numpy as np
import torch
from numpy.polynomial import Polynomial

import torch.nn.functional as F

from mlp_kprop.wick import *

torch.set_default_dtype(torch.float64)


def test_norm():
    assert np.allclose(norm_pdf(0.0), 1 / math.sqrt(2 * math.pi))
    assert np.allclose(norm_cdf(0.0), 0.5)
    # Check against scipy.stats.norm
    from scipy.stats import norm

    xs = np.linspace(-5, 5, 100)
    for x in xs:
        assert np.allclose(norm_pdf(x).item(), norm.pdf(x))
        assert np.allclose(norm_cdf(x).item(), norm.cdf(x))


def test_He():
    from scipy.special import eval_hermitenorm as spHe

    xs = torch.linspace(-5, 5, 100)
    for n in range(10):
        h1 = He(n, xs)
        h2 = torch.tensor(spHe(n, xs.numpy()))
        assert torch.allclose(h1.float(), h2.float())


def test_poly_wick_coefs():
    # coef, k, mean, var, expected
    tests = [
        ([1], 0, 0, 1, 1),
        ([0, 1], 0, 0, 1, 0),
        ([0, 0, 1], 0, 0, 1, 1),
        ([0, 0, 1], 1, 0, 1, 0),
        ([0, 0, 1], 2, 0, 1, 2),
        ([0, 0, 0, 0, 1], 0, 0, 1, 3),
        ([0, 0, 1, 0, 1], 0, 1, 4, 78),
        ([1, 3, 4, 1], 0, 2, 4, 71),
    ]
    for coef, k, mean, var, expected in tests:
        poly = Polynomial(coef)
        assert int(poly_wick_coef(poly, float(mean), float(var), k)) == int(expected)


def test_relu_wick_coefs_basic():
    assert np.allclose(relu_wick_coef(0.0, 1.0, 0), 1 / math.sqrt(2 * math.pi))
    assert np.allclose(relu_wick_coef(0.0, 1.0, 1), 0.5)
    assert np.allclose(relu_wick_coef(0.0, 1.0, 2), 1 / math.sqrt(2 * math.pi))
    assert np.allclose(relu_wick_coef(0.0, 1.0, 3), 0.0)
    assert np.allclose(relu_wick_coef(1.0, 2.0, 0), 1.19964)  # from wolfram alpha
    assert np.allclose(relu_wick_coef(1.0, 2.0, 1), 0.76025)  # from wolfram alpha


def test_relu_wick_coefs():
    # mu, sigma
    tests = [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5), (3.0, 1.5), (1.0, 0.01), (0.0, 0.01), (3.0, 10.0)]
    phi = lambda x: torch.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)
    p_z = lambda z, mu, sigma: phi((z - mu) / sigma) / sigma

    def dn(f, n):
        g = f
        for _ in range(n):
            g = torch.func.grad(g)
        return g

    for mu, sigma in tests:
        # test k=0, p=1 via numeric integration
        zs = torch.linspace(mu - 6 * sigma, mu + 6 * sigma, 10000)
        res = relu_wick_coef(mu, sigma**2, 0).float()
        exp = torch.trapz(torch.maximum(zs, torch.tensor(0.0)) * p_z(zs, mu, sigma), zs).float()
        assert torch.allclose(res, exp)

        # test k=1, p=1 using norm_cdf
        # E[∂ ReLU(Z)] = P(z >= 0) = Φ(mu/sigma)
        res = relu_wick_coef(mu, sigma**2, 1).float()
        exp = norm_cdf(mu / sigma).float()
        assert torch.allclose(res, exp)

        # test k>1, p=1 using autograd on p_Z
        # E[∂^k ReLU(Z)] = (-1)^{k-2} p_Z^{(k-2)}(0)
        for k in range(2, 6):
            m = k - 2
            p_zm = dn(lambda z: p_z(z, mu, sigma), m)
            exp = (-1) ** m * p_zm(torch.tensor(0.0)).float()
            res = relu_wick_coef(mu, sigma**2, k).float()
            assert torch.allclose(res, exp)

        # test k>1, p>=1 using autograd on analytic expression for E[ReLU(Z+h)^p]
        # E[∂^k ReLU(Z)^p] = E[d^k/dh^k ReLU(Z+h)^p] = d^k/dh^k E[ReLU(Z+h)^p]
        for k in range(2, 6):
            for p in range(1, 6):
                F = lambda h: relu_wick_coef(mu + h, sigma**2, k=0, p=p)
                Fk = dn(F, k)
                exp = Fk(torch.tensor(0.0)).float()
                res = relu_wick_coef(mu, sigma**2, k=k, p=p).float()
                assert torch.allclose(res, exp)

        # test k < p via numeric integration
        for p in range(2, 6):
            for k in range(p):
                res = relu_wick_coef(mu, sigma**2, k, p).float()
                zs = torch.linspace(mu - 8 * sigma, mu + 8 * sigma, 50000)
                exp = torch.trapz(
                    math.prod(range(p - k + 1, p + 1))
                    * torch.maximum(zs, torch.tensor(0.0)) ** (p - k)
                    * p_z(zs, mu, sigma),
                    zs,
                ).float()
                assert torch.allclose(res, exp)


def test_heaviside_wick_coefs():
    # mu, sigma
    tests = [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5), (3.0, 1.5), (1.0, 0.01), (0.0, 0.01), (3.0, 10.0)]
    phi = lambda x: torch.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)
    p_z = lambda z, mu, sigma: phi((z - mu) / sigma) / sigma

    def dn(f, n):
        g = f
        for _ in range(n):
            g = torch.func.grad(g)
        return g

    for mu, sigma in tests:
        # test k=0, p=1 using norm_cdf
        # E[H(Z)] = P(z >= 0) = Φ(mu/sigma)
        res = heaviside_wick_coef(mu, sigma**2, 0).float()
        exp = norm_cdf(mu / sigma).float()
        assert torch.allclose(res, exp)

        # test k>=1, p=1 using autograd on p_Z
        # E[∂^k H(Z)] = (-1)^{k-1} p_Z^{(k-1)}(0)
        for k in range(1, 6):
            m = k - 1
            p_zm = dn(lambda z: p_z(z, mu, sigma), m)
            exp = (-1) ** m * p_zm(torch.tensor(0.0)).float()
            res = heaviside_wick_coef(mu, sigma**2, k).float()
            assert torch.allclose(res, exp)


def test_hermgauss_relu_wick_coefs():
    # ReLU's kink at 0 limits quadrature accuracy, so we use more points and looser tolerance.
    tests = [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5), (3.0, 1.5), (3.0, 10.0)]
    relu = torch.nn.functional.relu

    for mu, sigma in tests:
        for k in range(5):
            for p in range(1, 5):
                res = hermgauss_wick_coef(relu, mu, sigma**2, k, p, deg=200)
                exp = relu_wick_coef(mu, sigma**2, k, p)
                assert torch.allclose(res.float(), exp.float(), atol=1e-4, rtol=0.02), \
                    f"mu={mu}, sigma={sigma}, k={k}, p={p}: got {res.item():.6f}, expected {exp.item():.6f}"


def test_sigmoid_wick_coefs():
    tests = [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5), (3.0, 1.5)]
    phi = lambda x: torch.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)
    p_z = lambda z, mu, sigma: phi((z - mu) / sigma) / sigma

    def dn(f, n):
        g = f
        for _ in range(n):
            g = torch.func.grad(g)
        return g

    for mu, sigma in tests:
        for k in range(4):
            for p in range(1, 3):
                res = sigmoid_wick_coef(mu, sigma**2, k, p)

                # Reference: autograd for ∂^k sigmoid(z)^p, trapz for E[...]
                zs = torch.linspace(mu - 10 * sigma, mu + 10 * sigma, 20000)
                dk_f = dn(lambda z: torch.sigmoid(z) ** p, k)
                dk_vals = torch.vmap(dk_f)(zs)
                exp = torch.trapz(dk_vals * p_z(zs, mu, sigma), zs)

                assert torch.allclose(res.float(), exp.float()), \
                    f"sigmoid: mu={mu}, sigma={sigma}, k={k}, p={p}: got {res.item():.6f}, expected {exp.item():.6f}"


def test_gelu_wick_coefs():
    tests = [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.5), (3.0, 1.5)]
    phi = lambda x: torch.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)
    p_z = lambda z, mu, sigma: phi((z - mu) / sigma) / sigma

    def dn(f, n):
        g = f
        for _ in range(n):
            g = torch.func.grad(g)
        return g

    for mu, sigma in tests:
        for k in range(4):
            for p in range(1, 3):
                res = gelu_wick_coef(mu, sigma**2, k, p)

                # Reference: autograd for ∂^k gelu(z)^p, trapz for E[...]
                zs = torch.linspace(mu - 10 * sigma, mu + 10 * sigma, 20000)
                dk_f = dn(lambda z: F.gelu(z) ** p, k)
                dk_vals = torch.vmap(dk_f)(zs)
                exp = torch.trapz(dk_vals * p_z(zs, mu, sigma), zs)

                assert torch.allclose(res.float(), exp.float()), \
                    f"gelu: mu={mu}, sigma={sigma}, k={k}, p={p}: got {res.item():.6f}, expected {exp.item():.6f}"
