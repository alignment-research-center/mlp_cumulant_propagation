import logging
import math
from functools import cache, partial
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)

from numpy.polynomial import Polynomial, hermite_e
from numpy.polynomial.hermite import hermgauss


def norm_pdf(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    return torch.special.ndtr(x)


def He(n: int, x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    return torch.special.hermite_polynomial_he(x, n)


@cache
def He_poly(n: int) -> Polynomial:
    return hermite_e.HermiteE.basis(n).convert(kind=Polynomial)


def eval_poly(poly: Polynomial, x: Float[Tensor, "n"]) -> Float[Tensor, "n"]:
    """
    Evaluates a numpy Polynomial at a torch tensor x.
    """
    x = torch.as_tensor(x)
    coef = torch.as_tensor(poly.coef, device=x.device, dtype=x.dtype)
    # Horner's method
    y = torch.zeros_like(x)
    for ck in coef.flip(0):
        y = y * x + ck
    return y


@cache
def _gauss_moment_poly(k: int) -> Polynomial:
    """
    Returns a polynomial P such that E[Z^k] = σ^k P(α) using the notation of relu_wick_coef.
    """
    alpha = Polynomial([0, 1])
    return sum(
        math.comb(k, 2 * j) * math.prod(range(1, 2 * j, 2)) * alpha ** (k - 2 * j)
        for j in range(k // 2 + 1)
    )


@cache
def _relu_wick_poly(p: int, k: int) -> tuple[Polynomial, Polynomial]:
    """
    Returns polynomials (P_1, P_2) such that
        E[∂^k ReLU(Z)^p] = σ^{p-k} (P_1(α)φ(α) + P_2(α)Φ(α))
    when k<p, where notation is as in relu_wick_coef and k < p. Explicitly,
        P_1(α) = (p)⤋k sum_{j=0}^{p-k} C(p-k, j) α^{p-k-j} ( sum_{m=0}^{(j-1)//2} C(j, 2m) (2m-1)!! He(j-2m-1, -α) )
        P_2(α) = (p)⤋k sum_{j=0}^{p-k} C(p-k, j) α^{p-k-j} ( 1_{j even} (j-1)!! )

    Derivation sketch:
    Rewrite as E[(p)_k (μ + σX)^{p-k} 1[X > -α]] for X ~ N(0, 1).
    The outer summation comes from the binomial expansion of (μ + σX)^{p-k}.
    The inner parts compute E[X^j 1[X > -α]] for X ~ N(0, 1) using the Hermite expansion of X^j and integration by parts.
    """
    alpha = Polynomial([0, 1])
    binom = lambda inner: math.prod(range(p - k + 1, p + 1)) * sum(
        math.comb(p - k, j) * alpha ** (p - k - j) * inner(j) for j in range(p - k + 1)
    )
    inner_1 = lambda j: sum(
        math.comb(j, 2 * m)
        * math.prod(range(1, 2 * m, 2))
        * He_poly(j - 2 * m - 1)
        * (-1) ** (j - 1)
        for m in range((j - 1) // 2 + 1)
    )
    inner_2 = lambda j: math.prod(range(1, j, 2)) if j % 2 == 0 else Polynomial([0])
    return binom(inner_1), binom(inner_2)


def relu_wick_coef(mean: Float[Tensor, "n"], var: Float[Tensor, "n"], k: int, p: int = 1) -> Float[Tensor, "n"]:
    """
    Computes E[∂^k ReLU(Z)^p] for Z ~ N(mean, var).
    Let σ = sqrt(var) and α = mean / σ, with φ the standard normal pdf and Φ the cdf.

    Formulas for p=1:
      k = 0: E[ReLU(Z)] = σ φ(α) + mean Φ(α).
      k = 1: E[H(Z)] = Φ(α).
      k ≥ 2: E[∂^k ReLU(Z)] = (-1)^{k-2} σ^{-(k-1)} He_{k-2}(α) φ(α),
             where He_n is the probabilists’ Hermite polynomial.

    For k >= p, ∂^k ReLU(Z)^p = p! ∂^{k-p} ReLU(Z), so we can reduce to the above.
    For k < p, see _relu_wick_poly.
    """
    mean, var = torch.as_tensor(mean), torch.as_tensor(var)
    if (var < 1e-10).any():
        logger.warning("Snapping negative variance to zero")
        var = torch.clamp(var, min=1e-10)
    sigma = var.sqrt()
    alpha = mean / sigma
    if k < p:
        P_1, P_2 = _relu_wick_poly(p, k)
        return sigma ** (p - k) * (
            eval_poly(P_1, alpha) * norm_pdf(alpha) + eval_poly(P_2, alpha) * norm_cdf(alpha)
        )
    elif p > 1:
        # Reduce to p=1 case by differentiating (p-1) times
        return math.factorial(p) * relu_wick_coef(mean, var, k - p + 1, 1)
    else:
        if k == 0:
            return sigma * norm_pdf(alpha) + mean * norm_cdf(alpha)
        elif k == 1:
            return norm_cdf(alpha)
        else:
            return (-1) ** (k - 2) * sigma ** (-(k - 1)) * He(k - 2, alpha) * norm_pdf(alpha)

def sgn_wick_coef(mean: Float[Tensor, 'n'], var: Float[Tensor, 'n'], k: int, p: int = 1) -> float:
    """
    Computes E[∂^k sgn(Z)^p] for Z ~ N(mean, var).
    For odd p, sgn^p = sgn. For even p, sgn^p = 1 (a.e.).
    """
    if p % 2 == 0:
        # sgn(z)^{2m} = 1 a.e., so E[∂^k 1] = 1 if k=0, else 0.
        return torch.as_tensor(1.0 if k == 0 else 0.0)
    # Use that sgn = 2*∂ReLU - 1
    if k == 0:
        return 2 * relu_wick_coef(mean, var, 1) - 1
    else:
        return 2 * relu_wick_coef(mean, var, k + 1)

def heaviside_wick_coef(mean: Float[Tensor, 'n'], var: Float[Tensor, 'n'], k: int, p: int = 1) -> float:
    """
    Computes E[∂^k 1[Z>0]^p] for Z ~ N(mean, var).
    Of course, output does not depend on p since 1^p=1.
    """
    # Use that 1[Z>0] = ∂ReLU
    return relu_wick_coef(mean, var, k + 1)

def poly_wick_coef(
    poly: Polynomial, mean: Float[Tensor, "n"], var: Float[Tensor, "n"], k: int, p: int = 1
) -> float:
    """
    Returns E[∂^k poly(Z)^p] for Z ~ N(mean, var) and poly a numpy Polynomial.
    """
    mean = torch.as_tensor(mean)
    var = torch.as_tensor(var)
    dk_p = (poly**p).deriv(k)
    sigma = var.sqrt()
    alpha = mean / sigma
    return sum(
        c * eval_poly(_gauss_moment_poly(i), alpha) * sigma**i for i, c in enumerate(dk_p.coef)
    )

def hermgauss_wick_coef(
    f: Callable,
    mean: Float[Tensor, 'n'],
    var: Float[Tensor, 'n'],
    k: int,
    p: int = 1,
    deg: int = 100,
    float64: bool = False,
) -> Float[Tensor, 'n']:
    """
    Computes E[∂^k (f(Z))^p] for Z ~ N(mean, var) using Gauss-Hermite quadrature.
    Uses E[∂^k g(Z)] = σ^{-k} E[He_k((Z-μ)/σ) g(Z)].
    If float64=True, cast to float64 for the computation, then cast back.
    """
    mean, var = torch.as_tensor(mean), torch.as_tensor(var)
    orig_dtype = mean.dtype
    if float64:
        mean, var = mean.double(), var.double()
    sigma = var.sqrt()

    nodes, weights = hermgauss(deg)
    nodes = torch.as_tensor(nodes, dtype=mean.dtype, device=mean.device)
    weights = torch.as_tensor(weights, dtype=mean.dtype, device=mean.device)

    # z_i = μ + σ√2 t_i, shape: (n, deg)
    z = mean.unsqueeze(-1) + sigma.unsqueeze(-1) * math.sqrt(2) * nodes
    he_vals = He(k, math.sqrt(2) * nodes)  # (deg,)

    # E[∂^k (f(Z))^p] ≈ (1/√π) σ^{-k} Σ_i w_i He_k(√2 t_i) f(μ + σ√2 t_i)^p
    result = (sigma ** (-k) / math.sqrt(math.pi)) * (f(z) ** p * he_vals * weights).sum(-1)
    if float64:
        result = result.to(orig_dtype)
    return result


sigmoid_wick_coef = partial(hermgauss_wick_coef, torch.sigmoid)
gelu_wick_coef = partial(hermgauss_wick_coef, torch.nn.functional.gelu)
tanh_wick_coef = partial(hermgauss_wick_coef, torch.tanh)


WICK_COEF_D = {
    "relu": relu_wick_coef,
    "heaviside": heaviside_wick_coef,
    "sgn": sgn_wick_coef,
    "square": partial(poly_wick_coef, Polynomial([0, 0, 1])),
    "cube": partial(poly_wick_coef, Polynomial([0, 0, 0, 1])),
    "sigmoid": sigmoid_wick_coef,
    "gelu": gelu_wick_coef,
    "tanh": tanh_wick_coef,
}
