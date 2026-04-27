import pytest
import torch
from mlp_kprop.factor_k4 import *
from mlp_kprop.tensor_utils import *
from mlp_kprop.diagslice import zero_repeated, diagslice
from mlp_kprop.kprop_harmonic import Kind, coerce_input, linear_kprop, nonlin_kprop

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

def test_dslice():
    parts = [
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2),
        (3, 1),
        (1, 3),
        (4,),
    ]
    n = 8
    r = 4

    for part in parts:
        factors = tuple(torch.randn(n, n, r) for _ in range(2))
        FT = FactoredTensor4(n, factors)
        computed = FT.get_dslice(part)
        expected = zero_repeated(diagslice(FT.to_tensor(), part))
        assert torch.allclose(computed, expected)

def test_contract_W():
    n = 8
    r = 4
    factors = tuple(torch.randn(n, n, r) for _ in range(2))
    FT = FactoredTensor4(n, factors)
    W = torch.randn(n, n)
    FT_W = FT.contract_W(W)
    T = FT.to_tensor()
    T_W = contract_W_basic(T, W)
    assert torch.allclose(FT_W.to_tensor(), T_W)

def test_from_dstensor():
    n = 10
    A = symmetrize(torch.randn(n, n, n, n))
    dsA = DSTensor.from_tensor(A)
    dsA.slices.pop((1, 1, 1, 1))
    fA = FactoredTensor4.from_dstensor(dsA)
    assert torch.allclose(dsA.to_tensor(), fA.to_tensor())

@pytest.mark.parametrize(
    "kind,use_avg_metric,use_pK",
    [
        (kind, use_avg_metric, use_pK)
        for kind, use_avg_metric in product([Kind.SIMPLE, Kind.AUGMENT, Kind.BASE], [True, False])
        for use_pK in ([True, False] if kind == Kind.BASE else [True])
    ]
)
def test_factored_kprop(kind, use_avg_metric, use_pK):
    # TODO: Results seem to differ if the variance estimate ever goes negative (small n, large depth)
    # Not a priority to fix since in that case the output is probably garbage anyway
    n = 8
    depth = 3
    K = {1: torch.zeros(n), 2: torch.eye(n)}
    K = coerce_input(K, k_max=4)
    KF = K
    for l in range(depth):
        W = torch.randn(n, n) * math.sqrt(2 / n)
        WK = linear_kprop(K, W, k_max=4, set_metric=2. * torch.ones(n) if use_avg_metric else None)
        K = nonlin_kprop(WK, nonlin_wick_coef=relu_wick_coef, k_max=4, kind=kind, use_pK=use_pK)
        WKF = linear_kprop(KF, W, k_max=4, set_metric=2. * torch.ones(n) if use_avg_metric else None)
        KF = factored_nonlin_kprop_k4(
            K_in=WKF,
            nonlin_wick_coef=relu_wick_coef,
            augment=(kind==Kind.AUGMENT),
            base=(kind==Kind.BASE),
            use_pK=use_pK,
        )
        for d in K.keys():
            assert torch.allclose(
                K[d].to_tensor(),
                KF[d].to_tensor(),
                atol=1e-5
            )

def test_repeated_wick():
    n, r = 8, 4
    wick = torch.randn(n)
    A, B = tuple((torch.randn(n, n, r) for _ in range(2)))
    FT = FactoredTensor4(n, (A, B))
    repeated = FT.get_repeated()
    print('1', FT.repeated.slices.keys())
    expected = einops.einsum(
        FT.to_tensor(), wick, wick, wick, wick,
        'i j k l, i, j, k, l -> i j k l'
    )
    print('2', FT.repeated.slices.keys())
    FT.contract_wick_(wick)
    assert torch.allclose(FT.to_tensor(), expected)
    for part in repeated.slices.keys():
        computed_slice = FT.get_dslice(part)
        expected_slice = zero_repeated(diagslice(expected, part))
        assert torch.allclose(computed_slice, expected_slice)
