import einops
import pytest
import torch
from functools import partial
from itertools import product

from mlp_kprop.harmonic import *
from mlp_kprop.harmonic import (
    _lap_m_prod_einexpr,
    _lap_m_prod,
    _lap_m_dslice,
)
from mlp_kprop.cumulants import symmetrize
from mlp_kprop.diagslice import _einsum_delta, zero_repeated, _diagslice_view, diagslice
from mlp_kprop.partitions import *

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

def test_proj_coef_2d():
    n = 10
    A = symmetrize(torch.randn(n, n))

    P0 = proj_coef(n, 2, 0)
    P1 = proj_coef(n, 2, 1)
    R = partial(rad, n=n, strict=True)
    L = partial(lap, strict=True)
    def eval_proj(P, A):
        return sum(
            compose([R] * j + [L] * j)(A) * coeff
            for j, coeff in enumerate(P)
        )
    H0 = eval_proj(P0, A)
    H1 = eval_proj(P1, A)
    assert torch.allclose(H0 + H1, A)
    assert torch.allclose(H1, A.diag().mean() * torch.eye(n))

def test_proj_coef():
    n = 10

    R = partial(rad, n=n, strict=True)
    L = partial(lap, strict=True)

    def eval_proj(P, A):
        return sum(
            compose([R] * j + [L] * j)(A) * coeff
            for j, coeff in enumerate(P)
        )

    for d in range(2, 6):
        H = {
            r: symmetrize(torch.randn(([n] * (d - 2 * r))))
            for r in range(d // 2 + 1)
        }
        for r in H:
            H[r] = zero_repeated(H[r])
            assert lap(H[r], strict=True).abs().max() < 1e-10
        A = sum(
            compose([R] * r)(H[r])
            for r in H
        )
        assert is_symmetric(A)
        for r in H:
            P = proj_coef(n, d, r)
            H_r = eval_proj(P, A)
            assert torch.allclose(H_r, compose([R] * r)(H[r]))

def test_proj_geq_r():
    n = 8
    R = partial(rad, n=n, strict=True)
    L = partial(lap, strict=True)
    for d in range(2, 6):
        H = {
            r: symmetrize(torch.randn(([n] * (d - 2 * r))))
            for r in range(d // 2 + 1)
        }
        for r in H:
            H[r] = zero_repeated(H[r])
            assert lap(H[r], strict=True).abs().max() < 1e-10
        A = sum(
            compose([R] * r)(H[r])
            for r in H
        )
        assert is_symmetric(A)
        for r in H:
            expected = sum(
                compose([R] * r2)(H[r2])
                for r2 in range(r, d // 2 + 1)
            )
            assert is_symmetric(expected)
            computed = proj_geq_r(A, n=n, r_out=r, strict=True).to_tensor(strict=True)
            assert torch.allclose(expected, computed)
            
def test_lap_m_prod_einexpr():
    # graph, aritys, expected
    tests = [
        ([((0, 1), 1)], [2, 2], 'i0_0 i0_1, i0_0 i1_1 -> i0_1 i1_1'),
        ([((0, 1), 2)], [2, 2], 'i0_0 i0_1, i0_0 i0_1 -> '),
        ([((0, 0), 1)], [2, 2], 'i0_0 i0_0, i1_0 i1_1 -> i1_0 i1_1'),
        ([((0, 0), 1), ((1, 1), 1)], [2, 2], 'i0_0 i0_0, i1_0 i1_0 -> '),
        ([((0, 0), 10)], [20], ' '.join([f'i0_{i} i0_{i}' for i in range(0, 20, 2)]) + ' -> '),
    ]
    for graph, aritys, expected in tests:
        einexpr = _lap_m_prod_einexpr(graph, aritys)
        assert einexpr == expected

def test_lap_m_prod():
    n = 5
    L = partial(lap, strict=True)
    # aritys, m
    tests = [
        ((1,), 0),
        ((2,), 0),
        ((1,), 1),
        ((2,), 2),
        ((1,), 3),
        ((2,), 3),
        ((2, 2), 0),
        ((2, 2), 1),
        ((2, 3), 0),
        ((2, 3), 1),
        ((3, 3), 0),
        ((3, 3), 1),
        ((1, 1, 1), 2),
        ((2, 3, 2), 1),
        ((3, 3, 3), 1),
    ]
    for aritys, m in tests:
        v = len(aritys)
        As = [symmetrize(torch.randn([n] * arity)) for arity in aritys]
        legs = [
            [
                f"i{i}_{j}"
                for j in range(arity)
            ]
            for i, arity in enumerate(aritys)
        ]
        out_legs = [legs[i][j] for i in range(v) for j in range(aritys[i])]
        in_expr = ', '.join(
            ' '.join(legs[i]) for i in range(v)
        )
        out_expr = ' '.join(out_legs)
        prod_A = symmetrize(einops.einsum(
            *As, f'{in_expr} -> {out_expr}'
        ))
        expected = compose([L] * m)(prod_A)
        computed = _lap_m_prod(m, As, strict=True)
        assert torch.allclose(expected, computed)

def test_contract_W_proj_r0():
    n = 5

    R = partial(rad, n=n, strict=True)
    L = partial(lap, strict=True)

    W = torch.randn(n, n)

    for s_in, r_in in product(range(0, 4), range(0, 3)):
        A = symmetrize(torch.randn([n] * s_in))
        RA = compose([R] * r_in)(A)
        computed = contract_W_proj(
            HTensor(A, r=r_in, n=n, strict=True),
            W,
            r_out=0,
            strict=True,
        ).core
        expected = contract_W_basic(RA, W)
        assert torch.allclose(expected, computed)


def test_htensor_metric_to_tensor():
    n = 6
    A = symmetrize(torch.randn(n, n))
    metric_diag = torch.rand(n)
    metric_full = symmetrize(torch.randn(n, n))

    H_diag = HTensor(A, r=1, n=n, metric=metric_diag, strict=True)
    H_full = HTensor(A, r=1, n=n, metric=metric_full, strict=True)

    expected_diag = rad(A, metric=metric_diag, strict=True)
    expected_full = rad(A, metric=metric_full, strict=True)

    assert torch.allclose(H_diag.to_tensor(strict=True), expected_diag)
    assert torch.allclose(H_full.to_tensor(strict=True), expected_full)


def test_contract_W_metric():
    n_in, n_out = 5, 4
    W = torch.randn(n_out, n_in)
    A = symmetrize(torch.randn(n_in, n_in))

    metric_full = symmetrize(torch.randn(n_in, n_in))
    H_full = HTensor(A, r=1, n=n_in, metric=metric_full, strict=True)
    WH_full = contract_W(H_full, W)
    assert torch.allclose(WH_full.core, contract_W_basic(A, W))
    assert torch.allclose(WH_full.metric, W @ metric_full @ W.T)

    metric_diag = torch.rand(n_in)
    H_diag = HTensor(A, r=1, n=n_in, metric=metric_diag, strict=True)
    WH_diag = contract_W(H_diag, W)
    assert torch.allclose(WH_diag.metric, W @ torch.diag(metric_diag) @ W.T)


def test_contract_W_set_metric_requires_identity():
    n_in, n_out = 5, 4
    W = torch.randn(n_out, n_in)
    A = symmetrize(torch.randn(n_in, n_in))
    W_var = torch.rand(n_out)

    H = HTensor(A, r=1, n=n_in, strict=True)
    WH = contract_W(H, W, set_metric=W_var)
    assert torch.allclose(WH.metric, W_var)

    H_non_id = HTensor(A, r=1, n=n_in, metric=torch.rand(n_in), strict=True)
    with pytest.raises(NotImplementedError):
        contract_W(H_non_id, W, set_metric=W_var)

def test_contract_W_proj_orth():
    n = 8

    W = torch.randn(n, n)
    W = torch.svd(W).U
    
    for s_in, r_in in product(range(0, 4), range(3)):
        H = symmetrize(torch.randn([n] * s_in))
        H = zero_repeated(H)
        for r_out in range(r_in + 1, (s_in + 2 * r_in) // 2 + 1):
            computed = contract_W_proj(
                HTensor(H, r=r_in, n=n, strict=True),
                W,
                r_out=r_out,
                strict=True,
            ).core
            expected = torch.zeros_like(computed)
            assert torch.allclose(expected, computed)
        WH = contract_W_basic(H, W)
        computed = contract_W_proj(
            HTensor(H, r=r_in, n=n, strict=True),
            W,
            r_out=r_in,
            strict=True,
        ).core
        assert torch.allclose(WH, computed)


def test_contract_W_proj_rejects_non_identity_metric():
    n = 5
    W = torch.randn(n, n)
    A = symmetrize(torch.randn(n, n))
    H = HTensor(A, r=1, n=n, metric=torch.rand(n), strict=True)
    with pytest.raises(NotImplementedError):
        contract_W_proj(H, W, r_out=0, strict=True)
            
def test_lap_m_dslice():
    parts = [
        (1, 1, 1),
        (2, 1),
        (3,),
        (2, 2),
        (3, 1),
        (4,),
    ]
    n = 8
    max_m = 3
    L = partial(lap, strict=True)
    for part, m in product(parts, range(max_m + 1)):
        dslice = torch.randn([n] * len(part))
        dslice = zero_repeated(dslice)

        expanded = torch.zeros([n] * sum(part))
        _diagslice_view(
            expanded, int_to_canonical_set_partition(part)
        ).add_(
            int_partition_coef(part) * dslice
        )
        expanded = symmetrize(expanded)

        expected = compose([L] * m)(expanded)
        computed = _lap_m_dslice(m, dslice, part)
        assert torch.allclose(expected, computed)

def test_DS_harmonic_proj():
    n = 8
    d_max = 4
    for d in range(1, d_max + 1):
        A = symmetrize(torch.randn([n] * d))
        dsA = DSTensor.from_tensor(A)
        for r_out in range((d // 2) + 1):
            computed = DS_harmonic_proj(dsA, r_out=r_out, strict=True).core
            expected = contract_W_proj(
                HTensor(A, r=0, n=n, strict=True),
                W=torch.eye(n),
                r_out=r_out,
                strict=True,
            ).core
            assert torch.allclose(expected, computed)

def test_diagslice():
    n = 8
    R = partial(rad, n=n, strict=True)
    parts = [
        (1,),
        (1, 1),
        (1, 1, 1),
        (1, 2, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (3, 1),
        (1, 3),
        (4,),
    ]
    for part in parts:
        d = sum(part)
        for r in range(0, d // 2 + 1):
            s = d - 2 * r
            A = symmetrize(torch.randn([n] * s))
            RA = compose([R] * r)(A)
            expected = zero_repeated(diagslice(RA, part))
            computed = harmonic_diagslice(HTensor(A, r=r, n=n, strict=True), part)
            assert torch.allclose(expected, computed)

    # Test with metric_diag
    W = torch.randn(n, n)
    metric = W @ W.T
    for part in parts:
        d = sum(part)
        for r in range(0, d // 2 + 1):
            s = d - 2 * r
            A = symmetrize(torch.randn([n] * s))
            RA = compose([R] * r)(A)
            WA = contract_W_basic(A, W)
            expected = zero_repeated(diagslice(contract_W_basic(RA, W), part))
            computed = harmonic_diagslice(HTensor(WA, r=r, n=n, metric=metric, strict=True), part)
            assert torch.allclose(expected, computed)


def test_htensor_get_dslice_cache_invalidates():
    n = 6
    part = (2,)
    core = symmetrize(torch.randn(n, n))
    H = HTensor(core, r=0, n=n, strict=True)

    before = H.get_dslice(part).clone()
    assert len(H.repeated.slices) == 1

    H.core += 1

    after = H.get_dslice(part)
    expected = harmonic_diagslice(H, part)

    assert torch.allclose(after, expected)
    assert not torch.allclose(before, after)

def test_upslice():
    n = 8
    A = symmetrize(torch.randn(n, n))
    hA = HTensor(A, r=1, n=n, strict=True)
    parts = [
        (1, 1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (3, 1),
        (1, 3),
        (2, 2),
        (4,)
    ]
    for part in parts:
        expected = diagslice(hA.to_tensor(), part)
        computed = diagslice(hA, part)
        assert torch.allclose(computed, expected)
