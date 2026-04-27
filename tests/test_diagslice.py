from itertools import product

import einops
import pytest
import torch

from mlp_kprop.cumulants import symmetrize
from mlp_kprop.diagslice import *
from mlp_kprop.diagslice import _diagslice_view, _einsum_delta, _merge_legs

torch.set_default_dtype(torch.float64)


# sp and sps are artifacts from when SetPartition was a different type, used for testing only.
# TODO: clean up
def sp(part):
    return tuple(frozenset(block) for block in part)


def sps(*parts):
    return tuple(sp(part) for part in parts)


def test_diagslice_view():
    n = 5
    A = torch.randn(n)
    assert torch.allclose(_diagslice_view(A, sp(((0,),))), A)
    A = torch.randn(n, n)
    assert torch.allclose(_diagslice_view(A, sp(((0, 1),))), A.diag())
    assert torch.allclose(_diagslice_view(A, sp(((0,), (1,)))), A)
    A = torch.randn(n, n, n)
    assert torch.allclose(
        _diagslice_view(A, sp(((0, 1, 2),))), A.diagonal(0, 0, 1).diagonal(0, 0, 1)
    )
    assert torch.allclose(_diagslice_view(A, sp(((0, 1), (2,)))), A.diagonal(0, 0, 1).T)
    assert torch.allclose(_diagslice_view(A, sp(((2,), (0, 1)))), A.diagonal(0, 0, 1))
    assert torch.allclose(_diagslice_view(A, sp(((0, 2), (1,)))), A.diagonal(0, 0, 2).T)
    assert torch.allclose(_diagslice_view(A, sp(((1,), (0, 2)))), A.diagonal(0, 0, 2))
    assert torch.allclose(_diagslice_view(A, sp(((1, 2), (0,)))), A.diagonal(0, 1, 2).T)
    assert torch.allclose(_diagslice_view(A, sp(((0,), (1, 2)))), A.diagonal(0, 1, 2))

    # Check that _diagslice_view is a view
    A = torch.eye(5)
    B = _diagslice_view(A, sp(((0, 1),)))
    B[:] = 2.0
    assert torch.allclose(A, torch.eye(5) * 2.0)


def test_diagslice():
    n = 5
    A = torch.randn(n)
    assert torch.allclose(diagslice(A, (1,)), A)
    X = torch.randn(n, n)
    A = X.T @ X  # symmetric 2-tensor
    assert torch.allclose(diagslice(A, (2,)), A.diag())
    assert torch.allclose(diagslice(A, (1, 1)), A)
    A = einops.einsum(A, A, A, "i a, i b, i c -> a b c")  # symmetric 3-tensor
    assert torch.allclose(diagslice(A, (3,)), A.diagonal(0, 0, 1).diagonal(0, 0, 1))
    assert torch.allclose(diagslice(A, (2, 1)), A.diagonal(0, 0, 1).T)
    assert torch.allclose(diagslice(A, (1, 2)), A.diagonal(0, 0, 1))
    assert torch.allclose(diagslice(A, (1, 1, 1)), A)


def test_to_from():
    A = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
            [[2.0, 4.0, 5.0], [4.0, 7.0, 8.0], [5.0, 8.0, 9.0]],
            [[3.0, 5.0, 6.0], [5.0, 8.0, 9.0], [6.0, 9.0, 10.0]],
        ]
    )
    assert is_symmetric(A)
    dst = DSTensor.from_tensor(A)
    expected = {
        (3,): torch.tensor([1.0, 7.0, 10.0]),
        (2, 1): torch.tensor([[0.0, 2.0, 3.0], [4.0, 0.0, 8.0], [6.0, 9.0, 0.0]]),
        (1, 1, 1): torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 5.0, 0.0]],
                [[0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
    }
    assert dst.slices.keys() == expected.keys()
    for part in dst.slices:
        assert torch.allclose(dst.slices[part], expected[part])
    assert torch.allclose(dst.to_tensor(), A)
    n = 8
    X = torch.randn(n, n)
    for d in range(2, 6):
        # Create a random symmetric d-tensor
        A = einops.einsum(
            *(X,) * d,
            ", ".join(f"i j{idx}" for idx in range(d))
            + " -> "
            + " ".join(f"j{idx}" for idx in range(d)),
        )
        assert A.shape == (n,) * d
        assert is_symmetric(A)
        assert torch.allclose(DSTensor.from_tensor(A).to_tensor(), A)


def test_einsum_delta():
    n = 5

    A = torch.randn(n, n)
    B = torch.randn(n, n)
    ret = _einsum_delta(A, B, "a b, b c -> a c")
    exp = A @ B
    assert torch.allclose(ret, exp)

    ret = _einsum_delta(A, B, "a b, b c -> a a")
    exp = torch.diagflat((A @ B).sum(dim=1))
    assert torch.allclose(ret, exp)

    ret = _einsum_delta(A, B, "a b, b c -> a c c")
    exp = torch.zeros((n, n, n))
    for a, b, c in product(range(n), repeat=3):
        exp[a, c, c] += A[a, b] * B[b, c]
    assert torch.allclose(ret, exp)

    ret = _einsum_delta(A, B, "a b, c d -> a a b b b c d d")
    exp = torch.zeros((n,) * 8)
    for a, b, c, d in product(range(n), repeat=4):
        exp[a, a, b, b, b, c, d, d] = A[a, b] * B[c, d]
    assert torch.allclose(ret, exp)

    A = torch.randn(n)
    for d in range(2, 6):
        ret = _einsum_delta(A, "a -> " + " ".join(("a",) * d))
        exp = torch.zeros((n,) * d)
        for i in range(n):
            exp[(i,) * d] = A[i]
        assert torch.allclose(ret, exp)


def test_merge_legs():
    # (einexpr, parts, expected)
    tests = [
        (
            "a b c -> a b c",
            sps(((0,), (1,), (2,)), ((0,), (1,), (2,))),
            "t0b0 t0b1 t0b2 -> t0b0 t0b1 t0b2",
        ),
        ("a b c -> a b c", sps(((0, 1), (2,)), ((0, 1), (2,))), "t0b0 t0b1 -> t0b0 t0b1"),
        ("a b c -> a b c", sps(((0, 1), (2,)), ((0,), (1, 2))), "t0b0 t0b0 -> t0b0 t0b0"),
        ("a b c -> a b c", sps(((0, 1, 2),), ((0,), (1,), (2,))), "t0b0 -> t0b0 t0b0 t0b0"),
        (
            "a b, b c -> a c",
            sps(((0,), (1,)), ((0,), (1,)), ((0,), (1,))),
            "t0b0 t0b1, t0b1 t1b1 -> t0b0 t1b1",
        ),
        (
            "a b, b c -> a c",
            sps(((0, 1),), ((0,), (1,)), ((0,), (1,))),
            "t0b0, t0b0 t1b1 -> t0b0 t1b1",
        ),
        ("a b, b c -> a c", sps(((0, 1),), ((0, 1),), ((0,), (1,))), "t0b0, t0b0 -> t0b0 t0b0"),
        # KX_{(0, 1), (2, 3)} -> KZ_{(0, 1), (2, 3)}
        (
            "a b c d, a i, b j, c k, d l -> i j k l",
            sps(
                ((0, 1), (2, 3)),
                ((0,), (1,)),
                ((0,), (1,)),
                ((0,), (1,)),
                ((0,), (1,)),
                ((0, 1), (2, 3)),
            ),
            "t0b0 t0b1, t0b0 t1b1, t0b0 t1b1, t0b1 t3b1, t0b1 t3b1 -> t1b1 t3b1",
        ),
        # KX_{(0, 1), (2, 3)} -> KZ_{(0, 2), (1, 3)}
        (
            "a b c d, a i, b j, c k, d l -> i j k l",
            sps(
                ((0, 1), (2, 3)),
                ((0,), (1,)),
                ((0,), (1,)),
                ((0,), (1,)),
                ((0,), (1,)),
                ((0, 2), (1, 3)),
            ),
            "t0b0 t0b1, t0b0 t1b1, t0b0 t2b1, t0b1 t1b1, t0b1 t2b1 -> t1b1 t2b1",
        ),
    ]

    for einexpr, parts, expected in tests:
        merged = _merge_legs(einexpr, parts)
        assert merged == expected, (
            f"Failed for {einexpr} with parts {parts}: got {merged}, expected {expected}"
        )


def test_diagslice_einsum():
    n = 5
    A, B, C = [torch.randn(n, n) for _ in range(3)]
    A, B, C = A.T + A, B.T + B, C.T + C  # make symmetric
    X, Y = torch.randn(n, n), torch.randn(n, n)
    D = einops.einsum(X, X, X, "i a, i b, i c -> a b c")
    E = einops.einsum(Y, Y, Y, "i a, i b, i c -> a b c")
    # NOTE: output needs to be symmetric; otherwise there is no guarantee of correctness
    exprs = [
        ((A, A, A), "a b, b c, c d -> a d"),
        ((A, B, C), "a b, a b, a b -> a b"),
        ((A, A, A), "a b, a c, a d -> b c d"),
        ((D, A, A, A), "a b c, a d, b e, c f -> d e f"),
        ((D, E), "a b c, a b c -> a b c"),
    ]
    for inputs, expr in exprs:
        ds_inputs = [DSTensor.from_tensor(X) for X in inputs]
        expected = einops.einsum(*inputs, expr)
        for x_inputs in product(*zip(inputs, ds_inputs)):
            ret = DSTensor.einsum(*x_inputs, expr)
            assert torch.allclose(ret.to_tensor(), expected), (
                f"Failed for expr {expr} with types {[type(x) for x in x_inputs]}"
            )

    # Mimic kprop.linear_kprop
    m = 6
    W = torch.randn(n, m)
    for d in range(1, 6):
        K = torch.randn(*(n,) * d)
        # symmetrize
        K = sum(K.permute(*p) for p in permutations(range(d))) / math.factorial(d)
        expr = " ".join(f"i{idx}" for idx in range(d))
        expr += ", " + ", ".join(f"i{idx} j{idx}" for idx in range(d))
        expr += " -> " + " ".join(f"j{idx}" for idx in range(d))
        tensors = (K,) + (W,) * d
        expected = einops.einsum(*tensors, expr)
        ds_tensors = (DSTensor.from_tensor(K),) + (W,) * d
        for in_symmetric in [False, True]:
            ret = DSTensor.einsum(*ds_tensors, expr, in_symmetric=in_symmetric)
            assert torch.allclose(ret.to_tensor(), expected), (
                f"Failed for linearity step with d={d} and in_symmetric={in_symmetric}"
            )


def test_get_slice():
    n, d = 5, 6
    A = symmetrize(torch.randn(*(n,) * d))
    dsA = DSTensor.from_tensor(A)
    A321 = einops.einsum(A, "a a a b b c -> a b c")
    A321 = zero_repeated(A321)
    assert torch.allclose(dsA.get_slice((3, 2, 1)), A321)
    assert torch.allclose(dsA.get_slice((2, 3, 1)), A321.permute(1, 0, 2))
    assert torch.allclose(dsA.get_slice((2, 1, 3)), A321.permute(1, 2, 0))
    assert torch.allclose(dsA.get_slice((1, 3, 2)), A321.permute(2, 0, 1))
    dsA = DSTensor.from_tensor(A, part_cond=lambda part: min(part) > 1)
    assert torch.allclose(dsA.get_slice((3, 2, 1), strict=False), torch.zeros_like(A321))
    with pytest.raises(AssertionError):
        dsA.get_slice((3, 2, 1))
    with pytest.raises(AssertionError):
        dsA.get_slice((2, 1, 1))
    with pytest.raises(AssertionError):
        dsA.get_slice((5, -5, 6))


def test_ops():
    n, d = 5, 6
    A, B = [symmetrize(torch.randn(*(n,) * d)) for _ in range(2)]
    assert torch.allclose(A + B, (DSTensor.from_tensor(A) + DSTensor.from_tensor(B)).to_tensor())
    assert torch.allclose(A * B, (DSTensor.from_tensor(A) * DSTensor.from_tensor(B)).to_tensor())
    assert torch.allclose(A / B, (DSTensor.from_tensor(A) / DSTensor.from_tensor(B)).to_tensor())
    assert torch.allclose(A - B, (DSTensor.from_tensor(A) - DSTensor.from_tensor(B)).to_tensor())
    assert torch.allclose(A + 2, (DSTensor.from_tensor(A) + 2).to_tensor())
    assert torch.allclose(A / 2, (DSTensor.from_tensor(A) / 2).to_tensor())
    dsA = DSTensor.from_tensor(A)
    dsA *= B
    assert torch.allclose(A * B, dsA.to_tensor())


def test_einsumcond1():
    # ((aritys, is_dstensor, out_symmetric), expected or len(expected))
    tests = [
        (
            ((2, 2), (True,), True),
            [
                sps(((0,), (1,)), ((0,), (1,))),
                sps(((0,), (1,)), ((0, 1),)),
                sps(((0, 1),), ((0,), (1,))),
                sps(((0, 1),), ((0, 1),)),
            ],
        ),
        (
            ((2, 2), (True,), False),
            [
                sps(((0,), (1,)), ((0,), (1,))),
                sps(((0,), (1,)), ((0, 1),)),
                sps(((0, 1),), ((0,), (1,))),
                sps(((0, 1),), ((0, 1),)),
            ],
        ),
        (
            ((2, 2), (False,), False),
            [sps(((0,), (1,)), ((0,), (1,))), sps(((0,), (1,)), ((0, 1),))],
        ),
        (((3, 3, 3), (True, True), False), 5 * 5 * 5),
        (((3, 3, 3), (True, True), True), 5 * 5 * 3),
        (((3, 4, 3), (True, False), True), 5 * 1 * 3),
        (((3, 4, 3), (False, True), True), 15 * 1 * 3),
    ]
    for (aritys, is_dstensor, out_symmetric), expected in tests:
        input_iparts = [
            int_partitions(aritys[i]) if b else ((1,) * aritys[i],)
            for i, b in enumerate(is_dstensor)
        ]
        arg_conds = [IntPartCond(parts=set(iparts)) for iparts in input_iparts]
        arg_conds.append(trivial_int_cond)
        ein_cond = EinsumCond(einsum_cond=lambda parts: True)
        ret = ein_cond.get_parts(
            aritys=tuple(aritys), arg_conds=tuple(arg_conds), out_symmetric=out_symmetric
        )
        ret = [t[0] for t in ret]
        if isinstance(expected, int):
            assert len(ret) == expected
        else:
            assert sorted(ret) == sorted(expected)


def test_einsumcond2():
    import random

    n, d = 5, 6
    int_parts = random.sample(list(int_partitions(d)), 5)
    set_parts = [p for p in set_partitions(d) if set_to_int_partition(p) in int_parts]
    A = symmetrize(torch.randn(*(n,) * d))
    dsA = DSTensor.from_tensor(A)
    args = " ".join(f"i{i}" for i in range(d))
    expr = f"{args} -> {args}"

    ret = DSTensor.einsum(dsA, expr)
    assert set(ret.slices.keys()) == set(int_partitions(d))
    assert torch.allclose(ret.to_tensor(), A)

    ret = DSTensor.einsum(dsA, expr, out_cond=IntPartCond(parts=int_parts))
    assert set(ret.slices.keys()) == set(int_parts)

    ret = DSTensor.einsum(
        dsA, expr, ein_cond=EinsumCond(einsum_cond=lambda parts: parts[-1] in set_parts)
    )
    assert set(ret.slices.keys()) == set(int_parts)

    ret = DSTensor.einsum(
        dsA, expr, ein_cond=EinsumCond(einsum_cond=lambda parts: parts[0] in set_parts)
    )
    assert set(ret.slices.keys()) == set(int_partitions(d))
    for part in int_partitions(d):
        if part in int_parts:
            assert torch.allclose(ret.slices[part], dsA.slices[part])
        else:
            assert torch.allclose(ret.slices[part], torch.zeros_like(dsA.slices[part]))


def test_einsum_part_contributions():
    torch.manual_seed(0)
    n, m = 3, 2
    K = symmetrize(torch.randn(n, n))
    dsK = DSTensor.from_tensor(K)
    W = torch.randn(m, n)
    expr = "i0 i1, j0 i0, j1 i1 -> j0 j1"
    ret, contribs = DSTensor.einsum(dsK, W, W, expr, return_part_contributions=True)
    assert isinstance(contribs, dict)
    for out_part, part_map in contribs.items():
        assert out_part in ret.slices
        total = torch.zeros_like(ret.slices[out_part])
        for in_parts, contrib in part_map.items():
            assert isinstance(in_parts, tuple)
            for set_part in in_parts:
                assert isinstance(set_part, tuple)
                for block in set_part:
                    assert isinstance(block, frozenset)
            total = total + contrib
        assert torch.allclose(total, ret.slices[out_part])
    ret_no_tracking = DSTensor.einsum(dsK, W, W, expr)
    assert torch.allclose(ret_no_tracking.to_tensor(), ret.to_tensor())


def test_dstower_pointwise_and_utils():
    t1 = DSTower.from_tower(
        {1: torch.tensor([1.0, 2.0]), 2: torch.tensor([[2.0, 0.5], [0.5, 3.0]])}
    )
    t2 = DSTower.from_tower({1: torch.tensor([3.0, 4.0]), 3: torch.ones((2, 2, 2))})

    added = t1 + t2
    assert isinstance(added, DSTower)
    assert set(added.keys()) == {1, 2, 3}
    assert torch.allclose(added[1].to_tensor(), t1[1].to_tensor() + t2[1].to_tensor())
    assert torch.allclose(added[2].to_tensor(), t1[2].to_tensor())
    assert torch.allclose(added[3].to_tensor(), t2[3].to_tensor())

    scaled = t1 * 2
    assert torch.allclose(scaled[1].to_tensor(), 2 * t1[1].to_tensor())
    assert torch.allclose(scaled[2].to_tensor(), 2 * t1[2].to_tensor())

    negated = -t2
    assert torch.allclose(negated[1].to_tensor(), -t2[1].to_tensor())

    tower = DSTower.from_tower(
        {1: torch.tensor([1.0, 2.0]), 2: torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    )
    assert tower.is_downward_closed()

    slices = {(1,): torch.tensor([1.0, 2.0]), (2,): torch.tensor([3.0, 4.0])}
    rebuilt = DSTower.from_slices(slices, autozero=True)
    assert set(rebuilt.keys()) == {1, 2}
    assert torch.allclose(rebuilt[1].to_tensor(), torch.tensor([1.0, 2.0]))
    assert torch.allclose(rebuilt[2].to_tensor(), torch.tensor([[3.0, 0.0], [0.0, 4.0]]))

def test_upslice():
    n = 8
    A = symmetrize(torch.randn(n, n, n, n))
    ds_A = DSTensor.from_tensor(A)
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
        expected = diagslice(A, part)
        computed = diagslice(ds_A, part)
        assert torch.allclose(computed, expected)