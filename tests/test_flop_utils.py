import pytest
import torch
import einops

from mlp_kprop.flop_utils import *

def test_named_flop_counter():
    n = 8
    def matmul():
        with flop_name("matmul"):
            A = torch.randn(n, n)
            B = torch.randn(n, n)
            return torch.matmul(A, B)

    @flop_name("einsum")
    def einsum():
        A = torch.randn(n, n, n)
        B = torch.randn(n, n)
        return einops.einsum(A, B, B, B, 'i j k, a i, b j, c k -> a b c')

    @flop_name("add")
    def add():
        A = torch.randn(n, n)
        B = torch.randn(n, n)
        return A + B

    @flop_name("sum_all")
    def sum_all():
        A = torch.randn(n, n, n)
        return A.sum()
    
    @flop_name("sum_1")
    def sum_1():
        A = torch.randn(n, n, n)
        return A.sum(dim=1)
    
    @flop_name("sum_2")
    def sum_2():
        A = torch.randn(n, n, n)
        return A.sum(dim=(1, 2))
    
    @flop_name("norm_2")
    def norm_2():
        A = torch.randn(n, n, n)
        return A.norm(p=2, dim=(1, 2))
        
    with NamedFlopCounter() as counter:
        matmul()
        einsum()
        matmul()
        add()
        sum_all()
        sum_1()
        sum_2()
        norm_2()
    flops = counter.flop_dict()
    assert flops['matmul'] == 2 * 2 * n**3
    assert 2 * 3 * n**4 <= flops['einsum'] <= 2 * 3 * n**4  + 30  # einops.einsum has 30 warmup flops; which may or may not be present depending on test order
    assert flops['add'] == n**2
    assert flops['sum_all'] == n**3 - 1
    assert flops['sum_1'] == n**3 - n**2
    assert flops['sum_2'] == n**3 - n
    assert flops['norm_2'] == 2 * n ** 3
    assert flops['resid'] == 0
    assert counter.total() == sum(v for k, v in flops.items())

    def add():
        A = torch.randn(n, n)
        B = torch.randn(n, n)
        return A + B

    # Test guard against gaps
    with pytest.raises(RuntimeError):
        with NamedFlopCounter(strict=True) as counter:
            add()

    # No guard if not strict
    with NamedFlopCounter() as counter:
        add()

    # Nested regions are accounted correctly
    with NamedFlopCounter() as counter:
        with flop_name("outer"):
            add()
            with flop_name("inner"):
                add()
            add()
    flops = counter.flop_dict()
    assert flops['outer'] == 2 * n**2
    assert flops['inner'] == n**2
    assert flops['resid'] == 0
    assert counter.total() == 3 * n**2

def test_slice_factor():
    # For large enough n, the slice factors should be close to the n->infinity limit
    # which is prod_c 1/c!.
    n = 1_000
    atol = 1e-3
    tests = [
        ((1, 1, 1, 1), 1 / 24),
        ((1, 1, 1), 1 / 6),
        ((1, 1), 1 / 2),
        ((1,), 1.),
        ((2, 1), 1.),
        ((2, 2), 1 / 2),
        ((2, 2, 1, 1), 1 / 4),
        ((2, 2, 2, 1), 1 / 6),
        ((2, 2, 3, 1), 1 / 2),
    ]
    for part, expected in tests:
        assert np.allclose(slice_factor(part, n), expected, atol=atol)
        x = slice_factor(part, n) * n ** len(part)
        assert np.allclose(x, round(x))

    # Exact tests
    n = 16
    atol = 1e-8
    tests = [
        ((1, 1), (n+1) / (2 * n)),
        ((2, 1), 1.),
    ]
    for part, expected in tests:
        assert np.allclose(slice_factor(part, n), expected, atol=atol)

def test_contract_factor():
    n = 1_000
    atol = 1e-3
    tests = [
        (1, 1.),
        (2, 3 / 4),
        (3, 7 / 18),
        (4, 5 / 32),
    ]
    for d, expected in tests:
        assert np.allclose(contract_factor(d, n), expected, atol=atol)
        x = contract_factor(d, n) * n**d
        assert np.allclose(x, round(x))

def test_sym_flop_total():
    n = 8
    A = torch.randn(n, n, n)
    W = torch.randn(n, n)
    with NamedFlopCounter() as counter:
        with flop_name('contract_W_basic d=3', factor=contract_factor(3, n=8)):
            einops.einsum(A, W, W, W, 'i1 i2 i3, j1 i1, j2 i2, j3 i3 -> j1 j2 j3')
        with flop_name('nonlin_sum part=(1,1,1)', factor=slice_factor((1,1,1), n=8)):
            A + A
        with flop_name('nonlin_sum part=(2,1,1)', factor=slice_factor((2,1,1), n=8)):
            A + A
        with flop_name('something'):
            einops.einsum(A, W, W, W, 'i1 i2 i3, j1 i1, j2 i2, j3 i3 -> j1 j2 j3')
            A + A

    assert np.allclose(counter.total(), 
        n**3 * (n+1) * 2 + n**2 * (n+1) * (n+2) / 3 +     # contract_W_basic d=3
        n * (n+1) * (n+2) / 6 +                  # nonlin_sum part=(1,1,1)
        n * n * (n + 1) / 2 +                  # nonlin_sum part=(2,1,1)
        6 * n**4 + n**3         # something
    )
    assert np.allclose(counter.total(), round(counter.total()))


def test_flop_poly_fit():
    def f(n: int, l: int):
        flops = 5 * n**2 * l**2 + 3 * n**2 * l + 2 * l + 7
        with flop_name("f"):
            x = torch.zeros(flops)
            return x + 1

    polys = flop_poly_fit(f, deg_n=2, deg_l=2)

    expected = np.zeros((3, 3))
    expected[2, 2] = 5
    expected[2, 1] = 3
    expected[0, 1] = 2
    expected[0, 0] = 7
    np.testing.assert_allclose(polys["f"].c, expected, atol=1e-8)

    with pytest.raises(AssertionError):
        flop_poly_fit(f, deg_n=1, deg_l=2)

    with pytest.raises(AssertionError):
        flop_poly_fit(f, deg_n=2, deg_l=1)


def test_poly2d_arithmetic():
    c1 = np.zeros((3, 2))
    c1[0, 0] = 1
    c1[1, 0] = -2
    c1[2, 1] = 5

    c2 = np.zeros((2, 4))
    c2[0, 0] = 3
    c2[0, 2] = -4
    c2[1, 3] = 7

    p1 = Poly2D(c1)
    p2 = Poly2D(c2)

    expected_sum = np.zeros((3, 4))
    expected_sum[: c1.shape[0], : c1.shape[1]] += c1
    expected_sum[: c2.shape[0], : c2.shape[1]] += c2

    expected_diff = np.zeros((3, 4))
    expected_diff[: c1.shape[0], : c1.shape[1]] += c1
    expected_diff[: c2.shape[0], : c2.shape[1]] -= c2

    np.testing.assert_allclose((p1 + p2).c, expected_sum, atol=1e-8)
    np.testing.assert_allclose((p1 - p2).c, expected_diff, atol=1e-8)
    np.testing.assert_allclose((-p1).c, -c1, atol=1e-8)

    expected_p1_plus_2 = c1.copy()
    expected_p1_plus_2[0, 0] += 2
    expected_p1_minus_2 = c1.copy()
    expected_p1_minus_2[0, 0] -= 2
    expected_2_minus_p1 = -c1
    expected_2_minus_p1[0, 0] += 2
    np.testing.assert_allclose((p1 + 2).c, expected_p1_plus_2, atol=1e-8)
    np.testing.assert_allclose((2 + p1).c, expected_p1_plus_2, atol=1e-8)
    np.testing.assert_allclose((p1 - 2).c, expected_p1_minus_2, atol=1e-8)
    np.testing.assert_allclose((2 - p1).c, expected_2_minus_p1, atol=1e-8)

    x, y = 4, -1
    assert np.allclose((p1 + p2)(x, y), p1(x, y) + p2(x, y))
    assert np.allclose((p1 - p2)(x, y), p1(x, y) - p2(x, y))
    assert np.allclose((-p1)(x, y), -p1(x, y))
