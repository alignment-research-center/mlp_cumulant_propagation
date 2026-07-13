"""Variable-elimination engine vs naive contraction (spec test L4)."""

import torch

from mlp_kprop.max_endpoint.treewidth import (
    FlopTally,
    VEFactor,
    contract_factors,
    find_elimination_order,
    naive_contract,
)

torch.manual_seed(0)
N = 4
Q = 3


def _mat():
    return torch.randn(N, N, dtype=torch.float64)


def _vec_batched():
    return torch.randn(Q, N, dtype=torch.float64)


def _check(factors, domains, expected_width=None):
    info = find_elimination_order(sorted(domains), [f.vars for f in factors])
    assert info.exact
    if expected_width is not None:
        assert info.width == expected_width, f"width={info.width} expected={expected_width}"
    tally = FlopTally()
    got = contract_factors(factors, domains, info.order, batch=Q, tally=tally)
    want = naive_contract(factors, domains, batch=Q)
    assert torch.allclose(got, want, rtol=1e-10, atol=1e-12), f"{got} vs {want}"
    assert tally.flops > 0 and tally.max_table > 0
    return info


def _unaries(vars_):
    return [VEFactor(vars=(v,), tensor=_vec_batched(), batched=True) for v in vars_]


def test_path():
    """Path 0-1-2-3: width 1."""
    factors = _unaries(range(4)) + [
        VEFactor(vars=(0, 1), tensor=_mat()),
        VEFactor(vars=(1, 2), tensor=_mat()),
        VEFactor(vars=(2, 3), tensor=_mat()),
    ]
    _check(factors, {v: N for v in range(4)}, expected_width=1)


def test_cycle():
    """4-cycle: width 2."""
    factors = _unaries(range(4)) + [
        VEFactor(vars=(0, 1), tensor=_mat()),
        VEFactor(vars=(1, 2), tensor=_mat()),
        VEFactor(vars=(2, 3), tensor=_mat()),
        VEFactor(vars=(3, 0), tensor=_mat()),
    ]
    _check(factors, {v: N for v in range(4)}, expected_width=2)


def test_triangle():
    factors = _unaries(range(3)) + [
        VEFactor(vars=(0, 1), tensor=_mat()),
        VEFactor(vars=(1, 2), tensor=_mat()),
        VEFactor(vars=(0, 2), tensor=_mat()),
    ]
    _check(factors, {v: N for v in range(3)}, expected_width=2)


def test_two_triangles_sharing_edge():
    """Vertices 0,1 shared; triangles (0,1,2) and (0,1,3): width 2."""
    factors = _unaries(range(4)) + [
        VEFactor(vars=(0, 1), tensor=_mat()),
        VEFactor(vars=(1, 2), tensor=_mat()),
        VEFactor(vars=(0, 2), tensor=_mat()),
        VEFactor(vars=(1, 3), tensor=_mat()),
        VEFactor(vars=(0, 3), tensor=_mat()),
    ]
    _check(factors, {v: N for v in range(4)}, expected_width=2)


def test_repeated_edges():
    """Two parallel edges between the same vertices (multigraph)."""
    factors = _unaries(range(2)) + [
        VEFactor(vars=(0, 1), tensor=_mat()),
        VEFactor(vars=(0, 1), tensor=_mat()),
    ]
    _check(factors, {v: N for v in range(2)}, expected_width=1)


def test_self_loop_as_unary():
    """A 'loop' (factor whose legs merged) arrives as a unary diagonal factor."""
    m = _mat()
    factors = _unaries(range(2)) + [
        VEFactor(vars=(0,), tensor=m.diagonal()),
        VEFactor(vars=(0, 1), tensor=_mat()),
    ]
    _check(factors, {v: N for v in range(2)}, expected_width=1)


def test_hyperedge():
    """3-hyperedge plus a pendant edge."""
    t3 = torch.randn(N, N, N, dtype=torch.float64)
    factors = _unaries(range(4)) + [
        VEFactor(vars=(0, 1, 2), tensor=t3),
        VEFactor(vars=(2, 3), tensor=_mat()),
    ]
    _check(factors, {v: N for v in range(4)}, expected_width=2)


def test_disconnected_components():
    """Two disjoint edges: componentwise elimination keeps width 1."""
    factors = _unaries(range(4)) + [
        VEFactor(vars=(0, 1), tensor=_mat()),
        VEFactor(vars=(2, 3), tensor=_mat()),
    ]
    _check(factors, {v: N for v in range(4)}, expected_width=1)


def test_mixed_domains_rank_variable():
    """Rank-decomposed 3-edge: coordinate vars (domain N) + rank var (domain R)."""
    r = 7
    a = torch.randn(N, r, dtype=torch.float64)
    b = torch.randn(N, r, dtype=torch.float64)
    c = torch.randn(N, r, dtype=torch.float64)
    factors = _unaries(range(3)) + [
        VEFactor(vars=(0, 3), tensor=a),
        VEFactor(vars=(1, 3), tensor=b),
        VEFactor(vars=(2, 3), tensor=c),
    ]
    domains = {0: N, 1: N, 2: N, 3: r}
    info = find_elimination_order(sorted(domains), [f.vars for f in factors])
    got = contract_factors(factors, domains, info.order, batch=Q)
    want = naive_contract(factors, domains, batch=Q)
    assert torch.allclose(got, want, rtol=1e-10)


def test_min_fill_fallback_on_larger_graph():
    """A 10-vertex path forces the heuristic branch; result must still be exact."""
    n_small = 2
    factors = [
        VEFactor(vars=(v,), tensor=torch.randn(Q, n_small, dtype=torch.float64), batched=True)
        for v in range(10)
    ] + [
        VEFactor(vars=(v, v + 1), tensor=torch.randn(n_small, n_small, dtype=torch.float64))
        for v in range(9)
    ]
    domains = {v: n_small for v in range(10)}
    info = find_elimination_order(sorted(domains), [f.vars for f in factors])
    assert not info.exact  # heuristic branch, flagged as such
    assert info.width == 1  # min-fill is optimal on a path
    got = contract_factors(factors, domains, info.order, batch=Q)
    want = naive_contract(factors, domains, batch=Q)
    assert torch.allclose(got, want, rtol=1e-10)
