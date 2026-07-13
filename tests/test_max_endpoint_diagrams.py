"""Diagram compiler vs brute-force direct sums (spec tests L3, L5).

For an operator term c * sum_slots T[slots] d^(slots) Psi, the direct value is

    c * sum_{labelings l} T[l] * (-1)^(p(l)-1) * prod_{distinct labels i} u[i, m_i(l)],

where m_i(l) is the multiplicity of label i and p(l) the number of distinct
labels (the endpoint derivative formula for beta(l)). The compiled quotient
diagrams must reproduce this exactly for random tensors and random unary
weight tables (the weights stand in for u_{i,k}(t) at a fixed t; equality is
then integrated pointwise, so pointwise equality is the right test).
"""

import itertools
import math
from fractions import Fraction

import pytest
import torch

from mlp_kprop.max_endpoint.diagrams import (
    FactorSpec,
    TermSpec,
    compile_term,
    diagrams_from_json,
    diagrams_to_json,
    term_c2,
    term_c2_squared_half,
    term_c3_generic,
    term_c3_trace,
    term_c4_core_metric,
    term_c4_dense,
)

torch.manual_seed(0)
N = 5
K_MAX_ORDER = 4


def _rand_sym(shape):
    t = torch.randn(*shape, dtype=torch.float64)
    # symmetrize over all axes
    d = t.ndim
    out = torch.zeros_like(t)
    for perm in itertools.permutations(range(d)):
        out += t.permute(perm)
    return out / math.factorial(d)


def _make_tensors():
    cov = _rand_sym((N, N))
    cov = cov - torch.diag(cov.diagonal())  # cov_od has zero diagonal
    return {
        "cov_od": cov,
        "k3": _rand_sym((N, N, N)),
        "k3_core_vec": torch.randn(N, dtype=torch.float64),
        "k3_metric": _rand_sym((N, N)),
        "k4_core": _rand_sym((N, N)),
        "k4_metric": _rand_sym((N, N)),
        "k4_dense": _rand_sym((N, N, N, N)),
    }


def _u_table():
    """u[i, k] for k = 1..4 (column k-1), random but O(1)."""
    return torch.randn(N, K_MAX_ORDER, dtype=torch.float64)


def brute_force_term(term: TermSpec, tensors: dict, u: torch.Tensor) -> float:
    slots = term.slots
    total = 0.0
    for labels in itertools.product(range(N), repeat=len(slots)):
        lab = dict(zip(slots, labels))
        # distinctness constraints
        if any(lab[a] == lab[b] for pair in term.distinct_pairs for a, b in [tuple(pair)]):
            continue
        val = 1.0
        for f in term.factors:
            val *= float(tensors[f.kind][tuple(lab[s] for s in f.legs)])
        mult: dict[int, int] = {}
        for s in slots:
            mult[lab[s]] = mult.get(lab[s], 0) + 1
        p = len(mult)
        w = math.prod(float(u[i, m - 1]) for i, m in mult.items())
        total += val * (-1.0) ** (p - 1) * w
    return float(term.coef) * total


def eval_compiled(diagrams, tensors: dict, u: torch.Tensor) -> float:
    """Direct dense evaluation of the compiled unrestricted contractions."""
    total = 0.0
    for d in diagrams:
        v = d.num_vertices
        for labels in itertools.product(range(N), repeat=v):
            val = 1.0
            for kind, legs in d.factors:
                val *= float(tensors[kind][tuple(labels[x] for x in legs)])
            for x, orders in enumerate(d.vertex_orders):
                for k in orders:
                    val *= float(u[labels[x], k - 1])
            total += d.coef * val
    return total


TERMS = {
    "C2": term_c2(),
    "C2sq_half": term_c2_squared_half(),
    "C3": term_c3_generic("k3"),
    "C3_trace": term_c3_trace(),
    "C4_trace": term_c4_core_metric(),
    "C4_dense": term_c4_dense(),
}


@pytest.mark.parametrize("name", sorted(TERMS))
def test_compiled_matches_brute_force(name):
    term = TERMS[name]
    tensors = _make_tensors()
    u = _u_table()
    direct = brute_force_term(term, tensors, u)
    compiled = compile_term(term)
    assert len(compiled) > 0
    got = eval_compiled(compiled, tensors, u)
    assert abs(got - direct) < 1e-9 * max(1.0, abs(direct)), (
        f"{name}: compiled={got} direct={direct}"
    )


def test_moebius_injective_sum():
    """Moebius inversion in isolation (spec test L5): the injective sum
    sum_{i,j,k distinct} A_i B_j C_k u_i u_j u_k as a compiled combination
    of unrestricted contractions."""
    term = TermSpec(
        name="inj3",
        coef=Fraction(1),
        factors=(FactorSpec("A", (0,)), FactorSpec("B", (1,)), FactorSpec("C", (2,))),
        distinct_pairs=frozenset(
            {frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 2))}
        ),
    )
    tensors = {k: torch.randn(N, dtype=torch.float64) for k in ("A", "B", "C")}
    u = _u_table()
    # direct injective sum, sign (+1) for p=3, weights u_{.,1} each
    direct = 0.0
    for i, j, k in itertools.product(range(N), repeat=3):
        if len({i, j, k}) < 3:
            continue
        direct += float(
            tensors["A"][i] * tensors["B"][j] * tensors["C"][k]
        ) * float(u[i, 0] * u[j, 0] * u[k, 0])
    compiled = compile_term(term)
    got = eval_compiled(compiled, tensors, u)
    assert abs(got - direct) < 1e-9 * max(1.0, abs(direct))


def test_c2_diagram_structure():
    """C2 compiles to exactly one diagram: the equality pattern is forced to
    all-distinct (i != j), giving sign (-1)^(2-1) = -1, and the Moebius
    merged-vertex companion contains a zero-diagonal covariance loop and is
    dropped at compile time. Net: a single 2-vertex edge with coef -1/2."""
    compiled = compile_term(term_c2())
    assert len(compiled) == 1
    d = compiled[0]
    assert d.num_vertices == 2
    assert abs(d.coef - (-0.5)) < 1e-12
    assert d.vertex_orders == ((1,), (1,))
    assert d.factors == (("cov_od", (0, 1)),)


def test_serialization_roundtrip(tmp_path):
    compiled = compile_term(term_c2_squared_half())
    path = tmp_path / "c2sq.json"
    diagrams_to_json(compiled, path, meta={"term": "C2sq_half"})
    loaded = diagrams_from_json(path)
    assert loaded == compiled
