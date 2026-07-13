"""
Offline diagram compiler for cumulant differential operators applied to the
product-Gaussian maximum Psi.

An operator term is

    O = c * sum_{s_1..s_m} T[s_1,...,s_m] d_{s_1} ... d_{s_m},

an unrestricted sum over "slots" (derivative positions), where T factors as a
product of small tensors (a covariance 2-edge, a third-cumulant 3-edge, the
core/metric pair of a retained fourth-cumulant trace sector, ...). Some factor
kinds are known to vanish when their own legs coincide (the off-diagonal
covariance); such merges are dropped at compile time.

Compilation pipeline (D1 of the experiment spec):

1. Equality patterns: enumerate set partitions pi of the slots (respecting
   declared distinctness). A block B of pi contributes derivative multiplicity
   |B| at its vertex, so the injective term over pi's blocks is

       sum_{injective labelings l} T[l o pi] * (-1)^(|pi|-1)
           * int B(t) prod_{B in pi} u_{l(B), |B|}(t) dt,

   using the endpoint derivative formula (hermite_endpoint).

2. Moebius inversion on the partition lattice converts each injective sum into
   unrestricted contractions: for coarsenings sigma of pi,

       sum_inj = sum_{sigma >= pi} mu(pi, sigma) * sum_unrestricted(sigma),
       mu(pi, sigma) = prod_{C in sigma} (-1)^(m_C - 1) (m_C - 1)!,

   where m_C is the number of pi-blocks merged into the sigma-block C. A merged
   vertex carries the *product* of the unary weights of its merged blocks.

3. Quotient diagrams are canonicalized (vertex relabeling + factor sorting) and
   identical diagrams have their coefficients summed.

The output is a list of CompiledDiagram: unrestricted sum-product contractions
evaluated by treewidth-bounded variable elimination (treewidth.py).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import permutations
from pathlib import Path

__all__ = [
    "FactorSpec",
    "TermSpec",
    "CompiledDiagram",
    "compile_term",
    "set_partitions_of",
    "term_c2",
    "term_c2_squared_half",
    "term_c3_generic",
    "term_c3_trace",
    "term_c4_core_metric",
    "term_c4_dense",
    "diagrams_to_json",
    "diagrams_from_json",
]


@dataclass(frozen=True)
class FactorSpec:
    """One tensor factor of an operator term.

    Attributes:
        kind: symbolic name resolved by the ARC bridge at contraction time
            (e.g. "cov_od", "k3", "k4_core", "k4_metric").
        legs: slot ids this factor's tensor legs attach to.
        zero_when_merged: if True, the factor vanishes whenever two of its own
            legs are assigned the same coordinate (used for the off-diagonal
            covariance); diagrams containing such a merge are dropped.
    """

    kind: str
    legs: tuple[int, ...]
    zero_when_merged: bool = False


@dataclass(frozen=True)
class TermSpec:
    """An operator term c * sum_slots (prod factors) d^(slots)."""

    name: str
    coef: Fraction
    factors: tuple[FactorSpec, ...]
    distinct_pairs: frozenset[frozenset[int]] = field(default_factory=frozenset)

    @property
    def slots(self) -> tuple[int, ...]:
        out: set[int] = set()
        for f in self.factors:
            out.update(f.legs)
        return tuple(sorted(out))


@dataclass(frozen=True)
class CompiledDiagram:
    """One unrestricted quotient contraction.

    Attributes:
        coef: scalar coefficient (operator constant x endpoint sign x Moebius).
        vertex_orders: per-vertex sorted tuple of Hermite orders; vertex v
            carries weight prod_j u_{i_v, vertex_orders[v][j]}(t).
        factors: (kind, vertex ids per leg) pairs.
    """

    coef: float
    vertex_orders: tuple[tuple[int, ...], ...]
    factors: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def num_vertices(self) -> int:
        return len(self.vertex_orders)


def set_partitions_of(items: tuple[int, ...]):
    """Yield all set partitions of `items` as tuples of tuples."""
    items = list(items)
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for part in set_partitions_of(tuple(rest)):
        # first in its own block
        yield ((first,),) + part
        # first added to each existing block
        for i in range(len(part)):
            yield part[:i] + ((first,) + part[i],) + part[i + 1 :]


def _moebius_factor(m: int) -> int:
    """mu contribution (-1)^(m-1) (m-1)! of a sigma-block merging m pi-blocks."""
    return (-1) ** (m - 1) * math.factorial(m - 1)


def _canonicalize(
    vertex_orders: list[tuple[int, ...]],
    factors: list[tuple[str, tuple[int, ...]]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    """Canonical form under vertex relabeling (small diagrams: try all perms)."""
    v = len(vertex_orders)
    best = None
    for perm in permutations(range(v)):
        vo = tuple(vertex_orders[perm.index(i)] for i in range(v))
        # perm maps old vertex -> new vertex id
        fs = tuple(
            sorted((kind, tuple(perm[x] for x in legs)) for kind, legs in factors)
        )
        cand = (vo, fs)
        if best is None or cand < best:
            best = cand
    assert best is not None
    return best


def compile_term(term: TermSpec) -> list[CompiledDiagram]:
    """Compile an operator term into unrestricted quotient diagrams.

    Every returned diagram is an unrestricted contraction; summing
    coef * contraction over the list reproduces the original constrained
    operator term applied to Psi (up to quadrature).
    """
    slots = term.slots
    acc: dict[tuple, float] = {}

    for pi in set_partitions_of(slots):
        # Respect distinctness constraints between slots.
        ok = True
        for block in pi:
            for a in block:
                for b in block:
                    if a < b and frozenset((a, b)) in term.distinct_pairs:
                        ok = False
        if not ok:
            continue
        p = len(pi)
        sign = (-1.0) ** (p - 1)

        # Moebius inversion: coarsenings sigma of pi (partitions of pi's blocks).
        for sigma in set_partitions_of(tuple(range(p))):
            mu = math.prod(_moebius_factor(len(c)) for c in sigma)
            # Build quotient diagram: one vertex per sigma-block.
            slot_to_vertex: dict[int, int] = {}
            vertex_orders: list[tuple[int, ...]] = []
            for v, c in enumerate(sigma):
                orders = tuple(sorted(len(pi[b]) for b in c))
                vertex_orders.append(orders)
                for b in c:
                    for s in pi[b]:
                        slot_to_vertex[s] = v
            factors = []
            dropped = False
            for f in term.factors:
                legs = tuple(slot_to_vertex[s] for s in f.legs)
                if f.zero_when_merged and len(set(legs)) < len(legs):
                    dropped = True
                    break
                factors.append((f.kind, legs))
            if dropped:
                continue
            vo, fs = _canonicalize(vertex_orders, factors)
            key = (vo, fs)
            acc[key] = acc.get(key, 0.0) + float(term.coef) * sign * mu

    out = [
        CompiledDiagram(coef=c, vertex_orders=vo, factors=fs)
        for (vo, fs), c in acc.items()
        if abs(c) > 1e-14
    ]
    # Deterministic order: by vertex count then repr.
    out.sort(key=lambda d: (d.num_vertices, d.vertex_orders, d.factors))
    return out


# ---------------------------------------------------------------------------
# Standard operator terms
# ---------------------------------------------------------------------------

def term_c2() -> TermSpec:
    """C2 = (1/2) sum_{i != j} Sigma_ij d_i d_j (diagonal absorbed into sigma)."""
    return TermSpec(
        name="C2",
        coef=Fraction(1, 2),
        factors=(FactorSpec("cov_od", (0, 1), zero_when_merged=True),),
        distinct_pairs=frozenset({frozenset((0, 1))}),
    )


def term_c2_squared_half() -> TermSpec:
    """(1/2) C2^2 = (1/8) sum_{i!=j} sum_{k!=l} Sigma_ij Sigma_kl d_i d_j d_k d_l."""
    return TermSpec(
        name="C2sq_half",
        coef=Fraction(1, 8),
        factors=(
            FactorSpec("cov_od", (0, 1), zero_when_merged=True),
            FactorSpec("cov_od", (2, 3), zero_when_merged=True),
        ),
        distinct_pairs=frozenset({frozenset((0, 1)), frozenset((2, 3))}),
    )


def term_c3_generic(kind: str = "k3") -> TermSpec:
    """C3 = (1/6) sum_{ijk} kappa3_{ijk} d_i d_j d_k (unrestricted).

    The single 3-edge is expanded by the ARC bridge (dense tensor at small n,
    rank-decomposed FactoredTensor or Sym(v (x) M) trace surrogate otherwise).
    Because the derivative contraction is fully symmetric, any surrogate whose
    full unrestricted contraction matches kappa3's may be substituted before
    equality-pattern decomposition.
    """
    return TermSpec(name=f"C3[{kind}]", coef=Fraction(1, 6), factors=(FactorSpec(kind, (0, 1, 2)),))


def term_c3_trace() -> TermSpec:
    """C3 for the trace sector kappa3 ~ Sym(v (x) M) (k_max=2 AUGMENT):

        C3 Psi = (1/6) sum_{ijk} v_i M_jk d_i d_j d_k.
    """
    return TermSpec(
        name="C3_trace",
        coef=Fraction(1, 6),
        factors=(FactorSpec("k3_core_vec", (0,)), FactorSpec("k3_metric", (1, 2))),
    )


def term_c4_dense() -> TermSpec:
    """C4 with a dense fourth cumulant (reference path, small n only)."""
    return TermSpec(
        name="C4_dense", coef=Fraction(1, 24), factors=(FactorSpec("k4_dense", (0, 1, 2, 3)),)
    )


def term_c4_core_metric() -> TermSpec:
    """Retained fourth-cumulant trace sector kappa4 ~ Sym(C (x) M):

        C4 Psi = (1/24) sum_{ijkl} C_ij M_kl d_i d_j d_k d_l.

    The symmetrization is dropped exactly (the unrestricted derivative sum is
    symmetric); equality patterns are taken on the surrogate C (x) M.
    For ARC's SIMPLE sector kappa4 ~ c * Sym(M (x) M), the bridge maps the
    "k4_core" factor to c * M.
    """
    return TermSpec(
        name="C4_trace",
        coef=Fraction(1, 24),
        factors=(FactorSpec("k4_core", (0, 1)), FactorSpec("k4_metric", (2, 3))),
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def diagrams_to_json(diagrams: list[CompiledDiagram], path: str | Path, meta: dict | None = None) -> None:
    payload = {
        "meta": meta or {},
        "diagrams": [
            {
                "coef": d.coef,
                "vertex_orders": [list(v) for v in d.vertex_orders],
                "factors": [[kind, list(legs)] for kind, legs in d.factors],
            }
            for d in diagrams
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1))


def diagrams_from_json(path: str | Path) -> list[CompiledDiagram]:
    payload = json.loads(Path(path).read_text())
    return [
        CompiledDiagram(
            coef=float(d["coef"]),
            vertex_orders=tuple(tuple(v) for v in d["vertex_orders"]),
            factors=tuple((kind, tuple(legs)) for kind, legs in d["factors"]),
        )
        for d in payload["diagrams"]
    ]
