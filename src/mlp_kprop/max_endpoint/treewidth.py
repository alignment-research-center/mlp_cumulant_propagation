"""
Sum-product variable elimination for compiled endpoint diagrams.

A diagram is a set of factors over discrete variables (diagram vertices plus
internal rank variables from factorized cumulants). Evaluation cost is
O(batch * n^(w+1)) where w is the induced width of the elimination order; the
templates here are tiny, so an exact minimum-width order is found by exhaustive
search over elimination orders (with a min-fill fallback for larger graphs,
recorded as non-exact).

Factors are either static (cumulant tensors, no batch dimension) or batched
over quadrature nodes (unary endpoint weights, leading dim Q). Batched and
static factors are mixed freely; a contraction result is batched iff any of
its inputs is.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "VEFactor",
    "EliminationInfo",
    "FlopTally",
    "find_elimination_order",
    "contract_factors",
    "naive_contract",
]


@dataclass
class VEFactor:
    """A factor over distinct variables.

    tensor shape: (Q,) + (domain[v] for v in vars) if batched
                  else    (domain[v] for v in vars).
    """

    vars: tuple[int, ...]
    tensor: Tensor
    batched: bool = False

    def __post_init__(self) -> None:
        assert len(set(self.vars)) == len(self.vars), "VEFactor vars must be distinct"
        expected = len(self.vars) + (1 if self.batched else 0)
        assert self.tensor.ndim == expected, (
            f"tensor ndim {self.tensor.ndim} != expected {expected} for vars={self.vars}"
        )


@dataclass
class EliminationInfo:
    order: tuple[int, ...]
    width: int  # induced width: max #neighbors at elimination time
    exact: bool


def _simulate(adj: dict[int, set[int]], order: tuple[int, ...]) -> int:
    """Induced width of an elimination order on (a copy of) the primal graph."""
    adj = {v: set(nb) for v, nb in adj.items()}
    width = 0
    for v in order:
        nbrs = adj[v]
        width = max(width, len(nbrs))
        for a in nbrs:
            adj[a].discard(v)
            adj[a].update(nbrs - {a})
        del adj[v]
    return width


def _primal_graph(all_vars: list[int], factor_vars: list[tuple[int, ...]]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {v: set() for v in all_vars}
    for fv in factor_vars:
        for a, b in itertools.combinations(fv, 2):
            adj[a].add(b)
            adj[b].add(a)
    return adj


def find_elimination_order(
    all_vars: list[int],
    factor_vars: list[tuple[int, ...]],
    exact_max_vertices: int = 8,
) -> EliminationInfo:
    """Minimum induced-width elimination order.

    Exhaustive search over all orders when len(all_vars) <= exact_max_vertices
    (endpoint templates have <= ~6 variables); min-fill heuristic otherwise,
    with the `exact` flag recording which was used.
    """
    adj = _primal_graph(all_vars, factor_vars)
    if len(all_vars) <= exact_max_vertices:
        best: tuple[int, tuple[int, ...]] | None = None
        for order in itertools.permutations(all_vars):
            w = _simulate(adj, order)
            if best is None or w < best[0]:
                best = (w, order)
        assert best is not None
        return EliminationInfo(order=best[1], width=best[0], exact=True)
    # Min-fill heuristic.
    work = {v: set(nb) for v, nb in adj.items()}
    order: list[int] = []
    while work:

        def fill(v: int) -> int:
            nbrs = work[v]
            return sum(1 for a, b in itertools.combinations(nbrs, 2) if b not in work[a])

        v = min(work, key=lambda x: (fill(x), len(work[x])))
        order.append(v)
        nbrs = work[v]
        for a in nbrs:
            work[a].discard(v)
            work[a].update(nbrs - {a})
        del work[v]
    return EliminationInfo(order=tuple(order), width=_simulate(adj, tuple(order)), exact=False)


class FlopTally:
    """Analytic FLOP counter for sum-product operations.

    Convention: joining k tables over a joint index space of size S (batch
    dimension included) costs (k-1)*S multiplies; summing out a variable from
    the joint table costs S adds. Cross-checked against torch's instrumented
    counter in tests up to this documented convention.
    """

    def __init__(self) -> None:
        self.flops = 0
        self.max_table = 0

    def table(self, size: int) -> int:
        self.max_table = max(self.max_table, size)
        return size


def _einsum_group(group: list[VEFactor], out_vars: tuple[int, ...]) -> VEFactor:
    """Multiply a group of factors and project onto out_vars (summing the rest)."""
    all_vars = tuple(sorted({x for f in group for x in f.vars}))
    letters = {x: chr(ord("a") + i) for i, x in enumerate(all_vars)}
    batched = any(f.batched for f in group)
    in_exprs = [
        ("q" if f.batched else "") + "".join(letters[x] for x in f.vars) for f in group
    ]
    out_expr = ("q" if batched else "") + "".join(letters[x] for x in out_vars)
    result = torch.einsum(",".join(in_exprs) + "->" + out_expr, *[f.tensor for f in group])
    return VEFactor(vars=out_vars, tensor=result, batched=batched)


def contract_factors(
    factors: list[VEFactor],
    domains: dict[int, int],
    order: tuple[int, ...],
    batch: int,
    tally: FlopTally | None = None,
) -> Tensor:
    """Contract all factors, eliminating variables in `order`.

    Args:
        factors: the diagram's factors; every variable of every factor must
            appear in `order`.
        domains: variable id -> domain size.
        order: elimination order covering all variables.
        batch: Q, the size of the batched leading dimension.
    Returns:
        (Q,) tensor (broadcast from a scalar if no factor was batched).
    """
    if tally is None:
        tally = FlopTally()
    live = list(factors)
    for v in order:
        group = [f for f in live if v in f.vars]
        live = [f for f in live if v not in f.vars]
        if not group:
            continue
        out_vars = tuple(sorted({x for f in group for x in f.vars} - {v}))
        joint_vars = out_vars + (v,)
        joint_batch = batch if any(f.batched for f in group) else 1
        joint_size = joint_batch
        for x in joint_vars:
            joint_size *= domains[x]
        tally.table(joint_size)
        tally.flops += (len(group) - 1) * joint_size + joint_size  # mults + sum-out adds
        live.append(_einsum_group(group, out_vars))
    # Remaining factors have no variables: multiply their values.
    out = torch.ones((), dtype=factors[0].tensor.dtype, device=factors[0].tensor.device)
    for f in live:
        assert f.vars == (), f"Variable(s) {f.vars} not covered by elimination order"
        out = out * f.tensor
        tally.flops += batch if f.batched else 1
    if out.ndim == 0:
        out = out.expand(batch)
    return out


def naive_contract(factors: list[VEFactor], domains: dict[int, int], batch: int) -> Tensor:
    """Reference contraction over the full joint space (tests only)."""
    all_vars = tuple(sorted({x for f in factors for x in f.vars}))
    letters = {x: chr(ord("a") + i) for i, x in enumerate(all_vars)}
    batched = any(f.batched for f in factors)
    in_exprs = [
        ("q" if f.batched else "") + "".join(letters[x] for x in f.vars) for f in factors
    ]
    out = torch.einsum(",".join(in_exprs) + "->" + ("q" if batched else ""), *[f.tensor for f in factors])
    if out.ndim == 0:
        out = out.expand(batch)
    return out
