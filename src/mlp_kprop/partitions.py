import itertools
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from functools import cache
from itertools import combinations, product
from typing import Any, Optional, Self

import numpy as np

logger = logging.getLogger(__name__)

# NOTE: Throughout this codebase "part" is shorthand for "partition".
# This differs from typical mathematical usage, where a "part" often denotes an element of a partition.
# We refer to elements of a partition as "blocks" instead.

# Our convention is that IntPartitions are always reverse-sorted.
# TODO: Maybe write a class to enforce this harder?
type IntPartition = tuple[int, ...]
type SetPartition[T] = tuple[frozenset[T], ...]
type Vec = tuple[int, ...]
type VecPartition = tuple[Vec, ...]


class IntPartCond:
    """
    Class for tracking which integer partitions we want to include.
    """

    def __init__(
        self,
        part_cond: Callable[[IntPartition], bool] | None = None,
        parts: Iterable[IntPartition] | None = None,
    ):
        assert (part_cond is not None) != (parts is not None), (
            "Exactly one of part_cond or parts must be specified."
        )
        self.part_cond = part_cond
        self.parts = parts

    def yield_parts(
        self, *, d_max: int | None = None, d: int | None = None
    ) -> Iterator[IntPartition]:
        """
        Yields integer partitions satisfying the condition.
        If part_cond is a callable, either d or d_max must be specified.
        """
        if self.part_cond is not None:
            logger.debug(f"Yielding parts meeting part_cond up to d_max={d_max}")
            assert (d_max is None) != (d is None), (
                "Exactly one of d or d_max must be specified if part_cond is a callable."
            )
            ds = [d] if d is not None else range(1, d_max + 1)
            for d in ds:
                for p in int_partitions(d):
                    if self.part_cond(p):
                        yield p
        else:
            yield from (p for p in self.parts if d_max is None or sum(p) <= d_max)

    @cache
    def get_parts(self, *, d_max: int | None = None, d: int | None = None) -> list[IntPartition]:
        return list(self.yield_parts(d_max=d_max, d=d))

    @cache
    def __call__(self, part: IntPartition) -> bool:
        if self.part_cond is not None:
            return self.part_cond(part)
        else:
            return part in self.parts

    def __and__(
        self, other: Self | Callable[[IntPartition], bool] | Iterator[IntPartition]
    ) -> Self:
        if isinstance(other, Callable):
            other = IntPartCond(part_cond=other)
        elif isinstance(other, Iterator):
            other = IntPartCond(parts=other)

        part_cond, parts = None, None
        if self.part_cond is not None and other.part_cond is not None:
            part_cond = lambda part: self.part_cond(part) and other.part_cond(part)
        elif self.part_cond is not None:
            part_cond = self.part_cond
        elif other.part_cond is not None:
            part_cond = other.part_cond

        if self.parts is not None and other.parts is not None:
            parts = list(set(self.parts).intersection(set(other.parts)))
        elif self.parts is not None:
            parts = self.parts
        elif other.parts is not None:
            parts = other.parts

        if part_cond is not None and parts is not None:
            parts = [p for p in parts if part_cond(p)]

        return IntPartCond(part_cond=part_cond, parts=parts)


trivial_int_cond = IntPartCond(part_cond=lambda part: True)


class VecPartCond:
    """
    Class for tracking which vector partitions we want to include.
    """

    def __init__(
        self,
        part_cond: Callable[[VecPartition], bool] | None = None,
        parts: Iterable[VecPartition] | None = None,
    ):
        assert (part_cond is not None) != (parts is not None), (
            "Exactly one of part_cond or parts must be specified."
        )
        self.part_cond = part_cond
        self.parts = parts

    def yield_parts(self, dim: int, sum_max: int | None = None) -> Iterator[VecPartition]:
        if self.part_cond is not None:
            assert sum_max is not None, "sum_max must be specified if part_cond is a callable."
            for v in vecs_sum_leq_k(dim, sum_max):
                for part in vector_partitions(v):
                    if self.part_cond(part):
                        yield part
        else:
            (p for p in self.parts if sum_max is None or sum(sum(v) for v in p) <= sum_max)

    @cache
    def get_parts(self, dim: int, sum_max: int | None = None) -> list[VecPartition]:
        return list(self.yield_parts(dim, sum_max))

    @cache
    def __call__(self, part: VecPartition) -> bool:
        if self.part_cond is not None:
            return self.part_cond(part)
        else:
            return part in self.parts


trivial_vec_cond = VecPartCond(part_cond=lambda part: True)


def _block_sort_key(block: frozenset[Any]) -> tuple[int, tuple[Any, ...]]:
    return (-len(block), tuple(sorted(block)))


def sort_set_partition(part: SetPartition[Any]) -> SetPartition[Any]:
    """
    Returns a canonical ordering of the blocks in part: descending by block size, breaking ties lexicographically.
    """
    return tuple(sorted(part, key=_block_sort_key))


@cache
def get_int_to_set_d(k: int) -> dict[IntPartition, list[SetPartition[int]]]:
    """
    Returns a mapping from integer partitions of k to the corresponding set partitions of [k].
    By convention we sort the blocks in decreasing order.
    (It's important for _diagslice_view that the ordering between int and set partitions is consistent.)
    """
    int_to_set_d = defaultdict(list)
    for part in set_partitions(k):
        part = sort_set_partition(part)
        int_to_set_d[set_to_int_partition(part)].append(part)
    return dict(int_to_set_d)


def int_to_set_partitions(part: IntPartition) -> list[SetPartition[int]]:
    """
    Returns list of all set partitions coresponding to the given int partition.
    """
    return get_int_to_set_d(sum(part))[part]

@cache
def int_partition_coef(part: IntPartition) -> int:
    """
    Returns the number of set partitions corresponding to the given int partition.
    """
    counts = defaultdict(int)
    for b in part:
        counts[b] += 1
    return math.factorial(sum(part)) // math.prod(
        math.factorial(size) ** count * math.factorial(count)
        for size, count in counts.items()
    )

def int_to_canonical_set_partition(part: IntPartition) -> SetPartition[int]:
    idx = 0
    blocks = []
    for b in part:
        blocks.append(frozenset(range(idx, idx + b)))
        idx += b
    return tuple(blocks)


@cache
def set_to_int_partition(part: SetPartition[int]) -> IntPartition:
    """
    Converts a set partition to the corresponding integer partition, with consistent ordering.
    """
    U = [x for block in part for x in block]
    assert set(U) == set(range(len(U))), "Partition must be of [d] for some d."
    return tuple(len(block) for block in part)


@cache
def set_to_vec_partition(part: SetPartition[int], v: Vec) -> VecPartition:
    """
    Given a set partition part of [n] and a vector v [n_1, ..., n_k] with sum n,
    returns the vector partition corresponding to applying part to the relabeling of
    [(1,1),...,(1,n_1),(2,1),...,(2,n_2),...,(k,1),...,(k,n_k)] as [n].
    """
    assert list(check_set_partition(part)) == list(range(sum(v))), (
        "part must be a partition of [n] where n = sum(v)."
    )
    vec_part = []
    for block in part:
        vec = [0] * len(v)
        for i in block:
            acc = 0
            for j, nj in enumerate(v):
                if i < acc + nj:
                    vec[j] += 1
                    break
                acc += nj
        vec_part.append(tuple(vec))
    return tuple(vec_part)


@cache
def set_partitions(U: int | tuple[Any, ...]) -> tuple[SetPartition[Any], ...]:
    """
    Returns all set partitions of U.

    Args:
        U: The universe to partition. Either an integer d, or a tuple of objects.
    Returns:
        Tuple of partitions, where each partition is a tuple of blocks (tuples).

    >>> set_partitions(0)
    ((),)
    >>> set_partitions(2)
    ((frozenset({0, 1}),), (frozenset({0}), frozenset({1})))
    >>> len(set_partitions(6))  # Bell number B_6
    203
    """
    base = tuple(range(U)) if isinstance(U, int) else tuple(U)
    blocks: list[list[Any]] = []
    parts: list[SetPartition[Any]] = []

    def build(i: int) -> None:
        if i == len(base):
            parts.append(tuple(frozenset(b) for b in blocks))
            return
        for b in blocks:
            b.append(base[i])
            build(i + 1)
            b.pop()
        blocks.append([base[i]])
        build(i + 1)
        blocks.pop()

    build(0)
    return tuple(parts)


def discrete_partition(n: int) -> SetPartition[int]:
    """
    Returns the discrete partition of [n], i.e., blocks of singletons in order.
    """
    return tuple(frozenset((i,)) for i in range(n))


@cache
def check_int_partition(part: IntPartition) -> int:
    """
    Checks that part is a valid integer partition and returns its sum.
    """
    assert all(isinstance(x, int) and x > 0 for x in part), "All parts must be positive integers."
    return sum(part)


@cache
def down_set(part: IntPartition) -> list[IntPartition]:
    """
    Returns all integer partitions below part in Young's lattice, *excluding the empty partition* since we never use it.
    """
    check_int_partition(part)
    assert sorted(part, reverse=True) == list(part), "part must be sorted in nonincreasing order."
    n = len(part)
    res: list[IntPartition] = []
    cur: list[int] = []

    def dfs(i: int, prev: int) -> None:
        if i == n:
            if len(cur) > 0:  # Exclude empty partition
                res.append(tuple(cur))
            return
        max_allowed = min(prev, part[i])
        for x in range(max_allowed, -1, -1):
            if x > 0:
                cur.append(x)
                dfs(i + 1, x)
                cur.pop()
            else:
                dfs(i + 1, 0)
                break

    if n == 0:
        return []  # Recall we exclude the empty partition
    dfs(0, part[0])
    return res


@cache
def check_set_partition(part: SetPartition[Any]) -> list[Any]:
    """
    Checks that part is a valid set partition and returns the sorted list of all elements.
    """
    all_elems = [e for block in part for e in block]
    assert len(all_elems) == len(set(all_elems)), "Duplicate elements found in partition."
    return sorted(all_elems)


@cache
def check_vec_partition(part: VecPartition, d: int) -> Vec:
    """
    Checks that part is a valid vector partition of a d-dimensional vector, and returns that vector.
    It is necessary to provide d for the all-zeros case.
    """
    assert all(len(v) == d for v in part), "All vectors in partition must have the same length."
    assert all(isinstance(v, tuple) for v in part), "Non-tuple found in partition."
    assert all(all(isinstance(x, int) and x >= 0 for x in v) for v in part), (
        "All entries in partition vectors must be non-negative integers."
    )
    return tuple(sum(x[i] for x in part) for i in range(d))


@cache
def weak_compositions(m: int, s: int) -> tuple[Vec, ...]:
    """
    Returns all weak compositions of s into m parts, i.e. tuples of m non-negative integers summing to s.
    """
    # Stars and bars
    comps: list[Vec] = []
    for cuts in combinations(range(s + m - 1), m - 1):
        prev = -1
        comp = []
        for c in cuts:
            comp.append(c - prev - 1)
            prev = c
        comp.append(s + m - 1 - prev - 1)
        comps.append(tuple(comp))
    return tuple(comps)

@cache
def multigraphs(v: int, e: int) -> list[list[tuple[tuple[int, int], int]]]:
    '''
    Returns all multigraphs on v vertices with e edges, loops allowed.
    Each multigraph is represented as a list of (edge, multiplicity) pairs, where
    edge is a tuple of vertex indices (i, j) with i <= j.
    '''
    if v == 0:
        return []
    edge_types = [
        (i, j)
        for i in range(v)
        for j in range(i, v)
    ]
    graphs = weak_compositions(len(edge_types), e)
    return list(sorted([
        [
            (edge_types[edge_idx], mult)
            for edge_idx, mult in enumerate(g)
            if mult > 0
        ]
        for g in graphs
    ]))

@cache
def vecs_sum_leq_k(m: int, k: int) -> tuple[Vec, ...]:
    """
    Returns all non-negative integer m-tuples summing to at most k.
    """
    vecs: list[Vec] = []
    for s in range(k + 1):
        vecs.extend(weak_compositions(m, s))
    return tuple(vecs)


@cache
def vector_partitions(n: Vec, prev: Vec | None = None) -> tuple[VecPartition, ...]:
    """
    Yields each distinct vector partition of the vector n.
    A vector partition of n is a tuple of nonzero vectors (tuples of non-negative integers)
    that sum elementwise to n. Also known as a partition of a multipartite number (MacMahon 1918).

    Args:
        n: The vector to partition.
        prev: All vectors in partition must be >= prev lexicographically.
    Returns:
        Tuple of partitions, where each partition is a tuple of Vec.
    """
    assert all(isinstance(x, int) and x >= 0 for x in n), (
        "n must be a vector of non-negative integers."
    )
    assert prev is None or len(prev) == len(n), (
        "prev must be None or a vector of the same length as n."
    )
    if all(x == 0 for x in n):
        return ((),)
    parts: list[VecPartition] = []
    for v in product(*[range(x + 1) for x in n]):
        if sum(v) == 0:
            continue
        if prev is not None and v < prev:
            continue
        resid = tuple(x - y for x, y in zip(n, v))
        for part in vector_partitions(resid, v):
            parts.append((v,) + part)
    return tuple(parts)


@cache
def int_partitions(n: int) -> tuple[IntPartition, ...]:
    """
    Special case of vector_partitions with 1d vectors.
    """
    if n < 0:
        raise ValueError("n must be a nonnegative integer.")
    elif n == 0:
        return ((),)
    parts: list[IntPartition] = []
    for vec_part in vector_partitions((n,)):
        parts.append(tuple(sorted((v[0] for v in vec_part), reverse=True)))
    return tuple(parts)


@cache
def count_vector_partitions(n: Vec, sum_all: bool = False) -> int:
    k = len(n)
    comps = [c for c in product(*[range(0, n[i] + 1) for i in range(k)]) if sum(c) > 0]

    dp = defaultdict(int)
    dp[tuple(0 for _ in range(k))] = 1

    def for_all_u_ge_c(c: Vec):
        # yields all u with c[i] ≤ u[i] ≤ n[i]
        ranges = [range(c[i], n[i] + 1) for i in range(k)]
        for u in product(*ranges):
            yield u

    for c in comps:
        for u in for_all_u_ge_c(c):
            prev = tuple(u[i] - c[i] for i in range(k))
            if prev in dp:
                dp[u] += dp[prev]

    if sum_all:
        return sum(dp.values())
    else:
        return dp[n]


@cache
def is_mixed(part: VecPartition, m: int | None) -> bool:
    """
    Checks whether vector partition part is m-mixed, i.e. each vector with sum no more than m has support on >1 indices.
    m=None is treated as infinity.
    """
    for v in part:
        if (m is None or sum(v) <= m) and len([i for i in v if i > 0]) <= 1:
            return False
    return True


@cache
def vec_part_coef(part: VecPartition, divide_fac=True) -> int:
    """
    Args:
        part: A vector partition of (n_1,...,n_k).
        divide_fac: If True, returns the coefficient divided by prod_i n_i!.
    Returns:
        The size of the fiber under the map from set partitions of sqcup_i [(i, 1),...,(i,n_i)]
        to vector partitions of (n_1, ..., n_k) that forgets the second coordinate,
        divided by prod_i n_i! if divide_fac is True.

    By orbit-stabilizer, this is equal to the reciprocal stabilizer size of any fiber element
    wrt the action of prod_i S_{n_i} on set partitions.
    Any stabilizing permutation must permute equivalent blocks among themselves and can arbitrarily
    permute elements within restrictions of each block to each index i.
    Thus the formula is
        prod_i n_i! / prod_v (c(v)! * prod_i (v_i)! ** (c(v))),
    where the product is over distinct vectors v in part, and c(v) is the number of times v appears in part.
    The prod_i n_i! factor is divided out to give the coefficient for the Wick expansion.
    """
    if len(part) == 0:
        return 1
    part_set = list(set(part))
    counts = {v: part.count(v) for v in part_set}
    n = [sum(v[i] for v in part) for i in range(len(part[0]))]
    ret = 1 / math.prod(
        math.factorial(c) * math.prod(math.factorial(v_i) for v_i in v) ** c
        for v, c in counts.items()
    )
    if not divide_fac:
        ret *= math.prod(math.factorial(ni) for ni in n)
        assert np.isclose(ret, round(ret)), "Coefficient should be integer."
        ret = int(ret)
    return ret


@cache
def disjoint_set_union(
    elems: tuple[Any, ...], unions: tuple[tuple[Any, Any], ...]
) -> dict[Any, Any]:
    """
    Performs disjoint set union on elems with given unions.
    Returns a mapping from each element to its root.
    """
    parent = {e: e for e in elems}
    size = dict.fromkeys(elems, 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a, b in unions:
        union(a, b)

    for e in elems:
        find(e)

    return parent


@cache
def is_connected(part: VecPartition, d: int) -> bool:
    """
    Checks whether the vector partition part is connected, i.e.
    whether its join with {(1,0,...,0), (0,1,0,...,0), ..., (0,...,0,1)} is the indiscrete partition.
    Uses the disjoint set union algorithm.
    As a special case, a partition of a vector with a 0 at any index is connected iff the vector has length 1.
    (This corresponds to the fact that the only nonzero cumulant of the empty Wick product (one) is the first one,
    and is why d needs to be provided as an argument.)
    """
    check_vec_partition(part, d)

    total = [0] * d
    for v in part:
        for j, x in enumerate(v):
            total[j] += x

    elems = tuple(range(d))
    unions: list[tuple[int, int]] = []
    for v in part:
        supp = [j for j, x in enumerate(v) if x > 0]
        if len(supp) >= 2:
            root = supp[0]
            for j in supp[1:]:
                unions.append((root, j))

    return len(set(disjoint_set_union(elems, tuple(unions)).values())) == 1

def vec_part_apply_perm(v: Vec, perm: tuple[int, ...]) -> Vec:
    return tuple(v[i] for i in perm)

@cache
def vec_part_isomorphic(p: VecPartition, q: VecPartition, vec: Optional[Vec] = None) -> bool:
    """
    Returns True iff there exists a permutation of coordinate indices that preserves vec and maps p to q.

    - Coordinate permutation acts on each vector v by v -> (v[perm[0]], ..., v[perm[k-1]]).
    - The order of vectors (“blocks”) inside a vector partition is treated as irrelevant
      (i.e., we compare multisets via sorting).
    """
    if len(p) != len(q):
        return False
    if len(p) == 0:
        return len(q) == 0

    assert all(len(v) == len(p[0]) for v in p), "p is not a partition of a fixed-dimensional vector."
    assert all(len(v) == len(q[0]) for v in q), "q is not a partition of a fixed-dimensional vector."
    k = len(p[0])

    q_norm = tuple(sorted(q))

    if vec is None:
        vec = (1,) * k
    assert len(vec) == k, "vec must have the same dimension as the vectors in the partitions."
    for perm in itertools.permutations(range(k)):
        if vec != tuple(vec[i] for i in perm):
            continue
        p_img = tuple(sorted(vec_part_apply_perm(v, perm) for v in p))
        if p_img == q_norm:
            return True

    return False

@cache
def canonical_vec_part(part: VecPartition, vec: Optional[Vec] = None, dim: int | None = None) -> VecPartition:
    """
    Canonical representative of the isomorphism class of `part` under coordinate permutations.

    Returns the lexicographically-minimal canonicalization:
      - apply a coordinate permutation to every vector in the partition
      - sort the vectors (since block order is irrelevant)
      - take the minimum over all permutations
    """
    if len(part) == 0:
        # If you might mix dimensions, pass dim and incorporate it into the key externally.
        return ()

    k = dim if dim is not None else len(part[0])
    assert all(len(v) == k for v in part), "All vectors must have the same dimension."

    best: VecPartition | None = None
    for perm in itertools.permutations(range(k)):
        if vec is not None and vec != tuple(vec[i] for i in perm):
            continue
        img = tuple(sorted(vec_part_apply_perm(v, perm) for v in part))
        if best is None or img < best:
            best = img
    return best  # type: ignore[return-value]

def vec_part_isos(
    parts: Iterable[VecPartition], vec: Optional[Vec] = None, dim: int | None = None
) -> dict[VecPartition, int]:
    """
    Returns {representative: count} where 'representative' is a canonical vec partition
    for each isomorphism class, and count is how many inputs fall in that class.
    """
    counts: defaultdict[VecPartition, int] = defaultdict(int)
    for p in parts:
        counts[canonical_vec_part(p, vec=vec, dim=dim)] += 1
    return dict(counts)
