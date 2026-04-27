import math
from collections import defaultdict

from mlp_kprop.partitions import *
from mlp_kprop.kprop_harmonic import get_all_terms

PART_NUMS = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]  # OEIS A000041
BELL_NUMS = [1, 1, 2, 5, 15, 52, 203, 877]  # OEIS A000110


def test_set_to_int_partition():
    assert set_to_int_partition((frozenset({0, 1, 2}),)) == (3,)
    assert set_to_int_partition((frozenset({0, 1}), frozenset({2}))) == (2, 1)
    assert set_to_int_partition((frozenset({2}), frozenset({0, 1}))) == (1, 2)
    assert set_to_int_partition((frozenset({1}), frozenset({0}), frozenset({2}))) == (1, 1, 1)


def test_set_partitions():
    assert list(set_partitions(0)) == [()]
    assert list(set_partitions(1)) == [(frozenset({0}),)]
    assert list(set_partitions(2)) == [(frozenset({0, 1}),), (frozenset({0}), frozenset({1}))]
    assert list(set_partitions(("a", "b"))) == [
        (frozenset({"a", "b"}),),
        (frozenset({"a"}), frozenset({"b"})),
    ]
    for d in range(len(BELL_NUMS)):
        assert len(list(set_partitions(d))) == BELL_NUMS[d]


def test_int_partitions():
    assert set(int_partitions(3)) == {(3,), (2, 1), (1, 1, 1)}
    assert set(int_partitions(4)) == {(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)}
    for n in range(len(PART_NUMS)):
        assert len(list(int_partitions(n))) == PART_NUMS[n]


def test_vec_partitions():
    # Test 1d vectors
    for n in range(len(PART_NUMS)):
        assert len(list(vector_partitions((n,)))) == PART_NUMS[n]

    # Test all-one vectors
    for n in range(len(BELL_NUMS)):
        assert len(list(vector_partitions((1,) * n))) == BELL_NUMS[n]

    # Misc small cases
    expected = {
        (1, 2): [((0, 1), (0, 1), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 0)), ((1, 2),)],
        (3, 1): [
            ((0, 1), (1, 0), (1, 0), (1, 0)),
            ((0, 1), (1, 0), (2, 0)),
            ((0, 1), (3, 0)),
            ((1, 0), (1, 0), (1, 1)),
            ((1, 0), (2, 1)),
            ((1, 1), (2, 0)),
            ((3, 1),),
        ],
    }
    for n, exp in expected.items():
        assert list(vector_partitions(n)) == exp
        for part in exp:
            assert check_vec_partition(part, len(n)) == n


def test_set_to_vec_partition():
    # (set parition, vec, expected vec partition)
    tests = [
        (({0, 1, 2},), (1, 2), ((1, 2),)),
        (({0}, {1}, {2}), (1, 2), ((1, 0), (0, 1), (0, 1))),
        (({0}, {1, 2}), (1, 2), ((1, 0), (0, 2))),
        (({0, 1}, {2}), (1, 2), ((1, 1), (0, 1))),
        (({0, 2}, {1}), (1, 2), ((1, 1), (0, 1))),
    ]
    for set_part, vec, vec_part in tests:
        set_part = tuple(frozenset(s) for s in set_part)
        assert set_to_vec_partition(set_part, vec) == vec_part


def test_is_mixed():
    tests = [  # (partition, m, expected)
        (((1, 0),), None, False),
        (((1, 0),), 0, True),
        (((1, 0),), 1, False),
        (((1, 1),), None, True),
        (((2, 0),), 1, True),
        (((2, 0),), 2, False),
        (((1, 1), (0, 2)), 1, True),
        (((1, 1), (0, 2)), 2, False),
        (((0, 0, 3), (1, 1, 0)), 2, True),
        (((0, 0, 3), (1, 0, 0)), 2, False),
        (((1, 1, 1),), None, True),
        (((0, 0),), None, False),
        (((0, 0),), 0, False),
        (((1, 0), (0, 1)), None, False),
        (((2, 0), (0, 2)), 1, True),
        (((1, 0), (1, 1)), 1, False),
        (((2, 0), (1, 1)), 1, True),
        ((), None, True),
    ]
    for part, m, exp in tests:
        assert is_mixed(part, m) == exp


def test_mixed_partitions():
    # (n, m, expected count)
    tests = [
        ((3,), 0, 3),
        ((3,), 1, 1),
        ((3,), 2, 1),
        ((3,), None, 0),
        ((1, 1, 1), 0, 5),
        ((1, 1, 1), 1, 1),
        ((1, 1, 1), 2, 1),
        ((1, 1, 1), None, 1),
        ((1, 2), 1, 1),
    ]
    for n, m, exp in tests:
        parts = [p for p in vector_partitions(n) if is_mixed(p, m)]
        assert len(parts) == exp, f"n={n}, m={m}, parts={parts}"


def test_vec_part_coef():
    """
    Checks that enumeration over set partitions and vec partitions give same coefs
    """

    def to_vec(block: list[tuple[int, int]], k: int) -> Vec:
        """
        Converts a subset of sqcup_{i=1}^k {(i,1),...,(i,n_i)} to a vector by forgetting the second coordinate.
        """
        ret = [0] * k
        for u, v in block:
            ret[u] += 1
        return tuple(ret)

    def to_vec_part(part: SetPartition[int], k: int) -> VecPartition:
        """
        Converts a set partition of sqcup_{i=1}^k {(i,1),...,(i,n_i)} to a vector partition by forgetting the second coordinate.
        """
        return tuple(to_vec(block, k) for block in part)

    def get_counts(n: Vec) -> dict[VecPartition, int]:
        """
        Enumerates set partitions of sqcup_{i=1}^k {(i,1),...,(i,n_i)} and counts occurrences of each corresponding vector partition.
        """
        counts = defaultdict(int)
        U = tuple((i, j) for i in range(len(n)) for j in range(n[i]))
        for part in set_partitions(U):
            counts[tuple(sorted(to_vec_part(part, len(n))))] += 1

        return counts

    def check_coefs(n: Vec):
        """
        Checks that manual enumeration via get_counts matches vec_part_coef.
        """
        counts = get_counts(n)
        coef = math.prod(math.factorial(x) for x in n)
        for vec_part, count in counts.items():
            try:
                assert int(vec_part_coef(vec_part) * coef) == count, (
                    f"n={n}, part={vec_part}, count={count}"
                )
                assert int(vec_part_coef(vec_part, divide_fac=False)) == count, (
                    f"n={n}, part={vec_part}, count={count}"
                )
            except:
                import pdb

                pdb.set_trace()

    ns = [(1,), (2,), (3,), (5,), (1, 1, 1), (1, 2, 1), (2, 2), (2, 2, 3), (3, 3, 3)]

    for n in ns:
        check_coefs(n)


def test_is_connected():
    # (d, partition, expected)
    tests = [
        (1, (), True),
        (2, (), False),
        (5, (), False),
        (1, ((2,),), True),
        (2, ((3, 0), (0, 2)), False),
        (2, ((3, 0), (0, 2), (1, 2)), True),
        (3, ((1, 1, 1),), True),
        (3, ((1, 1, 0), (0, 0, 1)), False),
        (3, ((1, 1, 0), (0, 1, 1)), True),
        (3, ((1, 1, 0), (0, 1, 1), (1, 0, 1)), True),
        (3, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), False),
        (3, ((1, 1, 0), (1, 1, 0), (0, 0, 1)), False),
        (4, ((1, 1, 0, 0), (1, 0, 1, 0), (0, 0, 1, 1)), True),
    ]
    for d, part, exp in tests:
        assert is_connected(part, d) == exp, f"d={d}, part={part}"


def test_count_vec_partitions():
    tests = [
        (0,),
        (0, 0, 0),
        (1,),
        (5,),
        (1, 1),
        (1, 1, 1),
        (2, 2),
        (2, 1, 3),
        (4, 2, 1, 3),
        (2, 2, 2, 2),
        (4, 4, 4),
    ]
    for test in tests:
        assert count_vector_partitions(test) == len(list(vector_partitions(test)))
        assert count_vector_partitions(test, sum_all=True) == sum(
            len(list(vector_partitions(u)))
            for u in product(*[range(0, test[i] + 1) for i in range(len(test))])
        )


def test_weak_compositions():
    tests = [
        (1, 1, [(1,)]),
        (1, 5, [(5,)]),
        (2, 3, [(0, 3), (1, 2), (2, 1), (3, 0)]),
        (3, 2, [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]),
        (
            3,
            3,
            [
                (0, 0, 3),
                (0, 1, 2),
                (0, 2, 1),
                (0, 3, 0),
                (1, 0, 2),
                (1, 1, 1),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
                (3, 0, 0),
            ],
        ),
    ]
    for m, s, expected in tests:
        assert set(weak_compositions(m, s)) == set(expected)


def test_vecs_sum_leq_k():
    tests = [
        (1, 1, [(0,), (1,)]),
        (1, 3, [(0,), (1,), (2,), (3,)]),
        (2, 2, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]),
        (2, 3, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)]),
        (
            3,
            2,
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 2),
                (0, 1, 0),
                (0, 1, 1),
                (0, 2, 0),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (2, 0, 0),
            ],
        ),
    ]
    for m, k, expected in tests:
        assert set(vecs_sum_leq_k(m, k)) == set(expected)


def test_down_set():
    tests = [
        ((), set()),
        ((1,), {(1,)}),
        ((2,), {(1,), (2,)}),
        ((1, 1), {(1,), (1, 1)}),
        ((2, 1), {(1,), (2,), (1, 1), (2, 1)}),
        ((2, 2), {(1,), (2,), (1, 1), (2, 1), (2, 2)}),
        ((3, 2), {(1,), (2,), (3,), (1, 1), (2, 1), (2, 2), (3, 1), (3, 2)}),
        ((3, 1, 1), {(1,), (2,), (3,), (1, 1), (2, 1), (3, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)}),
    ]
    for v, expected in tests:
        assert set(down_set(v)) == expected

def test_multigraph():
    # v, e, expected or count
    tests = [
        (0, 5, []),
        (1, 10, [[((0, 0), 10)]]),
        (2, 1, [[((0, 0), 1)], [((0, 1), 1)], [((1, 1), 1)]]),
        (10, 1, 55),
        (2, 2, [[((0, 0), 1), ((0, 1), 1)], [((0, 0), 1), ((1, 1), 1)], [((0, 0), 2)], [((0, 1), 1), ((1, 1), 1)], [((0, 1), 2)], [((1, 1), 2)]]),
    ]
    for v, e, expected in tests:
        graphs = multigraphs(v, e)
        for graph in graphs:
            assert sum(m for _, m in graph) == e
        if isinstance(expected, int):
            assert len(graphs) == expected
        else:
            assert sorted(graphs) == sorted(expected)

def test_int_partition_coef():
    tests = [
        (4,),
        (3, 1),
        (2, 2),
        (2, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1),
        (1,),
    ]
    for part in tests:
        assert len(list(int_to_set_partitions(part))) == int_partition_coef(part)

def test_vec_part_isomorphism():
    # p, q, vec, expected
    tests = [
        (
            ((2, 0), (1, 0)),
            ((0, 2), (0, 1)),
            None,
            True
        ),
        (
            ((2, 0), (1, 0)),
            ((0, 2), (1, 1)),
            None,
            False
        ),
        (
            ((0, 1, 2, 3),),
            ((2, 1, 3, 0),),
            None,
            True
        ),
        (
            ((0, 1, 2, 3),),
            ((2, 1, 3, 1),),
            None,
            False
        ),
        (
            ((1, 1),),
            ((1, 1, 1),),
            None,
            False
        ),
        (
            ((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1)),
            ((1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)),
            None,
            False
        ),
        (
            ((1, 1, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1)),
            ((1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)),
            None,
            True
        ),
        (
            ((1, 1, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1)),
            ((1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)),
            (2, 1, 1, 1),
            False
        ),
        (
            ((1, 1, 0, 0),),
            ((0, 0, 1, 1),),
            (1, 1, 2, 2),
            False
        ),
        (
            ((1, 1, 0, 0),),
            ((0, 0, 1, 1),),
            (2, 1, 2, 1),
            True
        ),
    ]
    for p, q, vec, expected in tests:
        assert vec_part_isomorphic(p, q, vec=vec) == expected

def test_vec_part_isos():
    vec_parts = [
        ((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1)),
        ((1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)),
        ((1, 1, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1)),
    ]
    isos = vec_part_isos(vec_parts)
    assert len(isos) == 2
    assert list(sorted(isos.values())) == [1, 2]
    isos = vec_part_isos(vec_parts, vec=(2, 1, 1, 1))
    assert len(isos) == 3
    assert list(isos.values()) == [1, 1, 1]
    
    terms = get_all_terms(3)
    terms = [vec_part for int_part, vec_part in terms if int_part == (1, 1, 1)]
    isos = vec_part_isos(terms)
    keys = list(isos.keys())
    for i, j in product(range(len(keys)), repeat=2):
        if i >= j:
            continue
        assert not vec_part_isomorphic(keys[i], keys[j])


    vec = (2, 2, 1)
    isos = vec_part_isos(terms, vec=vec)
    keys = list(isos.keys())
    for i, j in product(range(len(keys)), repeat=2):
        if i >= j:
            continue
        assert not vec_part_isomorphic(keys[i], keys[j], vec=vec)
