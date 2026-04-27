import logging
import operator as _op
import pprint
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from functools import cache, partial
from itertools import combinations, permutations, product
from typing import Self

import einops
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm

from mlp_kprop.partitions import *
from mlp_kprop.tensor_utils import *

logger = logging.getLogger(__name__)


def _zero_repeated(A: Float[Tensor, "*n"]) -> Float[Tensor, "*n"]:
    """
    Zeros out all entries of A where some indices are equal. In-place.
    """
    for i, j in combinations(range(A.ndim), 2):
        A.diagonal(dim1=i, dim2=j).zero_()
    return A

@flop_name('zero_repeated')
def zero_repeated(A: Float[Tensor, "*n"]) -> Float[Tensor, "*n"]:
    """
    Returns a copy of A with all entries where some indices are equal zeroed out.
    """
    return _zero_repeated(A.clone())


def _diagslice_view(A: Float[Tensor, "*n"], part: SetPartition[int]) -> Float[Tensor, "*n"]:
    """
    Returns a view of the diagonal slice of symmetric tensor A corresponding to set partition part.
    We respect the ordering of blocks in part (Be careful! This is the only place where ordering of a partition matters.)
    """
    if sum(len(b) for b in part) == 0:
        assert A.ndim == 0
        return A
    U = check_set_partition(part)
    assert set(U) == set(range(A.ndim)), f"Partition {part} must be of [{A.ndim}]"
    stride = A.stride()
    new_stride = [sum(stride[i] for i in block) for block in part]
    return A.as_strided(size=(A.shape[0],) * len(part), stride=new_stride)


def _get_sizes(in_expr: str, tensors: list[Tensor]) -> dict[str, int]:
    """
    Infers the size of each index in the einsum expression from the input tensors.
    Returns a mapping from index name to size.
    """
    sizes = {}
    assert len(tensors) == len(in_expr.split(",")), (
        "Number of input tensors does not match einsum expression."
    )
    for i, (A, A_expr) in enumerate(zip(tensors, in_expr.split(","))):
        assert len(A_expr.strip(" ").split(" ")) == A.ndim, (
            f"Dims mismatch at input {i}: tensor has shape {A.shape} but einsum expr is '{A_expr}'."
        )
        for name, size in zip(A_expr.strip(" ").split(" "), A.shape):
            if name in sizes:
                assert sizes[name] == size, (
                    f"Index {name} has inconsistent sizes {sizes[name]} and {size}."
                )
            else:
                sizes[name] = size
    return sizes


def _einsum_delta(*tensors_and_expr: Tensor | str) -> Tensor:
    """
    Extends torch.einsum to support repeated indices in the output.
    Assumes that indices are space-separated.
    """
    tensors = tensors_and_expr[:-1]
    expr = tensors_and_expr[-1]
    assert "_dup" not in expr, f"_dup in einsum index names is not allowed: {expr}"
    in_expr, out_expr = expr.split("->")[0].strip(" "), expr.split("->")[1].strip(" ")

    # output idx name -> [positions in output with that idx]
    groups = defaultdict(list)
    out_expr_split = out_expr.split(" ")
    for p, name in enumerate(out_expr_split):
        groups[name].append(p)

    sizes = _get_sizes(in_expr, list(tensors))

    # 1. Do einsum as if repeated indices were separate
    dupped = ["" for _ in range(len(out_expr_split))]
    for name in groups:
        for i, p in enumerate(groups[name]):
            dupped[p] = str(name) + (f"_dup{i}" if i > 0 else "")
    no_dupped = [name for name in dupped if "_dup" not in name]
    dropped = [name for name in dupped if "_dup" in name]
    dupped, no_dupped = " ".join(dupped), " ".join(no_dupped)
    # 1.1 Einsum with duplicate output indices omitted
    no_dupped_expr = in_expr + " -> " + no_dupped
    result = cached_einsum(*(tensors + (no_dupped_expr,)))
    # 1.2 Repeat to get duplicate output indices
    result = einops.repeat(
        result,
        no_dupped + " -> " + dupped,
        **{name: sizes[name.split("_dup")[0]] for name in dropped},
    )

    # 2. Deal with repeated indices by masking out off-diagonals
    idxs = torch.meshgrid(
        *[torch.arange(sizes[name], device=result.device) for name in out_expr_split], indexing="ij"
    )
    mask = torch.ones_like(result, dtype=torch.bool)
    for group in groups.values():
        for p in group[1:]:
            mask &= idxs[group[0]] == idxs[p]
    return result * mask

@cache
def _merge_legs(einexpr: str, parts: tuple[SetPartition[int], ...]) -> str:
    """
    Merges legs in einexpr according to parts.
    NOTE: May return an expr with repeat indices in the output, which is not valid einsum syntax; e.g. 'a -> a a', equivalent to torch.diagflat.
    Thus the returned expression should always be used with _einsum_delta.

    Args:
        einexpr: An einsum expression like 'i a, i b, j c -> a b c'
        parts: A list of set partitions, one for each tensor (both input and output) in einexpr, in order of appearance.

    Returns:
        The einsum expression with legs merged according to parts.
        More precisely, the return expression computes the contribution to the contraction from the diagonal slices specified in parts,
        restricted to the diagonal slice parts[-1].
    """
    in_legs = [s.strip(" ").split(" ") for s in einexpr.replace("->", ",").split(",")]
    for part, legs in zip(parts, in_legs):
        assert sum(len(block) for block in part) == len(legs), (
            f"Partition {part} does not match number of legs {len(legs)}."
        )

    # Tensor t, block b, index name i, index position p
    tb_l = [(t, b) for t, part in enumerate(parts) for b in range(len(part))]
    tp_b_d = {
        (t, p): b for t, part in enumerate(parts) for b, block in enumerate(part) for p in block
    }
    tbi_l = [(t, tp_b_d[(t, p)], i) for t, legs in enumerate(in_legs) for p, i in enumerate(legs)]
    ret_legs = tuple(f"t{t}b{b}" for t, b in tb_l)
    merges = tuple(
        (f"t{t1}b{b1}", f"t{t2}b{b2}")
        for (t1, b1, i1), (t2, b2, i2) in product(tbi_l, repeat=2)
        if (t1, b1) < (t2, b2) and i1 == i2
    )
    parents = disjoint_set_union(ret_legs, merges)
    ret_legs = [[parents[f"t{t}b{b}"] for b in range(len(parts[t]))] for t in range(len(parts))]
    return ", ".join(" ".join(legs) for legs in ret_legs[:-1]) + " -> " + " ".join(ret_legs[-1])


class EinsumCond:
    """
    For DSTensor.einsum, we need a condition on which tuples of set partitions (*input, output) to include.
    """

    def __init__(
        self,
        einsum_cond: Callable[[tuple[SetPartition[int], ...]], bool] | None = None,
        parts_coefs: Iterator[tuple[tuple[SetPartition[int], ...], float]] | None = None,
    ):
        """
        Provide one of:
            einsum_cond: A function taking in (in_part_0, ..., in_part_k, out_part) and
                returning whether to include the contribution from in_parts to out_part in the einsum.
            parts_coefs: A zip of the argument partitions to include and the coefficient for each in the einsum.
        """
        self.einsum_cond = einsum_cond
        self.parts_coefs = parts_coefs

    def yield_parts(
        self,
        aritys: tuple[int, ...] | None = None,
        arg_conds: tuple[IntPartCond, ...] | None = None,
        out_symmetric: bool = True,
        in_symmetric: bool = False,
    ) -> Iterator[tuple[tuple[SetPartition[int], ...], float]]:
        """
        Yields tuples of the form
            ((*in_parts, out_part), coef)
        where (*in_parts, out_parts) corresponds to a term of the DSTensor.einsum expansion satisfying the cond
        and coef is the coefficient for that term in the expansion.
        coef is always 1 unless in_symmetric is True.
        Args:
            aritys: Tuple of arities for each input and output tensor.
            arg_conds: If provided, a tuple of IntPartCond indicating which partitions are present for each input tensor.
                and which partitions are desired for the output tensor.
                If not present, we assume all partitions are present, which may be slow.
            out_symmetric: If True, only yield output partitions that are unique up to integer partition equivalence.
            in_symmetric: If True, only yield input partitions that are unique up to vector partition (of output integer partition) equivalence
                NOTE: This is only correct to use for the einsum(K, W, ..., W) pattern from kprop.linear_kprop
                TODO: Make this more general and/or organize better?
        Returns:
            An iterator over tuples of set partitions.
        """

        @cache
        def get_parts_i(arity_i: int, cond_i: IntPartCond) -> Iterable[SetPartition[int]]:
            return sum((int_to_set_partitions(p) for p in cond_i.yield_parts(d=arity_i)), [])

        @cache
        def get_parts_coefs_sym_i(
            arity_i: int, cond_i: IntPartCond, out_int_part: IntPartition
        ) -> Iterable[tuple[SetPartition[int], float]]:
            """
            Returns representative input set partitions (and coefficients) unique under
            vector-partition equivalence induced by the canonical output partition of
            type out_int_part.
            """
            parts = get_parts_i(arity_i, cond_i)
            parts_coefs: list[tuple[SetPartition[int], float]] = []
            vec_parts = set()

            for part in parts:
                vpart = list(set_to_vec_partition(part, out_int_part))
                # Canonicalize ordering of vectors (treat as multiset)
                vpart = tuple(sorted(vpart, key=lambda v: (-sum(v),) + tuple(v)))
                if vpart in vec_parts:
                    continue
                vec_parts.add(vpart)
                coef = vec_part_coef(vpart, divide_fac=False)
                parts_coefs.append((part, coef))
            return parts_coefs

        if self.einsum_cond is not None:
            logger.debug(f"Yielding parts meeting einsum_cond for aritys={aritys}")
            if arg_conds is None:
                arg_conds = tuple(trivial_int_cond for _ in range(len(aritys)))
            assert len(aritys) == len(arg_conds), "aritys and arg_conds must have the same length."
            out_int_parts = set()
            for out_part in get_parts_i(aritys[-1], arg_conds[-1]):
                out_part = sort_set_partition(out_part)
                out_int_part = tuple(set_to_int_partition(out_part))
                if out_int_part not in out_int_parts:
                    out_int_parts.add(out_int_part)
                elif out_symmetric:
                    continue
                # in_symmetric logic assumes out_part is the canonical representative of its integer partition class
                out_part = int_to_canonical_set_partition(out_int_part)
                parts_coefs_per_i = []
                for i in range(len(aritys) - 1):
                    if (
                        in_symmetric and aritys[i] == aritys[-1]
                    ):  # Hacky but fine for einsum(K, W, ..., W)
                        parts_coefs_per_i.append(
                            get_parts_coefs_sym_i(aritys[i], arg_conds[i], out_int_part)
                        )
                    else:
                        parts_coefs_per_i.append(
                            (p, 1.0) for p in get_parts_i(aritys[i], arg_conds[i])
                        )

                for p_coefs in product(*parts_coefs_per_i):
                    ps, coefs = zip(*p_coefs)
                    ps = ps + (out_part,)
                    coef = math.prod(coefs)
                    if self.einsum_cond(ps):
                        yield ps, coef

        else:
            for ps, coef in self.parts:
                assert aritys is None or [sum(p) for p in ps] == list(aritys), (
                    "Parts do not match specified aritys."
                )
                yield ps, coef

    @cache
    def get_parts(
        self,
        aritys: tuple[int, ...] | None = None,
        arg_conds: tuple[IntPartCond, ...] | None = None,
        out_symmetric: bool = True,
        in_symmetric: bool = False,
    ) -> list[tuple[tuple[SetPartition[int], ...], float]]:
        return list(
            self.yield_parts(
                aritys=aritys,
                arg_conds=arg_conds,
                out_symmetric=out_symmetric,
                in_symmetric=in_symmetric,
            )
        )

    @cache
    def __call__(self, parts: tuple[SetPartition[int], ...]) -> bool:
        if self.einsum_cond is not None:
            return self.einsum_cond(parts)
        else:
            return parts in self.parts


trivial_einsum_cond = EinsumCond(einsum_cond=lambda parts: True)


class DSTensor:
    """
    A DSTensor (diagonally sliced tensor) is a symmetric tensor that has been decomposed into a sum of diagonal slices.
    Slices not present are understood to be zero.
    """

    # TODO: It's needlessly restrictive to only allow symmetric DSTensors.
    # Maybe have DSTensor store slices by SetPartition, and a subclass SymmetricDSTensor that stores by IntPartition.
    slices: dict[IntPartition, Float[Tensor, "*n"]]
    n: int
    d: int
    device: torch.device
    dtype: torch.dtype

    @flop_name('DSTensor constructor')
    def __init__(
        self,
        slices: dict[IntPartition, Float[Tensor, "*n"]],
        *,
        autozero: bool = False,
        n: int | None = None,
        d: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.slices = slices
        if not slices:
            assert n is not None and d is not None and device is not None and dtype is not None, (
                "Must provide n, d, device, dtype when initializing empty DSTensor."
            )
            self.n = n
            self.d = d
            self.device = device
            self.dtype = dtype
        else:
            part, dslice = next(iter(slices.items()))
            self.n = dslice.shape[0] if n is None else n
            self.d = sum(part) if d is None else d
            self.device = dslice.device if device is None else device
            self.dtype = dslice.dtype if dtype is None else dtype
            for part, dslice in slices.items():
                assert sum(part) == self.d, (
                    f"Partition {part} does not match DSTensor order {self.d}."
                )
                assert sorted(part, reverse=True) == list(part), (
                    f"Partition {part} must be in descending order."
                )
                assert dslice.shape == (self.n,) * len(part), (
                    f"Diagonal slice for partition {part} has incorrect shape {dslice.shape}."
                )
                assert dslice.device == self.device, (
                    "All diagonal slices must be on the same device."
                )
                assert dslice.dtype == self.dtype, "All diagonal slices must have the same dtype."
                if autozero:
                    _zero_repeated(dslice)
                # To avoid double counting, check that all repeated index entries in each slice are zero
                for i, j in combinations(range(dslice.ndim), 2):
                    assert not (dslice.diagonal(dim1=i, dim2=j).abs() > 1e-12).any()

    def __repr__(self) -> str:
        return f"DSTensor(n={self.n}, d={self.d}, slices={str(self.slices)})"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n,) * self.d

    @property
    def ndim(self):
        return self.d

    def item(self):
        assert self.n == 1, "Can only call item() on DSTensor with n=1."
        return self.to_tensor().item()

    def clone(self):
        slices = {part: dslice.clone() for part, dslice in self.slices.items()}
        return DSTensor(slices, n=self.n, d=self.d, device=self.device, dtype=self.dtype)

    def prune(self: Self) -> Self:
        """
        Removes zero slices from self.
        """
        self.slices = {
            part: dslice for part, dslice in self.slices.items() if dslice.abs().sum() > 1e-12
        }
        return self

    @staticmethod
    def from_tensor(A: Float[Tensor, "*n"], part_cond: IntPartCond = trivial_int_cond) -> Self:
        """
        Constructs a DSTensor from a full tensor by extracting all diagonal slices, filtered by part_cond.
        Note that this is only guaranteed to be exact if part_cond accepts all partitions.
        Otherwise, missing slices are zeroed.
        NOTE: Assumes A is symmetric.
        """
        A = A.clone()
        d = A.ndim
        int_to_set_d = get_int_to_set_d(d)
        slices = {}
        # Length induces a topological sort on the integer partition poset
        # which will be necessary for iteratively subtracting off slices in the correct order.
        for int_part in sorted(int_to_set_d, key=lambda p: len(p)):
            if not part_cond(int_part):
                continue
            slices[int_part] = diagslice(A, int_part)
            for set_part in int_to_set_d[int_part]:
                _diagslice_view(A, set_part).zero_()
        return DSTensor(slices)

    @flop_name('DSTensor.to_tensor')
    def to_tensor(self) -> Float[Tensor, "*n"]:
        ret = torch.zeros((self.n,) * self.d, device=self.device, dtype=self.dtype)
        for int_part, dslice in self.slices.items():
            set_part = int_to_canonical_set_partition(int_part)
            # Multiply by int_partition_coef to ensure diagslice(D.to_tensor(), int_part) == D.slices[int_part]
            # Be careful to account for this coef in downstream formulas
            coef = int_partition_coef(int_part)
            _diagslice_view(ret, set_part).add_(dslice * coef)
        return symmetrize(ret)

    def has_slice(self, part: IntPartition) -> bool:
        """
        Checks if the DSTensor is tracking the diagonal slice corresponding to integer partition part.
        """
        part = tuple(part)
        assert check_int_partition(part) == self.d, (
            f"Partition {part} does not match DSTensor order {self.d}."
        )
        sorted_part = tuple(sorted(part, reverse=True))
        return sorted_part in self.slices

    def get_slice(self, part: IntPartition, strict: bool = True) -> Float[Tensor, "*n"]:
        """
        Returns the diagonal slice corresponding to integer partition part.
        """
        part = tuple(part)
        assert check_int_partition(part) == self.d, (
            f"Partition {part} does not match DSTensor order {self.d}."
        )
        sorted_part = tuple(sorted(part, reverse=True))
        tmp_sorted_part = list(sorted_part)
        permutation = []
        for b in part:
            idx = tmp_sorted_part.index(b)
            permutation.append(idx)
            tmp_sorted_part[idx] = -1  # So we don't reuse the same idx
        if strict:
            assert sorted_part in self.slices, (
                f"DSTensor does not have diagonal slice for partition {part}."
            )
        # TODO: It'd be more efficient to just return scalar 0 and rely on broadcasting when the slice is not present
        if sorted_part not in self.slices:
            # Return (1,)*len(part) zeros tensor and let broadcasting handle it
            return torch.zeros((1,) * len(part), device=self.device, dtype=self.dtype)
        return self.slices[sorted_part].permute(permutation)

    def get_dslice(self, part: IntPartition) -> Float[Tensor, "*n"]:
        """
        Alias for get_slice.
        """
        return self.get_slice(part)

    def to(self, device: torch.device, dtype: torch.dtype = None) -> Self:
        """
        Moves the DSTensor to the specified device and/or dtype.
        """
        dtype = dtype if dtype is not None else self.dtype
        slices = {
            part: dslice.to(device=device, dtype=dtype) for part, dslice in self.slices.items()
        }
        return DSTensor(slices)

    @staticmethod
    def einsum(
        *tensors_and_expr: Self | Tensor | str,
        ein_cond: EinsumCond = trivial_einsum_cond,
        out_cond: IntPartCond = trivial_int_cond,
        in_symmetric: bool = False,
        return_part_contributions: bool = False,
    ) -> Self | tuple[
        Self, dict[IntPartition, dict[tuple[SetPartition[int], ...], Float[Tensor, "*n"]]]
    ]:
        """
        Performs einsum over DSTensors and/or Tensors, outputting a DSTensor.
        NOTE: Assumes that *output* is symmetric. DSTensor inputs must also be symmetric by definition, but Tensor inputs need not be.
        Args:
            tensors_and_expr: A sequence of either Tensors or DSTensors followed by an einsum expression string.
            out_cond: Takes in (out_int_part) and returns whether to include out_part in the output DSTensor.
            ein_cond: Takes in (in_part_0, ..., in_part_k, out_part) and returns whether to include the contribution from in_parts to out_part.
            return_part_contributions: If True, also return a mapping from each output integer partition
                to tensors contributed by every input set-partition tuple (in_parts, out_part). Stored tensors
                already include the combinatorial coefficient applied during accumulation.
        Returns:
            A DSTensor representing the result of the einsum. If return_part_contributions is True,
            also returns the contribution mapping described above.
        """
        # TODO: Symmetry is a needlessly restrictive condition.
        #   We should implement general einsum (summing over set partitions) and add a flag for if we know the output is symmetric.

        tensors = tensors_and_expr[:-1]
        expr = tensors_and_expr[-1]
        in_expr, out_expr = expr.split("->")[0].strip(" "), expr.split("->")[1].strip(" ")
        sizes = _get_sizes(in_expr, tensors)
        out_n = sizes[out_expr.split(" ")[0]]
        for name in out_expr.split(" "):
            assert sizes[name] == out_n, (
                "Output indices must all have the same dimension for symmetric DSTensor output."
            )

        # Number of legs per tensor including output
        aritys = [len(x.strip(" ").split(" ")) for x in expr.replace("->", ",").split(",")]
        assert len(tensors) + 1 == len(aritys), (
            "Number of input tensors does not match einsum expression."
        )

        def get_part(A: DSTensor | Tensor, set_part: SetPartition[int]) -> Float[Tensor, "*n"]:
            if isinstance(A, DSTensor):
                int_part = set_to_int_partition(set_part)
                return A.get_slice(int_part, strict=False)
            else:
                assert set_part == discrete_partition(A.ndim), (
                    "Non-DSTensor inputs must use the discrete partition."
                )
                return A

        device = tensors[0].device
        dtype = tensors[0].dtype
        slices: dict[IntPartition, Float[Tensor, "*n"]] = {}
        part_contribs: defaultdict[
            IntPartition, dict[tuple[SetPartition[int], ...], Float[Tensor, "*n"]]
        ] | None = None
        if return_part_contributions:
            part_contribs = defaultdict(dict)

        input_iparts = [
            ((1,) * A.ndim,) if not isinstance(A, DSTensor) else tuple(A.slices.keys())
            for A in tensors
        ]
        input_conds = [IntPartCond(parts=set(iparts)) for iparts in input_iparts]

        arg_parts_coefs = ein_cond.get_parts(
            aritys=tuple(aritys),
            arg_conds=tuple(input_conds) + (out_cond,),
            in_symmetric=in_symmetric,
            out_symmetric=True,
        )

        logger.debug(
            f"Computing einsum contributions for {len(arg_parts_coefs)} arg parts for aritys={aritys}."
        )
        for arg_parts, coef in tqdm(
            arg_parts_coefs,
            disable=logger.getEffectiveLevel() > logging.DEBUG
            and not (logger.getEffectiveLevel() == logging.INFO and len(arg_parts_coefs) > 30),
            desc=expr,
        ):
            arg_parts = tuple(sort_set_partition(part) for part in arg_parts)
            in_parts, out_part = arg_parts[:-1], arg_parts[-1]
            out_int_part = set_to_int_partition(out_part)
            in_tensors = [get_part(A, part) for A, part in zip(tensors, in_parts)]
            out_tensor = _einsum_delta(*(in_tensors + [_merge_legs(expr, arg_parts)]))
            _zero_repeated(out_tensor)
            if out_int_part not in slices:
                slices[out_int_part] = torch.zeros(
                    (out_n,) * len(out_part), device=device, dtype=dtype
                )
            contrib_tensor = coef * out_tensor
            slices[out_int_part] += contrib_tensor
            if part_contribs is not None:
                # Store the contribution attributable to these input partitions.
                part_contribs[out_int_part][in_parts] = contrib_tensor.clone()
        ret = DSTensor(
            slices, n=out_n, d=len(out_expr.strip(" ").split(" ")), device=device, dtype=dtype
        )
        if part_contribs is not None:
            return ret, {k: dict(v) for k, v in part_contribs.items()}
        return ret

    def _check_compat(self, other: Self) -> None:
        assert self.d == other.d, "DSTensors must have the same order."
        assert self.n == other.n, "DSTensors must have the same dimension."
        assert self.device == other.device, "DSTensors must be on the same device."
        assert self.dtype == other.dtype, "DSTensors must have the same dtype."

    def _binary(self, other, op):
        # Fast path for DSTensor.
        if isinstance(other, DSTensor):
            self._check_compat(other)
            parts = set(self.slices) | set(other.slices)
            return DSTensor(
                {p: op(self.slices.get(p, 0.0), other.slices.get(p, 0.0)) for p in parts},
                autozero=True,
            )

        # Scalars
        if isinstance(other, (int, float)):
            s = float(other)
            return DSTensor({p: op(t, s) for p, t in self.slices.items()}, autozero=True)

        # torch.Tensor by coercion
        if isinstance(other, torch.Tensor):
            assert tuple(other.shape) == self.shape, "Shape mismatch."
            if other.device != self.device or other.dtype != self.dtype:
                other = other.to(device=self.device, dtype=self.dtype)
            return self._binary(DSTensor.from_tensor(other), op)

        return NotImplemented

    def _rbinary(self, other, op):
        if isinstance(other, DSTensor):
            self._check_compat(other)
            parts = set(self.slices) | set(other.slices)
            return DSTensor(
                {p: op(other.slices.get(p, 0.0), self.slices.get(p, 0.0)) for p in parts},
                autozero=True,
            )

        if isinstance(other, (int, float)):
            s = float(other)
            return DSTensor({p: op(s, t) for p, t in self.slices.items()}, autozero=True)

        if isinstance(other, torch.Tensor):
            assert tuple(other.shape) == self.shape, "Shape mismatch."
            if other.device != self.device or other.dtype != self.dtype:
                other = other.to(device=self.device, dtype=self.dtype)
            return DSTensor.from_tensor(other)._binary(self, op)

        return NotImplemented

    def _unary(self, op):
        return DSTensor({p: op(t) for p, t in self.slices.items()}, autozero=True)

    def _iinplace(self, other, op):
        # Mutate in place for +=, -=, etc.
        if isinstance(other, DSTensor):
            self._check_compat(other)
            parts = set(self.slices) | set(other.slices)
            for p in parts:
                a = self.slices.get(p, 0.0)
                b = other.slices.get(p, 0.0)
                self.slices[p] = op(a, b)
                _zero_repeated(self.slices[p])
            return self

        if isinstance(other, (int, float)):
            s = float(other)
            for p in list(self.slices.keys()):
                self.slices[p] = op(self.slices[p], s)
                _zero_repeated(self.slices[p])
            return self

        if isinstance(other, torch.Tensor):
            assert tuple(other.shape) == self.shape, "Shape mismatch."
            if other.device != self.device or other.dtype != self.dtype:
                other = other.to(device=self.device, dtype=self.dtype)
            return self._iinplace(DSTensor.from_tensor(other), op)

        return NotImplemented

    def clamp(self, min=None, max=None):
        return DSTensor({p: t.clamp(min=min, max=max) for p, t in self.slices.items()}, autozero=True)



######################################
## Basic arith dunders for DSTensor ##
######################################

for _name, _fn in {
    "__add__": _op.add,
    "__sub__": _op.sub,
    "__mul__": _op.mul,
    "__truediv__": _op.truediv,
    "__floordiv__": _op.floordiv,
    "__pow__": _op.pow,
    "__mod__": _op.mod,
}.items():
    setattr(DSTensor, _name, lambda self, other, fn=_fn: self._binary(other, fn))
    setattr(DSTensor, _name.strip("__"), lambda self, other, fn=_fn: self._binary(other, fn))
    setattr(
        DSTensor,
        _name.replace("__", "__r", 1),
        lambda self, other, fn=_fn: self._rbinary(other, fn),
    )
    setattr(
        DSTensor,
        _name.replace("__", "__i", 1),
        lambda self, other, fn=_fn: self._iinplace(other, fn),
    )

for _name, _fn in {"__neg__": _op.neg, "__pos__": _op.pos, "__abs__": _op.abs}.items():
    setattr(DSTensor, _name, lambda self, fn=_fn: self._unary(fn))

##############################


class DSTower(MutableMapping[int, DSTensor]):
    """Mapping from tensor order to :class:`DSTensor`.

    Provides convenience constructors, arithmetic operations, and utilities for
    reasoning about collections of diagonal slices across orders.
    """

    def __init__(self, mapping: Mapping[int, DSTensor | Float[Tensor, "*n"]] | None = None) -> None:
        self._data: dict[int, DSTensor] = {}
        if mapping is not None:
            for degree, value in mapping.items():
                self[degree] = value

    # ------------------------------------------------------------------
    # Mapping protocol
    # ------------------------------------------------------------------
    def __getitem__(self, degree: int) -> DSTensor:
        return self._data[degree]

    def __setitem__(self, degree: int, value: DSTensor | Float[Tensor, "*n"]) -> None:
        if isinstance(value, DSTensor):
            assert value.d == degree, (
                f"DSTensor for degree {degree} has mismatched order {value.d}."
            )
            dst = value
        elif isinstance(value, torch.Tensor):
            dst = DSTensor.from_tensor(value)
            assert dst.d == degree, f"Tensor for degree {degree} has mismatched order {dst.d}."
        else:
            raise TypeError(f"Unsupported DSTower value type {type(value)!r} for degree {degree}.")

        self._data[degree] = dst

    def __delitem__(self, degree: int) -> None:
        del self._data[degree]

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"DSTower({self._data!r})"

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------
    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, degree: int, default=None):
        return self._data.get(degree, default)

    def get_slice(self, part: IntPartition) -> Float[Tensor, "*n"]:
        return self._data[sum(part)].get_slice(part)

    def get_dslice(self, part: IntPartition) -> Float[Tensor, "*n"]:
        return self.get_slice(part)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_tower(cls, tower: Tower) -> "DSTower":
        return cls({degree: DSTensor.from_tensor(tensor) for degree, tensor in tower.items()})

    @classmethod
    @flop_name('DSTower.from_slices')
    def from_slices(
        cls, slices: Mapping[IntPartition, Float[Tensor, "*n"]], *, autozero: bool = False
    ) -> "DSTower":
        """Build a ``DSTower`` from diagonal slices of possibly different orders."""
        ret: dict[int, dict[IntPartition, Float[Tensor, "*n"]]] = defaultdict(dict)
        for part, dslice in slices.items():
            ret[sum(part)][part] = dslice
        return cls({degree: DSTensor(parts, autozero=autozero) for degree, parts in ret.items()})

    # ------------------------------------------------------------------
    # Copy helpers
    # ------------------------------------------------------------------
    def clone(self) -> "DSTower":
        return DSTower({degree: dst.clone() for degree, dst in self.items()})

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "DSTower":
        if not self:
            return DSTower()

        ref_dtype = dtype if dtype is not None else next(iter(self.values())).dtype
        return DSTower(
            {degree: dst.to(device=device, dtype=ref_dtype) for degree, dst in self.items()}
        )

    def coerce(
        self,
        *,
        prune: bool = True,
        part_cond: IntPartCond = trivial_int_cond,
        dim: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> "DSTower":
        """Return a sanitized copy that satisfies shape and partition constraints."""

        if not self:
            return DSTower()

        reference = next(iter(self.values()))
        dim = dim if dim is not None else reference.n
        dtype = dtype if dtype is not None else reference.dtype
        device = device if device is not None else reference.device

        coerced: dict[int, DSTensor] = {}
        for degree, tensor in self.items():
            assert tensor.d == degree, f"K[{degree}] should have order {degree}, got {tensor.d}"
            assert tensor.n == dim, f"K[{degree}] should have dimension {dim}, got {tensor.n}"

            filtered_slices: dict[IntPartition, Float[Tensor, "*n"]] = {}
            for part, dslice in tensor.slices.items():
                if not part_cond(part):
                    logger.debug(
                        f"Removing K[{degree}] slice for partition {part} due to part_cond."
                    )
                    continue
                filtered_slices[part] = dslice.to(device=device, dtype=dtype)

            coerced[degree] = DSTensor(
                filtered_slices,
                n=reference.n,
                d=degree,
                device=reference.device,
                dtype=reference.dtype,
            )
        ret = DSTower(coerced)
        if prune:
            ret.prune()
        return ret

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def _binary(self, other, op):
        if isinstance(other, DSTower):
            degrees = set(self.keys()) | set(other.keys())
            result = {}
            for degree in degrees:
                left = self._data.get(degree)
                right = other._data.get(degree)
                if left is None and right is None:
                    continue
                if left is None:
                    result[degree] = right._rbinary(0.0, op)
                elif right is None:
                    result[degree] = left._binary(0.0, op)
                else:
                    result[degree] = left._binary(right, op)
            return DSTower(result)

        if isinstance(other, (int, float)):
            scalar = float(other)
            return DSTower({degree: dst._binary(scalar, op) for degree, dst in self.items()})

        if isinstance(other, torch.Tensor):
            raise TypeError(
                "Pointwise operations between DSTower and Tensor are ambiguous; convert to DSTensor first."
            )

        return NotImplemented

    def _rbinary(self, other, op):
        if isinstance(other, DSTower):
            degrees = set(self.keys()) | set(other.keys())
            result = {}
            for degree in degrees:
                left = other._data.get(degree)
                right = self._data.get(degree)
                if left is None and right is None:
                    continue
                if left is None:
                    result[degree] = right._rbinary(0.0, op)
                elif right is None:
                    result[degree] = left._binary(0.0, op)
                else:
                    result[degree] = left._binary(right, op)
            return DSTower(result)

        if isinstance(other, (int, float)):
            scalar = float(other)
            return DSTower({degree: dst._rbinary(scalar, op) for degree, dst in self.items()})

        if isinstance(other, torch.Tensor):
            raise TypeError(
                "Pointwise operations between Tensor and DSTower are ambiguous; convert to DSTensor first."
            )

        return NotImplemented

    def _unary(self, op):
        return DSTower({degree: dst._unary(op) for degree, dst in self.items()})

    def _iinplace(self, other, op):
        if isinstance(other, DSTower):
            degrees = set(self.keys()) | set(other.keys())
            for degree in degrees:
                left = self._data.get(degree)
                right = other._data.get(degree)
                if left is None and right is None:
                    continue
                if left is None:
                    self._data[degree] = right._rbinary(0.0, op)
                elif right is None:
                    self._data[degree] = left._binary(0.0, op)
                else:
                    self._data[degree] = left._binary(right, op)
            return self

        if isinstance(other, (int, float)):
            scalar = float(other)
            for degree in list(self.keys()):
                self._data[degree] = self._data[degree]._binary(scalar, op)
            return self

        if isinstance(other, torch.Tensor):
            raise TypeError(
                "Pointwise operations between DSTower and Tensor are ambiguous; convert to DSTensor first."
            )

        return NotImplemented

    # ------------------------------------------------------------------
    # Utilities ---------------------------------------------------------
    # ------------------------------------------------------------------
    def prune(self):
        for A in self.values():
            A.prune()
        return self

    def is_downward_closed(self) -> bool:
        """Return ``True`` if every tracked slice has all of its sub-slices."""

        def decr(part: IntPartition, i: int) -> IntPartition:
            part_l = list(part)
            part_l[i] -= 1
            if part_l[i] == 0:
                part_l.pop(i)
            return tuple(sorted(part_l, reverse=True))

        all_parts = [part for _, tensor in self.items() for part in tensor.slices]
        for part in all_parts:
            if sum(part) <= 1:
                continue
            for i in range(len(part)):
                if decr(part, i) not in all_parts:
                    logger.warning(f"NOT DOWNWARD CLOSED: {part} -> {decr(part, i)}")
                    # logger.info("ALL PARTS:", all_parts)
                    return False
        return True

    def pprint(self) -> None:
        """Pretty-print the stored diagonal slices."""

        for degree in sorted(self.keys()):
            print(f"d={degree}:")
            pprint.pprint({part: self[degree].slices[part] for part in sorted(self[degree].slices)})
            print("-----")

    def to_tower(self) -> Tower:
        return {degree: dst.to_tensor() for degree, dst in self.items()}


# Register arithmetic dunder methods for DSTower ------------------------
for _name, _fn in {
    "__add__": _op.add,
    "__sub__": _op.sub,
    "__mul__": _op.mul,
    "__truediv__": _op.truediv,
    "__floordiv__": _op.floordiv,
    "__pow__": _op.pow,
    "__mod__": _op.mod,
}.items():
    setattr(DSTower, _name, lambda self, other, fn=_fn: self._binary(other, fn))
    setattr(
        DSTower, _name.replace("__", "__r", 1), lambda self, other, fn=_fn: self._rbinary(other, fn)
    )
    setattr(
        DSTower,
        _name.replace("__", "__i", 1),
        lambda self, other, fn=_fn: self._iinplace(other, fn),
    )

for _name, _fn in {"__neg__": _op.neg, "__pos__": _op.pos, "__abs__": _op.abs}.items():
    setattr(DSTower, _name, lambda self, fn=_fn: self._unary(fn))

def diagslice(A: Any, part: IntPartition, output_zero_repeated=False) -> Float[Tensor, "*n"]:
    """
    Returns a copy of the diagonal slice of symmetric tensor A corresponding to an integer partition part.
    A can be either a torch.Tensor or any object with a get_dslice method (e.g. DSTensor).
    If `output_zero_repeated`, zeroes out repeated indices in the diagonal slice.
    """
    assert sum(part) == A.ndim, "Partition does not match tensor order."
    if isinstance(A, Tensor):
        ret = _diagslice_view(A, int_to_canonical_set_partition(part)).clone()
        if output_zero_repeated:
            _zero_repeated(ret)
        return ret
    else:
        if output_zero_repeated:
            return zero_repeated(A.get_dslice(part))
        else:
            ret = A.get_dslice(part).clone()
            for meta in set_partitions(len(part)):
                if all(len(p) == 1 for p in meta):
                    continue
                # Form super-partition induced by meta-partition of part
                supr = []
                for block in meta:
                    supr.append(sum(part[i] for i in block))
                _diagslice_view(ret, meta).add_(zero_repeated(A.get_dslice(tuple(supr))))
            return ret

def expand_dslice(A: Any, vec: Vec, output_zero_repeated=True) -> Float[Tensor, "*n"]:
    """
    Let k = len(vec). Returns a tensor B such that
        B[i_0, ..., i_{k-1}] = A[(i_0,)*vec[i_0] + ... + (i_{k-1},)*vec[i_{k-1}]].
    if the i_0, ..., i_{k-1} are all distinct, and 0 otherwise.
    """
    nonzeros = [i for i, v in enumerate(vec) if v > 0]
    vec_nz = tuple(v for v in vec if v > 0)
    dslice = diagslice(A, vec_nz, output_zero_repeated=output_zero_repeated)
    return expand(dslice, nonzeros, len(vec))
            
@flop_name('eval_part')
def eval_part(
    K: dict[int, Any], vec_part: list[Vec], d: int, output_zero_repeated: bool = True
) -> Float[Tensor, "*n"]:
    """
    Evaluate the contribution from a single vector partition, *excluding* the Wick coefficient
    (which depends only on the vector being partitioned, not the particular partition),
    but including the combinatorial coefficient.
    TODO: More informative name?
    """
    check_vec_partition(vec_part, d)
    n = K[1].n
    if any(sum(v) not in K for v in vec_part):
        return None
    if not vec_part:
        # Edge case: empty partition returns all-ones tensor
        return torch.ones((n,) * d, device=K[1].device, dtype=K[1].dtype)
    factors = [expand_dslice(K[sum(v)], v, output_zero_repeated=output_zero_repeated) for v in vec_part]
    result = factors[0]
    for f in factors[1:]:
        result = result * f
    return vec_part_coef(vec_part, divide_fac=True) * result

def decompose_dslice(A: Float[Tensor, "*n"], part: IntPartition) -> DSTensor:
    """
    For A with possibly nonzero diagonals, returns a DSTensor B such that diagslice(B, part, output_zero_repeated=False) = A,
    and all other slices not in the up-set of part are zero.
    NOTE: Unused. (Might be potentially useful for kind=SIMPLE pK ablation if we want to do that.)
    """
    slices = {}
    for meta in set_partitions(len(part)):
        supr = []
        for block in meta:
            supr.append(sum(part[i] for i in block))
        if sorted(supr, reverse=True) != list(supr):
            continue
        S = _diagslice_view(A, meta).clone()
        if supr in slices:
            assert torch.allclose(supr[slices], S)
        slices[supr] = S
    return DSTensor(slices, n=A.shape[0], d=A.ndim, device=A.device, dtype=A.dtype, autozero=True)

