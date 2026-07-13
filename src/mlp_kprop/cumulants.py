import itertools
import logging
import math
from collections.abc import Callable, Generator, Iterator

import torch
from jaxtyping import Float
from torch import Tensor

from mlp_kprop.diagslice import *
from mlp_kprop.mlp import MLP
from mlp_kprop.partitions import set_partitions
from mlp_kprop.tensor_utils import *
from mlp_kprop.logging_utils import *

logger = logging.getLogger(__name__)

type Samples = Float[Tensor, "samples n"]  # Note this is the transpose of the torch.cov convention
type SampleStream = Iterator[Samples]  # samples -> (samples n)


def part_sum(
    A: Tower, coef: Callable[[SetPartition[int]], float], d: int | None = None
) -> Tower | Float[Tensor, "*n"]:
    """
    Transforms A by summing over set partitions with given coefficient function.
    I.e., returns tA[i_1,...,i_k] = sum_pi coef(pi) prod_{B in pi} A[B].
    """
    if d is None:
        return {d: part_sum(A, coef, d) for d in A}

    for i in range(1, d + 1):
        assert i in A, f"A must contain all orders up to d={d}."

    n = A[1].shape[0]
    dtype, device = A[1].dtype, A[1].device
    out = torch.zeros([n] * d, dtype=dtype, device=device)

    for part in set_partitions(d):
        out += coef(part) * math.prod(expand(A[len(block)], block, d) for block in part)

    assert list(out.shape) == [n] * d
    return out


def M_to_K(M: Tower, d: int | None = None) -> Tower | Float[Tensor, "*n"]:
    """
    Converts moment tensors M to dth cumulant tensor K via Moebius inversion.
    """
    coef = lambda part: math.factorial(len(part) - 1) * ((-1) ** (len(part) - 1))
    return part_sum(M, coef, d)


def K_to_M(K: Tower, d: int | None = None) -> Tower | Float[Tensor, "*n"]:
    """
    Converts cumulant tensors K to dth moment tensor M via standard partition expansion.
    """
    return part_sum(K, lambda part: 1, d)


def DS_part_sum(A: DSTower, coef: Callable[[VecPartition], float], strict: bool = True) -> DSTower:
    """
    Same as part_sum, but for DSTensors, computed per diagonal slice.
    We compute precisely the diagonal slices of the output that appear in A.
    Note that A has the necessary diagonal slices for this if it is downward closed.
    Partitions of a diagonal slice (n_1, ..., n_k) correspond to set partitions of sqcup_{i=1}^k [(i, 1),...,(i,n_i)],
    which correspond to vector partitions of (n_1, ..., n_k) multiplied by the fiber size vec_part_coef(divide_fac=False).
    """
    if strict:
        assert A.is_downward_closed(), "A must be downward closed."
    R = DSTower()  # to return

    def get_block(block: Vec) -> Float[Tensor, "*n"]:
        # Handle zero indices by dropping and expanding
        nonzeros = tuple([i for i, x in enumerate(block) if x > 0])
        block_nz = tuple(block[i] for i in nonzeros)
        try:
            # TODO: Change DSTensor.get_slice to return scalar 0 if slice is missing instead of full tensor (for efficiency)
            return expand(A[sum(block)].get_slice(block_nz, strict=True), nonzeros, len(block))
        except:
            assert not strict, f"Missing slice {block} in A."
            return 0.

    for d in range(1, max(A.keys()) + 1):
        Rd_slices = dict()
        for int_part in A[d].slices:
            Rd_slices[int_part] = torch.zeros_like(A[d].slices[int_part])
            for vpart in vector_partitions(int_part):
                Rd_slices[int_part] += (
                    math.prod(get_block(block) for block in vpart)
                    * vec_part_coef(vpart, divide_fac=False)
                    * coef(vpart)
                )
        R[d] = DSTensor(Rd_slices, autozero=True)
    return R


def DS_K_to_M(K: DSTower) -> DSTower:
    """
    Cumulants to moments via the usual partition formula, applied to each diagonal slice separately.
    We compute precisely the diagonal slices of M that appear in K.
    Note that K has the necessary diagonal slices for this if it is downward closed.
    Partitions of a diagonal slice (n_1, ..., n_k) correspond to set partitions of sqcup_{i=1}^k [(i, 1),...,(i,n_i)],
    which are vector partitions of (n_1, ..., n_k) multiplied by the fiber size vec_part_coef(divide_fac=False).
    """
    return DS_part_sum(K, lambda vpart: 1)


def DS_M_to_K(M: DSTower) -> DSTower:
    """
    Moments to cumulants via the usual Moebius inversion, applied to each diagonal slice separately.
    We use the same strategy as DS_K_to_M, just with the Moebius inversion coefficients.
    """
    coef = lambda part: math.factorial(len(part) - 1) * ((-1) ** (len(part) - 1))
    return DS_part_sum(M, coef)


def DS_pK_to_M(pK: DSTower) -> DSTower:
    """
    Converts power cumulants pK to moments M as a DSTensor.
    We compute precisely the diagonal slices of M that appear in pK.
    We form the moments via the standard partition formula applied to *each diagonal slice separately*,
        e.g. E[X_1^2 X_2 X_3] is treated as an entry in the 3rd moment tensor of (X^2, X, X), as opposed to the 4th of (X, X, X, X).
    """
    assert pK.is_downward_closed(), "pK must be downward closed."
    M = DSTower()
    for d in range(1, max(pK.keys()) + 1):
        # Compute moments for each partition separately
        # By symmetry we only need to compute one per integer partition
        Md_slices = dict()
        for int_part in pK[d].slices:
            # Do K_to_M formula, using the correct multiplicities of each pK. i.e., partition int_part itself
            Md_slices[int_part] = torch.zeros_like(pK[d].slices[int_part])
            for blocks in set_partitions(
                tuple(enumerate(int_part))
            ):  # need to track int block idx to know how to expand
                Md_slices[int_part] += math.prod(
                    expand(
                        pK.get_slice(sorted(tuple(t[1] for t in B), reverse=True)),
                        tuple(t[0] for t in B),  # idxs of entries in block B
                        len(int_part),
                    )
                    for B in blocks
                )
        M[d] = DSTensor(Md_slices, autozero=True)
    return M

@flop_name('pK_to_K')
def _DS_pK_to_K_old(pK: DSTower) -> DSTower:
    """
    Converts power cumulants pK to cumulants K as a DSTensor.
    We first convert pK to moments M via DS_pK_to_M, then M to K via DS_M_to_K.
    """
    # We want to skip as many (1, ..., 1) computations as possible, since pK->K is a no-op on them.
    # We greedily remove as many as possible while maintaining that pK is downward closed (which is required by pK->M).
    pK = pK.clone()
    ones_slices = dict()
    for d in sorted(pK.keys(), reverse=True):
        ones = (1,) * d
        if ones not in pK[d].slices:
            continue
        ones_slices[d] = pK[d].slices[ones]
        pK[d].slices.pop(ones)
        if not pK.is_downward_closed():
            # Put the slice back
            pK[d].slices[ones] = ones_slices[d]
            ones_slices.pop(d)
            break
    K = DS_M_to_K(DS_pK_to_M(pK))
    for d in ones_slices:
        K[d].slices[(1,) * d] = ones_slices[d]
    return K

@cache
def _pK_to_K_coef(vpart: VecPartition) -> float:
    '''
    Computes sum_{tau >= rho; sigma ∧ tau <= rho} mu(tau, 1)
    where mu is the Moebius function on the partition lattice.
    We think of (rho, sigma) as the diagram (i.e. vector partition) vpart in the usual way.
    '''
    def is_disconnected(tau: SetPartition[tuple[int]]) -> bool:
        for mblock in tau:
            for i in range(len(vpart[0])):
                if len([j for j in mblock if vpart[j][i] > 0]) > 1:
                    return False
        return True
    ret = 0
    for tau in set_partitions(len(vpart)):
        if is_disconnected(tau):
            ret += (-1) ** (len(tau) - 1) * math.factorial(len(tau) - 1)
    return ret

@flop_name('pK_to_K')
def DS_pK_to_K(pK: DSTower, strict=True) -> DSTower:
    """
    Does the pK -> K conversion directly.
    
    Composing the pK -> M and M -> K formulae yields:
        pK[X_{i_1},...,X_{i_d}] = sum_{rho} (prod_{B in rho} pK[X_{i_B}]) * (sum_{tau >= rho; sigma ∧ tau <= rho} mu(tau, 1))
    where sigma is the type of the partition (i_1, ..., i_d), and mu is the Moebius function on the partition lattice.
    """
    return DS_part_sum(pK, _pK_to_K_coef, strict=strict)

def stream_tensor(X: Samples, batch_size: int = 1024) -> SampleStream:
    """
    Create a stream of samples from a fixed sample tensor.
    """
    for i in range(0, X.shape[0], batch_size):
        yield X[i : i + batch_size, :]


def finish(gen, sentinel=None):
    try:
        gen.send(sentinel)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Generator did not stop")


def _moment_expr(d: int) -> str:
    idxs = [f"i{t}" for t in range(d)]
    left = ", ".join(f"j {idx}" for idx in idxs)
    right = " ".join(idxs)
    return f"{left} -> {right}"


def moment_gen_slice(
    part: int | IntPartition,
) -> Generator[None, Samples | None, Float[Tensor, "*n"]]:
    if isinstance(part, int):
        part = (1,) * part
    check_int_partition(part)
    logger.trace(f"Computing moment for part {part}")

    d = len(part)
    ret = None
    count = 0
    X = yield
    n, m = X.shape
    while True:
        if X is None:
            return ret / count
        assert X.shape[1] == m, "All batches must have same number of features."
        count += X.shape[0]
        if ret is None:
            ret = torch.zeros([m] * d, dtype=X.dtype, device=X.device)
            logger.trace(
                f"{ret.nelement() * ret.element_size() / 2**30:.2f} MiB allocated for moment tensor of part {part}"
            )
        ret += cached_einsum(*(X.pow(p) for p in part), _moment_expr(d))
        X = yield


def DS_moment_gen(
    part_cond: IntPartCond, d_max: int | None = None
) -> Generator[None, Samples | None, DSTower]:
    parts = part_cond.get_parts(d_max=d_max)
    logger.debug(f"Computing cumulants for {len(parts)} parts")

    inners = {p: moment_gen_slice(p) for p in parts}
    # Prime generators per slice
    for p in inners:
        next(inners[p])
    X = yield
    while True:
        if X is None:
            return DSTower.from_slices({p: finish(inners[p]) for p in inners}, autozero=True)
        for p in inners:
            inners[p].send(X)
        X = yield


def DS_cumulant_gen(
    part_cond: IntPartCond = trivial_int_cond, d_max: int | None = None
) -> Generator[None, Samples | None, DSTower]:
    parts = part_cond.get_parts(d_max=d_max)
    logger.debug(f"Computing cumulants for {len(parts)} parts")

    down = list(set().union(*(set(down_set(part)) for part in parts)))
    M = DS_moment_gen(part_cond=IntPartCond(parts=down))
    next(M)
    X = yield
    while True:
        if X is None:
            return DS_M_to_K(finish(M))
        M.send(X)
        X = yield


def DS_moment(
    X: Samples,
    part_cond: IntPartCond = trivial_int_cond,
    d_max: int | None = None,
    batch_size: int = 1024,
) -> DSTower:
    """
    Empirical raw moment tensor of powers of X.
    Streams over batches of size batch_size to avoid OOM.
    Note that this computes the sample moments, which are biased estimators of the population moments.
    """
    gen = DS_moment_gen(part_cond, d_max)
    next(gen)
    for Xb in stream_tensor(X, batch_size):
        gen.send(Xb)
    return finish(gen)


def DS_cumulant(
    X: Samples,
    part_cond: IntPartCond = trivial_int_cond,
    d_max: int | None = None,
    batch_size: int = 1024,
) -> DSTower:
    """
    Empirical dth-order cumulant of shape (n,)*d via the usual moment-cumulant Moebius inversion.
    Streams over batches of size batch_size to avoid OOM.
    Note that this computes the sample cumulants, which are biased estimators of the population cumulants.
    """
    gen = DS_cumulant_gen(part_cond, d_max)
    next(gen)
    for Xb in stream_tensor(X, batch_size):
        gen.send(Xb)
    return finish(gen)
