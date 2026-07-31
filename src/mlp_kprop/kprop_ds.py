import logging
import math
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from functools import partial
from itertools import product

import torch
from jaxtyping import Float
from numpy.polynomial import Polynomial
from torch import Tensor
from tqdm.auto import tqdm

from mlp_kprop.cumulants import *
from mlp_kprop.diagslice import *
from mlp_kprop.mlp import MLP
from mlp_kprop.partitions import *
from mlp_kprop.wick import *
from mlp_kprop.tensor_utils import *
from mlp_kprop.kprop_harmonic import (
    get_int_cond, get_vec_cond, get_all_terms, get_all_terms_iso, multiply_wicks
)

logger = logging.getLogger(__name__)

# NOTE: This is an deprecated version of the kprop algorithm. The current version is kprop_harmonic.py

"""
Given a k_max budget parameter, we have the following conditions:
    - get_int_cond(k_max): Tracked cumulants with multi-index a satisfy |ceil(a/2)| <= k_max
    - get_vec_cond(k_max): Diagrams summed over satisfy sum_a (|a| - 1 - 1[a even]) <= k_max - 1, where a ranges over all blocks in diagram
    - get_ein_cond(k_max): (Input, output) slices (a,b) satisfy |ceil((a∧b)/2)|<=k_max when contracting linear layers
"""

@cache
def get_ein_cond(k_max: int, extremal_only: bool = False):
    def ein_cond(parts: tuple[SetPartition[int], ...]) -> bool:
        K_part, out_part = parts[0], parts[-1]  # Ignore W parts in middle
        if extremal_only and len(K_part) > 1 and any(len(block) > 1 for block in out_part):
            return False
        a = 0
        # Compute |ceil((K_part∨out_part)/2)|
        for K_block, out_block in product(K_part, out_part):
            a += (len(K_block & out_block) + 1) // 2  # ceil(|K_block ∩ out_block| / 2)
            if a > k_max:
                return False
        return True

    return EinsumCond(einsum_cond=ein_cond)

def expand_block(K: DSTower, block: Vec, use_mean_var: bool = False) -> Float[Tensor, "*n"]:
    '''
    Let k = len(block). Returns a tensor B such that
        B[i_0, ..., i_{k-1}] = K[(i_0,)*block[i_0] + ... + (i_{k-1},)*block[i_{k-1}]],
    whenever i_0, ..., i_{k-1} are all distinct, and zero otherwise.
    '''
    # Handle zero indices by dropping and expanding
    nonzeros = tuple([i for i, x in enumerate(block) if x > 0])
    block_nz = tuple(block[i] for i in nonzeros)
    ret = K[sum(block)].get_slice(block_nz, strict=False)
    # If use_mean_var we need to subtract out mean neuron variance from 2-loops
    if use_mean_var and len(nonzeros) == 1 and sum(block) == 2:
        var = K[2].get_slice((2,), strict=False)
        assert ret.shape == var.shape
        ret = ret - var.mean()
    return expand(ret, nonzeros, len(block))

def eval_part_old(
    K: DSTower,
    vec_part: list[Vec],
    d: int,
    get_block: Callable[[DSTower, Vec], Float[Tensor, "*n"]],
    use_mean_var: bool = False,
) -> Float[Tensor, "*n"]:
    """
    Evaluate the contribution from a single vector partition, *excluding* the Wick coefficient
    (which depends only on the vector being partitioned, not the particular partition),
    but including the combinatorial coefficient.
    """
    check_vec_partition(vec_part, d)
    n = K[1].n
    if any(sum(v) not in K for v in vec_part):
        return None
    if not vec_part:
        # Edge case: empty partition returns all-ones tensor
        return vec_part_coef(vec_part, divide_fac=True) * torch.ones(
            (n,) * d, device=K[1].device, dtype=K[1].dtype,
        )
    # Avoid allocating an n^d ones tensor as math.prod start value
    factors = [get_block(K, v, use_mean_var=use_mean_var) for v in vec_part]
    result = factors[0]
    for f in factors[1:]:
        result = result * f
    return vec_part_coef(vec_part, divide_fac=True) * result

def nonlin_kprop(
    K_in: DSTower,
    nonlin_wick_coef: Callable[[float, float, int, int], float],
    k_max: int,
    use_mean_var: bool = False,
) -> DSTower:
    """
    Propagate cumulants through nonlinearity.
    We first compute mixed cumulants via Wick expansion around a Gaussian with matching mean and variance
    (so the sum is over 2-mixed partitions); then we convert back to ordinary cumulants.

    Args:
        K_in: Input cumulants
        k_max: Budget parameter. We want final error O(n^{-k_max}). This corresponds to:
            - Tracking cumulants with multi-index a satisfying |ceil(a/2)| <= k_max
            - Sum precisely the diagrams including cumulants with sum_a (|a| - 1 - 1[a even]) <= k_max - 1
            - NOTE: Internally, we also use k to refer to the index in the Wick expanison. This is unrelated to k_max.
        nonlin_wick_coef: 1d Wick coefficients wrt a Gaussian. (mean, var, k, p) -> E_{Z~N(mean,var)}[∂^k nonlin(Z)^p]
        use_mean_var: If true, expands Wick coef around Gaussian with var equal to the average variance estimate across neurons,
            as opposed to the estimated variance for that specific neuron.
            This is a proxy for using the analytic expected variance over weights.

    Returns:
        K_out: Output cumulants
    """
    if use_mean_var:
        logger.debug("Using mean variance over neurons for Wick expansion")

    int_cond = get_int_cond(k_max)
    K_in = K_in.coerce(part_cond=int_cond).clone()
    mean = K_in[1].get_slice((1,), strict=False)
    var = K_in[2].get_slice((2,), strict=False)
    if use_mean_var:
        var = var.mean(dim=0, keepdim=True)

    @cache
    def get_wick_coef(k: int, p: int) -> Float[Tensor, "n"]:
        return nonlin_wick_coef(mean=mean, var=var, k=k, p=p)

    # 1. Compute pK
    terms_iso = get_all_terms_iso(k_max)
    terms_iso = [
        (int_part, vec_part, count)
        for int_part, vec_part_dict in terms_iso.items()
        for vec_part, count in vec_part_dict.items()
    ]
    pK_slices = defaultdict(lambda: 0.0)
    for int_part, vec_part, count in tqdm(
        terms_iso,
        disable=logger.getEffectiveLevel() > logging.INFO,
        desc="nonlin-kprop",
    ):
        term = eval_part_old(K_in, vec_part, len(int_part), get_block=expand_block, use_mean_var=use_mean_var)
        if term is None:
            continue
        pK_slices[int_part] += count * multiply_wicks(
            term,
            check_vec_partition(vec_part, len(int_part)),  # check_vec_partition returns sum of partition vectors
            p=int_part,
            wick_lookup=get_wick_coef,
        )
    # Since we sum over iso classes * count instead of all terms, each slice is not symmetric wrt its int_part
    # So we symmetrize here
    for int_part in pK_slices:
        pK_slices[int_part] = symmetrize(pK_slices[int_part], vec=int_part)
    pK_out = DSTower.from_slices(pK_slices, autozero=True)

    # 2. Convert pK to K
    return DS_pK_to_K(pK_out)


def relu_kprop(K_in: DSTower, k_max: int, use_mean_var: bool = False) -> DSTower:
    return nonlin_kprop(
        K_in, nonlin_wick_coef=relu_wick_coef, k_max=k_max, use_mean_var=use_mean_var
    )


def poly_kprop(
    K_in: DSTower, poly: Polynomial, k_max: int, use_mean_var: bool = False
) -> DSTower:
    return nonlin_kprop(
        K_in, nonlin_wick_coef=partial(poly_wick_coef, poly), k_max=k_max, use_mean_var=use_mean_var
    )


def linear_kprop(
    K: DSTower,
    W: Float[Tensor, "out_dim in_dim"],
    k_max: int,
    extremal_only: bool = False,
    return_part_contributions: bool = False,
    **kwargs,
) -> DSTower | tuple[DSTower, dict[int, dict[IntPartition, dict[tuple[SetPartition[int], ...], Tensor]]]]:
    """
    Contracts each K[d] with W along each axis.
    - Computes diagonal slices with sum_a (|ceil(a/2)|) <= k_max
    - Does the contraction for (input, output) slices (a,b) satisfying |ceil((a∧b)/2)|<=k_max
    Args:
        return_part_contributions: If True, also return the raw per-partition tensors supplied by
            DSTensor.einsum for each order d during the linear step.

    Returns:
        Either the propagated cumulants (DSTower) or, if return_part_contributions is True,
        a tuple of (DSTower, {d: {out_part: {(in_parts): tensor}}}).
    """
    part_cond = get_int_cond(k_max)
    ein_cond = get_ein_cond(k_max, extremal_only=extremal_only)
    in_dim = W.shape[1]
    K = K.coerce(part_cond=part_cond, dim=in_dim).clone()
    einsum_part_contribs: dict[
        int, dict[IntPartition, dict[tuple[SetPartition[int], ...], Tensor]]
    ] | None = {} if return_part_contributions else None

    for d, Kd in K.items():
        Kd_idx = " ".join(f"i{t}" for t in range(d))
        W_idx = ", ".join(f"j{i} i{i}" for i in range(d))
        out_idx = " ".join(f"j{i}" for i in range(d))
        einsum_result = DSTensor.einsum(
            Kd,
            *(W,) * d,
            f"{Kd_idx}, {W_idx} -> {out_idx}",
            ein_cond=ein_cond,
            out_cond=part_cond,
            in_symmetric=True,
            return_part_contributions=return_part_contributions,
        )
        if return_part_contributions:
            assert isinstance(einsum_result, tuple)
            K[d], contribs = einsum_result
            einsum_part_contribs[d] = contribs
        else:
            assert isinstance(einsum_result, DSTensor)
            K[d] = einsum_result

    if return_part_contributions:
        assert einsum_part_contribs is not None
        return K, einsum_part_contribs
    return K


def mlp_kprop(
    mlp: MLP,
    K_in: DSTower,
    k_max: int,
    use_mean_var: bool = False,
    output_all: bool = False,
    extremal_only: bool = False,
    **kwargs,
) -> DSTower | dict[str, DSTower]:
    """
    Cumulant propagation through MLP layers.

    Args:
        mlp: MLP instance
        K_in: dictionary mapping cumulant order d to input cumulant tensor of shape (n, ..., n)
        k_max: budget parameter. We want final error O(n^{-k_max}).
        output_all: whether to return cumulants at all layers or just the final layer
            If output_all, then final layer cumulants are stored under key 'pre{num_layers-1}'.

    Returns:
        K_out: Dictionary mapping cumulant order d to cumulant tensor of shape (num_layers, n, ..., n)
            where num_layers is the number of layers in the MLP and there are d axes of size n=hidden_dim.
    """
    if mlp.has_bias():
        raise NotImplementedError("kprop_ds.mlp_kprop currently does not support bias")
    if len(mlp.nonlin_names) > 0 and len(set(mlp.nonlin_names)) > 1:
        raise NotImplementedError("kprop_ds.mlp_kprop currently only supports a single nonlinearity kind for all layers")
    nonlin = mlp.nonlin_names[0] if len(mlp.nonlin_names) > 0 else "relu"
    if nonlin == "relu":
        nonlin_wick_coef = relu_wick_coef
    elif nonlin == "sgn":
        nonlin_wick_coef = sgn_wick_coef
    elif nonlin == "square":
        nonlin_wick_coef = partial(poly_wick_coef, Polynomial([0, 0, 1]))
    elif nonlin == "cube":
        nonlin_wick_coef = partial(poly_wick_coef, Polynomial([0, 0, 0, 1]))
    else:
        raise NotImplementedError("Unsupported nonlinearity for mlp_kprop")
    if mlp.layernorm:
        raise NotImplementedError("mlp_kprop currently does not support layernorm")

    int_cond = get_int_cond(k_max)

    K = K_in.coerce(part_cond=int_cond, dim=mlp.input_dim).clone()
    K_by_layer = OrderedDict()
    for l, W_module in enumerate(mlp.Ws):
        W = W_module.weight
        K = linear_kprop(K, W, k_max=k_max, extremal_only=extremal_only)
        if output_all:
            K_by_layer[f"pre{l}"] = K.clone()
        if l < len(mlp.nonlins):
            K = nonlin_kprop(
                K, nonlin_wick_coef=nonlin_wick_coef, k_max=k_max, use_mean_var=use_mean_var
            )
            if output_all:
                K_by_layer[f"act{l}"] = K.clone()
    if output_all:
        return K_by_layer
    else:
        return K
