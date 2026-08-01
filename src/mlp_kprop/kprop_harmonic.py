import logging
import math
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from functools import partial
from typing import Optional
from enum import Enum

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
from mlp_kprop.harmonic import *

logger = logging.getLogger(__name__)

class Kind(Enum):
    OLD = 1
    SIMPLE = 2
    AUGMENT = 3
    BASE = 4    # Just for ablation tests; not expected to get good enough MSE.

OLD, SIMPLE, AUGMENT, BASE = Kind.OLD, Kind.SIMPLE, Kind.AUGMENT, Kind.BASE


@cache
def get_int_cond(k_max: int):
    def int_cond(int_part: IntPartition) -> bool:
        return sum(math.ceil(x / 2) for x in int_part) <= k_max

    return IntPartCond(part_cond=int_cond)


def vec_block_cost(v: Vec) -> int:
    """
    Power-counting cost of a cumulant block in the nonlinear-step diagram summation:
    per entry, |kappa_v| = O(n^{-cost/2}) since E[kappa_v^2] = O(n^{2-sum(v)}) when v is
    even and O(n^{1-sum(v)}) otherwise.
    """
    return sum(v) - 1 - all(x % 2 == 0 for x in v)


@cache
def get_vec_cond(k_max: int, augment: bool = False):
    # Include exactly the diagrams with k(nu) := 1 + sum_v cost(v) <= k_max
    # (equation (14) of the paper); dropped diagrams have squared size O(n^{-k_max}).
    # The augmented algorithm raises the cap by one, summing every diagram of squared
    # size Omega(n^{-k_max}), i.e. every diagram contributing at leading order to the
    # basic algorithm's MSE. This keeps the whole nonlinear step time-subleading
    # (O(n^{k_max}) vs the linear step's O(n^{k_max+1})) when unfactorized.
    budget = k_max - 1 + augment

    def vec_cond(vec_part: VecPartition) -> bool:
        return sum(vec_block_cost(v) for v in vec_part) <= budget

    return VecPartCond(part_cond=vec_cond)


@cache
def is_hypertree(vec_part: VecPartition) -> bool:
    """
    Whether the diagram's blocks form a hypertree (Berge-acyclic hypergraph) after
    merging parallel 2-support blocks into a single effective edge: pairwise support
    overlaps of at most one group, and sum(|support| - 1) == #groups - 1.
    Exactly these diagrams admit an O(n)-rank factored evaluation (see factor_k3/factor_k4);
    e.g. covariance triangles and kappa_top x edge products do not.
    """
    counts = Counter(frozenset(i for i, x in enumerate(v) if x) for v in vec_part)
    if any(c > 1 and len(s) != 2 for s, c in counts.items()):
        return False
    edges = list(counts)
    if any(len(a & b) >= 2 for i, a in enumerate(edges) for b in edges[i + 1 :]):
        return False
    verts = set().union(*edges) if edges else set()
    return sum(len(e) - 1 for e in edges) == max(len(verts) - 1, 0)


def factored_keeps_term(k_max: int, int_part: IntPartition, vec_part: VecPartition) -> bool:
    """
    Which terms the factored implementations (factor_k3/factor_k4) include.
    In augment mode this is a strict subset of the full term set: factored kind=AUGMENT
    drops the terms it cannot afford, namely
      - non-hypertree diagrams for the top slice int_part == (1,)*k_max, whose output
        must be produced in O(n)-rank factored form; and
      - diagrams that pointwise-multiply the all-distinct block (1,)*k_max of the
        degree-k_max cumulant with other blocks, since that input only exists in
        factored form (materializing it costs O(n^k_max * rank)).
    All other terms are evaluated densely within the factored FLOP budget.
    In simple/base mode this keeps everything (the paper-basic term set is all-hypertree).
    """
    if int_part == (1,) * k_max:
        return is_hypertree(vec_part)
    return len(vec_part) == 1 or (1,) * k_max not in vec_part


# TODO: Move somewhere else and rename to something more informative.
@cache
def get_all_terms(
    k_max: int,
    d_max: Optional[int] = None,
    use_mean_var: bool = False,
    augment: bool = False,
) -> Iterable[tuple[IntPartition, VecPartition]]:
    int_cond = get_int_cond(k_max)
    vec_cond = get_vec_cond(k_max, augment=augment)
    logger.debug("Enumerating all partitions and diagrams...")
    mix_cond = (
        (lambda vpart: is_mixed(vpart, m=1))
        if use_mean_var
        else (lambda vpart: is_mixed(vpart, m=2))
    )
    all_terms = []
    if d_max is None:
        d_max = 2 * k_max
    int_parts = int_cond.get_parts(d_max=d_max)
    block_cond = lambda block, d_max=d_max: sum(block) <= d_max
    pbar = tqdm(int_parts, desc="get_all_terms", disable=logger.getEffectiveLevel() > logging.DEBUG)
    for int_part in pbar:
        pbar.set_postfix({"int_part": int_part})
        for vec_part in vec_cond.get_parts(
            dim=len(int_part),
            # Admissible blocks satisfy sum(v) <= 2 * vec_block_cost(v), so vec_cond
            # already implies |k| <= 2 * budget. (For the basic algorithm this sits
            # within the |k| <= 2*k_max - 1 cap of equation (13) of the paper.)
            sum_max=2 * (k_max - 1 + augment),
        ):
            if (
                mix_cond(vec_part)
                and is_connected(vec_part, d=len(int_part))
                and all(block_cond(block) for block in vec_part)
            ):
                all_terms.append((int_part, vec_part))

    logger.debug(f"Enumerated {len(all_terms)} (int_part, vec_part) pairs.")
    return all_terms


@cache
def get_all_terms_iso(
    k_max: int,
    d_max: Optional[int] = None,
    use_mean_var: bool = False,
    augment: bool = False,
) -> dict[IntPartition, dict[VecPartition, int]]:
    terms = get_all_terms(k_max, d_max=d_max, use_mean_var=use_mean_var, augment=augment)
    ret = {}
    for int_part in set(t[0] for t in terms):
        vec_parts = [t[1] for t in terms if t[0] == int_part]
        ret[int_part] = vec_part_isos(vec_parts, vec=int_part, dim=len(int_part))
    return ret


def multiply_wicks(
    K_part: Float[Tensor, "*n"],
    k: Vec,
    p: Vec,
    wick_lookup: Callable[[int, int], Float[Tensor, "n"]],
) -> Float[Tensor, "*n"]:
    """
    Multiplies in the diagonal Wick coefficient tensors corresponding to E[∂^k nonlin(Z)^p].
    """
    d = len(k)
    assert d == K_part.dim()
    assert d == len(p)
    for axis, (k_i, p_i) in enumerate(zip(k, p)):
        wick_coef = wick_lookup(int(k_i), int(p_i))
        view_shape = [1] * d
        view_shape[axis] = -1
        K_part = K_part * wick_coef.reshape(view_shape)
    return K_part

@cache
def mean_field_var(layer: int) -> float:
    '''
    Computes the expectation wrt weights of the variance wrt inputs of the pre-activation at l, in the infinite-width limit.
    Hardcoded for ReLU only. Unused.
    '''
    @cache
    def s(l):
        if l == 0:
            return 0.
        prev = s(l-1)
        return 2/np.pi * (np.sqrt(1 - prev**2 / 4) + (np.pi - np.arccos(prev / 2)) * prev / 2)
    return 1 - s(layer)/2

def get_r_x(d: int, k_max: int, kind: Kind = SIMPLE) -> int:
    """
    Given budget parameter k_max, we track the d-th cumulant going into the linear step
    as Sym(A otimes I^{otimes r_x}), where A is an order d-2*r_x tensor.
    Power counting says r_x = d-k_max.
    Return value of -1 means the entire cumulant should be discarded.
    """
    if kind == SIMPLE:
        if d > k_max:
            if d == k_max + 1 and d % 2 == 0:
                return d // 2
            else:
                return -1
        else:
            return 0
    elif kind == AUGMENT:
        if d > k_max:
            if d == k_max + 1:
                return 1
            elif d == k_max + 2 and d % 2 == 0:
                return d // 2
            else:
                return -1
        else:
            return 0
    elif kind == OLD:
        r = max(d - k_max, 0)
        if 2 * r > d:
            return -1
        else:
            return r
    elif kind == BASE:
        if d > k_max:
            return -1
        return 0
    else:
        raise ValueError(f"Unknown kind: {kind}")


def get_d_max(k_max, kind: Kind) -> int:
    '''
    Returns the maximum cumulant degree tracked given budget parameter k_max and kind.
    '''
    if kind == SIMPLE:
        return k_max + 1 if k_max % 2 == 1 else k_max
    elif kind == AUGMENT:
        return k_max + 2 if k_max % 2 == 0 else k_max + 1
    elif kind == BASE:
        return k_max
    else:
        # Maximum possible degree of diagslice satisfying ceil(alpha/2) int_cond is 2*k_max
        return 2 * k_max

def _coerce_layer_bias(
    bias: Optional[Float[Tensor, "out_dim"]],
    *,
    out_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Float[Tensor, "out_dim"]]:
    if bias is None:
        return None
    bias = torch.as_tensor(bias, device=device, dtype=dtype)
    if bias.ndim != 1 or bias.shape[0] != out_dim:
        raise ValueError(f"bias must have shape ({out_dim},), got {tuple(bias.shape)}")
    return bias

def linear_kprop(
    K: HTower,
    W: Float[Tensor, "out_dim in_dim"],
    k_max: int,
    d_max: Optional[int] = None,
    *,
    set_metric: Optional[Tensor] = None,
    bias: Optional[Float[Tensor, "out_dim"]] = None,
) -> HTower:
    """
    Linear step of cumulant propagation: contracts each K[d] with W.
    Used before the nonlinear step.

    Args:
        set_metric: Metric to set on the output HTensors. If None, the metric
            is computed as W @ old_metric @ W^T (see contract_W).
            Set to diag(E[WW^T]) = mlp.init_scale[layer]*I when using average metric.
    """
    n_out = W.shape[0]

    if set_metric is None and k_max == 1:
        # When k=1, the only non-loop edge is in the leading order (1, 1) partition for variance
        # which gets traced out in the projection after the nonlinearity, so we only need the diagonal.
        # This also saves us from going over budget when k=1.
        with flop_name('metric'):
            set_metric = W.pow(2).sum(dim=1)

    WK: HTower = {}
    for d, K_d in K.items():
        if d_max is not None and d > d_max:
            continue
        if isinstance(K_d, HTensor):
            assert K_d.has_identity_metric(), f"linear_kprop expects identity metric on input HTensors, got {K_d}"
            WK[d] = K_d.contract_W(W, set_metric=set_metric)
        elif hasattr(K_d, "contract_W"):
            WK[d] = K_d.contract_W(W)
        else:
            raise TypeError(f"Unsupported tensor type in linear_kprop: {type(K_d)!r}")

    if bias is not None:
        bias_vec = _coerce_layer_bias(
            bias,
            out_dim=n_out,
            device=WK[1].core.device,
            dtype=WK[1].core.dtype,
        )
        if bias_vec is not None:
            WK[1] = WK[1].clone()
            WK[1].core = WK[1].core + bias_vec
    return WK


def nonlin_kprop(
    K_in: HTower,
    nonlin_wick_coef: Callable[[float, float, int, int], float],
    k_max: int,
    kind: Kind = SIMPLE,
    use_pK: bool = True,
    factor: bool = False,
) -> HTower:
    """
    Propagate cumulants through nonlinearity.
    We first compute power cumulants via Wick expansion around a Gaussian with matching mean and variance
    (so the sum is over 2-mixed partitions); then we convert back to ordinary cumulants.

    Args:
        K_in: Input cumulants (from the linear step; may have non-identity metric)
        k_max: Budget parameter. We want final error O(n^{-k_max}). This corresponds to:
            - See get_r_x for how k_max determines which pieces of the harmonic decomposition we track.
        nonlin_wick_coef: 1d Wick coefficients wrt a Gaussian. (mean, var, k, p) -> E_{Z~N(mean,var)}[∂^k nonlin(Z)^p]
        factor: Use a factorized representation for the top-degree cumulant.
            Only supported for k_max=3 or 4.
            WARNING: with kind=AUGMENT, factor=True is NOT equivalent to factor=False:
            the factored algorithm drops the augmented diagrams it cannot afford
            (see factored_keeps_term), which have the same Theta(n^-k_max) squared
            size as the augmented diagrams it keeps.

    Returns:
        K_out: Output cumulants (with identity metric)
    """
    if not use_pK and kind != BASE:
        # TODO: If we really want to, we can ablate use_pK separately from kind=BASE
        # by computing dslices of the nonlin expansion using some block-merging logic
        raise NotImplementedError("not use_pK ablation only implemented for kind=BASE.")

    n = K_in[1].n

    if factor:
        if k_max > 4:
            raise NotImplementedError("Factored nonlin_kprop only implemented for k_max=3 or 4")
        assert kind in (SIMPLE, AUGMENT, BASE), "Factored nonlin_kprop only implemented for kind=SIMPLE, AUGMENT, or BASE"
        if k_max == 3:
            from mlp_kprop.factor_k3 import factored_nonlin_kprop_k3
            return factored_nonlin_kprop_k3(
                K_in=K_in,
                nonlin_wick_coef=nonlin_wick_coef,
                augment=(kind==AUGMENT),
                base=(kind==BASE),
                use_pK=use_pK,
            )
        elif k_max == 4:
            from mlp_kprop.factor_k4 import factored_nonlin_kprop_k4
            return factored_nonlin_kprop_k4(
                K_in=K_in,
                nonlin_wick_coef=nonlin_wick_coef,
                augment=(kind==AUGMENT),
                base=(kind==BASE),
                use_pK=use_pK,
            )
        else:
            logger.debug("nonlin_kprop with factor=True called with k_max<=2. Identical to unfactored.")

    # 1. Get propagated mean and variance
    with flop_name('get_mean_var'):
        assert K_in[1].r == 0
        mean = K_in[1].core
        if k_max == 1:
            if 2 not in K_in:
                # This only happens on k=1 kind=BASE
                var = torch.ones_like(mean)
            else:
                assert K_in[2].r == 1
                var_metric = K_in[2].metric
                var = K_in[2].core * (var_metric if var_metric.ndim == 1 else var_metric.diag())
        else:
            assert K_in[2].r == 0
            var = K_in[2].core.diag()
        assert mean.ndim == 1, "Mean must be a vector."
        assert var.ndim == 1, "Variance must be a vector."

    @cache
    @flop_name('get_wick_coef')
    def get_wick_coef(k: int, p: int) -> Float[Tensor, "n"]:
        return nonlin_wick_coef(mean=mean, var=var, k=k, p=p)

    # 2. Compute pK
    terms_iso = get_all_terms_iso(k_max, d_max=get_d_max(k_max, kind), augment=(kind == AUGMENT))
    terms_iso = [
        (int_part, vec_part, count)
        for int_part, vec_part_dict in terms_iso.items()
        for vec_part, count in vec_part_dict.items()
        if all(p==1 for p in int_part) or use_pK   # If not use_pK, only need (1, ..., 1) int_parts since we're not going to zero the diagonal
    ]
    pK_slices = defaultdict(lambda: 0.0)
    for int_part, vec_part, count in tqdm(
        terms_iso,
        disable=logger.getEffectiveLevel() > logging.INFO,
        desc="nonlin-kprop",
    ):
        with flop_name(f'nonlin_sum', factor=slice_factor(int_part, n=n)):
            term = eval_part(K_in, vec_part, len(int_part), output_zero_repeated=use_pK)
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
    with flop_name(f'symmetrize'):
        for int_part in pK_slices:
            pK_slices[int_part] = symmetrize(pK_slices[int_part], vec=int_part)

    # If not use_pK, pK_slices already contain our cumulant estimate. So immediately project to harmonic and return.
    if not use_pK:
        ret = {}
        for d in range(1, get_d_max(k_max, kind) + 1):
            part = (1,) * d
            if part not in pK_slices:
                continue
            ret[d] = proj_geq_r(pK_slices[part], n=n, r_out=get_r_x(d, k_max, kind=kind))
        return ret

    # 3. Convert pK to K
    pK_out_ds = DSTower.from_slices(pK_slices, autozero=True)
    K_out_ds = DS_pK_to_K(pK_out_ds)

    # 4. Project to harmonic form
    K_out: HTower = {}
    for d, K_d_ds in K_out_ds.items():
        r_x = get_r_x(d, k_max, kind=kind)
        if r_x == -1:
            continue
        K_out[d] = DS_harmonic_proj(K_d_ds, r_x)
    return K_out

def relu_kprop(K_in: HTower, k_max: int, kind: Kind=SIMPLE) -> HTower:
    return nonlin_kprop(
        K_in, nonlin_wick_coef=relu_wick_coef, k_max=k_max, kind=kind
    )

def poly_kprop(
    K_in: HTower, poly: Polynomial, k_max: int, kind: Kind=SIMPLE
) -> HTower:
    return nonlin_kprop(
        K_in, nonlin_wick_coef=partial(poly_wick_coef, poly), k_max=k_max, kind=kind
    )

@flop_name('coerce_input')
def coerce_input(
    K: dict[int, Float[Tensor, "*n"]], k_max: int, kind: Kind=SIMPLE
) -> HTower:
    d_max = max(K.keys())
    device = K[d_max].device
    dtype = K[d_max].dtype
    n = K[d_max].shape[0]
    K_out: HTower = {}
    for d, K_d in K.items():
        assert K_d.shape == (n,) * d, f"K[{d}] must have shape (n,)*{d}."
        r = get_r_x(d, k_max, kind=kind)
        if r == -1:
            continue
        if isinstance(K_d, HTensor):
            K_out[d] = K_d.to(device=device, dtype=dtype)
        elif isinstance(K_d, DSTensor):
            K_out[d] = DS_harmonic_proj(K_d.to(device=device, dtype=dtype), r_out=r)
        else:
            assert isinstance(K_d, Tensor), f"K[{d}] must be a Tensor, HTensor, or DSTensor, got {type(K_d)!r}"
            K_out[d] = proj_geq_r(K_d.to(device=device, dtype=dtype), n=n, r_out=r)
    return K_out

@flop_name('clone_tower')
def clone_tower(K: HTower, d_max: Optional[int] = None) -> HTower:
    if d_max is None:
        d_max = max(K.keys())
    return {d: K_d.clone() for d, K_d in K.items() if d <= d_max}

def mlp_kprop(
    mlp: MLP,
    K_in: Tower,
    k_max: int,
    output_all: bool = False,
    kind: Kind = SIMPLE,
    use_avg_metric: bool = True,
    factor: bool = False,
    use_pK: bool = True,
    up_to_layer: Optional[str] = None,
    output_d_max: Optional[int] = None,
) -> HTower | dict[str, HTower]:
    """
    Cumulant propagation through MLP layers.

    Args:
        mlp: MLP instance
        K_in: dictionary mapping cumulant order d to input cumulant tensor of shape (n, ..., n)
        k_max: budget parameter. We want final error O(n^{-k_max}).
        output_all: whether to return cumulants at all layers or just the final layer
            If output_all, then final layer cumulants are stored under key 'pre{num_layers-1}'.
        kind: determines which cumulants to track based on k_max
            - SIMPLE: fewest pieces in harmonic decomposition to get O(n^{-k_max}) error.
            - AUGMENT: as many pieces as possible in harmonic decomposition while getting O(n^{k_max+1}) FLOPs (unfactorized).
                Also sums every Edgeworth diagram of squared size Omega(n^{-k_max})
                in the nonlinear step (see get_vec_cond), rather than just those of
                squared size Omega(n^{-k_max+1}).
            - BASE: Only cumulants up to k_max. (Doesn't get good MSE; just for ablation studies.)
            See get_r_x for details.
        use_avg_metric: whether to use the average-case metric E[WW^T] instead of WW^T
            Empirically, this doesn't have a large effect on MSE or FLOPs.
        factor: Use a factorized representation for the top-degree cumulant to cut a factor of n in FLOPs.
            Only supported for k_max=3 or 4.
            WARNING: factor=True with kind=AUGMENT drops the augmented diagrams that
            do not factor, so it is NOT equivalent to factor=False (see factored_keeps_term).
        use_pK: Whether to use pK_to_K logic in nonlinear expansion instead of directly computing K.
            (use_pK=False is only for ablation studies; it doesn't get good MSE.)
        up_to_layer: Output cumulants up to and including this layer.
            Takes a string f'pre{l}' or f'act{l}', interpreted as going up to the preactivation
            or activation labeled l, respectively. None means go through all layers.
        output_d_max: Max cumulant degree to output

    Returns:
        K_out: Dictionary mapping cumulant order d to cumulant tensor of shape (num_layers, n, ..., n)
            where num_layers is the number of layers in the MLP and there are d axes of size n=hidden_dim.
    """
    if mlp.layernorm:
        raise NotImplementedError("mlp_kprop currently does not support layernorm")

    nonlin_by_layer = mlp.nonlin_names
    nonlin_wick_coef_by_layer = [WICK_COEF_D[nonlin] for nonlin in nonlin_by_layer]
    init_scale_by_layer = mlp.init_scale

    K = coerce_input(K_in, k_max=k_max, kind=kind)
    K_by_layer = OrderedDict()
    for l, W_module in enumerate(mlp.Ws):
        W = W_module.weight
        layer_bias = W_module.bias
        if up_to_layer == f'pre{l}' or (l == len(mlp.nonlins) and up_to_layer is None):
            # Output preactivation (just linear_kprop, no projection needed).
            K = linear_kprop(K, W, k_max=k_max, d_max=output_d_max, bias=layer_bias)
            if output_all:
                K_by_layer[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            break
        if l < len(mlp.nonlins):
            metric = init_scale_by_layer[l] if use_avg_metric else None
            K = linear_kprop(
                K, W, k_max=k_max,
                set_metric=metric,
                bias=layer_bias,
            )
            if output_all:
                K_by_layer[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            K = nonlin_kprop(
                K,
                nonlin_wick_coef=nonlin_wick_coef_by_layer[l],
                k_max=k_max,
                kind=kind,
                use_pK=use_pK,
                factor=factor,
            )
            if output_all:
                K_by_layer[f"act{l}"] = clone_tower(K, d_max=output_d_max)
            if up_to_layer == f'act{l}':
                break
    if output_all:
        return K_by_layer
    else:
        return K
