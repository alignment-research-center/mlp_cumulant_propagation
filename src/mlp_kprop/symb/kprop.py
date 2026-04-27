import functools
import math
from fractions import Fraction
from typing import Callable, Optional
import itertools

import quimb.tensor as qtn
from mlp_kprop.diagslice import EinsumCond
from mlp_kprop.kprop_harmonic import get_all_terms, get_int_cond
from mlp_kprop.mlp import MLP
from mlp_kprop.partitions import (
    IntPartCond,
    IntPartition,
    SetPartition,
    VecPartition,
    check_vec_partition,
    set_partitions,
    set_to_int_partition,
    set_to_vec_partition,
    vec_part_coef,
)
from mlp_kprop.kprop_ds import get_ein_cond
from mlp_kprop.symb.diagslice import DSNetwork, merge_dsnetworks
from mlp_kprop.symb.lazy_tensor import EvalContext, create_relu_wick, create_weight
from mlp_kprop.symb.network import combine_networks, new_network, zero_diagonals
from mlp_kprop.symb.parallelize import multi_map
from tqdm import tqdm

# NOTE: This module implements the symbolic version of the deprecated kprop_ds algorithm.
# TODO: Switch the symbolic pipeline over to kprop_harmonic.

class PBar:
    def __init__(self, **default_kwargs):
        self.pbar = None
        self.default_kwargs = default_kwargs

    def new(self, **kwargs):
        kwargs = {**self.default_kwargs, **kwargs}
        if self.pbar is not None:
            self.close()
        self.pbar = tqdm(**kwargs)

    def update(self, n=1):
        if self.pbar is not None:
            self.pbar.update(n=n)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def update_with_iterable(self, iterable):
        for item in iterable:
            yield item
            self.update(1)


def product_exp_filter(*slices, k_max):
    if k_max is not None and k_max <= 0:
        return
    if not slices:
        yield (), (), ()
        return
    first, *rest = slices
    for mult, exp, net in first:
        if k_max is not None and exp is not None:
            rest_k_max = k_max + exp
        else:
            rest_k_max = None
        for mults, exps, nets in product_exp_filter(*rest, k_max=rest_k_max):
            yield (mult, *mults), (exp, *exps), (net, *nets)


def vec_part_coef_frac(part: VecPartition) -> Fraction:
    if len(part) == 0:
        return Fraction(1)
    n = [sum(v[i] for v in part) for i in range(len(part[0]))]
    num = vec_part_coef(part, divide_fac=False)
    denom = math.prod(math.factorial(ni) for ni in n)
    return Fraction(num, denom)


def kz_to_pkx(
    kz: DSNetwork,
    k_max: int,
    wick_fn: Callable,
    only_vecs: Optional[list[IntPartition]] = None,
    streaming: bool = False,
    pbar: Optional[PBar] = None,
) -> DSNetwork:
    terms = {}
    for pkx_vec, vec_part in get_all_terms(k_max):
        if only_vecs is not None and pkx_vec not in only_vecs:
            continue
        if pkx_vec not in terms:
            terms[pkx_vec] = []
        terms[pkx_vec].append((pkx_vec, vec_part))
        # Add all Wicks to the LazyContext prior to parallelizing,
        # otherwise they can be missing
        sum_vec_part = check_vec_partition(vec_part, len(pkx_vec))
        _ = [
            wick_fn(f"i{i}", order=order, power=power)
            for i, (order, power) in enumerate(zip(sum_vec_part, pkx_vec))
        ]

    if pbar is not None:
        pbar.new(total=sum(map(len, terms.values())))
        for pkx_vec in terms:
            terms[pkx_vec] = pbar.update_with_iterable(terms[pkx_vec])

    def multi_map_func(pkx_vec, vec_part):
        sum_vec_part = check_vec_partition(vec_part, len(pkx_vec))
        coef = vec_part_coef_frac(vec_part)
        for mults, _, nets in product_exp_filter(
            *[kz[kz_vec] for kz_vec in vec_part], k_max=k_max
        ):
            mult = math.prod(mults, start=Fraction(1)) * coef
            wicks = [
                wick_fn(f"i{i}", order=order, power=power)
                for i, (order, power) in enumerate(zip(sum_vec_part, pkx_vec))
            ]
            # Combining will copy tensors, so make virtual
            net = combine_networks([*nets, new_network(wicks, virtual=True)])
            yield (mult, None, net)

    slices = {}
    for pkx_vec in terms:
        slices[pkx_vec] = multi_map(multi_map_func, terms[pkx_vec])
        if not streaming:
            slices[pkx_vec] = list(slices[pkx_vec])
    return DSNetwork(slices)


@functools.cache
def pk_to_k_coef(vec_part: VecPartition) -> int:
    coef = 0
    for block_part in set_partitions(len(vec_part)):
        blocks = [[vec_part[i] for i in indices] for indices in block_part]
        if all(
            all(len([a for a in a_s if a > 0]) <= 1 for a_s in zip(*block))
            for block in blocks
        ):
            coef += (-1) ** (len(block_part) - 1) * math.factorial(len(block_part) - 1)
    return coef


def pkx_to_kx(
    pkx: DSNetwork,
    k_max: int,
    only_vecs: Optional[list[IntPartition]] = None,
    streaming: bool = False,
    pbar: Optional[PBar] = None,
) -> DSNetwork:
    terms = {}
    for k_vec in pkx.keys():
        if only_vecs is not None and k_vec not in only_vecs:
            continue
        if not pkx[k_vec]:
            continue
        for set_part in set_partitions(sum(k_vec)):
            vec_part = set_to_vec_partition(set_part, k_vec)
            if k_vec not in terms:
                terms[k_vec] = []
            terms[k_vec].append((vec_part,))
    if pbar is not None:
        pbar.new(total=sum(map(len, terms.values())))
        for k_vec in terms:
            terms[k_vec] = pbar.update_with_iterable(terms[k_vec])

    def multi_map_func(vec_part):
        coef = Fraction(pk_to_k_coef(vec_part))
        if coef == 0:
            return
        for mults, _, nets in product_exp_filter(
            *[pkx[pk_vec] for pk_vec in vec_part], k_max=k_max
        ):
            mult = math.prod(mults, start=Fraction(1)) * coef
            net = combine_networks(nets)
            yield (mult, None, net)

    slices = {}
    for k_vec in terms:
        slices[k_vec] = multi_map(multi_map_func, terms[k_vec])
        if not streaming:
            slices[k_vec] = list(slices[k_vec])
    return DSNetwork(slices)


def kx_to_kz(
    kx: DSNetwork,
    k_max: int,
    W: qtn.Tensor,
    only_vecs: Optional[list[IntPartition]] = None,
    streaming: bool = False,
    pbar: Optional[PBar] = None,
) -> DSNetwork:
    terms = {}
    for d in sorted(set([sum(vec) for vec in kx.keys()])):
        kx_cond = IntPartCond(parts=[vec for vec in kx.keys() if sum(vec) == d])
        kz_cond = get_int_cond(k_max)
        W_cond = IntPartCond(parts=[(1, 1)])
        for arg_parts, coef in get_ein_cond(k_max).get_parts(
            aritys=(d,) + (2,) * d + (d,),
            arg_conds=(kx_cond,) + (W_cond,) * d + (kz_cond,),
            in_symmetric=True,
            out_symmetric=True,
        ):
            kx_part = arg_parts[0]
            kz_part = arg_parts[-1]
            assert coef == int(coef)
            coef = int(coef)
            kz_vec = set_to_int_partition(kz_part)
            if only_vecs is not None and kz_vec not in only_vecs:
                continue
            if kz_vec not in terms:
                terms[kz_vec] = []
            terms[kz_vec].append((d, kx_part, kz_part, coef))
    if pbar is not None:
        pbar.new(total=sum(map(len, terms.values())))
        for kz_vec in terms:
            terms[kz_vec] = pbar.update_with_iterable(terms[kz_vec])

    def multi_map_func(d, kx_part, kz_part, coef):
        kx_vec = set_to_int_partition(kx_part)
        for kx_mult, _, kx_net in kx[kx_vec]:
            kz_mult = Fraction(coef) * kx_mult
            kx_net = zero_diagonals(kx_net)
            kx_net = kx_net.reindex({f"i{j}": f"j{j}" for j in range(d)})
            W_copies = []
            for k in range(d):
                [i] = [i for i in range(len(kz_part)) if k in kz_part[i]]
                [j] = [j for j in range(len(kx_part)) if k in kx_part[j]]
                W_copies.append(W.reindex({"X": f"j{j}", "Z": f"i{i}"}))
            # Reindexing already copied tensors, so make virtual
            kz_net = new_network([kx_net] + W_copies, virtual=True)
            yield (kz_mult, None, kz_net)

    slices = {}
    for kz_vec in terms:
        slices[kz_vec] = multi_map(multi_map_func, terms[kz_vec])
        if not streaming:
            slices[kz_vec] = list(slices[kz_vec])
    return DSNetwork(slices)


def get_only_vecs(
    ks_name: str,
    prev_vecs: set[IntPartition],
    layer_from_end: int,
    mean_only: bool,
    var_only: bool,
) -> set[IntPartition]:
    if not (mean_only or var_only):
        return None
    if layer_from_end == (-1 if ks_name == "kz" else -2):
        only_vecs = set()
        if mean_only:
            only_vecs.add((1,))
        if var_only:
            only_vecs.add((2,))
            if ks_name == "pkx" or ks_name == "kx":
                only_vecs.add((1, 1))
            if ks_name == "pkx":
                only_vecs.add((1,))
        return only_vecs
    elif layer_from_end == -2 and ks_name == "kz":
        if var_only:
            return set([vec for vec in prev_vecs if len(vec) <= 2])
        elif mean_only:
            return set([vec for vec in prev_vecs if len(vec) <= 1])
    else:
        return None


def abstract_kprop(
    depth: int,
    k_max: int,
    output_all: bool = False,
    mean_only: bool = False,
    var_only: bool = False,
    simplify: bool = False,
    prune: bool = False,
    split: Optional[bool] = None,
    verbose: bool = False,
) -> DSNetwork:
    """
    Cumulant propagation through a hypothetical MLP with a given depth.

    Returns a list of lazy tensors to be populated with the weights of the MLP
    and the requested cumulants.

    Arguments:
    - depth: depth of hypothetical MLP
    - k_max: budget parameter, we want final error O(n^{-k_max})
    - output_all: return kz, pkx and kx at all layers rather than only final kz
    - mean_only: compute the final mean but not necessarily other cumulants
    - var_only: compute the final variance but not necessarily other cumulants
    - simplify: group diagrams by isomorphism, recommended
    - prune: drop diagrams based on k_max, requires splitting
    - split: split into diagrams with distinct indices at every layer
    - verbose: display progress
    """
    if split is None:
        split = prune
    if prune and (not split):
        raise ValueError("Pruning requires splitting.")
    pbar = PBar(ncols=80, disable=(not verbose))
    log = print if verbose else (lambda s: None)
    width = 2
    layer = 0
    log(f"Layer {layer}:")
    Ws = [create_weight(layer=layer, width=width) for layer in range(depth)]
    kz_cov = new_network([Ws[0].reindex({"Z": "i0"}), Ws[0].reindex({"Z": "i1"})])
    kz_var = new_network([Ws[0].reindex({"Z": "i0"}), Ws[0].reindex({"Z": "i0"})])
    kz = DSNetwork(
        {(1, 1): [(Fraction(1), None, kz_cov)], (2,): [(Fraction(1), None, kz_var)]}
    )
    if output_all:
        all_ks = {"kz": [kz], "pkx": [], "kx": []}
    else:
        wick_kzs = [kz[[(1,), (2,)]]]
    for W in Ws[1:]:
        log(f"Applying kz_to_pkx to {kz.size()} diagrams...")
        only_vecs = get_only_vecs(
            "pkx",
            set(kz.keys()),
            layer - len(Ws),
            mean_only=mean_only,
            var_only=var_only,
        )
        pkx = kz_to_pkx(
            kz,
            k_max,
            functools.partial(create_relu_wick, layer=layer, width=width),
            only_vecs=only_vecs,
            streaming=split or simplify,
            pbar=pbar,
        )
        if simplify:
            pkx = pkx.simplify()
        if split:
            if simplify:
                pbar.close()
                log(f"Splitting {pkx.size()} diagrams...")
                pbar.new(total=pkx.size())
                pkx = pkx.vec_map(pbar.update_with_iterable)
            pkx = pkx.split(streaming=prune or simplify)
            if prune:
                pkx = pkx.prune(k_max, streaming=simplify)
            if simplify:
                pkx = pkx.simplify()
        pbar.close()
        if output_all:
            all_ks["pkx"].append(pkx)
        else:
            del kz

        log(f"Applying pkx_to_kx to {pkx.size()} diagrams...")
        only_vecs = get_only_vecs(
            "kx",
            set(pkx.keys()),
            layer - len(Ws),
            mean_only=mean_only,
            var_only=var_only,
        )
        kx = pkx_to_kx(
            pkx,
            k_max,
            only_vecs=only_vecs,
            streaming=split or simplify,
            pbar=pbar,
        )
        if simplify:
            kx = kx.simplify()
        if split:
            if simplify:
                pbar.close()
                log(f"Splitting {kx.size()} diagrams...")
                pbar.new(total=kx.size())
                kx = kx.vec_map(pbar.update_with_iterable)
            kx = kx.split(streaming=prune or simplify)
            if prune:
                kx = kx.prune(k_max, streaming=simplify)
            if simplify:
                kx = kx.simplify()
        pbar.close()
        if output_all:
            all_ks["kx"].append(kx)
        else:
            del pkx
        layer += 1
        log(f"Layer {layer}:")
        log(f"Applying kx_to_kz to {kx.size()} diagrams...")
        only_vecs = get_only_vecs(
            "kz", kx.keys(), layer - len(Ws), mean_only=mean_only, var_only=var_only
        )
        kz = kx_to_kz(
            kx, k_max, W, only_vecs=only_vecs, streaming=prune or simplify, pbar=pbar
        )
        # Splitting does nothing after kx_to_kz
        if prune:
            kz = kz.prune(k_max, streaming=simplify)
        if simplify:
            kz = kz.simplify()
        pbar.close()
        if output_all:
            all_ks["kz"].append(kz)
        else:
            if layer != len(Ws) - 1:
                wick_kzs.append(kz[[(1,), (2,)]])
            del kx

    log(f"Final kz has {kz.size()} diagrams.")
    return all_ks if output_all else (kz, wick_kzs)


def reduce_treewidth(
    ks: DSNetwork,
    k_max: int,
    treewidth: int,
    max_cycles: int = 8,
    wick_kzs: Optional[list[DSNetwork]] = None,
    simplify: bool = False,
    prune: bool = False,
    split: Optional[bool] = None,
    prune_slow: bool = True,
    verbose: bool = False,
):
    """
    Attempt to reduce the treewidth of all diagrams to the given target.

    Works by alternately recombining and splitting diagrams into diagrams
    without or with distinct indices. At each step, we set aside diagrams with
    based on treewidth, and if pruning, drop diagrams based on k_max.

    Only works for single-index cumulants that have been split.

    If wick_kzs are provided, also runs the same procedure on those.

    If prune_slow is set to True, then we also drop low-treewidth diagrams
    without distinct indices based on k_max. This reduces the total number of
    diagrams, but doesn't change whether the maximum treewidth is successfully
    reduced, and can be slow.

    Other arguments have the same meaning as for abstract_kprop. In particular,
    the function just checks treewidths unless split resolves to True.
    """
    if split is None:
        split = prune
    if prune and (not split):
        raise ValueError("Pruning requires splitting.")
    log = print if verbose else (lambda s: None)
    pbar = PBar(ncols=80, disable=(not verbose))
    wick_kzs_by_layer = enumerate(wick_kzs if wick_kzs is not None else [])
    reduced_ks = []
    for layer, ks_expen in [(None, ks)] + list(wick_kzs_by_layer):
        if layer is not None:
            log(f"Layer {layer} kz:")
        ks_cheaps = []
        ks_cheap, ks_expen = ks_expen.bucket_by_cost(treewidth)
        ks_cheaps.append(ks_cheap)
        cycles = 0
        while ks_expen.size() > 0 and cycles < max_cycles:
            if split:
                log(f"Recombining {ks_expen.size()} diagrams...")
                pbar.new(total=ks_expen.size())
                ks_expen = ks_expen.vec_map(pbar.update_with_iterable)
                ks_expen = ks_expen.recombine(streaming=simplify)
                if simplify:
                    ks_expen = ks_expen.simplify()
                pbar.close()
                ks_cheap, ks_expen = ks_expen.bucket_by_cost(treewidth)
                if prune and prune_slow:
                    log(f"Splitting and pruning {ks_cheap.size()} diagrams...")
                    pbar.new(total=ks_cheap.size())
                    ks_cheap = ks_cheap.vec_map(pbar.update_with_iterable)
                    ks_cheap = ks_cheap.prune(k_max, allow_split=True)
                    pbar.close()
                ks_cheaps.append(ks_cheap)
                if ks_expen.size() == 0:
                    break
                log(f"Splitting {ks_expen.size()} diagrams...")
                pbar.new(total=ks_expen.size())
                ks_expen = ks_expen.vec_map(pbar.update_with_iterable)
                ks_expen = ks_expen.split(streaming=prune or simplify)
                if prune:
                    ks_expen = ks_expen.prune(k_max, streaming=simplify)
                if simplify:
                    ks_expen = ks_expen.simplify()
                pbar.close()
                ks_cheap, ks_expen = ks_expen.bucket_by_cost(treewidth)
                ks_cheaps.append(ks_cheap)
            cycles += 1
        only_vecs = []
        for vec in ks_expen.keys():
            if len(ks_expen[vec]) > 0:
                log(
                    f"Failed to reduce ks[{vec}] to treewidth {treewidth} "
                    f"in {max_cycles} cycles."
                )
            else:
                only_vecs.append(vec)
                counts = [len(ks_cheap[vec]) for ks_cheap in ks_cheaps]
                while counts and counts[-1] == 0:
                    counts = counts[:-1]
                if len(counts) > 1:
                    counts = " + ".join(map(str, counts)) + f" = {sum(counts)}"
                    log(f"Reduced ks[{vec}] to {counts} diagrams.")
        ks_cheap = merge_dsnetworks(*ks_cheaps)[only_vecs]
        log(f"Reduced treewidth ks has {ks_cheap.size()} diagrams in total.")
        reduced_ks.append(ks_cheap)
    if wick_kzs is not None:
        return reduced_ks[0], reduced_ks[1:]
    else:
        return reduced_ks[0]


def reduce_treewidth_all(all_ks: dict, k_max: int, verbose: bool, **kwargs):
    """
    Apply reduce_treewidth to a collection of cumulants produced by
    abstract_kprop with output_all set to True.

    Restricts to means and variances of kzs and kxs only.
    """
    log = print if verbose else (lambda s: None)
    only_vecs = [(1,), (2,)]
    all_reduced = {"kz": [], "kx": []}
    for layer in range(len(all_ks["kz"])):
        log(f"Layer {layer} kz:")
        kz = all_ks["kz"][layer][only_vecs]
        all_reduced["kz"].append(
            reduce_treewidth(kz, k_max, wick_kzs=None, verbose=verbose, **kwargs)
        )
        if layer <= len(all_ks["kx"]) - 1:
            log(f"Layer {layer} kx:")
            kx = all_ks["kx"][layer][only_vecs]
            all_reduced["kx"].append(
                reduce_treewidth(kx, k_max, wick_kzs=None, verbose=verbose, **kwargs)
            )
    return all_reduced


def mlp_kprop(
    mlp: MLP,
    k_max: int,
    output_all: bool = False,
    mean_only: bool = False,
    var_only: bool = False,
    treewidth: Optional[int] = None,
    **kwargs,
):
    """
    Cumulant propagation through the given MLP.

    The treewidth argument determines whether to run reduce_treewidth. All
    arguments are passed through to abstract_kprop and reduce_treewidth.
    """
    if not (mlp.input_dim == mlp.hidden_dim == mlp.output_dim):
        raise NotImplementedError("Symbolic kprop requires constant width.")
    if treewidth is not None and (not mean_only) and (not var_only):
        raise ValueError("Can only reduce treewidth of means and variances.")
    kprop_kwargs = kwargs.copy()
    kprop_kwargs.pop("max_cycles", None)
    kprop_kwargs.pop("prune_slow", None)
    result = abstract_kprop(
        len(mlp.Ws),
        k_max=k_max,
        output_all=output_all,
        mean_only=mean_only,
        var_only=var_only,
        **kprop_kwargs,
    )
    if output_all:
        all_ks = result
        if treewidth is not None:
            all_ks = reduce_treewidth_all(
                all_ks,
                k_max=k_max,
                treewidth=treewidth,
                **kwargs,
            )
        wick_kzs = [kz[[(1,), (2,)]] for kz in all_ks["kz"][:-1]]
    else:
        kz, wick_kzs = result
        if treewidth is not None:
            kz, wick_kzs = reduce_treewidth(
                kz,
                k_max=k_max,
                treewidth=treewidth,
                wick_kzs=wick_kzs,
                **kwargs,
            )
    ctx = EvalContext(weights=[W.weight.data for W in mlp.Ws], wick_kzs=wick_kzs)
    if output_all:
        all_ks["ctx"] = ctx
        return all_ks
    else:
        return (kz, ctx)
