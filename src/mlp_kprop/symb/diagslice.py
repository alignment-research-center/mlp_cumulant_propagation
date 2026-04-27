import functools
from fractions import Fraction
from typing import Callable, Optional, Self

import quimb.tensor as qtn
import torch as th
from mlp_kprop.partitions import IntPartition
from mlp_kprop.symb.lazy_tensor import EvalContext
from mlp_kprop.symb.network import (
    canonize_network,
    draw_network,
    new_network,
    zero_diagonals,
)
from mlp_kprop.symb.parallelize import multi_map
from mlp_kprop.symb.pruning import (
    network_exponent,
    network_treewidth,
    recombine_network,
    split_network,
)


def evaluate_tensors(
    net: qtn.TensorNetwork, ctx: EvalContext, *, dtype=None, device=None
) -> qtn.TensorNetwork:
    tensors = []
    for tensor in net.tensor_map.values():
        tensor = tensor.copy()
        data = tensor.data.evaluate(ctx, dtype=dtype, device=device)
        tensor.modify(data=data)
        tensors.append(tensor)
    return new_network(tensors)


class DSNetwork:
    """
    A collection of diagonal slices of tensors stored as tensor networks.

    Slices are stored using reverse-sorted index partitions, but indexing
    supports other orders and zeros.

    Each slice is stored as a linear combination of tensor networks, with
    fractional multipliers and outer indices "i0", ..., f"i{len(vec) - 1}"
    that are assumed to be distinct.
    """

    def __init__(
        self,
        slices: dict[
            IntPartition, list[tuple[Fraction, Optional[int], qtn.TensorNetwork]]
        ],
    ):
        if not all(vec == tuple(sorted(vec, reverse=True)) for vec in slices.keys()):
            raise ValueError("Index partitions should be reverse-sorted.")
        self.slices = dict(slices)

    def keys(self):
        return self.slices.keys()

    def values(self):
        return self.slices.values()

    def items(self):
        return self.slices.items()

    def size(self):
        total = 0
        for vec in self.slices:
            try:
                total += len(self.slices[vec])
            except TypeError:
                return None
        return total

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    def __contains__(self, vec):
        return vec in self.slices

    def __getitem__(self, vec_or_vecs: IntPartition | list[IntPartition]):
        if isinstance(vec_or_vecs, list):
            slices = {}
            for vec in vec_or_vecs:
                num_nonzero = len(list(filter(bool, vec)))
                vec = tuple(sorted(vec, reverse=True)[:num_nonzero])
                if vec in self.keys():
                    slices[vec] = self.slices[vec]
            return DSNetwork(slices)
        else:
            vec = vec_or_vecs
        num_nonzero = len(list(filter(bool, vec)))
        new_indices = sorted(range(len(vec)), key=vec.__getitem__, reverse=True)
        index_map = {
            f"i{old}": f"i{new}" for old, new in enumerate(new_indices) if old != new
        }
        vec = tuple(sorted(vec, reverse=True)[:num_nonzero])
        if vec in self.keys():
            mults_exps_nets = self.slices[vec]
            if index_map:
                return [
                    (mult, exp, net.reindex(index_map))
                    for mult, exp, net in mults_exps_nets
                ]
            else:
                return mults_exps_nets
        else:
            return []

    def vec_map(self, func: Callable, inplace: bool = False) -> Self:
        slices = self.slices if inplace else {**self.slices}
        for vec in slices.keys():
            slices[vec] = func(slices[vec])
        return self if inplace else DSNetwork(slices)

    def simplify(self, inplace: bool = False) -> Self:
        slices = self.slices if inplace else {**self.slices}
        for vec in slices.keys():
            mults_exps_nets = slices[vec]
            mults_exps_nets_canonizations = multi_map(
                lambda mult, exp, net: [(mult, exp, net, canonize_network(net)[0])],
                mults_exps_nets,
            )
            mults_exps_nets = {}
            for mult, exp, net, canonization in mults_exps_nets_canonizations:
                if canonization not in mults_exps_nets:
                    mults_exps_nets[canonization] = [mult, exp, net]
                else:
                    mult_exp_net = mults_exps_nets[canonization]
                    mult_exp_net[0] += mult
                    if mult_exp_net[1] is None:
                        mult_exp_net[1] = exp
                    elif exp is not None:
                        assert mult_exp_net[1] == exp
            mults_exps_nets = list(map(tuple, mults_exps_nets.values()))
            slices[vec] = mults_exps_nets
        return self if inplace else DSNetwork(slices)

    def prune(
        self,
        k_max: int,
        allow_split: bool = False,
        inplace: bool = False,
        streaming: bool = False,
    ) -> Self:
        slices = self.slices if inplace else {**self.slices}
        net_exp = functools.partial(network_exponent, allow_split=allow_split)
        for vec in slices.keys():
            mults_exps_nets = slices[vec]
            mults_exps_nets = multi_map(
                lambda mult, exp, net: [
                    (mult, (net_exp(net) if exp is None else exp), net)
                ],
                mults_exps_nets,
            )
            mults_exps_nets = multi_map(
                lambda mult, exp, net: [(mult, exp, net)] if exp > -k_max else [],
                mults_exps_nets,
            )
            if not streaming:
                mults_exps_nets = list(mults_exps_nets)
            slices[vec] = mults_exps_nets
        return self if inplace else DSNetwork(slices)

    def split(self, inplace: bool = False, streaming: bool = False) -> Self:
        slices = self.slices if inplace else {**self.slices}
        for vec in slices.keys():
            mults_exps_nets = slices[vec]
            mults_exps_nets = multi_map(
                lambda mult, _, net: [
                    (mult, None, net_piece) for net_piece in split_network(net)
                ],
                mults_exps_nets,
            )
            if not streaming:
                mults_exps_nets = list(mults_exps_nets)
            slices[vec] = mults_exps_nets
        return self if inplace else DSNetwork(slices)

    def recombine(self, inplace: bool = False, streaming: bool = False) -> Self:
        slices = self.slices if inplace else {**self.slices}
        for vec in slices.keys():
            mults_exps_nets = slices[vec]
            mults_exps_nets = multi_map(
                lambda mult, _, net: [
                    (mult * Fraction(coef), None, net_group)
                    for coef, net_group in recombine_network(net)
                ],
                mults_exps_nets,
            )
            if not streaming:
                mults_exps_nets = list(mults_exps_nets)
            slices[vec] = mults_exps_nets
        return self if inplace else DSNetwork(slices)

    def bucket_by_cost(self, target_tw: int):
        cheap_slices = {}
        expensive_slices = {}
        for vec in self.keys():
            cheap_slices[vec] = []
            expensive_slices[vec] = []
            for mult, exp, net in self[vec]:
                if network_treewidth(net) <= target_tw:
                    cheap_slices[vec].append((mult, exp, net))
                else:
                    expensive_slices[vec].append((mult, exp, net))
        return DSNetwork(cheap_slices), DSNetwork(expensive_slices)

    def update_width(self, width: int):
        for vec in self.keys():
            for _, _, net in self[vec]:
                for tensor in net.tensor_map.values():
                    tensor.data.update_width(width)

    def contract(
        self, vec: IntPartition, ctx: EvalContext, *, dtype=None, device=None
    ) -> th.Tensor:
        self.update_width(ctx.width)
        output_inds = [f"i{i}" for i in range(len(vec)) if vec[i] != 0]
        if not self[vec]:
            result = th.zeros(
                (ctx.width,) * len(output_inds), dtype=dtype, device=device
            )
        else:
            result = 0
        for mult, _, net in self[vec]:
            net = zero_diagonals(net)
            net = evaluate_tensors(net, ctx=ctx, dtype=dtype, device=device)
            contraction = net.contract(output_inds=output_inds).data
            result += float(mult) * contraction
        result = result[
            tuple([None if vec[i] == 0 else slice(None, None) for i in range(len(vec))])
        ]
        return result

    def draw(
        self, hide_not_deltas: bool = False, depth: Optional[int] = None, **kwargs
    ):
        ref_nets = [
            net for mults_exps_nets in self.values() for _, _, net in mults_exps_nets
        ]
        for vec in self.keys():
            print(vec)
            for i, (mult, _, net) in enumerate(self[vec]):
                sign_str = "-" if mult.numerator < 0 else ("" if i == 0 else "+")
                mag_str = (
                    f"{abs(mult.numerator)}"
                    if mult.denominator == 1
                    else f"\\frac{{{abs(mult.numerator)}}}{{{mult.denominator}}}"
                )
                draw_network(
                    net,
                    ref_nets=ref_nets,
                    hide_not_deltas=hide_not_deltas,
                    depth=depth,
                    title=f"${sign_str}{mag_str}\\times$",
                    **kwargs,
                )

    def tensor_states(self):
        slices = {}
        for vec in self.slices:
            slices[vec] = []
            for mult, exp, net in self.slices[vec]:
                states = [tensor.__getstate__() for tensor in net.tensor_map.values()]
                slices[vec].append((mult, exp, states))
        return slices

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.tensor_states() == other.tensor_states()


def merge_dsnetworks(*dsnetworks):
    slices = {}
    for vec in dict.fromkeys([vec for d in dsnetworks for vec in d.keys()]):
        slices[vec] = []
        for d in dsnetworks:
            for mult, exp, net in d[vec]:
                slices[vec].append((mult, exp, net))
    return DSNetwork(slices)
