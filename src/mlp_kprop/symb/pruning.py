import itertools
import math

import quimb.tensor as qtn
from mlp_kprop.symb.comb import non_consuming_product, partitions
from mlp_kprop.symb.lazy_tensor import create_not_delta
from mlp_kprop.symb.network import (
    get_not_deltas,
    get_output_inds,
    get_W_ids,
    get_width,
    new_network,
)


def dedupe_not_deltas(
    net: qtn.TensorNetwork, *, virtual: bool = False
) -> qtn.TensorNetwork:
    not_delta_inds = set()
    tensor_ids_to_remove = set()
    for not_delta_id, not_delta in get_not_deltas(net).items():
        inds = tuple(sorted(not_delta.inds))
        if inds not in not_delta_inds:
            not_delta_inds.add(inds)
        else:
            tensor_ids_to_remove.add(not_delta_id)
    tensors_to_keep = [
        tensor
        for tensor_id, tensor in net.tensor_map.items()
        if tensor_id not in tensor_ids_to_remove
    ]
    return new_network(tensors_to_keep, virtual=virtual)


def split_network(net: qtn.TensorNetwork) -> list[qtn.TensorNetwork]:
    """
    Split a network into all possible combinations of equal or distinct indices
    at each layer.
    """
    width = get_width(net)
    W_ids = get_W_ids(net)
    not_delta_edges = [tensor.inds for tensor in get_not_deltas(net).values()]
    inds_dict = {}
    banned_dict = {}
    for W_id in W_ids:
        inds = [
            net.tensor_map[tensor_id].inds[1] for tensor_id in net.tag_map[f"W{W_id}"]
        ]
        inds_dict[W_id] = list(dict.fromkeys(inds))
        banned = []
        for not_delta_edge in not_delta_edges:
            if all(ind in inds_dict[W_id] for ind in not_delta_edge):
                banned.append(not_delta_edge)
        banned_dict[W_id] = list(dict.fromkeys(banned))
    net_pieces = []
    all_partitions = [partitions(inds_dict[W_id], banned_dict[W_id]) for W_id in W_ids]
    for partition_by_W_id in non_consuming_product(*all_partitions):
        index_map = {}
        new_not_deltas = []
        for W_id, partition in zip(W_ids, partition_by_W_id):
            for block_num, block in enumerate(partition):
                for ind in block:
                    index_map[ind] = block[0]
            for block1, block2 in itertools.combinations(partition, 2):
                new_not_deltas.append(
                    create_not_delta(block1[0], block2[0], width=width)
                )
        # Reindexing already copied tensors, so make virtual
        net_piece = new_network([net.reindex(index_map)] + new_not_deltas, virtual=True)
        net_pieces.append(dedupe_not_deltas(net_piece, virtual=True))
    return net_pieces


def recombine_network(net: qtn.TensorNetwork) -> list[tuple[int, qtn.TensorNetwork]]:
    """
    Recombine a network with distinct indices into networks with equal or
    either-equal-or-distinct indices.
    """
    if len(get_output_inds(net)) != 1:
        raise ValueError("Cannot recombine networks with multiple output indices.")
    W_ids = get_W_ids(net)
    not_delta_edges = [tensor.inds for tensor in get_not_deltas(net).values()]
    inds_dict = {}
    for W_id in W_ids:
        inds = [
            net.tensor_map[tensor_id].inds[1] for tensor_id in net.tag_map[f"W{W_id}"]
        ]
        inds_dict[W_id] = list(dict.fromkeys(inds))
    if not all(
        all(
            (id1, id2) in not_delta_edges or (id2, id1) in not_delta_edges
            for id1, id2 in itertools.combinations(inds_dict[W_id], 2)
        )
        for W_id in W_ids
    ):
        raise ValueError("Can only recombine split networks.")
    net_without_not_deltas = new_network(
        [tensor for tensor in net.tensor_map.values() if "~" not in tensor.tags]
    )
    net_groups = []
    all_partitions = [partitions(inds_dict[W_id]) for W_id in W_ids]
    for partition_by_W_id in non_consuming_product(*all_partitions):
        coef = 1
        index_map = {}
        for W_id, partition in zip(W_ids, partition_by_W_id):
            for block_num, block in enumerate(partition):
                coef *= (-1) ** (len(block) - 1) * math.factorial(len(block) - 1)
                for ind in block:
                    index_map[ind] = block[0]
        net_group = net_without_not_deltas.reindex(index_map)
        net_groups.append((coef, net_group))
    return net_groups


def network_exponent(net: qtn.TensorNetwork, allow_split: bool = False) -> int:
    """
    Compute the expected square of a tensor entry as an exponent of the width.
    """
    W_ids = get_W_ids(net)
    not_delta_edges = [tensor.inds for tensor in get_not_deltas(net).values()]
    inds_dict = {}
    for W_id in W_ids:
        inds = [
            net.tensor_map[tensor_id].inds[1] for tensor_id in net.tag_map[f"W{W_id}"]
        ]
        inds_dict[W_id] = list(dict.fromkeys(inds))
    if not all(
        all(
            (id1, id2) in not_delta_edges or (id2, id1) in not_delta_edges
            for id1, id2 in itertools.combinations(inds_dict[W_id], 2)
        )
        for W_id in W_ids
    ):
        if not allow_split:
            raise ValueError("Either split first or allow splitting.")
        return max(
            [
                network_exponent(net_piece, allow_split=False)
                for net_piece in split_network(net)
            ]
        )
    tensor_ids_to_ignore = set()
    for W_id in W_ids:
        tensor_ids_by_inds = {}
        for tensor_id in net.tag_map[f"W{W_id}"]:
            inds = net.tensor_map[tensor_id].inds
            if inds in tensor_ids_by_inds:
                tensor_ids_to_ignore.add(tensor_id)
                tensor_ids_to_ignore.add(tensor_ids_by_inds[inds])
                del tensor_ids_by_inds[inds]
            else:
                tensor_ids_by_inds[inds] = tensor_id
    component_count = 0
    for W_id in W_ids:
        for ind in inds_dict[W_id]:
            W_tensor_ids = [*net.tag_map[f"W{W_id}"]]
            if f"W{W_id - 1}" in net.tag_map:
                W_tensor_ids += [*net.tag_map[f"W{W_id - 1}"]]
            if all(
                tensor_id not in W_tensor_ids or tensor_id in tensor_ids_to_ignore
                for tensor_id in net.ind_map[ind]
            ):
                component_count += 2
            else:
                component_count += 1
    num_ws = sum(map(len, [net.tag_map[f"W{W_id}"] for W_id in W_ids]))
    return component_count - num_ws


def network_treewidth(net: qtn.TensorNetwork) -> int:
    if len(get_output_inds(net)) != 1:
        raise ValueError(
            "Treewidth is not a reliable indicator of "
            "contraction time for multiple output indices."
        )
    try:
        from sage.all import Graph
    except ImportError as exn:
        raise ImportError("Treewidth requires SageMath.") from exn

    graph = Graph(
        [tensor.inds for tensor in net.tensor_map.values() if len(tensor.inds) == 2]
    )
    return graph.treewidth()
