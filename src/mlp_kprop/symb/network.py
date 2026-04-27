import itertools
from typing import Hashable, Optional

import matplotlib.pyplot as plt
import quimb.tensor as qtn
from mlp_kprop.symb.lazy_tensor import create_not_delta


def get_output_inds(net: qtn.TensorNetwork) -> list[str]:
    return [ind for ind in net.ind_map.keys() if ind[:1] == "i" and ind[1:].isdigit()]


def get_W_ids(net: qtn.TensorNetwork) -> list[int]:
    W_ids = [
        int(tag[1:])
        for tag in net.tag_map.keys()
        if tag[:1] == "W" and tag[1:].isdigit()
    ]
    return sorted(W_ids)


def get_Wick_ids(net: qtn.TensorNetwork) -> list[int]:
    Wick_ids = [
        int(tag[4:])
        for tag in net.tag_map.keys()
        if tag[:4] == "Wick" and tag[4:].isdigit()
    ]
    return sorted(Wick_ids)


def get_not_deltas(net: qtn.TensorNetwork) -> list[qtn.Tensor]:
    if "~" in net.tag_map:
        return {tensor_id: net.tensor_map[tensor_id] for tensor_id in net.tag_map["~"]}
    else:
        return {}


def get_width(ref_net: qtn.TensorNetwork) -> int:
    output_inds = get_output_inds(ref_net)
    ref_tensor = ref_net.tensor_map[next(iter(ref_net.ind_map[output_inds[0]]))]
    return ref_tensor.shape[ref_tensor.inds.index(output_inds[0])]


def new_network(ts: list[qtn.TensorNetwork, qtn.Tensor], **kwargs) -> qtn.TensorNetwork:
    # We handle index mangling in combine_networks, so turn off check_collisions
    return qtn.TensorNetwork(ts, check_collisions=False, **kwargs)


def combine_networks(nets: list[qtn.TensorNetwork]) -> qtn.TensorNetwork:
    mangled_nets = []
    ind_counter = itertools.count()
    for net in nets:
        index_map = {
            ind: f"m{next(ind_counter)}"
            for ind in net.ind_map.keys()
            if not (ind[:1] == "i" and ind[1:].isdigit())
        }
        mangled_nets.append(net.reindex(index_map))
    # Reindexing already copied tensors, so make virtual
    return new_network(mangled_nets, virtual=True)


def zero_diagonals(net: qtn.TensorNetwork) -> qtn.TensorNetwork:
    width = get_width(net)
    not_deltas = []
    for ind1, ind2 in itertools.combinations(get_output_inds(net), 2):
        not_deltas.append(create_not_delta(ind1, ind2, width=width))
    return new_network([net] + not_deltas)


def canonize_network(net: qtn.TensorNetwork) -> tuple[Hashable, list[dict[str, str]]]:
    """
    Produce a canonized representation of a network that can be used to check
    isomorphism. Also returns a list of automorphisms of the network's indices.
    """
    # ind -> tags of attached tensors plus the ind for an output ind
    ind_tags = {}
    for ind, tensor_ids in net.ind_map.items():
        ind_tags[ind] = []
        for tensor_id in tensor_ids:
            ind_tags[ind].append(tuple(sorted(net.tensor_map[tensor_id].tags)))
        if ind[:1] == "i" and ind[1:].isdigit():
            ind_tags[ind].append((ind,))
        ind_tags[ind] = tuple(sorted(ind_tags[ind]))
    ind_adj_tags = {}

    # ind -> tags of adjacent inds
    for ind in net.ind_map.keys():
        ind_adj_tags[ind] = []
    for tensor_id, tensor in net.tensor_map.items():
        if len(tensor.inds) == 1:
            continue
        ind1, ind2 = tensor.inds
        ind_adj_tags[ind1].append(ind_tags[ind2])
        ind_adj_tags[ind2].append(ind_tags[ind1])
    for ind in net.ind_map.keys():
        ind_adj_tags[ind] = tuple(sorted(ind_adj_tags[ind]))

    # (tags, adjacent tags) -> list of inds
    tags_map = {}
    for ind in net.ind_map.keys():
        tags = (ind_tags[ind], ind_adj_tags[ind])
        if tags not in tags_map:
            tags_map[tags] = []
        tags_map[tags].append(ind)
    tags_map = sorted(tags_map.items())

    # edge list for each choice of permutations of the ind lists
    # we can ignore edge labels since they are determined by the tags
    edge_lists = []
    id_edge_list = None
    automorphisms = []
    for perms in itertools.product(
        *[itertools.permutations(range(len(inds))) for _, inds in tags_map]
    ):
        ind_poses = {}
        ind_perm = {}
        pos = 0
        for perm, (_, inds) in zip(perms, tags_map):
            for local_pos, ind in zip(perm, inds):
                ind_poses[inds[local_pos]] = pos
                ind_perm[inds[local_pos]] = ind
                pos += 1
        edge_list = []
        for tensor_id, tensor in net.tensor_map.items():
            if len(tensor.inds) == 1:
                continue
            ind1, ind2 = tensor.inds
            edge_list.append(tuple(sorted((ind_poses[ind1], ind_poses[ind2]))))
        edge_list = tuple(sorted(edge_list))
        edge_lists.append(edge_list)
        if all(perm == tuple(range(len(perm))) for perm in perms):
            id_edge_list = edge_list
        assert id_edge_list is not None
        if edge_list == id_edge_list:
            automorphisms.append(ind_perm)

    # canonization is ((tags, adjacent tags) -> num inds, lex min edge list)
    canonization = (
        tuple([(tags, len(inds)) for tags, inds in tags_map]),
        min(edge_lists),
    )
    return canonization, automorphisms


def color_map(ref_nets: list[qtn.TensorNetwork], depth: Optional[int] = None) -> dict:
    tags = set(itertools.chain(*[net.tags for net in ref_nets]))
    W_ids = [int(tag[1:]) for tag in tags if tag[:1] == "W" and tag[1:].isdigit()]
    Wick_ids = [int(tag[4:]) for tag in tags if tag[:4] == "Wick" and tag[4:].isdigit()]
    W_cmap = plt.get_cmap("Blues")
    Wick_cmap = plt.get_cmap("Greens")
    colors = {}
    for W_id in sorted(W_ids):
        W_depth = (max(W_ids) + 1) if depth is None else depth
        colors[f"W{W_id}"] = W_cmap(0.5 * (1 - W_id / W_depth))
    for Wick_id in sorted(Wick_ids):
        Wick_depth = (max(Wick_ids) + 1) if depth is None else depth
        colors[f"Wick{Wick_id}"] = Wick_cmap(0.5 * (1 - Wick_id / Wick_depth))
    if "~" in tags:
        colors["~"] = "lightcoral"
    return colors


def draw_network(
    net: qtn.TensorNetwork,
    ref_nets: Optional[list[qtn.TensorNetwork]] = None,
    hide_not_deltas: bool = False,
    depth: Optional[int] = None,
    **kwargs,
):
    if ref_nets is None:
        ref_nets = [net]
    colors = color_map(ref_nets, depth)
    if hide_not_deltas:
        net = new_network(
            [tensor for tensor in net.tensor_map.values() if "~" not in tensor.tags]
        )
    kwargs.setdefault("output_inds", get_output_inds(net))
    kwargs.setdefault("node_size", 2)
    kwargs.setdefault(
        "highlight_inds",
        () if kwargs["output_inds"] is None else kwargs["output_inds"],
    )
    kwargs.setdefault("highlight_inds_color", "orange")
    net.draw(
        color=colors.keys(),
        custom_colors=colors.values(),
        **kwargs,
    )
