import itertools
from collections import Counter

import pytest
import torch as th
from mlp_kprop.diagslice import DSTower
from mlp_kprop.kprop_ds import mlp_kprop as nonsymb_kprop
from mlp_kprop.mlp import MLP
from mlp_kprop.partitions import (
    int_partitions,
    is_connected,
    set_partitions,
    set_to_vec_partition,
)
from mlp_kprop.symb.kprop import mlp_kprop as symb_kprop
from mlp_kprop.symb.kprop import pk_to_k_coef


def frozen_multiset(iterable):
    c = Counter(iterable)
    items = tuple(sorted((k, v) for k, v in c.items() if v > 0))
    return frozenset(items)


@pytest.mark.parametrize("d", list(range(1, 7)))
@pytest.mark.parametrize("fwd_outer", [True, False])
def test_pk_to_k_coef(d, fwd_outer):
    for int_part_outer in int_partitions(d):
        mults_combined = {}
        for set_part_outer in set_partitions(d):
            vec_part_outer = set_to_vec_partition(set_part_outer, int_part_outer)
            if fwd_outer:
                mult_outer = (
                    1 if is_connected(vec_part_outer, len(int_part_outer)) else 0
                )
            else:
                mult_outer = pk_to_k_coef(vec_part_outer)
            if mult_outer == 0:
                continue
            set_parts_inner_iters = tuple(
                set_partitions(sum(int_part_inner)) for int_part_inner in vec_part_outer
            )
            for set_parts_inner in itertools.product(*set_parts_inner_iters):
                vec_part_combined = []
                mult_inner = 1
                for int_part_inner, set_part_inner in zip(
                    vec_part_outer, set_parts_inner
                ):
                    vec_part_inner = set_to_vec_partition(
                        set_part_inner, int_part_inner
                    )
                    vec_part_combined = [*vec_part_combined, *vec_part_inner]
                    if fwd_outer:
                        mult_inner *= pk_to_k_coef(vec_part_inner)
                    else:
                        filtered_vec_part_inner = tuple(
                            [
                                tuple(
                                    [b for a, b in zip(int_part_inner, block) if a != 0]
                                )
                                for block in vec_part_inner
                            ]
                        )
                        filtered_int_part_inner = [a for a in int_part_inner if a != 0]
                        mult_inner *= (
                            1
                            if is_connected(
                                filtered_vec_part_inner, len(filtered_int_part_inner)
                            )
                            else 0
                        )
                vec_part_combined = frozen_multiset(vec_part_combined)
                if vec_part_combined not in mults_combined:
                    mults_combined[vec_part_combined] = 0
                mults_combined[vec_part_combined] += mult_outer * mult_inner
        for vec_part_combined, mult_combined in mults_combined.items():
            assert mult_combined == (
                1 if vec_part_combined == frozen_multiset([int_part_outer]) else 0
            )


@pytest.mark.parametrize("k_max", [1, 2, 3])
@pytest.mark.parametrize("output_all", [True, False])
@pytest.mark.parametrize("mean_var_only", [True, False])
@pytest.mark.parametrize("split", [True, False])
def test_kprop(k_max, output_all, mean_var_only, split, width=4, depth=2):
    treewidth = k_max if mean_var_only else None
    simplify = split
    th.manual_seed(0)
    mlp = MLP(input_dim=width, hidden_dim=width, output_dim=width, num_layers=depth)
    K_in = DSTower.from_tower({1: th.zeros(width), 2: th.eye(width)})
    all_ks_nonsymb = nonsymb_kprop(mlp, K_in=K_in, k_max=k_max, output_all=output_all)
    all_ks_symb = symb_kprop(
        mlp,
        k_max=k_max,
        output_all=output_all,
        mean_only=mean_var_only,
        var_only=mean_var_only,
        simplify=simplify,
        split=split,
        treewidth=treewidth,
        verbose=True,
    )
    if output_all:
        for layer in range(depth):
            for z_or_x in ["z", "x"]:
                if layer == 0 and z_or_x == "z" and k_max == 1:
                    # nonsymb doesn't record input (1, 1) cumulants in this case
                    continue
                if layer == depth - 1 and z_or_x == "x":
                    continue
                pre_or_act = {"z": "pre", "x": "act"}[z_or_x]
                ks_nonsymb = all_ks_nonsymb[f"{pre_or_act}{layer}"]
                ks_symb = all_ks_symb[f"k{z_or_x}"][layer]
                assert (layer == 0 and z_or_x == "z") or (1,) in ks_symb
                for vec in ks_symb:
                    th.testing.assert_close(
                        ks_symb.contract(vec, all_ks_symb["ctx"]),
                        ks_nonsymb[sum(vec)].slices[vec],
                        rtol=2e-3,
                        atol=2e-4,
                    )
    else:
        ks_nonsymb = all_ks_nonsymb
        ks_symb, ctx = all_ks_symb
        for vec in ks_symb:
            th.testing.assert_close(
                ks_symb.contract(vec, ctx=ctx),
                ks_nonsymb[sum(vec)].slices[vec],
                rtol=2e-3,
                atol=2e-4,
            )
