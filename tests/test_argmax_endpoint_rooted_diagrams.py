"""Rooted contractions (spec 16.5): the autograd gradient of every compiled
scalar diagram contraction must equal the explicit rooted contraction built
from the multi-index shift d^{beta} -> d^{beta + e_r}.

The rooted reference iterates the root r over all coordinates, so each case
automatically covers roots that are separate from every cumulant label (new
endpoint-only vertex, extra u_{r,1}, sign flip) and roots coinciding with an
existing vertex (Hermite multiplicity + 1), including repeated-index equality
patterns of the unrestricted sums."""

import itertools
import math

import pytest
import torch

from mlp_kprop.max_endpoint.diagrams import (
    term_c2,
    term_c2_squared_half,
    term_c3_generic,
    term_c4_core_metric,
)
from mlp_kprop.max_endpoint.rooted_diagrams import (
    endpoint_workspace_on_grid,
    rooted_term_reference,
    scalar_term_reference,
    term_argmax_gradient_autograd,
)

torch.set_grad_enabled(False)

N = 4
MU = torch.tensor([0.2, -0.3, 0.05, 0.15], dtype=torch.float64)
SIGMA = torch.tensor([1.0, 0.8, 1.2, 0.9], dtype=torch.float64)
NODES = 1536


def _rand_sym(shape, gen):
    t = torch.randn(*shape, dtype=torch.float64, generator=gen)
    out = torch.zeros_like(t)
    for perm in itertools.permutations(range(t.ndim)):
        out += t.permute(perm)
    return out / math.factorial(t.ndim)


def _tensors(seed=0):
    gen = torch.Generator().manual_seed(seed)
    cov = 0.3 * _rand_sym((N, N), gen)
    cov = cov - torch.diag(cov.diagonal())
    k3 = 0.15 * _rand_sym((N, N, N), gen)
    # Strengthen repeated-index entries so merged equality patterns matter.
    for i in range(N):
        k3[i, i, i] += 0.2 * (-1) ** i
    return {
        "cov_od": cov,
        "k3": k3,
        "k4_core": 0.2 * _rand_sym((N, N), gen),
        "k4_metric": 0.25 * _rand_sym((N, N), gen) + torch.eye(N, dtype=torch.float64),
    }


CASES = {
    "one_cov_edge": term_c2(),
    "two_cov_edges": term_c2_squared_half(),
    "k3_hyperedge_with_repeats": term_c3_generic("k3"),
    "k4_trace_core_metric": term_c4_core_metric(),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_gradient_matches_rooted_contraction(case):
    term = CASES[case]
    tensors = _tensors()
    grad, corr = term_argmax_gradient_autograd(term, tensors, MU, SIGMA, num_nodes=NODES)
    ws = endpoint_workspace_on_grid(MU, SIGMA, num_nodes=NODES)
    rooted = rooted_term_reference(term, tensors, ws)
    scalar = scalar_term_reference(term, tensors, ws)
    # Same grid -> agreement to solver precision, not just quadrature accuracy.
    assert abs(corr - scalar) < 1e-11 * max(1.0, abs(scalar)), f"{case}: scalar mismatch"
    err = float((grad - rooted).abs().max())
    scale = max(1.0, float(rooted.abs().max()))
    assert err < 1e-10 * scale, f"{case}: grad vs rooted max err {err}"
