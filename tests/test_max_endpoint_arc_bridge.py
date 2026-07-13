"""ARC tensor bridge: factorized/HTensor contractions vs dense (spec test L6).

For n <= 8, every correction computed against ARC's factorized / trace-sector
representations must match the same correction computed against dense tensors
obtained via .to_tensor(). This test is mandatory before any n=512 run.
"""

import pytest
import torch

from mlp_kprop.harmonic import HTensor
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.max_endpoint.estimator import max_endpoint_estimate
from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)
N = 8
NUM_LAYERS = 3


def _kprop_tower(k_max: int, kind: Kind, factor: bool):
    torch.manual_seed(42)
    mlp = MLP(input_dim=N, hidden_dim=N, output_dim=N, num_layers=NUM_LAYERS)
    k_in = {1: torch.zeros(N), 2: torch.eye(N)}
    return mlp_kprop(mlp, k_in, k_max=k_max, kind=kind, factor=factor)


def _densify(K):
    """Replace K3/K4 by dense r=0 HTensors via .to_tensor()."""
    out = dict(K)
    if 3 in K:
        t3 = K[3].to_tensor() if not isinstance(K[3], HTensor) else K[3].to_tensor()
        out[3] = HTensor(t3.to(torch.float64), r=0)
    if 4 in K:
        out[4] = HTensor(K[4].to_tensor().to(torch.float64), r=0)
    return out


@pytest.mark.parametrize(
    "k_max,kind,factor",
    [(3, Kind.SIMPLE, True), (3, Kind.AUGMENT, True), (2, Kind.AUGMENT, False)],
)
def test_factorized_matches_dense(k_max, kind, factor):
    K = _kprop_tower(k_max, kind, factor)
    res_fac = max_endpoint_estimate(K)
    res_dense = max_endpoint_estimate(_densify(K))
    assert res_fac.corrections.keys() == res_dense.corrections.keys()
    # kprop towers are float32; the dense path symmetrizes in float32 before
    # the float64 cast while the factored path casts first, so agreement is
    # limited by float32 rounding of the cumulants, not by the contraction.
    for name in res_fac.corrections:
        a, b = res_fac.corrections[name], res_dense.corrections[name]
        assert abs(a - b) < 2e-6 * max(1.0, abs(a)), f"{name}: factored={a} dense={b}"
    for name in res_fac.estimates:
        a, b = res_fac.estimates[name], res_dense.estimates[name]
        assert abs(a - b) < 2e-6 * max(1.0, abs(a)), f"{name}: factored={a} dense={b}"


def test_factored_k3_matches_unfactored_kprop():
    """factor=True vs factor=False kprop should give near-identical estimates
    (identical algorithm up to float associativity)."""
    res_fac = max_endpoint_estimate(_kprop_tower(3, Kind.AUGMENT, True))
    res_unfac = max_endpoint_estimate(_kprop_tower(3, Kind.AUGMENT, False))
    for name in res_fac.estimates:
        a, b = res_fac.estimates[name], res_unfac.estimates[name]
        assert abs(a - b) < 1e-3 * max(1.0, abs(a)), f"{name}: {a} vs {b}"


def test_kmax1_only_e0_with_equivalences():
    res = max_endpoint_estimate(_kprop_tower(1, Kind.SIMPLE, False))
    assert set(res.estimates) == {"E0_product_gaussian"}
    assert res.equivalences["E1_cov1"] == "E0_product_gaussian"
    assert "cov_offdiag_unavailable" in res.status


def test_sector_metadata():
    res = max_endpoint_estimate(_kprop_tower(3, Kind.AUGMENT, True))
    assert res.k3_repr == "factored"
    assert res.k4_sector == "r1_traceful"
    res_s = max_endpoint_estimate(_kprop_tower(3, Kind.SIMPLE, True))
    assert res_s.k4_sector == "r2_double_trace"
    res_a2 = max_endpoint_estimate(_kprop_tower(2, Kind.AUGMENT, False))
    assert res_a2.k3_repr == "trace"
    assert res_a2.k4_sector == "r2_double_trace"
