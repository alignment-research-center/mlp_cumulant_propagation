"""ARC representation bridge for the argmax gradient (spec 16.10): dense and
factorized/HTensor contractions must give the same q on real kprop towers."""

import torch

from mlp_kprop.harmonic import HTensor
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.max_endpoint.argmax import argmax_endpoint_estimate
from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)


def _tower(n, k_max, kind, factor, seed=0):
    torch.manual_seed(seed)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=3)
    return mlp_kprop(
        mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=k_max, kind=kind, factor=factor
    )


def test_factored_vs_dense_argmax_gradient():
    """k_max=3 AUGMENT factor=True: FactoredTensor K3 + HTensor trace K4 vs the
    same tower densified via .to_tensor() (allowed at n=8)."""
    n = 8
    K = _tower(n, k_max=3, kind=Kind.AUGMENT, factor=True)
    res_fac = argmax_endpoint_estimate(K)
    K_dense = dict(K)
    K_dense[3] = HTensor(K[3].to_tensor().to(torch.float64), r=0)
    K_dense[4] = HTensor(K[4].to_tensor().to(torch.float64), r=0)
    res_dense = argmax_endpoint_estimate(K_dense)
    for name in res_fac.q_raw:
        # Dense path uses the full retained K4 core; the factored path uses the
        # identical trace surrogate, so C2/C3 sectors agree to solver precision
        # and C4 to representation-identical precision.
        err = float((res_fac.q_raw[name] - res_dense.q_raw[name]).abs().max())
        assert err < 2e-8, f"{name}: factored vs dense q max err {err}"


def test_simple_vs_dense_argmax_gradient():
    """k_max=3 SIMPLE factor=True (double-trace K4 sector)."""
    n = 8
    K = _tower(n, k_max=3, kind=Kind.SIMPLE, factor=True, seed=1)
    res_fac = argmax_endpoint_estimate(K)
    K_dense = dict(K)
    K_dense[3] = HTensor(K[3].to_tensor().to(torch.float64), r=0)
    K_dense[4] = HTensor(K[4].to_tensor().to(torch.float64), r=0)
    res_dense = argmax_endpoint_estimate(K_dense)
    for name in res_fac.q_raw:
        err = float((res_fac.q_raw[name] - res_dense.q_raw[name]).abs().max())
        assert err < 2e-8, f"{name}: factored vs dense q max err {err}"


def test_k2_augment_trace_sectors_run():
    """k_max=2 AUGMENT: trace-surrogate K3 (Sym(v x M)) and double-trace K4
    must produce a full nested family with sum-to-one raw vectors."""
    n = 8
    K = _tower(n, k_max=2, kind=Kind.AUGMENT, factor=False, seed=2)
    res = argmax_endpoint_estimate(K)
    assert res.k3_repr == "trace"
    for name, q in res.q_raw.items():
        assert abs(float(q.sum()) - 1.0) < 1e-9, f"{name}: {float(q.sum())}"
        assert torch.isfinite(q).all()


def test_k1_equivalences_match_scalar():
    """k_max=1: only variances tracked; E1/E2_cov equivalences recorded and only
    E0 emitted, exactly as in the scalar estimator."""
    n = 8
    K = _tower(n, k_max=1, kind=Kind.SIMPLE, factor=False, seed=3)
    res = argmax_endpoint_estimate(K)
    assert set(res.q_raw) == {"E0_product_gaussian"}
    assert res.equivalences["E1_cov1"] == "E0_product_gaussian"
    assert res.equivalences["E2_cov2"] == "E0_product_gaussian"
    assert abs(float(res.q_raw["E0_product_gaussian"].sum()) - 1.0) < 1e-10
