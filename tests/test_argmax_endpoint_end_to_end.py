"""End-to-end argmax validation on small fixed ReLU MLPs (spec 16.11)."""

import math

import pytest
import torch

from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.max_endpoint.argmax import argmax_endpoint_estimate
from mlp_kprop.max_endpoint.argmax_mse import (
    argmax_flops_per_sample,
    collision_probability,
    mse_unbiased,
    winner_counts,
)
from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)


@pytest.fixture
def float32_default():
    """Pin the default dtype: other test modules set float64 at import time,
    which changes the RNG draws for MLP init (and at width 8 produces towers
    with truncation-negative variances, a pathology tested elsewhere)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(prev)


@pytest.mark.parametrize("n", [8, 16])
def test_small_mlp_end_to_end(n, float32_default):
    torch.manual_seed(10 + n)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=4)
    assert not mlp.has_bias()
    m = 2_000_000
    counts = winner_counts(mlp, num_samples=m, seed=123, device=torch.device("cpu"))
    q_emp = counts.double() / m

    K = mlp_kprop(
        mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.AUGMENT, factor=True
    )
    res = argmax_endpoint_estimate(K)
    # Precondition for the accuracy/sum checks: a healthy tower. Towers with
    # truncation-negative (clamped) variances put unresolvable sigma ~ 1e-5
    # spikes under the quadrature; those are flagged, not asserted, in
    # production (num_clamped_var / simplex_residual).
    assert res.num_clamped_var == 0, res.status

    mc_se = math.sqrt(0.25 / m)  # per-coordinate MC uncertainty bound
    for name, q in res.q_raw.items():
        # Raw estimates: finite, sum to one, close to the empirical winners.
        assert torch.isfinite(q).all(), name
        assert abs(float(q.sum()) - 1.0) < 1e-9, f"{name}: sum {float(q.sum())}"
        u = mse_unbiased(q, counts)
        assert math.isfinite(u)
        direct = float(((q - q_emp) ** 2).sum())
        # The unbiased U-statistic agrees with the plug-in distance up to the
        # O(1/m) collision-bias correction and MC noise.
        assert abs(u - direct) < 200 * mc_se**2 * n + 5e-6, f"{name}: {u} vs {direct}"
        # Projection returns a valid probability vector near the raw one.
        p = res.q_projected[name]
        assert abs(float(p.sum()) - 1.0) < 1e-12
        assert float(p.min()) >= 0.0
        # At tiny widths the truncated expansion can go moderately negative;
        # the projection distance just has to stay bounded.
        assert float((p - q).norm()) < 0.3

    # Accuracy sanity bound: at tiny width the fixed-network winner
    # distribution is highly concentrated (strongly correlated outputs) and
    # the truncated expansion has genuinely large error — that width scaling
    # is exactly what the experiment measures. Bound it loosely per width.
    err_full = float((res.q_raw["E2_full"] - q_emp).norm())
    assert err_full < {8: 0.7, 16: 0.3}[n], f"E2_full l2 error {err_full}"

    # Diagnostics populated.
    assert res.flops_endpoint_total > res.flops_endpoint_forward > 0
    assert res.max_treewidth >= 1
    assert 0.0 < collision_probability(counts) < 1.0
    assert argmax_flops_per_sample(mlp) > 2 * n * n
