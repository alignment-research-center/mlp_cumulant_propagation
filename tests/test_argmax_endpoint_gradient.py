"""Argmax gradient validation: finite differences (spec 16.2), degree-5
coverage at K=3 (16.4), and the no-n-times-scalar-loop guarantee (16.6)."""

import torch

import mlp_kprop.max_endpoint.argmax as argmax_mod
from mlp_kprop.harmonic import HTensor
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.max_endpoint.argmax import argmax_endpoint_estimate
from mlp_kprop.max_endpoint.diagrams import term_c4_dense
from mlp_kprop.max_endpoint.estimator import max_endpoint_estimate
from mlp_kprop.max_endpoint.hermite_endpoint import EndpointWorkspace
from mlp_kprop.max_endpoint.quadrature import QuadratureCfg
from mlp_kprop.max_endpoint.rooted_diagrams import (
    endpoint_workspace_on_grid,
    rooted_term_reference,
    term_argmax_gradient_autograd,
)
from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)

N = 4
MU = torch.tensor([0.25, -0.15, 0.05, 0.1], dtype=torch.float64)
SIGMA = torch.tensor([1.0, 0.85, 1.15, 0.95], dtype=torch.float64)


def _tower(mu):
    cov = torch.diag(SIGMA**2).clone()
    cov[0, 1] = cov[1, 0] = 0.25
    cov[1, 2] = cov[2, 1] = -0.15
    cov[0, 3] = cov[3, 0] = 0.1
    k3 = torch.zeros(N, N, N, dtype=torch.float64)
    k3[0, 0, 0], k3[1, 1, 1] = 0.12, -0.08
    k3[0, 1, 2] = k3[0, 2, 1] = k3[1, 0, 2] = k3[1, 2, 0] = k3[2, 0, 1] = k3[2, 1, 0] = 0.05
    k4 = torch.zeros(N, N, N, N, dtype=torch.float64)
    k4[0, 0, 0, 0], k4[2, 2, 2, 2] = 0.06, 0.04
    return {
        1: HTensor(mu, r=0),
        2: HTensor(cov, r=0),
        3: HTensor(k3, r=0),
        4: HTensor(k4, r=0),
    }


# ---------------------------------------------------------------------------
# 16.2 finite differences
# ---------------------------------------------------------------------------

def test_finite_difference_gradient():
    """q_i must match [T(mu + h e_i) - T(mu - h e_i)] / 2h for every nested
    estimator: the additive shift changes only the mean (all other cumulants
    of the tower are left untouched), for several step sizes h."""
    res = argmax_endpoint_estimate(_tower(MU.clone()))
    for h in (1e-4, 3e-5):
        fd = {name: torch.zeros(N, dtype=torch.float64) for name in res.q_raw}
        for i in range(N):
            for sgn in (+1.0, -1.0):
                mu_s = MU.clone()
                mu_s[i] += sgn * h
                r = max_endpoint_estimate(_tower(mu_s))
                for name in fd:
                    fd[name][i] += sgn * r.estimates[name] / (2 * h)
        for name, q in res.q_raw.items():
            err = float((q - fd[name]).abs().max())
            assert err < 5e-7, f"{name} h={h}: max FD deviation {err}"


# ---------------------------------------------------------------------------
# 16.4 degree-5 coverage at K=3
# ---------------------------------------------------------------------------

def test_degree_five_endpoint_exercised():
    """The C4 term with all four derivative slots merged carries u_{i,4}; its
    mu_i-gradient needs the degree-5 endpoint weight u_{i,5} (He_4). The
    production gradient must match the explicit rooted contraction that uses
    u_5, and must NOT match a variant truncated at endpoint degree 4."""
    n = 2
    mu = torch.tensor([0.2, -0.1], dtype=torch.float64)
    sigma = torch.tensor([1.0, 0.8], dtype=torch.float64)
    k4 = torch.zeros(n, n, n, n, dtype=torch.float64)
    k4[0, 0, 0, 0], k4[1, 1, 1, 1] = 0.3, 0.2
    term = term_c4_dense()
    tensors = {"k4_dense": k4}

    grad, _ = term_argmax_gradient_autograd(term, tensors, mu, sigma, num_nodes=2048)
    ws = endpoint_workspace_on_grid(mu, sigma, num_nodes=2048)
    rooted = rooted_term_reference(term, tensors, ws)
    assert float((grad - rooted).abs().max()) < 1e-10

    # Truncation guard: dropping every derivative of order 5 must change the
    # answer by a visible margin, so an implementation capped at degree 4
    # cannot pass the comparison above.
    from mlp_kprop.max_endpoint.rooted_diagrams import derivative_from_workspace, _labelings

    q_trunc = torch.zeros(n, dtype=torch.float64)
    for lab in _labelings(term, n):
        beta = {}
        for s in term.slots:
            beta[lab[s]] = beta.get(lab[s], 0) + 1
        tv = float(tensors["k4_dense"][tuple(lab[s] for s in term.factors[0].legs)])
        for r in range(n):
            rooted_beta = dict(beta)
            rooted_beta[r] = rooted_beta.get(r, 0) + 1
            if max(rooted_beta.values()) >= 5:
                continue  # degree-4 truncation
            q_trunc[r] += tv * derivative_from_workspace(ws, rooted_beta)
    q_trunc = float(term.coef) * q_trunc
    gap = float((rooted - q_trunc).abs().max())
    assert gap > 1e-4, f"degree-5 sector unexpectedly negligible (gap={gap})"
    assert float((grad - q_trunc).abs().max()) > 0.5 * gap


# ---------------------------------------------------------------------------
# 16.6 one reverse pass, not n scalar evaluations
# ---------------------------------------------------------------------------

def test_no_per_coordinate_scalar_loop(monkeypatch):
    """The full q vector must come from a fixed number of workspace builds
    (Q and 2Q grids) and endpoint bisections, independent of n, with the same
    forward contraction FLOPs and treewidth as the scalar estimator."""
    counts = {"workspace": 0, "endpoints": 0}
    real_ws = EndpointWorkspace
    real_fe = argmax_mod.find_endpoints

    def counting_ws(*args, **kwargs):
        counts["workspace"] += 1
        return real_ws(*args, **kwargs)

    def counting_fe(*args, **kwargs):
        counts["endpoints"] += 1
        return real_fe(*args, **kwargs)

    monkeypatch.setattr(argmax_mod, "EndpointWorkspace", counting_ws)
    monkeypatch.setattr(argmax_mod, "find_endpoints", counting_fe)

    per_n = {}
    for n in (8, 16):
        torch.manual_seed(0)
        mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=3)
        K = mlp_kprop(
            mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.AUGMENT, factor=True
        )
        counts["workspace"] = counts["endpoints"] = 0
        quad = QuadratureCfg(num_nodes=64)
        res = argmax_endpoint_estimate(K, quad_cfg=quad)
        assert counts["workspace"] == 2, counts  # Q and 2Q grids only
        assert counts["endpoints"] == 1, counts
        scalar = max_endpoint_estimate(K, quad_cfg=quad)
        # Same forward contraction work and treewidth as the scalar endpoint:
        # no accidental extra factor of n anywhere in the argmax path. The
        # argmax forward integrates once per diagram instead of once per term
        # (needed to free each diagram's graph right after its backward),
        # which adds (num_diagrams - num_terms) * 2Q adds: < 1% and width-free.
        assert scalar.flops_endpoint <= res.flops_endpoint_forward <= 1.01 * scalar.flops_endpoint
        assert res.max_treewidth == scalar.max_treewidth
        assert res.flops_endpoint_total == 3 * res.flops_endpoint_forward
        per_n[n] = res.flops_endpoint_total
        # Byproduct scalar estimates must agree with the scalar pipeline.
        for name, v in scalar.estimates.items():
            assert abs(res.scalar_estimates[name] - v) < 1e-10
    # Doubling n must not blow up by an extra factor n (allow the factored-k3
    # rank growth: total should scale strictly slower than n^3).
    assert per_n[16] < per_n[8] * 8
