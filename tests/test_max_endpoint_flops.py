"""Analytic contraction FLOPs vs torch's instrumented counter (spec test L9).

Convention (treewidth.FlopTally): joining k tables over a joint index space of
size S costs (k-1)*S multiplies plus S adds for summing out the eliminated
variable. On a pairwise matrix-chain einsum this matches torch's counter
exactly (2*S per step); on group joins torch may use a different association
order, so we check on cases where the schedule is unambiguous.
"""

import torch

from mlp_kprop.flop_utils import ExtendedFlopCounterMode
from mlp_kprop.max_endpoint.treewidth import FlopTally, VEFactor, contract_factors

torch.manual_seed(0)


def test_pairwise_chain_matches_instrumented():
    """Path contraction u0 -A- u1: each elimination joins 2 tables."""
    n, q = 32, 8
    factors = [
        VEFactor(vars=(0,), tensor=torch.randn(q, n), batched=True),
        VEFactor(vars=(1,), tensor=torch.randn(q, n), batched=True),
        VEFactor(vars=(0, 1), tensor=torch.randn(n, n)),
    ]
    domains = {0: n, 1: n}
    tally = FlopTally()
    with ExtendedFlopCounterMode(display=False) as mode:
        contract_factors(factors, domains, order=(0, 1), batch=q, tally=tally)
    instrumented = mode.get_total_flops()
    # Analytic: eliminate 0 -> join {u0, A} over (q, n, n): S = q n^2, cost 2S.
    #           eliminate 1 -> join {u1, prev} over (q, n): S = q n, cost 2S.
    #           final scalar accumulation: q.
    expected = 2 * q * n * n + 2 * q * n + q
    assert tally.flops == expected
    # torch's einsum on 'qa,ab->qb' is a matmul: 2 q n^2 (+ small bookkeeping).
    assert abs(instrumented - expected) <= 0.25 * expected, (
        f"analytic={expected} instrumented={instrumented}"
    )


def test_analytic_flops_scale_with_width():
    """Doubling n must multiply edge-contraction FLOPs by ~4 (O(n^2) tables)."""

    def flops_at(n: int) -> int:
        q = 4
        factors = [
            VEFactor(vars=(0,), tensor=torch.randn(q, n), batched=True),
            VEFactor(vars=(1,), tensor=torch.randn(q, n), batched=True),
            VEFactor(vars=(0, 1), tensor=torch.randn(n, n)),
        ]
        tally = FlopTally()
        contract_factors(factors, {0: n, 1: n}, order=(0, 1), batch=q, tally=tally)
        return tally.flops

    f1, f2 = flops_at(16), flops_at(32)
    ratio = f2 / f1
    assert 3.5 < ratio < 4.5, f"ratio {ratio}"
