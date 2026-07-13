"""Unbiased Brier MSE estimation (spec 16.8) and MC formula (16.9)."""

import math

import torch

from mlp_kprop.max_endpoint.argmax_mse import (
    collision_probability,
    mc_mse_predicted,
    mse_unbiased,
    mse_unbiased_two_sample,
)

torch.set_grad_enabled(False)

Q_TRUE = torch.tensor([0.45, 0.25, 0.2, 0.07, 0.03], dtype=torch.float64)
Q_HAT = torch.tensor([0.4, 0.3, 0.18, 0.09, 0.03], dtype=torch.float64)
TRUE_MSE = float(((Q_HAT - Q_TRUE) ** 2).sum())


def test_mse_unbiased_mean_matches_truth():
    gen = torch.Generator().manual_seed(0)
    reps, m = 20_000, 64
    idx = torch.multinomial(Q_TRUE.expand(reps, -1), m, replacement=True, generator=gen)
    vals = []
    for r in range(reps):
        counts = torch.bincount(idx[r], minlength=Q_TRUE.numel())
        vals.append(mse_unbiased(Q_HAT, counts))
    vals_t = torch.tensor(vals, dtype=torch.float64)
    se = float(vals_t.std()) / math.sqrt(reps)
    assert abs(float(vals_t.mean()) - TRUE_MSE) < 5 * se, (
        f"mean {float(vals_t.mean())} vs true {TRUE_MSE} (se={se})"
    )
    # Negative single-block values are legitimate and must not be clipped.
    assert float(vals_t.min()) < 0.0


def test_two_sample_estimator_agrees():
    gen = torch.Generator().manual_seed(1)
    reps = 400_000
    ij = torch.multinomial(Q_TRUE.expand(2, -1), reps, replacement=True, generator=gen)
    vals = torch.tensor(
        [mse_unbiased_two_sample(Q_HAT, int(ij[0, r]), int(ij[1, r])) for r in range(4000)],
        dtype=torch.float64,
    )
    se = float(vals.std()) / math.sqrt(vals.numel())
    assert abs(float(vals.mean()) - TRUE_MSE) < 5 * se
    # Exact expectation check over all (i, j) pairs, weighted by q x q.
    exact = 0.0
    for i in range(Q_TRUE.numel()):
        for j in range(Q_TRUE.numel()):
            exact += float(Q_TRUE[i] * Q_TRUE[j]) * mse_unbiased_two_sample(Q_HAT, i, j)
    assert abs(exact - TRUE_MSE) < 1e-12


def test_collision_estimator_unbiased():
    gen = torch.Generator().manual_seed(2)
    reps, m = 20_000, 48
    idx = torch.multinomial(Q_TRUE.expand(reps, -1), m, replacement=True, generator=gen)
    vals = torch.tensor(
        [
            collision_probability(torch.bincount(idx[r], minlength=Q_TRUE.numel()))
            for r in range(reps)
        ],
        dtype=torch.float64,
    )
    truth = float((Q_TRUE**2).sum())
    se = float(vals.std()) / math.sqrt(reps)
    assert abs(float(vals.mean()) - truth) < 5 * se


def test_mc_mse_formula():
    """E||q_hat_MC - q||^2 = (1 - ||q||^2)/m for empirical winner frequencies."""
    gen = torch.Generator().manual_seed(3)
    n = Q_TRUE.numel()
    for m in (32, 256):
        reps = 8_000
        idx = torch.multinomial(Q_TRUE.expand(reps, -1), m, replacement=True, generator=gen)
        errs = []
        for r in range(reps):
            freq = torch.bincount(idx[r], minlength=n).double() / m
            errs.append(float(((freq - Q_TRUE) ** 2).sum()))
        errs_t = torch.tensor(errs, dtype=torch.float64)
        predicted = mc_mse_predicted(float((Q_TRUE**2).sum()), m)
        se = float(errs_t.std()) / math.sqrt(reps)
        assert abs(float(errs_t.mean()) - predicted) < 5 * se, (
            f"m={m}: empirical {float(errs_t.mean())} vs predicted {predicted}"
        )
