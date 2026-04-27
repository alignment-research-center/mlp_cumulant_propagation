from collections import defaultdict
from itertools import product

import einops
import torch

from mlp_kprop.cumulants import *
from mlp_kprop.cumulants import _DS_pK_to_K_old, _pK_to_K_coef

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

def test_cumulants():
    X = torch.randn(1000, 5)
    for batch_size in [1, 10, 100, 1000]:
        K = DS_cumulant(X, d_max=4, batch_size=batch_size)
        assert torch.allclose(K[1].to_tensor(), X.mean(dim=0))
        assert torch.allclose(K[2].to_tensor(), torch.cov(X.T, correction=0))
        # Third cumulant == third central moment
        X_c = X - X.mean(dim=0, keepdim=True)
        K3 = einops.einsum(X_c, X_c, X_c, "t i, t j, t k -> i j k") / X.shape[0]
        assert torch.allclose(K[3].to_tensor(), K3, atol=1e-6)
        # Just check it has right shape
        assert K[4].to_tensor().shape == (5, 5, 5, 5)


def test_K_to_M():
    n = 5
    d_max = 4
    X = torch.randn(1000, n) @ torch.randn(n, n)
    K = DS_cumulant(X, d_max=d_max).to_tower()
    M = DS_moment(X, d_max=d_max).to_tower()
    for d in range(1, 5):
        assert torch.allclose(K_to_M(K, d), M[d])
        assert torch.allclose(M_to_K(M, d), K[d])


def test_DS_K_to_M():
    n = 5
    d_max = 5
    X = torch.randn(1000, n) @ torch.randn(n, n)
    dsK = DS_cumulant(X, d_max=d_max)
    dsM = DS_moment(X, d_max=d_max)
    for d in range(1, d_max + 1):
        assert torch.allclose(DS_K_to_M(dsK)[d].to_tensor(), dsM[d].to_tensor())
        assert torch.allclose(DS_M_to_K(dsM)[d].to_tensor(), dsK[d].to_tensor())


def test_pK_to_K():
    n = 5
    N = 1000
    d_max = 5
    Xs = [
        torch.randn(N, n) + torch.randn(1, n) * 5,
        torch.randn(N, n) @ torch.randn(n, n) + torch.randn(1, n) * 5,
    ]
    for X in Xs:
        K = DS_cumulant(X, d_max=d_max).to_tower()
        X_pows = torch.concat([X**k for k in range(1, d_max + 1)], dim=1)
        K_pows = DS_cumulant(X_pows, d_max=d_max).to_tower()
        pK = {d: torch.zeros_like(K[d]) for d in range(1, d_max + 1)}
        for d in range(1, d_max + 1):
            # Not tensorized, but fine for testing
            for idxs in product(range(n), repeat=d):
                counts = defaultdict(int)
                for i in idxs:
                    counts[i] += 1
                pK[d][idxs] = K_pows[len(set(idxs))][tuple(i + (counts[i] - 1) * n for i in counts)]
        dspK = DSTower({d: DSTensor.from_tensor(pK[d]) for d in pK})
        dsK_old = _DS_pK_to_K_old(dspK)
        dsK = DS_pK_to_K(dspK)
        for d in range(1, d_max + 1):
            assert torch.allclose(dsK_old[d].to_tensor(), K[d])
            assert torch.allclose(dsK[d].to_tensor(), K[d])
