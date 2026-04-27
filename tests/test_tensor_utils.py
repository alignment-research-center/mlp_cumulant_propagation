import torch
import itertools

from mlp_kprop.tensor_utils import *

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

def test_is_symmetric():
    n = 10
    A = torch.randn(n, n)
    assert not is_symmetric(A)
    assert is_symmetric(A, vec=(1, 2))
    A_sym = (A + A.T) / 2
    assert is_symmetric(A_sym)

    A = torch.randn(n, n, n)
    B = (A + A.permute(1, 0, 2)) / 2
    C = symmetrize(A)

    assert not is_symmetric(A)
    assert not is_symmetric(A, vec=(1, 2, 2))
    assert is_symmetric(A, vec=(1, 2, 3))
    assert not is_symmetric(B)
    assert is_symmetric(B, vec=(1, 1, 2))
    assert not is_symmetric(B, vec=(1, 2, 2))
    assert is_symmetric(B, vec=(1, 2, 3))
    assert is_symmetric(C)
    assert is_symmetric(C, vec=(1, 1, 2))

def test_symmetrize():
    def slow_symmetrize(A):
        d = A.dim()
        assert list(A.shape) == [A.shape[0]] * d, "A must be an order-d tensor of shape (n,)*d."
        perms = list(set(itertools.permutations(range(d))))
        A_sym = torch.zeros_like(A)
        for p in perms:
            A_sym += A.permute(p)
        A_sym /= len(perms)
        return A_sym

    n = 10
    for d in range(1, 5):
        A = torch.randn(*([n] * d))
        A_sym1 = symmetrize(A)
        A_sym2 = slow_symmetrize(A)
        assert torch.allclose(A_sym1, A_sym2), f"Symmetrize failed for d={d}"

    A = torch.randn(n, n, n)
    B = (A + A.permute(1, 0, 2)) / 2
    C = (A + A.permute(0, 2, 1)) / 2
    D = (A + A.permute(2, 1, 0)) / 2
    assert torch.allclose(symmetrize(A, vec=(1, 1, 2)), B)
    assert torch.allclose(symmetrize(A, vec=(1, 2, 2)), C)
    assert torch.allclose(symmetrize(A, vec=(1, 2, 1)), D)
    assert torch.allclose(symmetrize(A, vec=(1, 2, 3)), A)
