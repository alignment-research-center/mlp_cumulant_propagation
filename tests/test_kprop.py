import gc
import math

import pytest
import torch
from numpy.polynomial import Polynomial
from pathlib import Path

from mlp_kprop.diagslice import *
from mlp_kprop.mlp import MLP
from mlp_kprop import kprop_harmonic
kprop = kprop_harmonic
from mlp_kprop.kprop_harmonic import Kind
from mlp_kprop.kprop_harmonic import coerce_input as harmonic_coerce_input

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KPROP_KINDS = ["harmonic_old", "simple", "augment"]
H_KIND_DICT = {
    "harmonic_old": Kind.OLD,
    "simple": Kind.SIMPLE,
    "augment": Kind.AUGMENT,
}

def test_harmonic_linear_bias_shifts_mean():
    n = 4
    K = harmonic_coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=2, kind=Kind.SIMPLE)
    W = torch.eye(n)
    bias = torch.linspace(-0.3, 0.3, n)
    K_out = kprop_harmonic.linear_kprop(K, W, k_max=2, bias=bias)
    assert torch.allclose(K_out[1].to_tensor(), bias)

def test_harmonic_nonlin_list_support():
    n = 1
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=3, nonlin=["square", "cube"])
    for W in mlp.Ws:
        W.weight.data.fill_(1.0)
    K_in = harmonic_coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE)
    K_out = kprop_harmonic.mlp_kprop(
        mlp,
        K_in,
        k_max=4,
        kind=Kind.SIMPLE,
        use_avg_metric=True,
        up_to_layer='act1',
    )
    # x ~ N(0,1), then square then cube gives x^6, whose mean is 15.
    assert torch.allclose(K_out[1].to_tensor(), torch.as_tensor([15.0]))

@pytest.mark.parametrize("kprop_kind", KPROP_KINDS)
def test_relu_kprop(kprop_kind):
    kind = H_KIND_DICT.get(kprop_kind, None)
    N = 1_000_000
    n = 32
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=1, num_layers=2)
    x = torch.randn(N, n)
    out = mlp(x)
    z = out.out  # samples n
    mean_true = z.mean(0).item()
    stderr = z.std(0).item() / math.sqrt(N)

    k_max = 2

    K_in = harmonic_coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=k_max, kind=kind)
    K_z1 = kprop.linear_kprop(
        K_in, mlp.Ws[0].weight.data, k_max=k_max,
        set_metric=mlp.init_scale[0],
    )
    K_x1 = kprop.relu_kprop(K_z1, k_max=k_max, kind=kind)
    K_z2 = kprop.linear_kprop(K_x1, mlp.Ws[1].weight.data, k_max=k_max)
    assert abs(K_z2[1].to_tensor().item() - mean_true) < 6.0 * stderr

@pytest.mark.parametrize("kprop_kind", KPROP_KINDS)
def test_square_kprop_1d(kprop_kind):
    kind = H_KIND_DICT.get(kprop_kind, None)
    n = 1
    depth = 3
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth, nonlin="square")
    for i in range(depth):
        mlp.Ws[i].weight.data = torch.ones_like(mlp.Ws[i].weight.data)

    k_max = 2
    sq = Polynomial([0, 0, 1])

    K = harmonic_coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=k_max, kind=kind)

    for i in range(depth - 1):
        K = kprop.linear_kprop(
            K, mlp.Ws[i].weight.data, k_max=k_max,
            set_metric=mlp.init_scale[i],
        )
        K = kprop.poly_kprop(K, poly=sq, k_max=k_max, kind=kind)
        assert torch.allclose(
            K[1].to_tensor(),
            torch.as_tensor(math.prod(range(1, 2 ** (i + 1), 2)), dtype=torch.float64),
        )

@pytest.mark.parametrize("kprop_kind", KPROP_KINDS)
def test_square_kprop(kprop_kind):
    kind = H_KIND_DICT.get(kprop_kind, None)
    N = 10_000_000
    n = 8
    depth = 2
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=1, num_layers=depth, nonlin="square")
    x = torch.randn(N, n)
    out = mlp(x)
    z = out.out  # samples n
    mean_true = z.mean(0).item()
    stderr = z.std(0).item() / math.sqrt(N)

    k_max = 3
    sq = Polynomial([0, 0, 1])

    K = harmonic_coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=k_max, kind=kind)

    for i in range(depth - 1):
        K = kprop.linear_kprop(
            K, mlp.Ws[i].weight.data, k_max=k_max,
            set_metric=mlp.init_scale[i],
        )
        K = kprop.poly_kprop(K, poly=sq, k_max=k_max, kind=kind)
    K = kprop.linear_kprop(K, mlp.Ws[-1].weight.data, k_max=k_max)
    assert abs(K[1].to_tensor().item() - mean_true) < 6.0 * stderr

@pytest.mark.parametrize("kprop_kind", ["simple", "augment"])
def test_goldens(kprop_kind):
    kind = H_KIND_DICT.get(kprop_kind, None)

    goldens_dir = Path(__file__).parent / "goldens" / f"mlp_kprop_{kprop_kind}"

    n = 16
    depth = 3
    k_maxs = range(1, 5)

    K_in = {1: torch.zeros(n, device=device), 2: torch.eye(n, device=device)}

    torch.manual_seed(0)
    mlp = MLP(n, n, n, depth).to(device)

    for k_max in k_maxs:
        # TODO: Write a wrapper class for HTower to make clone and to easier
        K_in = {k: v.clone() for k, v in K_in.items()}
        gc.collect()
        torch.cuda.empty_cache()
        K_by_layer = kprop.mlp_kprop(mlp, K_in, k_max=k_max, output_all=True, kind=kind)
        golden_path = f"{str(goldens_dir)}/n{n}/depth{depth}/kmax{k_max}/K_by_layer.pt"
        golden = torch.load(golden_path, map_location="cpu", weights_only=False)
        for layer in golden.keys():
            if 'pre' in layer:
                continue
            golden[layer] = {d: golden[layer][d].to(device) for d in golden[layer].keys()}
        for layer in K_by_layer.keys():
            if 'pre' in layer:
                continue
            assert layer in golden, f"Layer {layer} not in golden"
            for d in K_by_layer[layer].keys():
                assert torch.allclose(
                    K_by_layer[layer][d].to_tensor(), golden[layer][d].to_tensor().to(device)
                )
