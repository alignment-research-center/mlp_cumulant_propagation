import logging
import math

import torch

from mlp_kprop.cumulants import DS_cumulant
from mlp_kprop.mlp import MLP
from mlp_kprop.wick import relu_wick_coef

torch.set_grad_enabled(False)
LOG = logging.getLogger(__name__)


@torch.no_grad()
def test_mlp_manual_prop():
    torch.manual_seed(0)
    batch, width, layers = 8, 64, 5
    atol, rtol = 1e-6, 1e-5

    def manual_ln(x, eps=1e-5):
        # x: (B, D) or (B, L, D). normalize on last dim
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + eps)

    for layernorm, ln_before_act in [(False, False), (True, False), (True, True)]:
        mlp = MLP(
            input_dim=width,
            hidden_dim=width,
            output_dim=width,
            num_layers=layers,
            nonlin="relu",
            layernorm=layernorm,
            ln_before_act=ln_before_act,
        )

        x0 = torch.randn(batch, width)
        out = mlp(x0, output_acts=True)
        pre, act, y = out.pre, out.act, out.out

        assert pre is not None and act is not None
        assert pre.shape == (batch, layers - 1, width)
        assert act.shape == (batch, layers - 1, width)

        # Check act from pre for each configuration
        if not layernorm:
            # act_l = ReLU(pre_l)
            assert torch.allclose(act, torch.relu(pre), rtol=rtol, atol=atol)
        elif ln_before_act:
            # act_l = ReLU(LN(pre_l))
            pre_ln = manual_ln(pre)
            assert torch.allclose(act, torch.relu(pre_ln), rtol=rtol, atol=atol)
        else:
            # LN applied after activation, so act_l = ReLU(pre_l)
            assert torch.allclose(act, torch.relu(pre), rtol=rtol, atol=atol)

        # Linear propagation: pre_{l+1} from act_l (with or without LN after act)
        for l in range(layers - 2):
            if layernorm and not ln_before_act:
                nxt_in = manual_ln(act[:, l, :])
            else:
                nxt_in = act[:, l, :]
            expected_next_pre = mlp.Ws[l + 1](nxt_in)
            assert torch.allclose(pre[:, l + 1, :], expected_next_pre, rtol=rtol, atol=atol)

        # Final output: y from last act (LN after act if needed)
        if layernorm and not ln_before_act:
            last_in = manual_ln(act[:, -1, :])
        else:
            last_in = act[:, -1, :]
        expected_out = mlp.Ws[-1](last_in)
        assert torch.allclose(y, expected_out, rtol=rtol, atol=atol)


@torch.no_grad()
def test_mlp_batch_layernorm():
    torch.manual_seed(0)
    batch, width, layers = 6, 32, 4
    atol, rtol = 1e-6, 1e-5
    mlp = MLP(
        input_dim=width,
        hidden_dim=width,
        output_dim=width,
        num_layers=layers,
        nonlin="relu",
        batch_layernorm=True,
    )

    x0 = torch.randn(batch, width)
    out = mlp(x0, output_acts=True)
    pre, act, y = out.pre, out.act, out.out
    assert pre is not None and act is not None
    inp = x0
    for layer in range(layers - 1):
        expected_pre = mlp.Ws[layer](inp)
        assert torch.allclose(pre[:, layer, :], expected_pre, rtol=rtol, atol=atol)
        raw_act = torch.relu(expected_pre)
        avg_norm = raw_act.norm(dim=1).mean().clamp_min(1e-12) / (raw_act.shape[1] ** 0.5)
        expected_act = raw_act / avg_norm
        assert torch.allclose(act[:, layer, :], expected_act, rtol=rtol, atol=atol)
        inp = expected_act
    expected_out = mlp.Ws[-1](inp)
    assert torch.allclose(y, expected_out, rtol=rtol, atol=atol)


@torch.no_grad()
def test_mlp_init_gaussian():
    torch.manual_seed(0)
    width, layers = 1024, 3
    mlp = MLP(
        input_dim=width,
        hidden_dim=width,
        output_dim=width,
        num_layers=layers,
        nonlin="relu",
        init_kind="manual",
        w_var=2.0,
    )

    for i, W_module in enumerate(mlp.Ws):
        W = W_module.weight.detach()
        N = W.numel()
        fan_in = W.shape[1]
        sigma2 = 2.0 / fan_in
        sigma = math.sqrt(sigma2)
        X = W.view(1, -1)  # shape (n=1, samples=N)

        # sample stats
        mean = X.mean().item()
        var = X.var(unbiased=False).item()
        K = DS_cumulant(X.T, d_max=4)
        own_var = K[2].item()

        # 6-sigma tolerances
        tol_mean = 6.0 * math.sqrt(sigma2 / N)  # SE(mean) = sigma / sqrt(N)
        tol_var = 6.0 * math.sqrt(2.0) * sigma2 / math.sqrt(N)  # SE(var)  ~ sqrt(2)*sigma^2/sqrt(N)

        LOG.debug(f"mean: diff {abs(mean):.2g}, tol {tol_mean:.2g}")
        assert abs(mean) < tol_mean
        LOG.debug(f"var: diff {abs(var - sigma2):.2g}, tol {tol_var:.2g}")
        assert abs(var - sigma2) < tol_var
        assert abs(own_var - var) < 1e-8

        # higher-order cumulants near zero for Gaussian init
        c3 = K[3].item()
        c4 = K[4].item()

        # See https://mathematica.stackexchange.com/questions/229568/expressions-for-moments-of-sample-cumulants
        tol_c3 = (
            6.0 * math.sqrt(6.0) * (sigma**3) / math.sqrt(N)
        )  # SE(c3) ~ sqrt(6)*sigma^3/sqrt(N)
        tol_c4 = (
            6.0 * math.sqrt(24.0) * (sigma**4) / math.sqrt(N)
        )  # SE(c4) ~ sqrt(24)*sigma^4/sqrt(N)

        LOG.debug(f"c3: diff {abs(c3):.2g}, tol {tol_c3:.2g}")
        assert abs(c3) < tol_c3
        LOG.debug(f"c4: diff {abs(c4):.2g}, tol {tol_c4:.2g}")
        assert abs(c4) < tol_c4

def test_bias():
    torch.manual_seed(0)

    width, layers = 64, 5
    mlp = MLP(
        input_dim=width,
        hidden_dim=width,
        output_dim=width,
        num_layers=layers,
        nonlin="relu",
        b_mean=[1., 0., -1., 0., 0.],
    )

    # check biases
    assert torch.allclose(mlp.Ws[0].bias, torch.ones_like(mlp.Ws[0].bias))
    assert mlp.Ws[1].bias is None
    assert torch.allclose(mlp.Ws[2].bias, -torch.ones_like(mlp.Ws[2].bias))
    assert mlp.Ws[3].bias is None

    w_p1 = relu_wick_coef(mean=1., var=1., k=0, p=2).item()
    w_m1 = relu_wick_coef(mean=-1., var=1., k=0, p=2).item()

    # check init_scales
    def check_scale(W, expected):
        var = W.weight.var()
        assert torch.allclose(var, torch.as_tensor(expected), atol=3e-3)

    scales = [1/width, 1/width/w_p1, 1/width*2, 1/width/w_m1, 1/width*2]
    for W, scale in zip(mlp.Ws, scales):
        check_scale(W, scale)

@torch.no_grad()
def test_nonlin_list():
    torch.manual_seed(0)
    n = 5
    mlp = MLP(
        input_dim=n,
        hidden_dim=n,
        output_dim=n,
        num_layers=4,
        nonlin=["relu", "square", "cube"],
    )
    for W in mlp.Ws:
        W.weight.data.copy_(torch.eye(n))
        if W.bias is not None:
            W.bias.zero_()

    x0 = torch.randn(3, n) - 0.5
    out = mlp(x0, output_acts=True)
    assert out.act is not None
    assert torch.allclose(out.act[:, 0, :], torch.relu(x0))
    assert torch.allclose(out.act[:, 1, :], torch.relu(x0).pow(2))
    assert torch.allclose(out.act[:, 2, :], torch.relu(x0).pow(6))
    assert torch.allclose(out.out, out.act[:, 2, :])


import pytest

@pytest.mark.parametrize("q_star,expected_sw2,expected_sb2", [
    # Pennington et al. "Resurrecting the sigmoid in deep learning through dynamical isometry"
    (0.0259, 1.05, 2.01e-5),
    (0.8214, 2.0, 0.104),
])
def test_crit_init_tanh(q_star, expected_sw2, expected_sb2):
    width = 4096
    net = MLP(hidden_dim=width, num_layers=3, nonlin="tanh", init_kind="critical", q_star=q_star)
    W = net.Ws[1].weight
    b = net.Ws[1].bias
    assert W.var().item() == pytest.approx(expected_sw2 / width, rel=0.1)
    assert b.var().item() == pytest.approx(expected_sb2, rel=0.1)


@pytest.mark.parametrize("q_star", [0.5, 1.0, 2.0])
def test_crit_init_relu(q_star):
    width = 4096
    net = MLP(hidden_dim=width, num_layers=3, nonlin="relu", init_kind="critical", q_star=q_star)
    W = net.Ws[1].weight
    assert W.var().item() == pytest.approx(2.0 / width, rel=0.1)
    assert net.Ws[1].bias is None
