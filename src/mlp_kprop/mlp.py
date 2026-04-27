import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Any

import torch
from jaxtyping import Float
from torch import Tensor, nn

from mlp_kprop.wick import *

logger = logging.getLogger(__name__)


class Nonlin(nn.Module):
    """Absolute-value activation."""

    __constants__ = ("inplace",)

    def __init__(self, nonlin: Callable, name: str) -> None:
        super().__init__()
        self.nonlin = nonlin
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        return self.nonlin(x)

    def extra_repr(self) -> str:
        return self.name

# These need to be at global scope for MLP pickling to work
def SQUARE(x):
    return x.pow(2)

def CUBE(x):
    return x.pow(3)

def HEAVISIDE(x):
    return (x > 0).to(x.dtype)

NONLIN_D = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "abs": lambda: Nonlin(torch.abs, "abs"),
    "sgn": lambda: Nonlin(torch.sign, "sgn"),
    "square": lambda: Nonlin(SQUARE, "square"),
    "cube": lambda: Nonlin(CUBE, "cube"),
    "heaviside": lambda: Nonlin(HEAVISIDE, "heaviside"),
}

NONLIN_DERIV_FN = {
    "tanh": lambda x: 1 - torch.tanh(x) ** 2,
    "sigmoid": lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)),
    "relu": lambda x: (x > 0).to(x.dtype),
    "gelu": lambda x: torch.special.ndtr(x) + x * norm_pdf(x),
    "square": lambda x: 2 * x,
    "cube": lambda x: 3 * x ** 2,
    "abs": torch.sign,
}


def _expand_per_layer(
    value: Any,
    *,
    num_layers: int,
    name: str,
) -> list[Any]:
    if value is None:
        return [0.0] * num_layers
    if isinstance(value, list):
        ret = list(value)
    else:
        ret = [value] * num_layers
    if len(ret) != num_layers:
        raise ValueError(
            f"{name} must be a scalar or list of length {num_layers} (got {len(ret)})"
        )
    return ret

def _coerce_nonlin(nonlin: str | list[str], num_layers: int) -> list[str]:
    num_hidden = max(num_layers - 1, 0)
    if isinstance(nonlin, list):
        if len(nonlin) == num_layers:
            # If given one nonlinearity per linear layer, ignore the output entry.
            logger.warning(
                f"Got nonlin of length {len(nonlin)} equal to num_layers; dropping the last entry."
            )
            nonlin = nonlin[:-1]
        elif len(nonlin) != num_hidden:
            raise ValueError(
                f"nonlin must be a string or list of length {num_hidden} or {num_layers}"
                f" (got {len(nonlin)})"
            )
    nonlin_by_layer = _expand_per_layer(nonlin, num_layers=num_hidden, name="nonlin")
    for i, nonlin_i in enumerate(nonlin_by_layer):
        if nonlin_i not in NONLIN_D:
            raise ValueError(f"Unsupported nonlinearity nonlin[{i}]={nonlin_i!r}")
    return nonlin_by_layer

def he_w_var(nonlin: str, bias: float = 0.0) -> float:
    if nonlin in WICK_COEF_D:
        wick_fn = WICK_COEF_D[nonlin]
        mean = torch.as_tensor(bias)
        var = torch.as_tensor(1.)
        return 1 / wick_fn(mean=mean, var=var, k=0, p=2).item()
    else:
        assert bias == 0.0, "For unknown nonlinearities, only bias=0.0 is supported for He initialization (got bias=%s)" % bias
        try:
            return nn.init.calculate_gain(nonlin) ** 2
        except ValueError:
            logger.warning(f"Unknown nonlinearity {nonlin}, defaulting to w_var=1.0")
            return 1.0

def _solve_critical(q_star: float, nonlin_name: str) -> tuple[float, float]:
    """Solve for (σ_w², σ_b²) at the edge of chaos.

    Solves:
        q* = σ_w² E[g(√q* Z)²] + σ_b²
        1  = σ_w² E[g'(√q* Z)²]
    See Pennington et al. "Resurrecting the sigmoid in deep learning
    through dynamical isometry: theory and practice"
    """
    if nonlin_name not in NONLIN_DERIV_FN:
        raise ValueError(
            f"Critical init requires a nonlinearity with a known derivative "
            f"(got {nonlin_name!r}). Supported: {list(NONLIN_DERIV_FN.keys())}"
        )

    wick_fn = WICK_COEF_D[nonlin_name]
    g_prime = NONLIN_DERIV_FN[nonlin_name]
    mean = torch.tensor(0.0)
    var = torch.tensor(q_star)

    eg_prime_sq = hermgauss_wick_coef(g_prime, mean=mean, var=var, k=0, p=2).item()
    eg_sq = wick_fn(mean=mean, var=var, k=0, p=2).item()

    sigma_w_sq = 1.0 / eg_prime_sq
    sigma_b_sq = q_star - sigma_w_sq * eg_sq

    if sigma_b_sq < -1e-6:
        raise ValueError(
            f"Critical initialization infeasible: σ_b² = {sigma_b_sq:.6f} < 0 "
            f"for q_star={q_star}, nonlin={nonlin_name!r}"
        )
    sigma_b_sq = max(sigma_b_sq, 0.0)
    return sigma_w_sq, sigma_b_sq

@dataclass
class MLPOutput:
    out: Float[Tensor, "sample output_dim"]
    pre: Float[Tensor, "sample layer hidden_dim"]
    act: Float[Tensor, "sample layer hidden_dim"]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        nonlin: str | list[str] = "relu",
        init_kind: str = "he",
        q_star: Optional[float] = None,
        w_var: Optional[float | list[float]] = None,
        b_var: Optional[float | list[float]] = None,
        b_mean: Optional[float | list[float]] = None,
        layernorm: bool = False,
        ln_before_act: bool = True,
        batch_layernorm: bool = False,
    ):
        """
        A simple feedforward MLP with all hidden layer dims equal.
        If input_dim or output_dim is omitted, it defaults to hidden_dim.
        If ln_before_act, applies LayerNorm before nonlinearity. Else, after.
        act is always stored immediately after nonlinearity (and optional
        batch-layer normalization); pre is always stored immediately after
        linear layer.

        init_kind:
          - 'he': He initialization (default). w_var computed automatically.
          - 'critical': critical initialization (edge of chaos). Requires q_star.
          - 'manual': explicit w_var (required), optional b_var/b_mean.
          Weights are centered gaussians with variance w_var/fan_in.
          Biases are Gaussian with mean b_mean and variance b_var.
        """
        super().__init__()
        if hidden_dim is None:
            raise TypeError("hidden_dim must be provided")
        if num_layers is None:
            raise TypeError("num_layers must be provided")
        if input_dim is None:
            input_dim = hidden_dim
        if output_dim is None:
            output_dim = hidden_dim

        self.Ws = nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlins = nn.ModuleList()
        self.layernorm = layernorm
        self.layernorms = nn.ModuleList()
        self.ln_before_act = ln_before_act
        self.batch_layernorm = batch_layernorm

        nonlin_by_layer = _coerce_nonlin(nonlin, num_layers)
        self.nonlin_names = nonlin_by_layer

        # --- Validation ---
        if init_kind == "he":
            assert w_var is None, "w_var must not be specified for init_kind='he'"
            assert q_star is None, "q_star must not be specified for init_kind='he'"
        elif init_kind == "critical":
            assert w_var is None, "w_var must not be specified for init_kind='critical'"
            assert b_var is None, "b_var must not be specified for init_kind='critical'"
            assert b_mean is None, "b_mean must not be specified for init_kind='critical'"
            assert q_star is not None, "q_star is required for init_kind='critical'"
            if len(set(nonlin_by_layer)) > 1:
                raise ValueError(
                    f"init_kind='critical' requires a uniform nonlinearity across layers "
                    f"(got {nonlin_by_layer})"
                )
        elif init_kind == "manual":
            assert q_star is None, "q_star must not be specified for init_kind='manual'"
            assert w_var is not None, "w_var is required for init_kind='manual'"
        else:
            raise ValueError(f"init_kind must be 'he', 'critical', or 'manual' (got {init_kind!r})")

        # --- Compute per-layer variance/mean lists ---
        if init_kind == "he":
            b_var_by_layer = _expand_per_layer(b_var, num_layers=num_layers, name="b_var")
            b_mean_by_layer = _expand_per_layer(b_mean, num_layers=num_layers, name="b_mean")
            w_var_by_layer = [1.0]
            for i in range(1, num_layers):
                w_var_by_layer.append(he_w_var(nonlin_by_layer[i - 1], bias=b_mean_by_layer[i - 1]))
        elif init_kind == "critical":
            sigma_w_sq, sigma_b_sq = _solve_critical(q_star, nonlin_by_layer[0])
            # Set first layer w_var to q_star-sigma_b_sq so that first (and thus all) layers have pre variance q_star.
            w_var_by_layer = [q_star - sigma_b_sq] + [sigma_w_sq] * (num_layers - 1)
            b_var_by_layer = [sigma_b_sq] * num_layers
            b_mean_by_layer = [0.0] * num_layers
        else:  # manual
            w_var_by_layer = _expand_per_layer(w_var, num_layers=num_layers, name="w_var")
            b_var_by_layer = _expand_per_layer(b_var, num_layers=num_layers, name="b_var")
            b_mean_by_layer = _expand_per_layer(b_mean, num_layers=num_layers, name="b_mean")

        # --- Create layers ---
        for i in range(num_layers):
            has_bias_i = b_var_by_layer[i] > 0 or b_mean_by_layer[i] != 0
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            self.Ws.append(nn.Linear(in_dim, out_dim, bias=has_bias_i))
            self.layernorms.append(nn.LayerNorm(out_dim) if layernorm else nn.Identity())
            if i < num_layers - 1:
                self.nonlins.append(NONLIN_D[nonlin_by_layer[i]]())

        # --- Initialize weights and biases ---
        for i, W in enumerate(self.Ws):
            nn.init.kaiming_normal_(W.weight, nonlinearity="linear")
            W.weight.data *= w_var_by_layer[i] ** 0.5
            if W.bias is not None:
                if b_var_by_layer[i] > 0:
                    nn.init.normal_(W.bias, mean=b_mean_by_layer[i], std=math.sqrt(b_var_by_layer[i]))
                else:
                    nn.init.constant_(W.bias, b_mean_by_layer[i])
        self.init_scale = w_var_by_layer

    def has_bias(self) -> bool:
        return any(W.bias is not None for W in self.Ws)

    @staticmethod
    def _apply_batch_layernorm(x: Tensor, eps: float = 1e-12) -> Tensor:
        """Divide activations by the batch-average L2 norm to stabilize scale."""
        avg_norm = x.norm(dim=1).mean().clamp_min(eps) / (x.shape[1] ** 0.5)
        return x / avg_norm

    def forward(
        self,
        x: Float[Tensor, "sample input_dim"],
        output_acts: bool = False,
        up_to_layer: Optional[str] = None,
    ) -> MLPOutput:
        """
        Forward pass through the MLP.

        Args:
            up_to_layer: Stop after this layer. Takes a string f'pre{l}' or f'act{l}',
                interpreted as go up to the preactivation or activation labeled l,
                respectively. None means go through all layers.
        """
        pre: list[Tensor] = []
        act: list[Tensor] = []
        stopped = False
        for i, nonlin in enumerate(self.nonlins):
            x = self.Ws[i](x)
            layernorm = self.layernorms[i]
            if output_acts:
                pre.append(x)
            if up_to_layer == f'pre{i}':
                stopped = True
                break
            if self.ln_before_act:
                x = nonlin(layernorm(x))
                if self.batch_layernorm:
                    x = self._apply_batch_layernorm(x)
                if output_acts:
                    act.append(x)
            else:
                x = nonlin(x)
                if self.batch_layernorm:
                    x = self._apply_batch_layernorm(x)
                if output_acts:
                    act.append(x)
                x = layernorm(x)
            if up_to_layer == f'act{i}':
                stopped = True
                break

        # NOTE: pre does not include output value (thus pre and act have the same length).
        if not stopped:
            x = self.Ws[-1](x)

        if output_acts:
            if pre:
                pre_out = torch.stack(pre, dim=1)
                act_out = torch.stack(act, dim=1)
            else:
                shape = (x.shape[0], 0, self.hidden_dim)
                pre_out = x.new_empty(shape)
                act_out = x.new_empty(shape)
        else:
            pre_out, act_out = None, None
        return MLPOutput(out=x, pre=pre_out, act=act_out)
