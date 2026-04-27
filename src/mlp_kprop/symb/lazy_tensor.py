from abc import ABC, abstractmethod

import quimb.tensor as qtn
import torch as th
from mlp_kprop.wick import relu_wick_coef


class EvalContext:
    def __init__(self, *, weights, wick_kzs):
        assert len(weights) == len(wick_kzs) + 1
        self.width = weights[0].shape[0]
        if not all(weight.shape == (self.width, self.width) for weight in weights):
            raise RuntimeError("Weights must be square matrices of the same width.")
        self.weights = weights
        self.wick_kzs = wick_kzs
        self.reset()

    def reset(self):
        self.kz_means_vars = [None] * len(self.wick_kzs)
        self.wick_coefs = [{} for _ in range(len(self.wick_kzs))]
        self.not_delta = None

    def kz_mean_var(self, layer: int, *, dtype, device):
        if self.kz_means_vars[layer] is None:
            wick_kz = self.wick_kzs[layer]
            mean = wick_kz.contract((1,), ctx=self, dtype=dtype, device=device)
            var = wick_kz.contract((2,), ctx=self, dtype=dtype, device=device)
            self.kz_means_vars[layer] = (mean, var)
        mean, var = self.kz_means_vars[layer]
        mean = mean.to(dtype=dtype, device=device)
        var = var.to(dtype=dtype, device=device)
        return mean, var


class LazyTensor(ABC):
    def __init__(self, width: int):
        self.dtype = "lazy"
        self.update_width(width)

    @abstractmethod
    def evaluate(self, ctx: EvalContext, *, dtype=None, device=None):
        pass

    @abstractmethod
    def update_width(self, width: int):
        pass

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.__dict__ == other.__dict__


class Weight(LazyTensor):
    def __init__(self, layer: int, width: int):
        super().__init__(width)
        self.layer = layer

    def evaluate(self, ctx: EvalContext, *, dtype=None, device=None):
        weight = ctx.weights[self.layer]
        return weight.to(dtype=dtype, device=device)

    def update_width(self, width: int):
        self.shape = (width, width)


def create_weight(layer: int, *, width: int):
    weight_tensor = Weight(layer=layer, width=width)
    return qtn.Tensor(weight_tensor, inds=["Z", "X"], tags=qtn.oset([f"W{layer}"]))


class Wick(LazyTensor):
    def __init__(self, layer: int, width: int, *, order: int, power: int):
        super().__init__(width)
        self.layer = layer
        self.order = order
        self.power = power

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def wick_coef(self, mean, var, order, power):
        pass

    def evaluate(self, ctx: EvalContext, *, dtype=None, device=None):
        key = (self.name(), self.order, self.power)
        if key not in ctx.wick_coefs[self.layer]:
            mean, var = ctx.kz_mean_var(self.layer, dtype=dtype, device=device)
            ctx.wick_coefs[self.layer][key] = self.wick_coef(
                mean, var, self.order, self.power
            )
        return ctx.wick_coefs[self.layer][key].to(dtype=dtype, device=device)

    def update_width(self, width: int):
        self.shape = (width,)


class ReLUWick(Wick):
    def name(self):
        return "relu"

    def wick_coef(self, mean, var, order, power):
        return relu_wick_coef(mean, var, order, power)


def create_relu_wick(ind: str, *, layer: int, width: int, order: int, power: int):
    wick_tensor = ReLUWick(layer=layer, width=width, order=order, power=power)
    return qtn.Tensor(
        wick_tensor,
        inds=[ind],
        tags=qtn.oset([f"Wick{layer}", f"ord={order}|pow={power}"]),
    )


class NotDelta(LazyTensor):
    def __init__(self, width: int):
        super().__init__(width)

    def evaluate(self, ctx: EvalContext, *, dtype=None, device=None):
        if ctx.not_delta is None:
            ctx.not_delta = 1 - th.eye(ctx.width, dtype=dtype, device=device)
        return ctx.not_delta.to(dtype=dtype, device=device)

    def update_width(self, width: int):
        self.shape = (width, width)


def create_not_delta(ind1: str, ind2: str, *, width: int):
    return qtn.Tensor(NotDelta(width), inds=[ind1, ind2], tags=qtn.oset(["~"]))
