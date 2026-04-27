from __future__ import annotations

from collections import defaultdict
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from numbers import Real
from pathlib import Path
from typing import Iterator, Iterable, Optional, Any, Callable
import time
import torch
import math
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyvander2d, polyval2d
from itertools import product
from functools import cache

from joblib import Memory

from mlp_kprop.partitions import IntPartition

from torch.utils.flop_counter import FlopCounterMode

logger = logging.getLogger(__name__)

aten = torch.ops.aten

def _shape_numel(shape: Any) -> int:
    """
    Best-effort numel for the shapes produced by torch.utils.flop_counter.shape_wrapper.

    - Tensor -> Tensor.shape (torch.Size)
    - tuple/list of ints -> shape
    - tuple/list of shapes (multi-output) -> sum of numel of each output
    """
    if shape is None:
        return 0
    if isinstance(shape, torch.Size):
        shape = tuple(shape)

    if isinstance(shape, (tuple, list)):
        # scalar tensor has shape ()
        if len(shape) == 0:
            return 1

        # multi-output case: e.g. (torch.Size([...]), torch.Size([...]))
        if any(isinstance(x, (tuple, list, torch.Size)) for x in shape):
            return sum(_shape_numel(x) for x in shape)

        n = 1
        for d in shape:
            n *= int(d)
        return int(n)

    # non-tensor scalar output or unknown
    return 0


def _is_tensor_shape(x: Any) -> bool:
    # Under shape_wrapper, tensors become torch.Size / tuple/list; scalars stay scalars.
    return isinstance(x, (torch.Size, tuple, list))


def _out_numel(out_shape: Any) -> int:
    return _shape_numel(out_shape)


def _group_size(in_shape: Any, out_shape: Any) -> int:
    in_n = _shape_numel(in_shape)
    out_n = _shape_numel(out_shape)
    if in_n == 0 or out_n == 0:
        return 0
    # For reductions, out_n divides in_n (except some empty/degenerate cases).
    return in_n // out_n


# -----------------------
# Elementwise FLOP models
# -----------------------

def _elementwise_1flop(*args, out_shape=None, **kwargs) -> int:
    # Broadcasting is already reflected in out_shape.
    return _out_numel(out_shape)


def _add_like_flop(self_shape, other, alpha=1, out_shape=None, **kwargs) -> int:
    """
    For add/sub:
      - If other is a tensor and alpha != 1, count alpha*other + self as 2 flops/element.
      - Otherwise count as 1 flop/element.
    """
    n = _out_numel(out_shape)
    if n == 0:
        return 0

    # alpha may be SymInt/0-d tensor in some edge cases; best-effort compare.
    alpha_is_one = (alpha == 1)
    if (not alpha_is_one) and _is_tensor_shape(other):
        return 2 * n
    return n


# -------------------
# Reduction FLOP models
# -------------------

def _sum_like_flop(self_shape, *args, out_shape=None, **kwargs) -> int:
    """
    Sum/prod-style reductions: per output element reduce r items -> (r-1) ops.
    """
    out_n = _shape_numel(out_shape)
    r = _group_size(self_shape, out_shape)
    if out_n == 0 or r <= 0:
        return 0
    return out_n * max(r - 1, 0)


def _mean_flop(self_shape, *args, out_shape=None, **kwargs) -> int:
    """
    Mean: (r-1) adds + 1 division per output -> r ops/output.
    """
    out_n = _shape_numel(out_shape)
    r = _group_size(self_shape, out_shape)
    if out_n == 0 or r <= 0:
        return 0
    return out_n * r


def _norm_flop(self_shape, p_or_ord=None, *args, out_shape=None, **kwargs) -> int:
    """
    Best-effort FLOP model for vector/matrix norms.

    Uses r = (#reduced elements per output element).

    - p/ord == 2 (default): r muls (square) + (r-1) adds + 1 sqrt  => 2r flops/output
    - p/ord == 1: r abs + (r-1) adds                               => (2r-1) flops/output
    - p/ord == inf/-inf: r abs (comparisons ignored as FLOPs)      => r flops/output
    - other numeric p: r abs + r pow + (r-1) adds + 1 pow           => 3r flops/output (approx)
    - ord string:
        'fro'/'f' treated like p=2
        'nuc' not supported (returns 0)
    """
    out_n = _shape_numel(out_shape)
    r = _group_size(self_shape, out_shape)
    if out_n == 0 or r <= 0:
        return 0

    p = p_or_ord
    if p is None:
        p = 2

    if isinstance(p, str):
        p = p.lower()
        if p in {"fro", "f"}:
            p = 2
        elif p in {"nuc"}:
            return 0  # nuclear norm is not a "basic reduction" kernel
        else:
            # unknown string ord; best-effort fallback
            p = 2

    # bool is a subclass of int; keep it sane
    if isinstance(p, bool):
        p = int(p)

    if isinstance(p, (int, float)):
        if p == 0:
            return 0
        if p == 1:
            return out_n * (2 * r - 1)
        if p == 2:
            return out_n * (2 * r)
        if math.isinf(p):
            return out_n * r
        return out_n * (3 * r)

    # unknown p type (complex, etc.) -> conservative fallback
    return out_n * (2 * r)


# -----------------------
# Op packet -> formula map
# -----------------------

def _maybe(ops: Iterable[str]) -> list[Any]:
    out = []
    for name in ops:
        pkt = getattr(aten, name, None)
        if pkt is not None:
            out.append(pkt)
    return out


def _build_basic_mapping() -> dict[Any, Callable[..., int]]:
    mapping: dict[Any, Callable[..., int]] = {}

    # Arithmetic binaries (1 flop/output element)
    for pkt in _maybe([
        "mul", "mul_",
        "div", "div_",
        "true_divide", "true_divide_",
        "pow", "pow_",
        "remainder", "remainder_",
        "fmod", "fmod_",
        "floor_divide", "floor_divide_",
    ]):
        mapping[pkt] = _elementwise_1flop

    # Add/sub need alpha handling
    for pkt in _maybe(["add", "add_", "sub", "sub_"]):
        mapping[pkt] = _add_like_flop

    # Unary elementwise (modeled as 1 flop/output element)
    for pkt in _maybe([
        "neg", "neg_",
        "abs", "abs_",
        "reciprocal", "reciprocal_",
        "sqrt", "sqrt_",
        "rsqrt", "rsqrt_",
        "exp", "exp_",
        "log", "log_",
        "log1p", "log1p_",
        "sin", "sin_",
        "cos", "cos_",
        "tanh", "tanh_",
        "sigmoid", "sigmoid_",
        "relu", "relu_",
    ]):
        mapping[pkt] = _elementwise_1flop

    # Reductions
    for pkt in _maybe(["sum", "prod"]):
        mapping[pkt] = _sum_like_flop
    for pkt in _maybe(["mean"]):
        mapping[pkt] = _mean_flop

    # Norms (vector/matrix norms)
    # - torch.norm -> aten.norm (p argument)
    # - torch.linalg.vector_norm -> aten.linalg_vector_norm (ord argument)
    # - torch.linalg.norm -> aten.linalg_norm (ord can be scalar or string)
    for pkt in _maybe(["norm"]):
        mapping[pkt] = _norm_flop
    for pkt in _maybe(["linalg_vector_norm"]):
        mapping[pkt] = _norm_flop
    for pkt in _maybe(["linalg_norm"]):
        mapping[pkt] = _norm_flop

    return mapping


BASIC_FLOP_MAPPING: dict[Any, Callable[..., int]] = _build_basic_mapping()


class ExtendedFlopCounterMode(FlopCounterMode):
    """
    FlopCounterMode + basic elementwise ops + common reductions.

    Broadcasting is handled by using the op's realized output shape (out_shape).
    No special handling for backward-only kernels is added here.
    TODO: Add backward support if we need them
    """
    def __init__(
        self,
        mods=None,
        depth: int = 2,
        display: bool = True,
        custom_mapping: Optional[dict[Any, Any]] = None,
    ):
        merged = dict(BASIC_FLOP_MAPPING)
        if custom_mapping:
            merged.update(custom_mapping)
        super().__init__(mods=mods, depth=depth, display=display, custom_mapping=merged)


# Holds the currently-active counter for this logical execution context.
_ACTIVE_COUNTER: contextvars.ContextVar[Optional["NamedFlopCounter"]] = contextvars.ContextVar(
    "active_named_flop_counter", default=None
)

@dataclass
class _Frame:
    key: str
    start: int
    child_inclusive: int = 0  # sum of inclusive FLOPs from nested flop_name regions


@contextmanager
def flop_name(key: str, factor: float = 1.0) -> Iterator[None]:
    """
    - Nested flop_name is allowed.
    - Accounting is EXCLUSIVE (non-overlapping): parent gets total - sum(children).
    - Hard guard against gaps: if the counter is active and strict, FLOPs may not increase while
      we're outside any flop_name region (depth == 0). Detected at the next boundary.
    - Factor multiplies the counted FLOPs in this region by m (for symmetry adjustments).
    """
    c = _ACTIVE_COUNTER.get()
    if c is None:
        yield
        return

    cur = c.mode.get_total_flops()

    # Gap guard: only meaningful at depth==0 (outside any region).
    if not c._stack and c.strict:
        if cur != c._gap_baseline:
            raise RuntimeError(
                f"Unattributed FLOPs while not in flop_name: {cur - c._gap_baseline} "
                f"(before entering {key!r})."
            )

    frame = _Frame(key=key, start=cur)
    c._stack.append(frame)

    try:
        yield
    finally:
        # frame.start and frame.child_inclusive track raw flops
        # We apply factor only when accumulating to c._accum
        end = c.mode.get_total_flops()
        popped = c._stack.pop()
        assert popped is frame

        inclusive = end - frame.start
        exclusive = inclusive - frame.child_inclusive

        acc_key = frame.key
        if not c.aggregate:
            acc_key = f"{acc_key}_{str(time.time())[-10:]}"

        assert np.allclose(exclusive * factor, round(exclusive * factor)), f"Non-integer FLOP count for {acc_key}: {exclusive * factor}"
        c._accum[acc_key] = c._accum.get(acc_key, 0) + int(round(exclusive * factor))
        c._raw_accum[acc_key] = c._raw_accum.get(acc_key, 0) + exclusive

        # Inform parent about this region's inclusive cost.
        if c._stack:
            c._stack[-1].child_inclusive += inclusive
        else:
            # Leaving the outermost region: advance baseline for gap checking.
            c._gap_baseline = end


@dataclass
class NamedFlopCounter:
    mode: ExtendedFlopCounterMode = field(default_factory=lambda: ExtendedFlopCounterMode(display=False))
    aggregate: bool = True
    strict: bool = False
    _accum: dict[str, int] = field(default_factory=dict)  # flops * factor
    _raw_accum: dict[str, int] = field(default_factory=dict)  # raw flops (no factor)
    _token: Optional[contextvars.Token] = None

    # New internals for nesting + gap guard
    _stack: list[_Frame] = field(default_factory=list, init=False, repr=False)
    _gap_baseline: int = field(default=0, init=False, repr=False)

    def __enter__(self) -> "NamedFlopCounter":
        self.mode.__enter__()
        self._stack.clear()
        self._gap_baseline = self.mode.get_total_flops()
        self._token = _ACTIVE_COUNTER.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        try:
            # Don’t mask an exception already in flight.
            if exc_type is None:
                if self._stack:
                    raise RuntimeError(
                        "NamedFlopCounter exited while inside flop_name (unbalanced contexts)."
                    )
                cur = self.mode.get_total_flops()
                if cur != self._gap_baseline and self.strict:
                    raise RuntimeError(
                        f"Unattributed FLOPs while not in flop_name: {cur - self._gap_baseline} "
                        f"(after last region)."
                    )
            return self.mode.__exit__(exc_type, exc, tb)
        finally:
            # Restore previous active counter (supports nesting).
            prev: "NamedFlopCounter | None" = None
            if self._token is not None:
                _ACTIVE_COUNTER.reset(self._token)
                self._token = None
                prev = _ACTIVE_COUNTER.get()

            # If we re-activated an outer counter, advance its baseline so it
            # doesn’t blame FLOPs accrued while it was inactive as "gaps".
            if prev is not None and not prev._stack:
                prev._gap_baseline = prev.mode.get_total_flops()

    def flops(self, key: str) -> int:
        return self._accum.get(key, 0)

    def flop_dict(self) -> dict[str, int]:
        ret = self._accum.copy()
        ret["resid"] = self.mode.get_total_flops() - sum(self._raw_accum.values())
        return ret
    
    def raw_flop_dict(self) -> dict[str, int]:
        ret = self._raw_accum.copy()
        ret["resid"] = self.mode.get_total_flops() - sum(self._raw_accum.values())
        return ret

    def total(self) -> int:
        return sum(self.flop_dict().values())

    def raw_total(self) -> int:
        return sum(self.raw_flop_dict().values())

@cache
def slice_factor(part: IntPartition, n: int) -> float:
    '''
    Returns the multiplicative factor by which a `part` dslice is cheaper 
    when stored as a (theoretical, unimplemented) symmetric tensor vs a full tensor.
    Let part = (k_1,...,k_d).
    The factor is dim((R^n)^{otimes d}/~) / n^d = prod_c dim S^c(R^n) / n^d = prod_c comb(n+c-1, c) / n^c (stars and bars),
    where ~ is the equivalence relation generated by transposing dimensions (i,j) with part[i]=part[j].
    and the RHS product is over the counts c_i of distinct block sizes in part.
    '''
    classes = defaultdict(int)
    for v in part:
        classes[v] += 1
    return math.prod(math.comb(n+c-1, c) for c in classes.values()) / n**len(part)

@cache
def contract_factor(d: int, n: int) -> float:
    '''
    Returns the multiplicative factor by which a contract_W_basic operation of order d 
    is cheaper when applied to a (theoretical, unimplemented) symmetric tensor vs a full tensor.
    '''
    def i_part(i: int) -> IntPartition:
        return (1,) * i + (2,) * (d - i)
    ret = 0.
    for i in range(1, d+1):
        ret += slice_factor(i_part(i), n)
    return ret / d

@dataclass(frozen=True)
class Poly2D:
    # c[i, j] corresponds to x^i y^j
    c: np.ndarray  # shape (degx+1, degy+1)

    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return polyval2d(x, y, self.c)

    @property
    def degrees(self):
        return (self.c.shape[0] - 1, self.c.shape[1] - 1)

    @staticmethod
    def _pad_coeffs(c: np.ndarray, shape: tuple[int, int], dtype: np.dtype) -> np.ndarray:
        out = np.zeros(shape, dtype=dtype)
        out[: c.shape[0], : c.shape[1]] = c
        return out

    def _binary(self, other: Any, op: Callable[[Any, Any], Any], *, reverse: bool = False):
        if isinstance(other, Poly2D):
            shape = (
                max(self.c.shape[0], other.c.shape[0]),
                max(self.c.shape[1], other.c.shape[1]),
            )
            dtype = np.result_type(self.c, other.c)
            left = self._pad_coeffs(self.c, shape, dtype)
            right = self._pad_coeffs(other.c, shape, dtype)
            return Poly2D(op(right, left) if reverse else op(left, right))

        if isinstance(other, Real):
            dtype = np.result_type(self.c, other)
            scalar_c = np.zeros_like(self.c, dtype=dtype)
            scalar_c[0, 0] = other
            left, right = (scalar_c, self.c) if reverse else (self.c, scalar_c)
            return Poly2D(op(left, right))

        return NotImplemented

    def __add__(self, other):
        return self._binary(other, np.add)

    def __radd__(self, other):
        return self._binary(other, np.add, reverse=True)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __rsub__(self, other):
        return self._binary(other, np.subtract, reverse=True)

    def __neg__(self):
        return Poly2D(-self.c)

    def to_str(
        self,
        var1_name: str = "x",
        var2_name: str = "y",
        *,
        min_deg1: int = 0,
        min_deg2: int = 0,
        zero_tol: float = 1e-8,
        int_tol: float = 1e-8,
        sigfig: int = 3,
    ) -> str:
        c = np.asarray(self.c, dtype=float)
        dx, dy = self.degrees

        if min_deg1 < 0 or min_deg2 < 0:
            raise ValueError("min_deg1 and min_deg2 must be non-negative")

        def fmt_num(a: float) -> str:
            if a >= 10**(sigfig - 1):
                return str(int(round(a)))
            return f"{a:.{sigfig}g}"

        superscript_digits = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

        def fmt_power(var_name: str, exp: int) -> str:
            if exp == 0:
                return ""
            if exp == 1:
                return var_name
            return f"{var_name}{str(exp).translate(superscript_digits)}"

        terms = []
        truncated = False
        for i in range(dx + 1):
            for j in range(dy + 1):
                a = c[i, j]
                if abs(a) <= zero_tol:
                    continue
                if i < min_deg1 or j < min_deg2:
                    truncated = True
                    continue
                # sort later; store exponents + coefficient
                terms.append((i, j, a))

        if not terms:
            return "0 + ..." if truncated else "0"

        # graded lex: higher total degree first, then i, then j
        terms.sort(key=lambda t: (t[0] + t[1], t[0], t[1]), reverse=True)

        pieces = []
        for k, (i, j, a) in enumerate(terms):
            sign = " - " if a < 0 else " + "
            aa = abs(a)

            has_vars = (i != 0) or (j != 0)

            # coefficient: omit 1 for non-constant variable terms
            if has_vars and np.isclose(aa, 1.0, atol=int_tol, rtol=0.0):
                coef_str = ""
            else:
                coef_str = fmt_num(aa)

            monom = ""
            if i != 0:
                monom += fmt_power(var1_name, i)
            if j != 0:
                monom += fmt_power(var2_name, j)

            term_str = coef_str + monom if monom else coef_str  # constant term if monom == ""
            if k == 0:
                pieces.append(("-" if a < 0 else "") + term_str)
            else:
                pieces.append(sign + term_str)
        ret = "".join(pieces)
        if truncated:
            ret += " + ..."
        return ret


# ---------------------------------------------------------------------------
# Cached FLOP polynomial fitting
# ---------------------------------------------------------------------------

def flop_poly_fit(f: Callable[[int, int], Any], deg_n: int, deg_l: int) -> Polynomial:
    # More points than determines the polynomial so we can verify the fit is exact
    ns = list(range(10, deg_n + 13))
    ls = list(range(1, deg_l + 4))

    flops = defaultdict(lambda: [0] * (len(ns) * len(ls)))
    names = {'total'}
    f(ns[0], ls[0])  # einops.einsum needs some warmup flops
    for i, (n, l) in enumerate(product(ns, ls)):
        with NamedFlopCounter(strict=True) as counter:
            f(n, l)
        for name in counter.flop_dict():
            names.add(name)
            flops[name][i] = counter.flops(name)
        flops['total'][i] = counter.total()

    # Fit 2d poly for each flop name
    n_samples = np.array([n for n, l in product(ns, ls)])
    l_samples = np.array([l for n, l in product(ns, ls)])
    V = polyvander2d(n_samples, l_samples, [deg_n, deg_l])  # (len(samples), (n_deg+1)*(l_deg+1))
    polys = {}
    for name in names:
        coef_flat, resid, *_ = np.linalg.lstsq(V, flops[name], rcond=None)
        c = coef_flat.reshape(deg_n + 1, deg_l + 1)     # c[i,j] = coeff of x^i y^j
        polys[name] = Poly2D(c)
        assert np.allclose(polys[name](n_samples, l_samples), flops[name], atol=1e-8), f"Fit failed for {name}"
    return polys

_FLOP_POLY_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "flop_poly_cache"
_flop_poly_memory = Memory(_FLOP_POLY_CACHE_DIR, verbose=0)


@_flop_poly_memory.cache
def _fit_flop_polys(mlp_kwargs, kprop_kwargs):
    """Fit FLOP polynomials for kprop. Cached to data/flop_poly_cache by joblib."""
    from mlp_kprop.mlp import MLP
    from mlp_kprop.kprop_harmonic import mlp_kprop as _mlp_kprop

    k_max = kprop_kwargs['k_max']
    factor = kprop_kwargs.get('factor', False)
    if k_max < 3:
        factor = False

    deg_n = k_max + (0 if factor else 1)
    deg_l = 2 if factor else 1

    # Force output_d_max=1 (only total FLOPs matter for the fit)
    kprop_kw = {**kprop_kwargs, 'output_d_max': 1}

    @torch.no_grad()
    def f(n: int, l: int):
        with flop_name('mlp setup'):
            mlp = MLP(hidden_dim=n, num_layers=l + 2, **mlp_kwargs)
            K_in = {1: torch.zeros(n), 2: torch.eye(n)}
        _mlp_kprop(mlp, K_in, up_to_layer=f'act{l}', **kprop_kw)

    return flop_poly_fit(f, deg_n=deg_n, deg_l=deg_l)


def fit_flop_polys(mlp_kwargs=None, kprop_kwargs=None):
    """Fit or load cached FLOP polynomials for kprop.

    Returns dict of {component_name: Poly2D}.
    """
    if mlp_kwargs is None:
        mlp_kwargs = {}
    if kprop_kwargs is None:
        kprop_kwargs = {}

    mlp_kwargs = dict(mlp_kwargs)
    # These params don't affect FLOP count
    for key in ('init_kind', 'q_star', 'w_var', 'b_var', 'b_mean'):
        mlp_kwargs.pop(key, None)

    print("Fitting FLOP polynomials for mlp_kwargs=", mlp_kwargs, "kprop_kwargs=", kprop_kwargs)

    # Warn about keys that are parameterized over internally
    for key in ('hidden_dim', 'num_layers', 'input_dim', 'output_dim'):
        if key in mlp_kwargs:
            logger.warning(
                "mlp_kwargs[%r] is ignored: width and depth are parameterized over in the poly fit",
                key,
            )
            mlp_kwargs = {k: v for k, v in mlp_kwargs.items() if k != key}

    if 'up_to_layer' in kprop_kwargs:
        logger.warning(
            "kprop_kwargs['up_to_layer'] is ignored: layer is parameterized over in the poly fit"
        )
        kprop_kwargs = {k: v for k, v in kprop_kwargs.items() if k != 'up_to_layer'}

    if 'output_d_max' in kprop_kwargs:
        logger.warning(
            "kprop_kwargs['output_d_max'] is ignored: fit_flop_polys forces output_d_max=1"
        )
        kprop_kwargs = {k: v for k, v in kprop_kwargs.items() if k != 'output_d_max'}

    return _fit_flop_polys(mlp_kwargs, kprop_kwargs)


def get_flops(n, layer, mlp_kwargs, kprop_kwargs, name=None):
    """Evaluate total FLOP polynomial for kprop at given settings and layer.

    Polynomials are fit for act layers. For pre{l}: FLOPs = act_poly(n, l-1) + 2*n**2 (propagate mean through linearity).
    n can be a scalar or array.

    Careful: l=0 means up_to_layer="act0", i.e. do the first linear+nonlinear propagation.
    So l=-1 means "do nothing".
    The poly fit is invalid at l=-1 (since the input is gaussian, the 0-indexed layer has different FLOP counts from the rest)
    So we just hardcode pre0 to 2n^2.
    """
    if name is None:
        return get_flops(n, layer, mlp_kwargs, kprop_kwargs, name='total') - get_flops(n, layer, mlp_kwargs, kprop_kwargs, name='mlp setup')

    polys = fit_flop_polys(mlp_kwargs, kprop_kwargs)
    poly = polys[name]

    n_arr = np.asarray(n, dtype=float)

    if layer == 'pre0':
        return 2 * n_arr ** 2 if name == 'total' else 0
    elif layer.startswith('pre'):
        l = int(layer[3:])
        return poly(n_arr, np.full_like(n_arr, l - 1)) + (2 * n_arr ** 2 if name == 'total' else 0)
    elif layer.startswith('act'):
        l = int(layer[3:])
        return poly(n_arr, np.full_like(n_arr, l))
    else:
        raise ValueError(f"Unknown layer format: {layer}")
