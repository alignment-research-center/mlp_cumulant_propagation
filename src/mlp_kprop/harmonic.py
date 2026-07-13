import logging
import math
from functools import cache, partial
import einops
import torch
from jaxtyping import Float
from torch import Tensor
from typing import Any, Callable, Optional, Union
logger = logging.getLogger(__name__)

from mlp_kprop.flop_utils import *

"""
Notation convention for docstrings/comments in this file:
- L is the Laplacian operator
- R is the multiplication by |x|^2 operator
- Harmonic decomposition P_d^n = ⊕_{2r leq d} R^r H_{d-2r}^n is indexed by radial index r
- a⤊n and a⤋n are the rising and falling factorials, respectively
- d and n are polynomial degree and ambient dimension, respectively
    - opposite convention from Dai and Xu
- Cumulants are stored as (A, r, M) where A is a symmetric tensor, r>=0 is an integer,
  and M is a metric (matrix or diagonal), interpreted as Sym(A otimes M^{otimes r}).
    - We call A the "core" and r the "radial index"
    - This is the truncation of the cumulant to ⊕_{r' >= r} R^{r'} H_{d-2r'}^n
"""

from mlp_kprop.diagslice import (
    _diagslice_view, DSTensor, _einsum_delta, zero_repeated, expand_dslice
)
from mlp_kprop.partitions import *
from mlp_kprop.tensor_utils import *

def check_symmetric_or_warn(A: Float[Tensor, '*n'], strict=False) -> Float[Tensor, '*n']:
    # The symmetry check often fails due to numerical issues
    # So by default we just warn and symmetrize
    if strict:
        assert is_symmetric(A), "Input tensor must be symmetric."
        return A
    else:
        if not is_symmetric(A):
            logger.warning("Input tensor is not symmetric, symmetrizing.")
        return symmetrize(A)


@cache
def _identity_metric(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "n"]:
    return torch.ones((n,), device=device, dtype=dtype)


@cache
def _identity_metric_matrix(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "n n"]:
    return torch.eye(n, device=device, dtype=dtype)


def _coerce_metric(
    metric: Union[None, Float[Tensor, "n"], Float[Tensor, "n n"]],
    *,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Union[Float[Tensor, "n"], Float[Tensor, "n n"]]:
    if metric is None:
        return _identity_metric(n, device=device, dtype=dtype)
    metric = torch.as_tensor(metric, device=device, dtype=dtype)
    if metric.ndim == 0:
        metric = metric.expand(n)
    if metric.ndim == 1:
        assert metric.shape == (n,), f"Vector metric must have shape ({n},)."
    elif metric.ndim == 2:
        assert metric.shape == (n, n), f"Matrix metric must have shape ({n}, {n})."
        if (metric - metric.diag().diag()).abs().max() < 1e-10:
            metric = metric.diag()
    else:
        raise ValueError(f"Metric must have ndim 1 or 2, got shape {tuple(metric.shape)}.")
    return metric


def metric_is_identity(
    metric: Union[Float[Tensor, "n"], Float[Tensor, "n n"]],
    *,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    metric = _coerce_metric(metric, n=n, device=device, dtype=dtype)
    if metric.ndim == 1:
        return torch.allclose(metric, _identity_metric(n, device=device, dtype=dtype))
    return torch.allclose(metric, _identity_metric_matrix(n, device=device, dtype=dtype))

class HTensor:
    '''
    Represents a tensor as a pair (core, r) where core is a symmetric tensor.
    Interpreted as Sym(core otimes metric^{otimes r}).
    '''
    def __init__(
        self,
        core: Float[Tensor, '*n'],
        r: int = 0,
        n: Optional[int] = None,
        metric: Union[None, Float[Tensor, "n"], Float[Tensor, "n n"]] = None,
        strict: bool = False,
    ):
        core = torch.as_tensor(core)
        core = check_symmetric_or_warn(core, strict=strict)
        self.core = core
        self.r = r
        if core.ndim == 0:
            assert n is not None, "Must specify n when A is a scalar."
            self.n = n
        else:
            self.n = core.shape[0]
            if n is not None:
                assert n == self.n, "Inconsistent n."
        self.metric = _coerce_metric(
            metric,
            n=self.n,
            device=self.core.device,
            dtype=self.core.dtype,
        )
        self.clear_repeated()

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        # Any mutation of the HTensor state invalidates cached diagonal slices.
        # This also catches augmented assignment like `A.core += 1`.
        if (
            name in {"core", "metric", "r", "n"}
            and "_repeated_cache_state" in self.__dict__
        ):
            object.__setattr__(self, "_repeated_cache_state", None)
            if "repeated" in self.__dict__:
                self.repeated.slices.clear()

    def _current_repeated_cache_state(self) -> tuple[int, int, int, int, int, int]:
        return (
            id(self.core),
            int(self.core._version),
            id(self.metric),
            int(self.metric._version),
            self.r,
            self.n,
        )

    def clear_repeated(self) -> None:
        self.repeated = DSTensor(
            d=self.d,
            n=self.n,
            slices=dict(),
            device=self.device,
            dtype=self.dtype,
        )
        self._repeated_cache_state = self._current_repeated_cache_state()

    def _sync_repeated_cache(self) -> None:
        if (
            "repeated" not in self.__dict__
            or "_repeated_cache_state" not in self.__dict__
            or self._repeated_cache_state != self._current_repeated_cache_state()
        ):
            self.clear_repeated()

    def __getattr__(self, name: str):
        # Backward compatibility for pickled HTensor objects created before metric was added.
        if name == "metric":
            metric = _identity_metric(self.n, device=self.core.device, dtype=self.core.dtype)
            object.__setattr__(self, "metric", metric)
            return metric
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @property
    def d(self) -> int:
        return self.core.ndim + 2 * self.r

    @property
    def ndim(self) -> int:
        return self.d

    @property
    def s(self) -> int:
        return self.core.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n,) * self.d

    @property
    def device(self) -> torch.device:
        return self.core.device

    @property
    def dtype(self) -> torch.dtype:
        return self.core.dtype

    def __repr__(self) -> str:
        if self.has_identity_metric():
            metric_str = "id"
        else:
            metric_str = str(tuple(self.metric.shape))
        return (
            f"HTensor(core={self.core.shape}, d={self.d}, r={self.r}, n={self.n}, "
            f"metric={metric_str})"
        )

    def to(self, *args, **kwargs) -> "HTensor":
        return HTensor(
            self.core.to(*args, **kwargs),
            r=self.r,
            n=self.n,
            metric=self.metric.to(*args, **kwargs),
        )

    def clone(self) -> "HTensor":
        return HTensor(self.core.clone(), r=self.r, n=self.n, metric=self.metric)

    def has_identity_metric(self) -> bool:
        return metric_is_identity(
            self.metric,
            n=self.n,
            device=self.device,
            dtype=self.dtype,
        )

    def to_tensor(self, strict: bool = False) -> Float[Tensor, '*n']:
        '''
        Converts to a standard symmetric tensor by expanding out metric factors.
        '''
        return compose([partial(rad, metric=self.metric, strict=strict)] * self.r)(self.core)

    def get_dslice(
        self,
        part: IntPartition,
    ) -> Float[Tensor, '*n']:
        part = tuple(part)
        assert check_int_partition(part) == self.d, (
            f"Partition {part} does not match HTensor order {self.d}."
        )
        self._sync_repeated_cache()
        sorted_part = tuple(sorted(part, reverse=True))
        if sorted_part not in self.repeated.slices:
            self.repeated.slices[sorted_part] = harmonic_diagslice(self, sorted_part)
        return self.repeated.get_slice(part)

    def contract_W(
        self,
        W: Float[Tensor, "n_out n_in"],
        set_metric: Union[None, Float[Tensor, "n_out"], Float[Tensor, "n_out n_out"]] = None,
    ) -> "HTensor":
        return contract_W(self, W, set_metric=set_metric)

def lap(A: Float[Tensor, '*n'], strict=False) -> Float[Tensor, '*n']:
    """
    Computes the Laplacian of the symmetric tensor A treated as a polynomial.
    This reduces arity by 2.
    """
    A = torch.as_tensor(A)
    if A.ndim < 2:
        return torch.tensor(0., device=A.device, dtype=A.dtype)
    A = check_symmetric_or_warn(A, strict=strict)
    n, d = A.shape[0], A.ndim
    return d * (d - 1) * einops.einsum(A, 'i i ... -> ...')

def rad(
    A: Float[Tensor, '*n'],
    n: Optional[int] = None,
    metric: Union[None, Float[Tensor, "n"], Float[Tensor, "n n"]] = None,
    strict: bool = False,
) -> Float[Tensor, '*n']:
    """
    Multiplies the symmetric tensor A (treated as a polynomial) by the squared radius.
    This increases arity by 2.
    """
    A = torch.as_tensor(A)
    A = check_symmetric_or_warn(A, strict=strict)
    d = A.ndim
    if metric is None:
        if d == 0:
            assert n is not None, "Must specify n when A is a scalar."
        else:
            n = A.shape[0]
        metric = _identity_metric(n, device=A.device, dtype=A.dtype)
    else:
        metric = torch.as_tensor(metric, device=A.device, dtype=A.dtype)
        if d > 0:
            n = A.shape[0]
        elif n is None:
            n = metric.shape[0]
        metric = _coerce_metric(metric, n=n, device=A.device, dtype=A.dtype)
    if metric.ndim == 1:
        metric = torch.diag(metric)
    return symmetrize(
        einops.einsum(
            A, metric,
            '..., i j -> ... i j'
        )
    )

def proj_coef(n: int, d: int, r: int) -> Float[Tensor, "k"]:
    """Dtype-aware wrapper around _proj_coef_impl: the coefficient tensor is
    built with the *current* default dtype, so the cache must key on it
    (otherwise a test or script that flips torch.set_default_dtype poisons
    the cache for every later caller)."""
    return _proj_coef_impl(n, d, r, torch.get_default_dtype())


@cache
def _proj_coef_impl(n: int, d: int, r: int, dtype: torch.dtype) -> Float[Tensor, "k"]:
    """
    Returns the coefficients vector [a_0, a_1, ..., a_{floor(d/2)}] such that
    projection onto the space R^r H_{d-2r}^n is given by
        P_{n,d,r} = sum_{j=0}^{floor(d/2)} a_j R^j L^j.

    Formula:
    Let c = d - r + n/2 - 1.
    P_{n,d,r} = sum_{j=r}{floor(d/2)} (-1)^r * (c - r) / (4^j * r! * (j - r)! * c * (1 - c)⤊j) * R^j L^j
    
    Proof sketch:
    r=0 case is Dai and Xu Lemma 1.2.1. (Recall their notation swaps the roles of d and n.)
        P_{n,d,0} = sum_{j=0}^{floor(d/2)} 1 / (4^j * j! * (1 - c)⤊j) * R^j L^j.
    Now let p = sum_{t geq 0} R^t h_t, where h_t is a harmonic of degree d - 2t.
    Using the relation LR-RL = 2nI + 4dE (where I is identity and E=<x, nabla> is the Euler operator) and induction,
        L^r p = a_r * h_r + R * (...),
    where a_r = 4^r * r! * (c - r+ 1)_r.
    Thus 
        P_{n,d,r} p = R^r h_r = (1 / a_r) R^r P_0 L^r p,
    where the P_0 above is on degree d - 2r polynomials.
    Now substitute the r=0 formula for P_0 on degree d - 2r polynomials.
        (Note the c inside P_0 changes to c - r since the degree is now d - 2r and we reindex j to j-r.)
    """
    c = d - r + n / 2 - 1
    return torch.tensor(
        [0 for _ in range(r)] +
        [
            (-1) ** r * (c - r) / 4 ** j / math.factorial(r) / math.factorial(j - r) / c / math.prod([1 - c + m for m in range(j)])
            for j in range(r, d // 2 + 1)
        ],
        dtype=dtype,
    )

def _multigraph_coef(graph: list[tuple[tuple[int, int], int]], aritys: list[int], lap_coef: bool = True) -> float:
    """
    Given a multigraph on m edges and v vertices, compute the coefficient of the contraction
    corresponding to that multigraph in the expansion of L^m (prod_i f_i), where there
    are v tensors f_i with arity aritys[i].

    Let l be the number of loops in the multigraph.
    Let r_i be the degree of vertex i (counting loops twice).
    Let c_{i,j} be the multiplicity of the edge between vertices i and j.
    Let d_i = aritys[i] and d= sum_i d_i.

    Formula:
        m! * 2^{m - l} * prod_{i=1}^v d_i⤋r_i  / prod_{i<=j} c_{i,j}!

    Proof sketch:
        1. L^m is d⤋(2m) * (m caps)
        2. Number of ways to apply caps = num m-partial pairings of v = d⤋(2m) / 2^m / m!
        3. Number of pairings corresponding to each multigraph = 
            prod_{i=1}^v d_i⤋r_i / 2^l / prod_{i<=j} c_{i,j}!
            (Count ways to attach edge (i, j) to (d_i, d_j), and decrement after taking those legs.
            Then adjust for overcounting.)
        Formula is (1) * (3) / (2).
    """
    m = sum(mult for edge, mult in graph)
    v = len(aritys)
    l = sum(mult for (a, b), mult in graph if a == b)
    r = [0 for _ in range(v)]
    d = sum(aritys)
    for (a, b), mult in graph:
        r[a] += mult
        r[b] += mult
    # TODO: enumerate only graphs that meet the arity condition instead of filtering after
    if any(r[i] > aritys[i] for i in range(v)):
        return 0.
    
    fac = math.factorial
    ret = fac(m) * (2 ** (m - l))
    for i in range(v):
        ret *= math.prod(range(aritys[i] - r[i] + 1, aritys[i] + 1))
    for _, mult in graph:
        ret /= fac(mult)
    if not lap_coef:
        ret /= math.prod(range(d - 2 * m + 1, d + 1))
    return ret

def _lap_m_prod_einexpr(
    graph: list[tuple[tuple[int, int], int]],
    aritys: list[int]
) -> str | None:
    # TODO: Allow specifying which factors are represented as diagonal only and modify einexpr accordingly
    legs = [
        [
            f"i{i}_{j}"
            for j in range(arity)
        ]
        for i, arity in enumerate(aritys)
    ]
    out_legs = set(legs[i][j] for i in range(len(aritys)) for j in range(aritys[i]))
    cur_idx = [0 for _ in aritys]
    for (a, b), mult in graph:
        for _ in range(mult):
            # This works for u == v  and u != v
            idx1 = cur_idx[a]
            cur_idx[a] += 1
            idx2 = cur_idx[b]
            cur_idx[b] += 1
            if idx1 >= aritys[a] or idx2 >= aritys[b]:
                # TODO: enumerate only graphs that meet the arity condition instead of filtering after
                return None
            out_legs.remove(legs[a][idx1])
            out_legs.remove(legs[b][idx2])
            legs[b][idx2] = legs[a][idx1]
    in_expr = ', '.join(
        ' '.join(legs[i]) for i in range(len(aritys))
    )
    out_expr = ' '.join(sorted(out_legs))
    return f"{in_expr} -> {out_expr}"

def _lap_m_prod(m: int, As: list[Float[Tensor, '*n']], strict: bool = False) -> Float[Tensor, '*n']:
    """
    Computes L^m Sym(A_1 otimes A_2 otimes ... otimes A_v) where each A_i is a symmetric tensor.

    TODO: Could be optimized by caching einsum intermediates.
    But this is hard and only affects subleading performance.
    """
    for A in As:
        A = check_symmetric_or_warn(A, strict=strict)
    ret = torch.tensor(0., device=As[0].device, dtype=As[0].dtype)
    aritys = [A.ndim for A in As]
    for graph in multigraphs(len(As), m):
        coef = _multigraph_coef(graph, aritys)
        einexpr = _lap_m_prod_einexpr(graph, aritys)
        if einexpr is not None:
            ret = ret + coef * cached_einsum(*As, einexpr)
    return symmetrize(ret)

def compose(fs: list[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    '''
    Returns the composition of a list of functions (applied right to left).
    '''
    def comp_f(x):
        for f in fs[::-1]:
            x = f(x)
        return x
    return comp_f

@flop_name('harmonic contract_W')
def contract_W(
    A: HTensor,
    W: Float[Tensor, "n_out n_in"],
    set_metric: Union[None, Float[Tensor, "n_out"], Float[Tensor, "n_out n_out"]] = None,
) -> HTensor:
    """
    Contracts every leg of ``A.core`` with ``W`` and updates ``A.metric``.

    By default, metric updates as ``W metric W^T`` (or ``W diag(metric) W^T`` if ``metric`` is 1D).
    If ``set_metric`` is provided, ``A.metric`` must be identity and the output metric is set to ``set_metric``.

    ``set_metric`` is used to replace the metric with E[WW^T]=2/n instead of WW^T when use_avg_metric is True in kprop_harmonic
    """
    assert W.shape[1] == A.n, "W must have input dim matching A.n."
    core = contract_W_basic(A.core, W)
    n_out = W.shape[0]
    if set_metric is None:
        if A.metric.ndim == 1:
            metric = cached_einsum(
                W, A.metric, W,
                "o i, i, p i -> o p",
            )
        else:
            metric = W @ A.metric @ W.T
    else:
        if not A.has_identity_metric():
            raise NotImplementedError(
                "contract_W with set_metric requires HTensor.metric to be identity."
            )
        metric = _coerce_metric(
            set_metric,
            n=n_out,
            device=W.device,
            dtype=W.dtype,
        )
    return HTensor(core=core, r=A.r, n=n_out, metric=metric)

def contract_W_proj(
    A: HTensor,
    W: Float[Tensor, 'n n'],
    r_out: int,
    strict: bool = False,
) -> HTensor:
    """
    Computes P_{geq r_out} W^{otimes d} R^{A.r} A.core as an HTensor with radial index r_out,
    """
    if not A.has_identity_metric():
        raise NotImplementedError(
            "contract_W_proj currently only supports HTensors with identity metric."
        )
    A_core, r_in, n_in, d = A.core, A.r, A.n, A.d
    assert W.shape[1] == n_in, "W must have input dim n"
    n_out = W.shape[0]
    assert r_out <= d // 2, "r_out must be at most d//2."
    P = sum(
        proj_coef(n_out, d, r) for r in range(r_out, d // 2 + 1)
    )
    factors = [contract_W_basic(A_core, W)]
    if r_in > 0:
        if d == 2 and r_in == 1 and r_out == 1:
            # Special handling to avoid n^3 matmul for k_max=1
            # The sole WWT factor will be traced out when projecting, so only need the diagonal
            WWT = W.pow(2).sum(dim=1).diag()
        else:
            WWT = W @ W.T
        factors += [WWT] * r_in
    # if A_core.ndim == 0 and r_in == 1:
    #     import pdb; pdb.set_trace()
    factors = [symmetrize(factor) for factor in factors]
    ret = torch.tensor(0.0, device=A.device, dtype=A.dtype)
    R = partial(rad, n=n_out, strict=strict)
    for r, coef in enumerate(P):
        if r < r_out:
            assert torch.allclose(coef, torch.zeros_like(coef)), "Coefficient should be zero."
            continue
        ret = ret + coef * compose([R] * (r - r_out))(
            _lap_m_prod(r, factors, strict=strict)
        )
    return HTensor(symmetrize(ret), r=r_out, n=W.shape[0], strict=strict)

def proj_geq_r(A: Float[Tensor, '*n'], n: int, r_out: int, strict=False) -> HTensor:
    '''
    Computes P_{geq r} A, where A is a standard tensor.
    where P_{geq r} = sum_{r' >= r} P_{r'} projects onto harmonic components with radial index >= r.
    Output is an HTensor with radial index r_out.
    '''
    if r_out == 0:
        return HTensor(A, r=0, n=n, strict=strict)
    with flop_name(f'proj_geq_r r_out={r_out}'):
        A = check_symmetric_or_warn(A, strict=strict)
        d = A.ndim
        if d > 0:
            assert A.shape[0] == n, "A must have dimension n."
        P = sum(
            proj_coef(n, d, r) for r in range(r_out, d // 2 + 1)
        )
        L = partial(lap, strict=strict)
        R = partial(rad, n=n, strict=strict)
        ret = torch.tensor(0.0, device=A.device, dtype=A.dtype)
        for r, coef in enumerate(P):
            if r < r_out:
                assert torch.allclose(coef, torch.zeros_like(coef)), "Coefficient should be zero."
                continue
            ret = ret + coef * compose([R] * (r - r_out) + [L] * r)(A)
    return HTensor(ret, r=r_out, n=n, strict=strict)
    
def _lap_m_dslice(m: int, dslice: Float[Tensor, '*n'], part: IntPartition) -> Float[Tensor, '*n']:
    '''
    Computes L^m dslice where dslice is a diagonal slice corresponding to the partition part.
    Output is a standard tensor.
    Combinatorially, L can be thought of as the sum over all ways to place a cap on the legs of the tensor, with order of legs mattering, so d(d-1) total.
    Thus L^m is the sum over all ways to place m caps.
    Only caps that connect legs within the same block have nonzero.
    Hence, the sum is over multigraphs on the blocks of part with m edges.
    '''
    # Only loops-only graphs contribute
    # And applying a cap to a block just reduces its size by 2
    graphs = weak_compositions(len(part), m)
    d_out = sum(part) - 2 * m
    ret = torch.zeros(
        [dslice.shape[0]] * d_out,
        device=dslice.device,
        dtype=dslice.dtype
    )
    for graph in graphs:
        if any(2 * graph[i] > part[i] for i in range(len(part))):
            # TODO: enumerate only graphs that meet the arity condition instead of filtering after
            continue
        coef = _multigraph_coef(
            [((i, i), graph[i]) for i in range(len(part)) if graph[i] > 0],
            part
        )
        L_part = tuple(part[i] - 2 * graph[i] for i in range(len(part)))
        # Sum over fully capped legs
        capped = [i for i in range(len(part)) if L_part[i] == 0]
        if capped:
            to_add = dslice.sum(dim=capped)
        else:
            # Annoying edge case: tensor.sum(dim=[]) sums over all dims
            to_add = dslice
        L_part = tuple(b for b in L_part if b > 0)
        _diagslice_view(
            ret, int_to_canonical_set_partition(L_part)
        ).add_(coef * to_add)

    return symmetrize(ret) * int_partition_coef(part)

def DS_harmonic_proj(
    A: DSTensor,
    r_out: int,
    geq: bool = True,
    strict: bool = False,
) -> HTensor:
    '''
    Computes P_{geq r_out} A an an HTensor with radial index r_out,
    where A is represented as a DSTensor.
    If geq is False, computes P_{r_out} A instead.
    '''
    if r_out == 0:
        return HTensor(A.to_tensor(), r=0, n=A.n, strict=strict)
    with flop_name(f'DS_harmonic_proj'):
        n, d = A.n, A.d
        assert r_out <= d // 2, "r_out must be at most d//2."
        if geq:
            P = sum(
                proj_coef(n, d, r) for r in range(r_out, d // 2 + 1)
            )
        else:
            P = proj_coef(n, d, r_out)
        ret = torch.tensor(0., device=A.device, dtype=A.dtype)
        R = partial(rad, n=n, strict=strict)
        for part, dslice in A.slices.items():
            for r, coef in enumerate(P):
                if r < r_out:
                    assert torch.allclose(coef, torch.zeros_like(coef)), "Coefficient should be zero."
                    continue
                ret = ret + coef * compose([R] * (r - r_out))(
                    _lap_m_dslice(r, dslice, part)
                )
        # In principle ret is already symmetric, but do it again bc of numerical issues
        ret = symmetrize(ret)
    return HTensor(ret, r=r_out, n=n, strict=strict)

def _harmonic_diagslice_einexpr(
    graph: list[tuple[tuple[int, int], int]],
    part: IntPartition,
    r: int,
    s: int,
) -> str | None:
    core_legs = [f'core{i}' for i in range(s)]
    edge_legs = [
        [f'gi{i}', f'gj{i}'] for i in range(r)
    ]
    out_legs = [
        f'out{i}' for i in range(len(part))
    ]
    cur_edge = 0
    cur_out_idx = [0 for _ in part]
    for (a, b), mult in graph:
        for _ in range(mult):
            edge_legs[cur_edge][0] = out_legs[a]
            cur_out_idx[a] += 1
            edge_legs[cur_edge][1] = out_legs[b]
            cur_out_idx[b] += 1
            cur_edge += 1
    cur_core_idx = 0
    for i in range(len(part)):
        while cur_out_idx[i] < part[i]:
            core_legs[cur_core_idx] = out_legs[i]
            cur_out_idx[i] += 1
            cur_core_idx += 1
    assert cur_core_idx == s, "Not all core legs used."
    in_expr = ' '.join(
        core_legs
    ) + ', ' + ', '.join(
        ' '.join(edge_legs[i]) for i in range(r)
    )
    out_expr = ' '.join(out_legs)
    in_expr = in_expr.strip(', ')
    return f"{in_expr} -> {out_expr}"

@flop_name('harmonic diagslice')
def harmonic_diagslice(
    A: HTensor, 
    part: IntPartition,
) -> Float[Tensor, '*n']:
    '''
    Returns diagslice of HTensor A corresponding to part.
    A is interpreted as Sym(A.core otimes A.metric^{otimes A.r}).

    TODO: We can save some n^2 computations when metric is diagonal by keeping metric as a 1d vector and 
    modifing _harmonic_diagslice_einexpr to use _einsum_delta for those edges.
    '''
    r, n, s = A.r, A.n, A.s
    ret = torch.zeros(
        [n] * len(part),
        device=A.device,
        dtype=A.dtype
    )
    metric = A.metric
    metric = _coerce_metric(metric, n=n, device=A.device, dtype=A.dtype)
    diag_metric = False
    if metric.ndim == 1:
        logger.debug("Using diagonal metric in harmonic_diagslice.")
        metric = torch.diag(metric)
        diag_metric = True
    elif metric is _identity_metric_matrix(n, device=A.device, dtype=A.dtype):
        logger.debug("Using identity metric in harmonic_diagslice.")
        diag_metric = True
    if diag_metric:
        # Only loops-only graphs contribute
        graphs = weak_compositions(len(part), r)
        graphs = [
            [((i, i), mult) for i, mult in enumerate(graph) if mult > 0]
            for graph in graphs
        ]
    else:
        graphs = multigraphs(len(part), r)
    for graph in graphs:
        coef = _multigraph_coef(
            graph,
            list(part),
            lap_coef=False
        )
        if coef == 0.:
            continue

        # Edge case: einsum doesn't like 0-ary inputs, so we multiply in manually
        einargs = ([A.core] if A.s > 0 else []) + [metric] * r
        einexpr = _harmonic_diagslice_einexpr(
            graph,
            part,
            r=r, s=s,
        )
        term = coef * cached_einsum(*einargs, einexpr)
        if A.s == 0:
            term *= A.core
        ret += term

    return zero_repeated(ret)

# An HTower is a mapping from degree to HTensor
type HTower = dict[int, HTensor]
