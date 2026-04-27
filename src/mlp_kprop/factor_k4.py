import itertools
import einops
import torch
from torch import Tensor
from jaxtyping import Float
import math
from typing import Any, Optional
from tqdm.auto import tqdm
from functools import lru_cache

from mlp_kprop.partitions import *
from mlp_kprop.tensor_utils import *
from mlp_kprop.diagslice import *
from mlp_kprop.harmonic import *
from mlp_kprop.diagslice import _einsum_delta
from mlp_kprop.flop_utils import *
from mlp_kprop.cumulants import *
from mlp_kprop.wick import relu_wick_coef
from mlp_kprop.kprop_harmonic import (
    multiply_wicks,
    get_all_terms_iso,
)

logger = logging.getLogger(__name__)

def _factored_get_dslice(A: Float[Tensor, 'n n r'], B: Float[Tensor, 'n n r'], part: IntPartition) -> Float[Tensor, "*n"]:
    assert tuple(part) == tuple(sorted(part, reverse=True)), f"Partition {part} must be sorted."
    n, r = A.shape[0], A.shape[2]
    assert A.shape == (n, n, r)
    assert B.shape == (n, n, r)
    assert sum(part) == 4
    with flop_name(f"FactoredTensor4.get_dslice"):  # TODO: figure flop factor
        if part == (4,):
            ret = cached_einsum(
                A, B, 'i i r, i i r -> i'
            )
        elif part == (3, 1):
            ret = 1/2 * cached_einsum(
                A, B, 'i i r, i j r -> i j'
            ) + 1/2 * cached_einsum(
                A, B, 'i j r, i i r -> i j'
            )
        elif part == (2, 2):
            ret = 2/3 * cached_einsum(  # 4/6 ways
                A, B, 'i j r, i j r -> i j'
            ) + 1/6 * cached_einsum(    # 1/6 ways
                A, B, 'i i r, j j r -> i j'
            ) + 1/6 * cached_einsum(    # 1/6 ways
                A, B, 'j j r, i i r -> i j'
            )
        elif part == (2, 1, 1):
            ret = 1/3 * cached_einsum(  # 4/12 ways to assign i,i,j,k to legs are equivalent to this after noticing individual symmetries of A and B
                A, B, 'i j r, i k r -> i j k'
            ) + 1/3 * cached_einsum(   # 4/12 ways
                A, B, 'i k r, i j r -> i j k'
            )
            ret += 1/6 * cached_einsum(   # 2/12 ways
                A, B, 'i i r, j k r -> i j k'
            ) + 1/6 * cached_einsum(   # 2/12 ways
                A, B, 'j k r, i i r -> i j k'
            )
        elif part == (1, 1, 1, 1):
            raise NotImplementedError(
                "You shouldn't need to do this (materializing the FactoredTensor is too slow)." +
                " If you need this for testing, use zero_repeated(FT.to_tensor())."
            )
        else:
            assert False, f"Invalid partition {part} for d=4"
    return zero_repeated(ret)

class FactoredTensor4:
    '''
    A symmetric 4-tensor in factored form:
    T_{i, j, k, l} = Sym(sum_{r=1}^R A_{i, j, r} B_{k, l, r}).
    '''
    def __init__(
        self,
        n: int,
        factors: Optional[tuple[Tensor, Tensor]] = None,
        repeated: Optional[DSTensor] = None,
        device=None,
        dtype=None,
        assume_symmetric: bool = False,
    ):
        self.n = n
        if device is None or dtype is None:
            assert factors is not None, "Must specify device and dtype if factors not given"
            device = factors[0].device
            dtype = factors[0].dtype
        self.device = device
        self.dtype = dtype
        self.d = 4
        if factors is None:
            self._factors = (
                torch.zeros((n, n, 0), device=device, dtype=dtype),
                torch.zeros((n, n, 0), device=device, dtype=dtype),
            )
        else:
            if assume_symmetric:
                self._factors = (
                    factors[0].to(device=device, dtype=dtype),
                    factors[1].to(device=device, dtype=dtype),
                )
            else:
                # Symmetrize first two indices of each factor for convenience
                f0 = factors[0].to(device=device, dtype=dtype)
                f1 = factors[1].to(device=device, dtype=dtype)
                self._factors = (
                    (f0 + f0.transpose(0, 1)) / 2,
                    (f1 + f1.transpose(0, 1)) / 2,
                )

        # dslice cache
        # This contains information redundant with self._factors
        # But some slices are slow to compute from factors, so we cache them here
        if repeated is not None:
            assert repeated.d == 4
            assert repeated.n == n
            self.repeated = repeated
        else:
            self.repeated = DSTensor(d=4, n=n, slices=dict(), device=self.device, dtype=self.dtype)

    def clear_repeated(self) -> None:
        self.repeated = DSTensor(d=4, n=self.n, slices=dict(), device=self.device, dtype=self.dtype)

    @property
    def factors(self) -> tuple[Tensor, Tensor]:
        return tuple(A.clone() for A in self._factors)

    @property
    def ndim(self) -> int:
        return 4

    def set_A(self, value: Tensor, clear_cache: bool=True) -> None:
        self._factors = (
            (value + value.transpose(0, 1)) / 2,
            self._factors[1],
        )
        if clear_cache:
            self.clear_repeated()
        
    def set_B(self, value: Tensor, clear_cache: bool=True) -> None:
        self._factors = (
            self._factors[0],
            (value + value.transpose(0, 1)) / 2,
        )
        if clear_cache:
            self.clear_repeated()
    
    @property
    def A(self) -> Tensor:
        return self._factors[0].clone()
    
    @property
    def B(self) -> Tensor:
        return self._factors[1].clone()

    def add_factors_(self, factors: tuple[Tensor, Tensor]) -> None:
        '''
        Add factors and update self.repeated (mutates).
        '''
        new_A, new_B = factors
        new_A = (new_A + new_A.transpose(0, 1)) / 2
        new_B = (new_B + new_B.transpose(0, 1)) / 2
        self._factors = (
            torch.cat((self._factors[0], new_A), dim=2),
            torch.cat((self._factors[1], new_B), dim=2),
        )
        for part in self.repeated.slices:
            self.repeated.slices[part] += _factored_get_dslice(new_A, new_B, part)

    def add_factors(self, factors: tuple[Tensor, Tensor]) -> 'FactoredTensor4':
        '''
        Add factors and update self.repeated (clones).
        '''
        new = self.clone()
        new.add_factors_(factors)
        return new

    def __add__(self, other: 'FactoredTensor4') -> 'FactoredTensor4':
        # TODO: Figure out how to deal with the repeated cache instead of just discarding it
        # For now this doesn't matter because a contract_W immediately follows the only place where __add__ is used
        assert self.n == other.n
        new_factors = (
            torch.cat((self._factors[0], other._factors[0]), dim=2),
            torch.cat((self._factors[1], other._factors[1]), dim=2),
        )
        return FactoredTensor4(
            self.n,
            new_factors,
            device=self.device,
            dtype=self.dtype,
            assume_symmetric=True,
        )

    def to_tensor(self) -> Float[Tensor, "n n n n"]:
        # Symmetrize is slightly wasteful because (i, j) and (k, l) are already symmetric. But whatever.
        return symmetrize(
            cached_einsum(
                self._factors[0], self._factors[1],
                "i j r, k l r -> i j k l"
            )
        )
    
    def get_dslice(self, part: IntPartition) -> Float[Tensor, "*n"]:
        sorted_part = tuple(sorted(part, reverse=True))
        if sorted_part not in self.repeated.slices:
            self.repeated.slices[sorted_part] = _factored_get_dslice(
                self._factors[0], self._factors[1], sorted_part
            )
        return self.repeated.get_slice(part)

    @flop_name("FactoredTensor4.contract_W")
    def contract_W(self, W: Float[Tensor, "n_out n_in"]) -> 'FactoredTensor4':
        A, B = self._factors
        einexpr = 'i1 j1 r, i2 i1, j2 j1 -> i2 j2 r'
        A_new = cached_einsum(A, W, W, einexpr)
        B_new = cached_einsum(B, W, W, einexpr)
        return FactoredTensor4(
            self.n,
            (A_new, B_new),
            device=self.device,
            dtype=self.dtype,
            assume_symmetric=True,
        )

    @flop_name("FactoredTensor4.contract_wick_")
    def contract_wick_(self, wick: Float[Tensor, 'n']):
        # TODO: add support for distinct wick coefs on each leg (necessary for factoring the full harmonic algo, probably)
        for part in list(self.repeated.slices):
            letters = 'ijkl'
            slice_expr = ' '.join(letters[:len(part)])
            wick_expr= ', '.join(
                ', '.join(letters[i] for _ in range(part[i]))
                for i in range(len(part))
            )
            einexpr = f'{slice_expr}, {wick_expr} -> {slice_expr}'
            self.repeated.slices[part] = cached_einsum(
                self.repeated.slices[part],
                wick, wick, wick, wick,
                einexpr
            )
        self._factors = (
            self._factors[0]
            * wick[:, None, None]
            * wick[None, :, None],
            self._factors[1]
            * wick[:, None, None]
            * wick[None, :, None],
        )

    def contract_wick(self, wick: Float[Tensor, 'n'] | tuple[Float[Tensor, 'n'], ...]) -> 'FactoredTensor4':
        new = self.clone()
        new.contract_wick_(wick)
        return new

    def clone(self) -> 'FactoredTensor4':
        new_factors = tuple(
            factor.clone() for factor in self._factors
        )
        return FactoredTensor4(
            n=self.n,
            factors=new_factors,
            repeated=self.repeated.clone(),
            device=self.device,
            dtype=self.dtype,
            assume_symmetric=True,
        )
        
    def get_repeated(self) -> DSTensor:
        '''
        Returns a DSTensor B satisfying
            zero_repeated(self.to_tensor()) + B.to_tensor() = self.to_tensor()
        This is different from just getting self.repeated because it forces all dslices to be computed.
        '''
        slices = dict()
        for part in int_partitions(self.d):
            # Skip all-distinct slice
            if all(p == 1 for p in part):
                continue
            slices[part] = self.get_dslice(part)
        return DSTensor(d=self.d, n=self.n, slices=slices, device=self.device, dtype=self.dtype)

    @staticmethod
    @flop_name("FactoredTensor4.from_dstensor")
    def from_dstensor(ds: DSTensor) -> 'FactoredTensor4':
        assert ds.d == 4
        assert (1, 1, 1, 1) not in ds.slices, "DSTensor has all-distinct slice; cannot convert to FactoredTensor"
        # All dslice diagrams can be factored with a three-identity on the left
        # So we can sum into a single term.
        # Coefficients are partition.int_partition_coef(part), due to DSTensor.to_tensor scaling
        A = _einsum_delta(
            torch.ones((ds.n,), device=ds.device, dtype=ds.dtype), 'i -> i i i'
        )
        # Make sure 3rd leg of B is the inner leg of the factorization
        B = (
            1 * _einsum_delta(
                ds.get_slice((4,)), 'i -> i i i'
            ) +
            4 * _einsum_delta(
                ds.get_slice((3, 1)), 'i j -> j i i'
            ) +
            3 * _einsum_delta(
                ds.get_slice((2, 2)), 'i j -> j j i'
            ) +
            6 * ds.get_slice((2, 1, 1)).permute(1, 2, 0)
        )
        repeated = ds
        repeated.slices.pop((1, 1, 1, 1), None)
        return FactoredTensor4(
            n=ds.n,
            factors=(A, B),
            repeated=repeated,
            device=ds.device,
            dtype=ds.dtype
        )

type FacHTower = dict[int, FactoredTensor4| HTensor]

def factored_nonlin_kprop_k4(
    K_in: FacHTower,
    nonlin_wick_coef: Callable[[float, float, int, int], float],
    augment: bool = False,
    base: bool = False,
    use_pK: bool = True,
) -> FacHTower:
    '''
    Nonlinear step of cumulant propagation for k_max=4.
    K_in should be the output of linear_kprop (with non-identity metric and bias already applied).
    '''
    assert not (base and augment), "base and augment modes are mutually exclusive"
    if not use_pK and not base:
        raise NotImplementedError("use_pK=False only implemented for base=True")
    WK = K_in
    with flop_name('setup'):
        n = WK[1].n
        # Get propagated mean and variance
        assert WK[1].r == 0
        mean = WK[1].core
        assert WK[2].r == 0
        var = WK[2].core.diag()
        assert mean.ndim == 1, "Mean must be a vector."
        assert var.ndim == 1, "Variance must be a vector."

    # 3.0 Setup for nonlinearity
    @cache
    @flop_name('get_wick_coef')
    def get_wick_coef(k: int, p: int) -> Float[Tensor, "n"]:
        return nonlin_wick_coef(mean=mean, var=var, k=k, p=p)

    pK_slices = defaultdict(lambda: 0.)

    # 3.1 Compute pK slices that don't need to be factored
    terms_iso = get_all_terms_iso(k_max=4, d_max=6 if augment else 4)
    terms_iso = [
        (int_part, vec_part, count)
        for int_part, vec_part_dict in terms_iso.items()
        for vec_part, count in vec_part_dict.items()
        if len(int_part) <= 4
        and (use_pK or all(p == 1 for p in int_part))  # If not use_pK, only need (1, ..., 1) int_parts
        and (augment or len(int_part) < 4)
        and int_part != (1, 1, 1, 1)   # Factor this manually
        and (int_part, set(vec_part)) != ((2, 1, 1, 1), {(1, 1, 1, 1,)})   # Mult wick coefs and carry over to K2111_contrib manually
        and int_part != (2, 2, 1, 1)   # Contribution to H(d=6,r=3) is zero bc of distinct indices (output metric is always identity!)
    ]
    pK_slices = defaultdict(lambda: 0.0)
    for int_part, vec_part, count in tqdm(
        terms_iso,
        disable=logger.getEffectiveLevel() > logging.INFO,
        desc="nonlin-kprop",
    ):
        with flop_name(f'nonlin_sum', factor=slice_factor(int_part, n=n)):
            term = eval_part(WK, vec_part, len(int_part), output_zero_repeated=use_pK)
            if term is None:
                continue
            pK_slices[int_part] += count * multiply_wicks(
                term,
                check_vec_partition(
                    vec_part, len(int_part)
                ),  # check_vec_partition returns sum of partition vectors
                p=int_part,
                wick_lookup=get_wick_coef,
            )
    # Since we sum over iso classes * count instead of all terms, each slice is not symmetric wrt its int_part
    # So we symmetrize here
    for int_part in pK_slices:
        pK_slices[int_part] = symmetrize(pK_slices[int_part], vec=int_part)

    # The (1,1,1,1) factored section below only accesses <=3-block dslices,
    # so 4-dslices are no longer needed.
    # import pdb; pdb.set_trace()
    if 5 in WK:
        WK[5].repeated.slices.pop((2, 1, 1, 1), None)
    if 6 in WK:
        WK[6].repeated.slices.pop((2, 2, 1, 1), None)

    # 3.2 Compute pK slices that do need to be factored: just (1, 1, 1, 1)
    def dsWK(*part: IntPartition) -> Float[Tensor, "*n"]:
        '''
        Convenience function to get dslices of WK
        '''
        d = sum(part)
        if d not in WK:
            return None
        elif d < 4:
            assert WK[d].r == 0
            ret = diagslice(WK[d].core, part)
        elif d == 4:
            assert isinstance(WK[d], FactoredTensor4)
            ret = diagslice(WK[d], part, output_zero_repeated=use_pK)
        else:
            assert d in [5, 6]
            assert isinstance(WK[d], HTensor)
            ret = expand_dslice(WK[d], part, output_zero_repeated=use_pK)

        # Hacky way to incorporate vec_part_coef
        # Since there are no multiplicities in the vector partitions we consider,
        # the vector partition coefficient is just 1 / prod_v v!
        # where the product is over all vectors in the partition,
        # Thus the coefficient factors by edge (i.e. by vector in partition).
        return ret / math.prod(math.factorial(p) for p in part)

    w = lambda k: get_wick_coef(k, p=1)

    # (1, 1, 1, 1) contrib
    if 4 in WK:
        assert isinstance(WK[4], FactoredTensor4)
        pK_1111 = WK[4].clone()
        pK_1111.contract_wick_(w(1))
    else:
        pK_1111 = FactoredTensor4(n=n, device=mean.device, dtype=mean.dtype)

    with flop_name("nonlin_sum 1111 factored"):
        # Isolate out a 2-vertex edge with only one vertex incident to the rest of the graph
        # and group by the incidence of the edge with that vertex (by convention, vertex j)
        # Each B factor is the sum over the three possible graphs (up to permutation): path, star, 3+2
        # Keep new factors separate to avoid taking dslices of old factors repeatedly
        tmp = torch.empty((n, n, n), device=mean.device, dtype=mean.dtype)
        for A_j_inc in (1, 2):
            A_term = None
            dsA1 = dsWK(1, A_j_inc)
            if dsA1 is not None:
                A_term = dsA1 * w(1)[:, None]
            dsA2 = dsWK(2, A_j_inc)
            if dsA2 is not None:
                if A_term is None:
                    A_term = dsA2 * w(2)[:, None]
                else:
                    A_term = A_term + dsA2 * w(2)[:, None]
            if A_term is None:
                continue
            A = _einsum_delta(A_term, 'i j -> i j j')
            B = torch.zeros((n, n, n), device=mean.device, dtype=mean.dtype)
            B_jkl = B.permute(2, 0, 1)
            # Path with edges A(i, j), X(j, k), Y(k, l)
            for X_j_inc, X_k_inc, Y_k_inc, Y_l_inc in product((1, 2), repeat=4):
                X_jk = dsWK(X_j_inc, X_k_inc)
                if X_jk is None:
                    continue
                Y_kl = dsWK(Y_k_inc, Y_l_inc)
                if Y_kl is None:
                    continue
                torch.mul(X_jk[:, :, None], Y_kl[None, :, :], out=tmp)
                tmp.mul_(w(A_j_inc + X_j_inc)[:, None, None])
                tmp.mul_(w(X_k_inc + Y_k_inc)[None, :, None])
                tmp.mul_(w(Y_l_inc)[None, None, :])
                B_jkl.add_(tmp, alpha=12)  # number of path graphs
            # Star with edges A(i, j), X(j, k), Y(j, l)
            for X_j_inc, X_k_inc, Y_j_inc, Y_l_inc in product((1, 2), repeat=4):
                X_jk = dsWK(X_j_inc, X_k_inc)
                if X_jk is None:
                    continue
                Y_jl = dsWK(Y_j_inc, Y_l_inc)
                if Y_jl is None:
                    continue
                torch.mul(X_jk[:, :, None], Y_jl[:, None, :], out=tmp)
                tmp.mul_(w(A_j_inc + X_j_inc + Y_j_inc)[:, None, None])
                tmp.mul_(w(X_k_inc)[None, :, None])
                tmp.mul_(w(Y_l_inc)[None, None, :])
                B_jkl.add_(tmp, alpha=4)  # number of star graphs
            # 3+2 with edges A(i, j), B(j, k, l)
            for B_j_inc, B_k_inc, B_l_inc in product((1, 2), repeat=3):
                B_jkl_term = dsWK(B_j_inc, B_k_inc, B_l_inc)
                if B_jkl_term is None:
                    continue
                tmp.copy_(B_jkl_term)
                tmp.mul_(w(A_j_inc + B_j_inc)[:, None, None])
                tmp.mul_(w(B_k_inc)[None, :, None])
                tmp.mul_(w(B_l_inc)[None, None, :])
                B_jkl.add_(tmp, alpha=12)  # number of 3+2 graphs
            pK_1111.add_factors_((A, B))

        # WK2111 -> pK1111
        if augment and 5 in WK:
            assert WK[5].r == 1
            core = WK[5].core
            metric = WK[5].metric
            I = torch.eye(n, device=mean.device, dtype=mean.dtype)
            # We're taking the 2111 diagslice of a H(d=5, r=1) tensor. There are 3 possibilities for where the 2-block goes:
            # 1. 2-block goes on core (sym_coef=3/10 of possible pairings)
            #    core_{iij} metric_{kl}     (already factored)
            # 2. 2-block bridges core and metric (sym_coef=6/10 of possible pairings)
            #    core_{ijk} metric_{kl} = sum_r core_{ijr} metric_{kl} I_{kr}
            # 3. 2-block goes on metric (sym_coef=1/10 of possible pairings)
            #    core_{ijk} metric_{ll} = sum_r core_{ijr} metric_{ll} I_{kr}
            # Note 2 and 3 can be combined
            # Then there's another factor of
            #   terms_iso[(1,1,1,1)][((2,1,1,1),)] *  vec_part_coef(((2,1,1,1),), divide_fac=True) = 4 * 1/2 = 2
            # So the final coef is 2 * sym_coef
            # (Note that dsWK took care of vec_part_coef in previous steps,
            #  but can't be used here bc we need to factor the 2111 dslice, not materialize it)

            if metric.ndim == 1:
                metric_full = metric.diagflat()
                metric_diag = metric
            elif metric.ndim == 2:
                metric_full = metric
                metric_diag = metric.diagonal()
            else:
                raise ValueError(f"Invalid metric shape {tuple(metric.shape)}")

            # 3 and (if metric nondiag) 2 combined
            A = core * w(1)[:, None, None] * w(1)[None, :, None]
            B = I[:, None, :] * metric_diag[None, :, None] * w(1)[:, None, None] * w(2)[None, :, None] / 5
            if metric.ndim == 2:
                B += I[:, None, :] * metric_full[:, :, None] * w(2)[:, None, None] * w(1)[None, :, None] * 12/10
            pK_1111.add_factors_((A, B))

            if metric.ndim == 2:
                #1
                A = zero_repeated(diagslice(core, (2, 1)))[:,:,None] * w(2)[:,None,None] * w(1)[None,:,None] 
                B = metric_full[:, :, None] * 6/10 * w(1)[:, None, None] * w(1)[None, :, None]
                pK_1111.add_factors_((A, B))

        # WK2211 -> pK1111
        # This is zero unless metric is nondiagonal
        if augment and 6 in WK and WK[6].metric.ndim == 2:
            assert WK[6].r == 3
            core = WK[6].core
            metric = WK[6].metric
            I = torch.eye(n, device=mean.device, dtype=mean.dtype)

            # Four possibilities out of 45 total ways to assign two 2-blocks to (metric)^{otimes 3}
            # 1. Each 2-block goes on a metric (sym_coef=3/45)
            #    metric_{ii} metric_{jj} metric_{kl}    (already factored)
            # 2. Both 2-blocks bridge the same pair of metrics   (sym_coef=6/45)
            #    metric_{ij} metric_{ij} metric_{kl}    (already factored)
            # 3. One 2-block goes on a metric; the other bridges two other metrics   (sym_coef=12/45)
            #    metric_{ii} metric_{jk} metric_{kl} = sum_r (metric_{ii} metric_{jr}) (metric_{kl} I_{kr})
            # 4. Each 2-block bridges a different pair of metrics   (sym_coef=24/45)
            #    metric_{ij} metric_{jk} metric_{kl} = sum_r (metric_{ij} metric_{jr}) (metric_{kl} I_{kr})
            # Note (1, 2) and (3, 4) can each be combined.
            # Then everything is multiplied by 
            #   terms_iso[(1,1,1,1)][((2,2,1,1),)] *  vec_part_coef(((2,2,1,1),), divide_fac=True) = 6 * 1/4 = 3/2
            metric_diag = metric.diagonal()
            metric_full = metric

            # 1 and 2
            A = w(2)[:,None,None] * w(2)[None,:,None] * (
                metric_diag[:,None,None] * metric_diag[None,:,None] * (3/45) +
                metric_full[:,:,None] * metric_full[:,:,None] * (6/45)
            )
            B = w(1)[:,None,None] * w(1)[None,:,None] * metric_full[:,:,None] * (core * 3/2)
            pK_1111.add_factors_((A, B))
            
            # 3 and 4
            A = w(2)[:,None,None] * w(1)[None,:,None] * metric_diag[:,None,None] * metric_full[None,:,:] * (12/45) +\
                w(1)[:,None,None] * w(2)[None,:,None] * metric_full[:,:,None] * metric_full[None,:,:] * (24/45)
            B = w(2)[:,None,None] * w(1)[None,:,None] * metric_full[:,:,None] * I[:,None,:] * (core * 3/2)
            pK_1111.add_factors_((A, B))

    # If not use_pK, pK_slices already contain our cumulant estimate. Project to harmonic and return.
    if not use_pK:
        K_out: FacHTower = {}
        for d in range(1, 4):
            part = (1,) * d
            if part in pK_slices:
                K_out[d] = proj_geq_r(pK_slices[part], n=n, r_out=0)
        K_out[4] = pK_1111
        return K_out

    # 4. Convert pK to K
    with flop_name('pK_to_K'):
        pK_ds = DSTower.from_slices(pK_slices, autozero=True)
        del pK_slices
        K_ds = DS_pK_to_K(pK_ds, strict=not augment)
        # Free n^4 slices from pK_ds (degree 1 is kept for K2111_contrib below)
        pK1 = pK_ds[1].get_slice((1,))
        del pK_ds
        # Subtract the repeated part without materializing a second DSTensor.
        if 4 in K_ds:
            for part in list(K_ds[4].slices.keys()):
                K_ds[4].slices[part].sub_(pK_1111.get_dslice(part))

    # 4.1 Project degree 5 and 6 early to free their n^4 diagonal slices
    K_out: FacHTower = {}
    if augment:
        with flop_name('DS_harmonic_proj d=5,6'):
            K_out[5] = DS_harmonic_proj(K_ds[5], r_out=1)
            del K_ds[5]
            K_out[6] = DS_harmonic_proj(K_ds[6], r_out=3)
            del K_ds[6]

    # 4.2 Account for contribution from pK(1,1,1,1) to the H(r>=1) projection of K(2,1,1,1)
    if augment:
        K2111_contrib = 0.
        with flop_name('pK1111 -> K2111'):
            A, B = pK_1111._factors
            # Subtract out repeated indices
            rep_factors = FactoredTensor4.from_dstensor(pK_1111.get_repeated()).factors
            A = torch.cat((A, rep_factors[0]), dim=2)
            B = torch.cat((B, -rep_factors[1]), dim=2)
            pK1111_K2111 = symmetrize(
                einops.einsum(pK1, A, B, 'i, i j r, k l r -> j k l') / 2 +
                einops.einsum(pK1, B, A, 'i, i j r, k l r -> j k l') / 2
            )
            # Coef from pK_to_K formula
            #     vpart = ((1, 1, 1, 1), (1, 0, 0, 0))
            #     vec_part_coef(vpart, divide_fac=False) * _pK_to_K_coef(vpart) = 2 * (-1) = -2
            # Coef from DSTensor.to_tensor scaling
            #     int_partition_coef((2, 1, 1, 1)) = 10
            # Coef from harmonic projection
            #     harmonic._multigraph_coef([((0, 0), 1)], vpart) * harmonic.proj_coef(n, 5, 1)[1] = 2 / (2n + 12)
            #     (Note we only need the r=1 coef bc higher r are all 0 for (2,1,1,1) partition)
            pK1111_K2111 *= -2 * 10 * 2 /(2 * n + 12)
            K2111_contrib += pK1111_K2111

        if 4 in WK:
            with flop_name('pK2111 -> K2111'):
                A, B = WK[4].factors
                rep_factors = FactoredTensor4.from_dstensor(WK[4].get_repeated()).factors
                A = torch.cat((A, rep_factors[0]), dim=2)
                B = torch.cat((B, -rep_factors[1]), dim=2)
                w1, w2 = get_wick_coef(1, 1), get_wick_coef(1, 2)  # Careful! not the same as w(2)=get_wick_coef(2, 1)
                pK2111_K2111 = symmetrize(
                    einops.einsum(w2, w1, w1, w1, A, B, 'i, j, k, l,  i j r, k l r -> j k l') / 2 +
                    einops.einsum(w2, w1, w1, w1, B, A, 'i, j, k, l,  i j r, k l r -> j k l') / 2
                )

                # Coef from pK_to_K formula
                #     vpart = ((2, 1, 1, 1),)
                #     vec_part_coef(vpart, divide_fac=False) * _pK_to_K_coef(vpart) = 1 * 1 = 1
                # Coef from DSTensor.to_tensor scaling
                #     int_partition_coef((2, 1, 1, 1)) = 10
                # Coef from harmonic projection
                #     harmonic._multigraph_coef([((0, 0), 1)], vpart) * harmonic.proj_coef(n, 5, 1)[1] = 2 / (2n + 12)
                #     (Note we only need the r=1 coef bc higher r are all 0 for (2,1,1,1) partition)
                pK2111_K2111 *= 10 * 2 /(2 * n + 12)
                K2111_contrib += pK2111_K2111

        # There's no contribution from K(2,2,1,1) to the H(d=6,r=3) part, and K(2,2,1,1)
        # is the only degree 6 term that pK(1,1,1,1) contributes to. So there's nothing to do for d=6
        with flop_name('K2111_contrib', factor=slice_factor((1,1,1), n=n)):
            K_out[5].core += K2111_contrib

    # 5. Convert remaining degrees back to FacHTower
    with flop_name('DS_harmonic_proj'):
        for d in range(1, 4):
            # No need to project bc assume simple mode
            K_out[d] = HTensor(core=K_ds[d].to_tensor(), r=0)
        K_out[4] = pK_1111 + FactoredTensor4.from_dstensor(K_ds[4])
    return K_out
