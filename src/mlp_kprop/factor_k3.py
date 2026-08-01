import itertools
import einops
import torch
from torch import Tensor
from jaxtyping import Float
import math
import string
from typing import Any, Optional
from tqdm.auto import tqdm

from mlp_kprop.partitions import *
from mlp_kprop.tensor_utils import *
from mlp_kprop.diagslice import *
from mlp_kprop.harmonic import *
from mlp_kprop.cumulants import *
from mlp_kprop.wick import relu_wick_coef
from mlp_kprop.kprop_harmonic import (
    multiply_wicks,
    get_all_terms_iso,
    factored_keeps_term,
)

logger = logging.getLogger(__name__)

def _unfactor(factors):
    in_expr = ','.join(f'i{m} j' for m in range(len(factors)))
    out_expr = ' '.join(f'i{m}' for m in range(len(factors)))
    einexpr = f'{in_expr}->{out_expr}'
    return cached_einsum(*factors, einexpr)

def _factored_get_dslice(factors: tuple[Float[Tensor, 'n r'], ...], part: IntPartition) -> Float[Tensor, "*n"]:
    if set(part) == {1}:
        raise NotImplementedError(
            "You shouldn't need to do this (materializing the FactoredTensor is too slow)." +
            " If you need this for testing, use zero_repeated(FT.to_tensor())."
        )
    assert tuple(part) == tuple(sorted(part, reverse=True)), f"Partition {part} must be sorted."
    d = len(factors)
    assert sum(part) == d
    perms = list(itertools.permutations(range(d)))
    # TODO: There's surely a d! / part! way to just enumerate the needed block perms instead of filtering
    block_perms = set(
        tuple(map(frozenset, group_by_partition(p, part))) for p in perms
    )
    ret = torch.as_tensor(0., device=factors[0].device, dtype=factors[0].dtype)
    for perm in block_perms:
        perm_factors = []
        for block in perm:
            perm_factors.append(
                math.prod(factors[i] for i in block)
            )
        ret = ret + _unfactor(perm_factors)
    coef = math.prod(math.factorial(b) for b in part) / math.factorial(d)
    return coef * zero_repeated(ret)

def group_by_partition(items: list[Any], part: IntPartition):
    groups = []
    cur = 0
    for block in part:
        groups.append(items[cur : cur + block])
        cur += block
    return groups

# Unused
def perms_mod_part(part: IntPartition):
    '''
    List of representatives of S_d mod the Young subgroup S_{part_1} x S_{part_2} x ...
    where d = sum(part)
    '''
    d = sum(part)
    perms = list(itertools.permutations(range(d)))
    unique_perms = set(
        tuple(map(frozenset, group_by_partition(p, part))) for p in perms
    )
    return [tuple(itertools.chain(*blocks)) for blocks in unique_perms]

class FactoredTensor:
    '''
    A symmetric tensor in factored form:
    T_{i_1, ..., i_d} = Sym(sum_{r=1}^R (A1)_{i_1, r} (A2)_{i_2, r} ... (Ad)_{i_d, r})

    NOTE: Although this is written for general d, we only use it for kprop with k_max=3.
    For larger k_max, a different factorized form is need.
    '''

    def __init__(
        self, 
        n: int, 
        d: int, 
        factors: tuple[Float[Tensor, 'n r']] | None = None,
        repeated: Optional[DSTensor] = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.n = n
        self.d = d
        if factors is not None:
            assert len(factors) == d
            if device is None:
                device = factors[0].device
            if dtype is None:
                dtype = factors[0].dtype
            r = factors[0].shape[1]
            for factor in factors:
                assert factor.shape[0] == n
                assert factor.shape[1] == r
            self._factors = [
                factor.to(device=device, dtype=dtype) for factor in factors
            ]
        else:
            base = torch.zeros((n, 0), device=device, dtype=dtype)
            device = base.device
            dtype = base.dtype
            self._factors = [base.clone() for _ in range(d)]
        self.device = device
        self.dtype = dtype
        if repeated is not None:
            assert repeated.d == d
            assert repeated.n == n
            self.repeated = repeated
        else:
            self.repeated = DSTensor(d=d, n=n, slices=dict(), device=self.device, dtype=self.dtype)

    def clear_repeated(self) -> None:
        self.repeated = DSTensor(d=self.d, n=self.n, slices=dict(), device=self.device, dtype=self.dtype)

    @property
    def factors(self) -> tuple[Float[Tensor, 'n r']]:
        # Prevent external mutation of factors to protect cache
        return tuple(A.clone() for A in self._factors)

    @property
    def ndim(self) -> int:
        return self.d

    def add_factors_(self, factors: tuple[Float[Tensor, 'n r']]) -> None:
        assert len(factors) == self.d
        new_factors = [
            factor.to(device=self.device, dtype=self.dtype) for factor in factors
        ]
        self._factors = [
            torch.cat([self._factors[i], new_factors[i]], dim=1)
            for i in range(self.d)
        ]
        for part in self.repeated.slices:
            self.repeated.slices[part] += _factored_get_dslice(tuple(new_factors), part)

    def add_factors(self, factors: tuple[Float[Tensor, 'n r']]) -> 'FactoredTensor':
        new = self.clone()
        new.add_factors_(factors)
        return new

    def __add__(self, other: 'FactoredTensor') -> 'FactoredTensor':
        # TODO: Figure out how to deal with the repeated cache instead of just discarding it
        # For now this doesn't matter because a contract_W immediately follows the only place where __add__ is used
        assert self.n == other.n
        assert self.d == other.d
        new_factors = tuple(
            torch.cat([f1.to(device=self.device, dtype=self.dtype), f2.to(device=self.device, dtype=self.dtype)], dim=1)
            for f1, f2 in zip(self._factors, other._factors)
        )
        return FactoredTensor(
            n=self.n,
            d=self.d,
            factors=new_factors,
            device=self.device,
            dtype=self.dtype
        )

    def to_tensor(self) -> Float[Tensor, '*n']:
        return symmetrize(_unfactor(self._factors))

    @flop_name('FactoredTensor3.get_dslice')
    def get_dslice(self, part: IntPartition) -> Float[Tensor, '*n']:
        part = tuple(part)
        sorted_part = tuple(sorted(part, reverse=True))
        if sorted_part not in self.repeated.slices:
            self.repeated.slices[sorted_part] = _factored_get_dslice(self._factors, sorted_part)
        return self.repeated.get_slice(part)

    @flop_name('FactoredTensor3.contract_W')
    def contract_W(self, W: Float[Tensor, 'n_out n_in']) -> 'FactoredTensor':
        new_factors = tuple(
            W @ factor for factor in self._factors
        )
        return FactoredTensor(
            n=self.n,
            d=self.d,
            factors=new_factors,
            device=self.device,
            dtype=self.dtype
        )

    def contract_wick_(self, wick: Float[Tensor, 'n']) -> None:
        # TODO: Support distinct wick indices on different legs (needed if we want to factor the full harmonic algo, not just simple)
        self._factors = [
            factor * wick[:, None] for i, factor in enumerate(self._factors)
        ]
        if not self.repeated.slices:
            return
        letters = string.ascii_lowercase + string.ascii_uppercase
        for part in list(self.repeated.slices):
            if len(part) > len(letters):
                self.repeated.slices[part] = _factored_get_dslice(self._factors, part)
                logger.warning("Exceeded letter limit in einsum for contracting wick for cache; falling back to factored_get_dslice.")
                continue
            slice_expr = ' '.join(letters[:len(part)])
            wick_expr = ', '.join(
                ', '.join(letters[i] for _ in range(part[i]))
                for i in range(len(part))
            )
            einexpr = f'{slice_expr}, {wick_expr} -> {slice_expr}'
            self.repeated.slices[part] = cached_einsum(
                self.repeated.slices[part],
                *([wick] * self.d),
                einexpr
            )

    @flop_name('FactoredTensor3.contract_wick')
    def contract_wick(self, wick: Float[Tensor, 'n'] | tuple[Float[Tensor, 'n'], ...]) -> 'FactoredTensor':
        new = self.clone()
        new.contract_wick_(wick)
        return new

    def clone(self) -> 'FactoredTensor':
        new_factors = tuple(
            factor.clone() for factor in self._factors
        )
        return FactoredTensor(
            n=self.n,
            d=self.d,
            factors=new_factors,
            repeated=self.repeated.clone(),
            device=self.device,
            dtype=self.dtype
        )

    def get_repeated(self) -> DSTensor:
        '''
        Returns a DSTensor B satisfying
            zero_repeated(self.to_tensor()) + B.to_tensor() = self.to_tensor()
        '''
        slices = dict()
        for part in int_partitions(self.d):
            # Skip all-distinct slice
            if all(p == 1 for p in part):
                continue
            slices[part] = self.get_dslice(part)
        return DSTensor(d=self.d, n=self.n, slices=slices, device=self.device, dtype=self.dtype)

    @staticmethod
    @flop_name('FactoredTensor3.from_dstensor')
    def from_dstensor(ds: DSTensor) -> 'FactoredTensor':
        if ds.d != 3:
            raise NotImplementedError("Only implemented for d=3")
        assert (1, 1, 1) not in ds.slices, "DSTensor has 111 slice, cannot convert to FactoredTensor"
        eye = torch.eye(ds.n, device=ds.device, dtype=ds.dtype)
        factors = (
            (
                ds.slices[(3,)][:, None] * eye
                # *3 because of weird DSTensor.to_tensor scaling
                # Note ds.slices[(2, 1)] already has diagonal zeroed
                + ds.slices[(2, 1)].T * 3
            ),
            eye,
            eye
        )
        return FactoredTensor(
            n=ds.n,
            d=ds.d,
            factors=factors,
            repeated=ds,
            device=ds.device,
            dtype=ds.dtype
        )

type FacHTower = dict[int, FactoredTensor| HTensor]

def factored_nonlin_kprop_k3(
    K_in: FacHTower,
    nonlin_wick_coef: Callable[[float, float, int, int], float],
    augment: bool = False,
    base: bool = False,
    use_pK: bool = True,
) -> FacHTower:
    '''
    Nonlinear step of cumulant propagation for k_max=3, in O(n^3) time instead of O(n^4).
    K_in should be the output of linear_kprop (with non-identity metric and bias already applied).

    ==========================================================================================
    WARNING: with augment=True this is NOT equivalent to the unfactored nonlin_kprop!
    The augmented Edgeworth term set includes diagrams with no O(n)-rank factorization
    (e.g. the covariance triangle and kappa_3 x edge products), which this implementation
    DROPS: it sums exactly the terms kept by kprop_harmonic.factored_keeps_term. Since the
    dropped terms have the same Theta(n^-3) squared size as the extra terms that are kept,
    factored-augment and unfactored-augment estimates differ at the leading order of their
    (common-order) MSE. tests/test_factor_k3.py checks equivalence against an unfactored
    reference restricted to the same term set.
    ==========================================================================================
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
    # Note that this includes all the d=4 slices we need:
    # Since we only take the scalar ("pure radial") part, which forces the slice to be (2, 2) or a coarsening
    terms_iso = get_all_terms_iso(k_max=3, d_max=3 if base else 4, augment=augment)
    terms_iso = [
        (int_part, vec_part, count)
        for int_part, vec_part_dict in terms_iso.items()
        for vec_part, count in vec_part_dict.items()
        if len(int_part) <= 3
        and (use_pK or all(p == 1 for p in int_part))  # If not use_pK, only need (1, ..., 1) int_parts
        and (augment or int_part not in  [(3, 1), (2, 1, 1)])   # Skip in simple mode bc no contribution to d=4, r=2
        and int_part != (1, 1, 1)   # Factor this manually
        and (int_part, set(vec_part)) != ((2, 1, 1), {(1, 1, 1,)}) # Mult wick coefs and carry over to K211_contrib manually
        and factored_keeps_term(3, int_part, vec_part)  # Drop terms needing kappa_3's factored all-distinct block
    ]
    pK_slices = defaultdict(lambda: 0.0)
    for int_part, vec_part, count in tqdm(
        terms_iso,
        disable=logger.getEffectiveLevel() > logging.INFO,
        desc="nonlin-kprop",
    ):
        with flop_name('nonlin_sum', factor=slice_factor(int_part, n=n)):
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

    # 3.2 Compute pK slices that do need to be factored: just (1, 1, 1)
    # In simple/base mode (budget k_max - 1 = 2) the only (1, 1, 1) diagrams are the
    # single kappa_3 block [cost 2] and the light two-leg path (1, 1) + (1, 1)
    # [cost 1 + 1]. Augment mode (budget 3) adds every *hypertree* diagram of cost 3:
    # two-leg paths with one heavy leg (a kappa_3 (2, 1)-slice, kappa_4 (2, 2)-slice,
    # or doubled kappa_2 edge) and the single kappa_4 (2, 1, 1)-slice block.
    # (The non-hypertree cost-3 diagrams -- the covariance triangle and
    # kappa_3 x edge products -- are dropped; see factored_keeps_term.)
    with flop_name('nonlin_sum 111 factored'):
        w = lambda k: get_wick_coef(k, 1)
        budget = 2 + augment  # must match get_vec_cond(k_max=3, augment=augment)
        WK_11 = WK[2].core
        if use_pK:
            WK_11 = zero_repeated(WK_11)

        # (1, 1, 1) contrib
        if 3 in WK:
            assert isinstance(WK[3], FactoredTensor)
            pK_111 = WK[3].clone()
            pK_111.contract_wick_(w(1))
        else:
            pK_111 = FactoredTensor(n=n, d=3, device=mean.device, dtype=mean.dtype)

        # Two-leg path diagrams with edges (i, j) and (j, k), factored as
        # A_{ij}B_{jk} = A_{ir}I_{jr}B^T_{kr}.
        # Leg types are (near_mult, far_mult, slice / prod(mult!), block cost).
        legs = [(1, 1, WK_11, 1)]
        if augment:
            if 3 in WK:
                WK_21 = diagslice(WK[3], (2, 1), output_zero_repeated=use_pK) / 2
                legs += [(2, 1, WK_21, 2), (1, 2, WK_21.T, 2)]
            if 4 in WK:
                assert isinstance(WK[4], HTensor)
                WK_22 = diagslice(WK[4], (2, 2), output_zero_repeated=use_pK) / 4
                legs.append((2, 2, WK_22, 2))
            # Doubled light edge: two parallel (1, 1) kappa_2 blocks (1/2! for the repeated block)
            legs.append((2, 2, WK_11 * WK_11 / 2, 2))
        fac2 = torch.eye(n) * 3  # 3 = number of 3 vertex 2 edge graphs
        for a_l, b_l, s_l, c_l in legs:
            rights = [(a_r, b_r, s_r) for a_r, b_r, s_r, c_r in legs if c_l + c_r <= budget]
            if not rights:
                continue
            fac1 = w(a_l)[:, None] * s_l
            fac3 = sum(w(b_l + a_r)[:, None] * w(b_r)[None, :] * s_r for a_r, b_r, s_r in rights)
            pK_111.add_factors_((fac1, fac2, fac3.T))

        # 211 H(d=4,r=1) -> 111 (cost 3, so only within the augment budget)
        if augment and 4 in WK:
            # Three possibilities:
            # 1. 2-block goes on core (sym_coef=1/6 of possible pairings)
            #    w(2)_i core_{ii} w(1)_j w(1)_k metric_{jk} = sum_r w(2)_i core_{ii} w(1)_j metric_{jr} w(1)_k Id_{kr}
            # 2. 2-block bridges core and metric  (sym_coef=4/6 of possible pairings)
            #    w(1)_i w(2)_j w(1)_k core_{ij} metric_{jk} = sum_r w(1)_i core_{ir} w(2)_j Id_{jr} w(1)_k metric_{kr}
            # 3. 2-block goes on metric  (sym_coef=1/6 of possible pairings)
            #    w(1)_i w(1)_j w(2)_k core_{ij} metric_{kk} = sum_r w(1)_i core_{ir} w(1)_j Id_{jr} w(2)_k metric_{kk}
            core = WK[4].core
            metric = WK[4].metric
            if metric.ndim == 1:
                metric_full = metric.diagflat()   # n, n
                metric_diag = metric              # n
            elif metric.ndim == 2:
                metric_full = metric              # n, n
                metric_diag = metric.diagonal()   # n
            else:
                raise ValueError(f"metric must be 1d or 2d, got shape {metric.shape}")
            ones = torch.ones_like(metric_diag)
            I = torch.eye(n, device=mean.device, dtype=mean.dtype)

            # 1
            fac1 = w(2)[:, None] * core.diagonal()[:, None] * ones[None, :]
            fac2 = w(1)[:, None] * metric_full
            fac3 = w(1)[:, None] * I
            fac3 /= 4 # vec_part_coef(((2, 1, 1),)) * |iso_class| * sym_coef = 1/2 * 3 * 1/6 = 1/4
            pK_111.add_factors_((fac1, fac2, fac3))

            # 2
            if metric.ndim == 2:
                # This term is zero when metric is diagonal
                fac1 = w(1)[:, None] * core
                fac2 = w(2)[:, None] * I
                fac3 = w(1)[:, None] * metric_full
                # vec_part_coef(((2, 1, 1),)) * |iso_class| * sym_coef = 1/2 * 3 * 4/6 = 1
                # so no need to multiply
                pK_111.add_factors_((fac1, fac2, fac3))

            # 3
            fac1 = w(1)[:, None] * core
            fac2 = w(1)[:, None] * I
            fac3 = w(2)[:, None] *  metric_diag[:, None] * ones[None, :]
            fac3 /= 4 # vec_part_coef(((2, 1, 1),)) * |iso_class| * sym_coef = 1/2 * 3 * 1/6 = 1/4
            pK_111.add_factors_((fac1, fac2, fac3))

    # If not use_pK, pK_slices already contain our cumulant estimate. Project to harmonic and return.
    if not use_pK:
        K_out: FacHTower = {}
        K_out[1] = proj_geq_r(pK_slices[(1,)], n=n, r_out=0)
        K_out[2] = proj_geq_r(pK_slices[(1, 1)], n=n, r_out=0)
        K_out[3] = pK_111
        return K_out

    # 4. Convert pK to K
    with flop_name('pK_to_K'):
        pK_ds = DSTower.from_slices(pK_slices, autozero=True)
        K_ds = DS_pK_to_K(pK_ds, strict=not augment)
        K_ds[3] -= pK_111.get_repeated()  # K_111 is a FactoredTensor. So we need to zero repeated by subtracting from the ds part

    # 4.1 Account for contribution from pK_111 and pK_211 to the H(r=1) projection of K_211
    if augment:
        K211_contrib = 0.
        with flop_name('pK_111 -> K_211'):
            A, B, C = pK_111.factors
            # Subtract out the repeated part of pK(1,1,1)
            rep_factors = list(FactoredTensor.from_dstensor(pK_111.get_repeated()).factors)
            A = torch.cat([A, -rep_factors[0]], dim=1)
            B = torch.cat([B, rep_factors[1]], dim=1)
            C = torch.cat([C, rep_factors[2]], dim=1)
            pK1 = pK_ds[1].slices[(1,)]
            # The contribution to K(2, 1, 1) is the [(1, 1, 1), (1, 0, 0)] vec partition.
            # After tracing out the 2-index this is
            # sum_i sum_r pK1_i A_{i,r} B_{j, r} C_{k, r} averaged over permutations of A,B,C
            pK111_K211 = symmetrize(
                ((pK1[:,None] * A).sum(dim=0) * B) @ C.T +
                ((pK1[:,None] * B).sum(dim=0) * C) @ A.T +
                ((pK1[:,None] * C).sum(dim=0) * A) @ B.T
            ) / 3.

            # Coef from pK_to_K formula:
            # vpart = ((1, 1, 1), (1, 0, 0))
            #   vec_part_coef(vpart, divide_fac=False) * _pK_to_K_coef(vpart) = 2 * (-1) = -2
            # Coef from DSTensor.to_tensor scaling:
            #   int_partition_coef((2, 1, 1)) = 6
            # Coef from harmonic projection
            #   harmonic._multigraph_coef([((0, 0), 1)], vpart) * harmonic.proj_coef(n, 4, 1)[1] = 2/(2n+8)
            pK111_K211 *= (-2 * 6 * 2 / (2 * n + 8))
            K211_contrib += pK111_K211

        if 3 in WK:
            with flop_name('pK_211 -> K_211'):
                A, B, C = WK[3].factors
                rep_factors = list(FactoredTensor.from_dstensor(WK[3].get_repeated()).factors)
                A = torch.cat([A, -rep_factors[0]], dim=1)
                B = torch.cat([B, rep_factors[1]], dim=1)
                C = torch.cat([C, rep_factors[2]], dim=1)

                w1, w2 = get_wick_coef(1, 1), get_wick_coef(1, 2)  # Careful! not the same as w(2)=get_wick_coef(2, 1)
                pK211_K211 = symmetrize(
                    ((w2[:,None] * A).sum(dim=0) * w1[:,None]* B) @ (w1[:,None] * C).T +
                    ((w2[:,None] * B).sum(dim=0) * w1[:,None]* C) @ (w1[:,None] * A).T +
                    ((w2[:,None] * C).sum(dim=0) * w1[:,None]* A) @ (w1[:,None] * B).T
                ) / 3
                # vpart = ((2, 1, 1))
                # Coef from pK_to_K formula
                #   vec_part_coef(vpart, divide_fac=False) * _pK_to_K_coef(vpart) = 1 * 1 = 1
                # Coef from DSTensor.to_tensor scaling
                #   int_partition_coef((2, 1, 1)) = 6
                # Coef from harmonic projection
                #   harmonic._multigraph_coef([((0, 0), 1)], vpart) * harmonic.proj_coef(n, 4, 1)[1] = 2/(2n+8)
                pK211_K211 *= 6 * 2 / (2 * n + 8)
                K211_contrib += pK211_K211

    # 5. Convert back to FacHTower
    with flop_name('DS_harmonic_proj'):
        K_out: FacHTower = {}
        K_out[3] = pK_111 + FactoredTensor.from_dstensor(K_ds[3])
        K_out[1] = HTensor(core=K_ds[1].to_tensor(), r=0)
        K_out[2] = HTensor(core=K_ds[2].to_tensor(), r=0)
        if augment:
            K_out[4] = DS_harmonic_proj(K_ds[4], r_out=1)
            K_out[4].core += K211_contrib
        elif not base:
            K_out[4] = DS_harmonic_proj(K_ds[4], r_out=2)
    return K_out
