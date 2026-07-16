# Direct dense Edgeworth max — final report (2026-07-16)

Direct (dense-tensor, plain-einsum, no treewidth) Cumulant/Edgeworth estimator
of T(theta) = E_{X~N(0,I_n)}[max_i M_theta(X)_i] for bias-free He ReLU MLPs,
testing whether successive Edgeworth terms improve the MSE-vs-width scaling.

## 1-3. Commits and preservation
- Branch `experiment/max-endpoint`; final code commit `3bc0fa7` (earlier run
  commits recorded per row: smoke at `f8f8bea`, all pilot/full rows at
  `3bc0fa7`). Upstream ARC commit `6e80f7f2af0d33e252731ad9611dff17880b12fb`.
- The tree was clean at start; no existing files were reset/cleaned/stashed.
  The only pre-existing code touched is none — all changes are additive; the
  existing treewidth max-endpoint pipeline is reused untouched (quadrature,
  endpoint weights, cumulant extraction, ground truth, FLOP counter) and
  serves as an independent cross-check.

## 4. Files added
- `src/mlp_kprop/max_endpoint/direct_dense.py` (estimator)
- `tests/test_direct_dense_{derivatives,contractions,end_to_end}.py` (27 tests)
- `scripts/direct_edgeworth_max_experiment.py` (driver; reuses reference /
  MC-validation / sharding / checkpoint infra from max_endpoint_experiment.py)
- `scripts/run_direct_edgeworth_max_{smoke,pilot,full,d4probe}.py`,
  `configs/direct_edgeworth_max_{smoke,pilot,full,d4probe}.json`
- `scripts/plot_direct_edgeworth_max.py`
- `docs/direct_edgeworth_max_status.md`, this report.

## 5-6. Exact commands
Local: `uv sync`; `uv run pytest -q --ignore=tests/symb` (225 passed; symb
needs MPI/Sage — pre-existing); local smoke
`uv run python scripts/run_direct_edgeworth_max_smoke.py --run-name ..._smoke_local`.
Cluster (instance `kunalc`, recreated with express user permission after the
user killed the previous one; 1x A100-40GB, verda/helsinki):

    python -m arc_infra.cli launch kunalc --num-gpus 1 --auto --config infra/arc_infra_config.py
    python -m arc_infra.cli blob-auth kunalc --config infra/arc_infra_config.py
    python -m arc_infra.cli run kunalc scripts/run_direct_edgeworth_max_smoke.py \
        --run-name direct_edgeworth_max_smoke --num-procs 1 --config infra/arc_infra_config.py
    python -m arc_infra.cli run kunalc scripts/run_direct_edgeworth_max_pilot.py \
        --run-name direct_edgeworth_max_pilot2 --num-procs 1 --config infra/arc_infra_config.py
    python -m arc_infra.cli run kunalc scripts/run_direct_edgeworth_max_full.py \
        --run-name direct_edgeworth_max_full --num-procs 1 --config infra/arc_infra_config.py
    uv run python scripts/plot_direct_edgeworth_max.py <results_dir> [--common-widths ...]

Results: `~/run/mlp_cumulant_propagation/2026-07-16/<run>/results/depth<d>/`
on kunalc, mirrored (incl. plots) to
`gs://arc-ml/kunalchawla/run/mlp_cumulant_propagation/2026-07-16/<run>/results/`,
local copies under `data/direct_edgeworth_max/{smoke,pilot2,full}_kunalc/` and
`data/direct_edgeworth_max/merged_depth4/`.

## 7. Tests
225 passed locally (198 pre-existing + 27 new). New tests cover: D2/D3/D4
vs the per-multi-index reference integral for every equality pattern; D2-D4 vs
independent nested autograd of the scalar quadrature; FD recursion D3->D4;
symmetry; sum_i D1_i = 1; memory-guard refusal; einsum contractions vs nested
loops (n=5); coefficients 1/2 and 1/8 vs the closed-form bivariate Gaussian
max (residual scales as O(c^3), 1/4-instead-of-1/8 guard); coefficient 1/6 vs
a synthetic skewed vector with exact cumulants; Psi vs MC; small-MLP
end-to-end vs MC; direct == treewidth pipeline per correction (rel <= 1e-8);
graceful D4 refusal; k_max=1/2 sector handling. In-experiment treewidth
cross-checks: 58 defined comparisons, max rel diff 1.33e-08.

## 8. Formulas implemented
Psi = hi - int_lo^hi prod_i Phi((t-mu_i)/sigma_i) dt via log_ndtr and
Gauss-Legendre (Q and 2Q, difference reported). Endpoint derivatives
d^beta Psi = (-1)^(p-1) int B(t) prod_{i in S} u_{i,beta_i}(t) dt with
u_{i,m} = phi(a_i) He_{m-1}(a_i) / (sigma_i^m Phi(a_i)). Dense integrated
D_m (m = 2,3,4) assembled per set partition of the m slots (exact equality
patterns; label collisions zeroed via pairwise diagonals; embedding by
as_strided partition-diagonal views; node-chunked einsum accumulation).
Corrections: C2 Psi = (1/2) sum_{i!=j} C_ij D2_ij;
(1/2)C2^2 Psi = (1/8) sum C_ij C_kl D4_ijkl (i!=j, k!=l via zero-diagonal C);
C3 Psi = (1/6) sum kappa3 . D3; C4_trace Psi = (1/24) sum core_ij metric_kl
D4_ijkl (Sym dropped — exact against the symmetric D4). Estimators are the
nested partial sums E0, E1, E2_cov, E2_k3, E2_k3_k4trace.

## 9-10. ARC k_max interpretation (verified in code and empirically)
- k_max=1 SIMPLE: K[2] is HTensor r=1 (scalar core, vector metric) —
  variances only; E1/E2 collapse to E0 (recorded as equivalences).
  The product-Gaussian baseline needs marginal variances, which k_max=1 DOES
  provide, so E0 runs at k_max=1; k_max=2 is the lowest order with any
  off-diagonal information.
- k_max=2 SIMPLE: K[2] r=0 full covariance; no K[3]/K[4].
- k_max=3 factor=True: full kappa3 as a rank decomposition (densified here via
  .to_tensor()); K[4] retained trace sector only:
  SIMPLE -> r=2, kappa4 ~ c Sym(M x M) (double trace);
  AUGMENT -> r=1, kappa4 ~ Sym(C x M) (P_{>=1}, traceless H4 dropped).
  E2_k3_k4trace is therefore a *trace-sector* correction, never called a full
  kappa4 correction.

## 11. Grids completed
- Smoke (kunalc + local): depths {2,4}, widths {8,12,16}, 2 seeds, Q=64.
- Pilot `pilot2`: depths {2,4,6}, widths {16,24,32,48,64}, 8 seeds, Q=128.
- Full (user-specified grid): depth 4 (= 4 linear layers = 3 hidden layers,
  repo depth convention), 5 seeds, Q=256, widths {128,256,512,1024}:
  all methods at 128; at 256 dense D4 refused by the memory guard (recorded),
  E0/E1 kept; 512/1024 E0/E1 only (dense_orders=(2,)).
- Cost: pilot ~35 min, full ~4 min on the A100 (references dominate).

## 12. MSE by method and width (depth 4; cross-fidelity mean; pilot 8 seeds for n<=64, full 5 seeds for n>=128)

| n | E0 | E1 | E2_cov | E2_k3 | E2_k3_k4trace_aug | E2_k3_k4trace_simple |
|---|---|---|---|---|---|---|
| 16 | 1.2e-01 | 1.1e-02 | 8.5e-03 | 1.2e-02 | 9.3e-04 | 1.2e-02 |
| 24 | 5.8e-02 | 9.9e-03 | 6.8e-03 | 1.8e-03 | 8.1e-04 | 2.0e-03 |
| 32 | 3.9e-02 | 1.0e-02 | 7.4e-03 | 5.0e-04 | 9.7e-04 | 3.7e-04 |
| 48 | 9.7e-03 | 6.4e-03 | 4.3e-03 | 3.9e-04 | 2.6e-04 | 2.2e-04 |
| 64 | 1.6e-02 | 3.8e-03 | 1.9e-03 | 1.6e-04 | 9.3e-05 | 6.7e-05 |
| 128 | 7.5e-03 | 1.3e-03 | 7.0e-04 | 2.7e-05 | 1.1e-05 | 1.0e-05 |
| 256 | 6.4e-04 | 4.6e-04 | — | — | — | — |
| 512 | 6.7e-04 | 1.5e-04 | — | — | — | — |
| 1024 | 2.3e-04 | 5.5e-05 | — | — | — | — |

## 13. Fitted MSE slopes (log MSE = a + b log n; bootstrap 95% CI over seeds)

Depth 4, shared range n=16..128 (all methods comparable):
- E0 -1.37 [-1.66,-1.11] · E1 -1.05 [-1.24,-0.88] · E2_cov -1.25 [-1.54,-1.07]
- E2_k3 -2.76 [-3.33,-1.64] · k4trace_simple **-3.29 [-3.93,-2.31]** ·
  k4trace_augment -2.24 [-2.97,-1.69]

Depth 4, full grid n=128..1024 (5 seeds): E0 -1.50 [-2.73,-0.76],
E1 -1.54 [-1.75,-1.35].

Pilot depth 2 (16..64): E0 -0.87, E1 -1.22, E2_cov -1.47, E2_k3 -1.67,
k4trace_aug -3.12 [-3.95,-2.58], k4trace_simple -2.37.
Pilot depth 6 (16..64, k3 towers valid only n>=32): E0 -1.23, E1 -1.47,
E2_cov -1.59, E2_k3 -3.98 [-5.52,-3.03], k4trace_simple -3.98 [-5.75,-3.05].

Hypotheses vs data: E0 ~ n^-1: roughly consistent (-1.2..-1.5). E1 ~ n^-2:
NOT observed — the first covariance correction lowers the constant ~5x but
its fitted slope stays ~-1..-1.5. E2 ~ n^-3: observed (within CI) only once
the kappa4 trace term is included (E2_k3_k4trace_simple -3.29 at depth 4);
E2_k3 alone sits between (-2.8 with wide CI). No n^-K theorem is claimed.

## 14. FLOPs (modeled arithmetic; fits over merged depth-4 grid)
- E0 total ~n^1.39 (kprop n^1.82, endpoint n^1.22 — quadrature Theta(Qn)).
- E1 total ~n^2.53 (kprop n^2.95, endpoint n^2.09 — D2 build Theta(Qn^2)).
- E2_cov / E2_k3 / k4trace totals ~n^4.1, endpoint-dominated: the dense D4
  build is Theta(Q n^4) (at n=128: D4 6.6e11 of 6.8e11 total FLOPs; kprop
  4.7e8). This is the price of the *direct* implementation — the treewidth
  pipeline evaluates the same corrections in ~Theta(Q n^2).

## 15. Correction sizes (RMS over networks, k3_simple, n=128)
C2 Psi 8.3e-02 · (1/2)C2^2 Psi 1.25e-02 · C3 Psi 1.66e-02 ·
C4_trace Psi 2.6e-03. The quadratic covariance and third-cumulant terms are
the SAME order (C3 slightly larger); both must be included together, matching
the block-weight-2 counting. (correction_sizes plot: pilot + full.)

## 16. Matched-budget Monte Carlo (Var/m at equal modeled FLOPs, depth 4)
MSE_MC/MSE_det (>1 favors deterministic): E0: 5.7 (n=128) -> 307 (n=1024).
E1: 0.44 (128) -> 1.08 (1024) — crosses MC around n~1024. Dense-D4 towers at
n=128: E2_cov 1e-4, E2_k3 2.5e-3, k4trace ~6e-3 — the n^4 dense endpoint
loses to MC by 150-10000x despite the best MSE; matched-budget viability of
the k3 towers requires the factorized/treewidth contraction (cf.
docs/max_endpoint_report.md where the same corrections cost ~n^2 and win at
depth 2-4). MC MSE=Var/m validated by explicit repeated runs: ratios
0.51-1.34 at n=128 (32 repeats, ~1 +- sqrt(2/32)).

## 17. Reference diagnostics
Spherical Rao-Blackwellized streams; every stream hit its SE target
(3e-5 at n=128 [3.3e8 samples/stream] .. 1e-4 at 512/1024). Max |A-B|/se =
1.93 over 20 networks (N(0,1)-consistent). Noise floor se_A*se_B is >= 5.6e3x
below the smallest reported MSE at every width — all plotted full-run points
resolved; no clipping of negative cross-errors (none of the aggregates were
noise-dominated).

## 18. Dense D4 practicality ceiling
n=128: 2.15 GB tensor, 8.6 GB peak GPU, 0.47 s endpoint wall — comfortable.
n=256: refused by the guard with recorded reason: needs ~36.5 GB
(34.36 GB tensor + 2.15 GB chunk intermediate) > 30 GB budget on the 40 GB
card; the guard-limited maximum is n ~= 245. Runtime was NOT the binding
constraint at 128 (einsum lowers to GEMM); memory is. The refusal is recorded
per (width,seed,variant) row (status `dense_refused`), and E0/E1 degrade
gracefully at those widths.

## 19. Deviations from the requested experiment
- Full grid changed by user instruction mid-task to depth 4, 5 seeds, widths
  {128,256,512,1024} (original spec §15.3 grid superseded); pilot kept the
  original depths {2,4,6} per the same instruction ("for the full
  experiments, that is"). The earlier 3-depth pilot launch was killed (own
  job) and relaunched as `pilot2` after the grid change.
- "Depth" follows the repo convention: num_layers = linear layers (depth 4 =
  3 hidden layers). The spec's "four hidden layers" reading was superseded by
  the user's depths {2,4,6}.
- E2_k3 and E2_k3_k4trace at n=256 were attempted and refused by the memory
  guard (not silently skipped); the k2_augment variant of the original
  pipeline was not rerun (not requested in §9).
- The d4probe run (widths 160-224) was prepared but not launched, respecting
  "only widths 128,256,512,1024"; the ceiling is established by the guard
  arithmetic + the measured n=128 footprint.
- MC-validation ratio at one (width,seed) was 0.51 (32 repeats; ~2 sigma of
  the chi^2 spread) — noted, not hidden.

## 20. Remaining uncertainties
- E2_k3-tower slopes at depth 4 carry wide CIs (5-8 seeds, 6 widths, and the
  small-width points are near the divergence edge); the -3 reading rests on
  the shared 16..128 range and on the k4-trace term being included.
- k_max=3 towers genuinely diverge (kprop_nonfinite / all-variance clamping)
  at depth 4 n<=12 and depth 6 n<=24 — a property of the truncated cumulant
  expansion (documented upstream), flagged per row and excluded from fits.
- SIMPLE vs AUGMENT k4-trace sectors are statistically indistinguishable in
  constant at n>=48 and their slope difference (-3.29 vs -2.24) is within
  overlapping CIs — read as noise, not structure.
- The direct dense estimator is a correctness instrument: its n^4 endpoint
  cost is inherent; scaling beyond n~245 (or matched-budget competitiveness)
  requires the existing factorized/treewidth path, which this work validated
  to 1e-8.
