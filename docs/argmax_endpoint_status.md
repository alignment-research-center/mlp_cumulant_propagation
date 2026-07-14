# Argmax-endpoint implementation log

Extension of the scalar max-endpoint experiment to the expected one-hot argmax
vector q_i(theta) = P_X(i = argmax_j M_theta(X)_j), X ~ N(0, I_n), for fixed
randomly initialized bias-free He ReLU MLPs. Primary metric: total Brier MSE
E_theta ||q_hat - q||_2^2 (per-coordinate MSE = total/n also reported).

Tie policy: `torch_argmax_first_index; ties otherwise ignored`. No jitter,
tie detection, or tie corrections anywhere.

## Method

q_hat[E] = grad_mu E for each nested scalar estimator E in
{E0, E1_cov1, E2_cov2, E2_k3, E2_full} (identity q_i = dT/da_i at a=0 for the
output-mean shift T(a) = E[max_j (Y+a)_j]). Implementation
(`src/mlp_kprop/max_endpoint/argmax.py`):

- The exact scalar pipeline (compiled quotient diagrams, variable-elimination
  contractions, Gauss-Legendre grid) is rebuilt with mu as an autograd leaf.
- **Cumulants are fixed**: ArcBridge detaches every cumulant tensor at
  construction; sigma is re-detached; only the endpoint unary weights depend
  on mu. Nothing differentiates through cumulant propagation.
- **Quadrature is differentiable**: endpoints [lo, hi] come from the scalar
  bisection on *detached* parameters and are then held constant (neglected
  d(hi)/dmu ~ 1e-25 tail mass); nodes/weights are mu-independent constants;
  the integrand uses torch.special.log_ndtr (stable autograd backward).
  Convergence checked at Q vs 2Q nodes for both the scalar value and q
  (linf / l2 recorded).
- **One reverse pass per diagram** (width-independent count), never per
  coordinate; a single backward yields all n coordinates. Per-diagram
  backward + graph release keeps autograd memory at one diagram's tables
  (the shared O(Qn) workspace is retained via retain_graph).
- **Degree 5 at K=3**: autograd applies du_{i,k}/dmu_i = u_{i,k+1} +
  u_{i,1} u_{i,k} to the merged degree-4 vertex weights, exercising endpoint
  Hermite degree 2K-1 = 5 (He_4 via the hermite_he recurrence). Verified by
  an explicit test that fails under a degree-4 truncation.
- **Treewidth unchanged**: the root adds only unary endpoint factors, no new
  cumulant edge, so scalar_treewidth == argmax_treewidth (asserted in tests;
  both recorded).
- FLOP convention: `flops_endpoint_backward = 2 * flops_endpoint_forward`
  (documented constant-factor model for reverse-mode multilinear
  contractions; identical width exponent by construction). Forward FLOPs use
  the scalar FlopTally/EndpointWorkspace counters; the argmax forward
  integrates once per diagram instead of once per term (+<1%, width-free).
  Wall-clock and peak GPU memory recorded separately.

Simplex: raw gradient sums to 1 by construction (sum_i d_i Psi = F(hi)-F(lo)
~ 1; corrections are invariant under uniform mean shifts, so their gradients
sum to 0). Raw is the primary estimator, never silently renormalized;
Euclidean simplex projection is the secondary `_projected` variant
(`project_to_simplex`). Diagnostics per estimator: q_sum, q_min, q_max, q_l1,
q_l2_sq, simplex_residual, num_negative_coordinates, raw-projected distance.

## Evaluation (unbiased Brier MSE)

`src/mlp_kprop/max_endpoint/argmax_mse.py`: winner counts per shared input
block; U-statistic mse_unbiased = ||q_hat||^2 - 2 sum q_i c_i/m +
sum c_i(c_i-1)/(m(m-1)); may be negative, never clipped; float64 throughout.
Two-observation variant for validation. MC baseline
MSE_MC(m) = (1-||q||^2)/m with the collision U-statistic for ||q||^2;
validated by explicit repeated MC (mcval tasks). Matched budget:
m = floor(flops_total / flops_per_argmax_sample); no spherical baseline
(radius does not affect argmax for zero-bias positively homogeneous nets).

## Files added

- src/mlp_kprop/max_endpoint/argmax.py — differentiable pipeline, product
  Gaussian argmax (grad + explicit reference), projection, diagnostics.
- src/mlp_kprop/max_endpoint/rooted_diagrams.py — explicit rooted
  contractions via the multi-index shift (test reference), DenseTermBridge,
  per-term autograd gradient.
- src/mlp_kprop/max_endpoint/argmax_mse.py — winner counts, unbiased Brier
  U-statistics, MC formula, argmax FLOPs/sample.
- scripts/argmax_endpoint_experiment.py + run_argmax_endpoint_{smoke,pilot,
  full}.py + plot_argmax_endpoint.py.
- configs/argmax_endpoint_{smoke,pilot,full}.json.
- tests/test_argmax_endpoint_{base,gradient,rooted_diagrams,mse,arc_bridge,
  end_to_end}.py (spec 16.1-16.11).

## Shared files modified

- src/mlp_kprop/max_endpoint/estimator.py — `_nested_estimates` uses
  `running = running + x` instead of `+=` (float behavior identical; makes
  the helper safe for elementwise tensor reuse by argmax).
- src/mlp_kprop/max_endpoint/__init__.py — re-export argmax API.
- (pre-existing uncommitted, preserved: GIT_COMMIT content, infra config
  `.python-version` upload entry.)

## Assumptions about existing code

- ArcBridge detaches all cumulant tensors (verified: `prep` calls
  `.detach()`); mu enters the contraction only through EndpointWorkspace.
- Scalar `find_endpoints` reads mu only through float() (no graph).
- k4 sector semantics documented in docs/max_endpoint_experiment.md are
  reused verbatim: k_max=3 AUGMENT = P_{>=1} kappa4 (R^1 H_2 (+) R^2 H_0,
  Sym(C x M)); k_max=3 SIMPLE / k_max=2 AUGMENT = P_{>=2} = R^2 H_0
  (c Sym(M x M)). "k4trace" method names refer to those sectors, not the full
  fourth cumulant.
- Known pathology inherited from the scalar pipeline: truncation-negative
  output variances are clamped to 1e-10 and flagged
  (`negative_variance_clamped`, num_clamped_var). For argmax this puts an
  unresolvable sigma ~ 1e-5 spike under the fixed grid, visible as a large
  simplex_residual — flagged, not hidden.

## Remote test environment notes (kunalc-2)

- Full suite on kunalc-2 (A100 40GB, GPU-enabled): 221 passed initially;
  tests/symb failures diagnosed as (a) rsynced macOS `__pycache__` baking
  local paths into co_filename (remote pycache removed), (b) OpenMPI on the
  denvr calgary image failing MPI_Init with default transports — fixed with
  OMPI_MCA_btl=tcp,self OMPI_MCA_pml=ob1 (added to RUN_ENV), after which
  tests/symb/test_parallelize::test_multi_map passes; (c) the remaining 13
  symb-kprop tests require SageMath (`from sage.all import Graph`), an
  optional conda-only dependency the repo's `uv sync` env does not provide
  (pre-existing; the symbolic pipeline is not used by the scalar or argmax
  endpoints). Runnable suite: 222 passed, 0 failed.

## Tests run

- Local (macOS): `uv run pytest -q --ignore=tests/symb` — 197 passed
  (tests/symb needs an MPI runtime unavailable locally). Baseline before the
  argmax changes: 173 passed with the same exclusion.
- Remote (kunalc-2, A100 40GB, GPU): full `uv run pytest -q` — 222 passed;
  the 13 remaining tests/symb failures need SageMath (optional conda-only
  dependency, pre-existing; see remote-environment notes above).

## Depth setting (user-directed change)

Originally smoke depth 4, pilot/full depth 16. Per user instruction
(2026-07-13): no depths >= 9; pilot and full run at num_layers in {2, 4, 6}
(one resumable run directory per depth: <run_name>_depth<d>; the pilot/full
wrappers loop over the "depths" list in the config). Smoke stays at depth 4.

## Cluster jobs

- Per user instruction, experiments run on a NEW instance `kunalc-2`
  (1x A100 80GB), NOT on `kunalc` (in use by another agent; left untouched).
  First kunalc-2 attempt (hyperstack montreal) died at creation: "no
  deployable capacity"; record deleted, relaunched on denvr.
- smoke: `argmax_endpoint_smoke` (2026-07-13, kunalc-2, 1 GPU) — 152 rows,
  0 failed, simplex residual <= 2e-15, quad convergence <= 2e-15,
  dense-vs-factorized q linf 2.6e-9, MC-formula ratio 1.08, plots generated.
- pilot: `argmax_endpoint_pilot` (first attempt) crashed in depth6 at w16:
  diverged k_max=3 towers produced NaN q and project_to_simplex crashed
  inside the *unwrapped* dense_check_task. Fixes: project_to_simplex now
  propagates NaN for non-finite input (+ regression test);
  dense_check_task / mc_validation_task exception-wrapped.
  `argmax_endpoint_pilot_r2` (2026-07-14): depth2/4 clean (0 failed, all
  aggregates resolved); depth6 has 5 failed w16 k3 towers (recorded rows,
  kprop_nonfinite — genuine divergence) and 25/76 noisier aggregates; best
  method at depth6 is E2_cov2 (k3/k4 corrections stop helping at depth 6).
  Depth-4 fitted total-Brier width slopes (bootstrap 95%):
  E0 -1.30, +cov1 -1.67, +cov2 -2.01, +k3 -2.75, +k4trace_simple -3.01.
  Peak GPU mem <= 134 MB at widths <= 128; endpoint FLOPs ~ n^2.
- blob sync: 12h blob-auth token expired mid-pilot (upload_loop 401s;
  results unaffected, local + rsync). Re-run `blob-auth kunalc-2` before
  long runs.
- full: `argmax_endpoint_full_relu` (2026-07-14, kunalc-2, 1 GPU, depths
  2/4/6, widths 16..512, 20 seeds, 16 blocks x 262144 eval samples) —
  COMPLETED. Results + plots:
  `kunalc-2:~/run/mlp_cumulant_propagation/2026-07-14/argmax_endpoint_full_relu/results/depth{2,4,6}`
  and `gs://arc-ml/kunalchawla/run/mlp_cumulant_propagation/2026-07-14/argmax_endpoint_full_relu/results/`.
  depth2: 8360 rows, 0 failed. depth4: 8360 rows, 0 failed (15 raw rows from
  clamped-variance towers at w16/k3_simple flagged). depth6: 8171 rows, 21
  failed tasks (w16/w23 k3 towers, kprop_nonfinite — genuine divergence,
  recorded) and 97 flagged clamped rows at w16-32. Plot aggregates/fits
  exclude clamped-tower rows with an explicit printed/recorded note; the
  simplex-diagnostics figure shows all rows. Peak GPU memory 2.9 GB at
  n=512; endpoint fwd/bwd FLOPs both ~ n^2.0 (no extra width power vs the
  scalar endpoint); kprop ~ n^3.0 with a small constant.
  Fitted total-Brier width slopes (raw, 95% bootstrap):
    depth2: E0 -1.14, cov1 -1.75, cov2 -1.89, k3 -2.31, k4simple -2.93,
            k4augment -3.01
    depth4: E0 -1.04, cov1 -1.72, cov2 -2.02, k3 -2.53, k4simple -3.00,
            k4augment -2.82
    depth6: E0 -1.07, cov1 -1.76, cov2 -2.01, k3 -2.30, k4simple -2.53,
            k4augment -2.54
  Matched-budget MC ratio (MC MSE at equal FLOPs / det MSE, >1 favors det):
  product-Gaussian argmax crosses 1 at n~45-64 (250x at n=512 depth2);
  k4trace_simple crosses at n~128-181 (depth2), n~362-512 (depth4), not yet
  at n=512 for depth6 (0.45).
- Memory headroom check (local, CPU): factored K3 rank R = 13n; VE joins
  reduce to matmuls, largest materialized table ~ 2Q x 13n (~42 MB at n=512,
  Q=400); endpoint FLOPs scale ~ n^2 (measured 3.97x from n=32 to 64).

## Open issues

- (tracked here as discovered)
