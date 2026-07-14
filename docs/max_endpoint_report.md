# Max-endpoint experiment — final report (2026-07-13)

## 1–2. Commits
- Branch: `experiment/max-endpoint` (local clone at `~/code/max_kprop/mlp_cumulant_propagation`).
- Upstream ARC commit: `6e80f7f2af0d33e252731ad9611dff17880b12fb`.
- Report generated at branch head (see `git log`); the exact commit used by each
  run is recorded per-row (`git_commit`) and in each run's `manifest.json`.

## 3. Files added / modified
- Added `src/mlp_kprop/max_endpoint/`: `quadrature.py`, `hermite_endpoint.py`,
  `diagrams.py`, `treewidth.py`, `arc_bridge.py`, `estimator.py`,
  `ground_truth.py`, `flop_accounting.py` (plus a collaborator-added argmax
  extension: `argmax.py`, `argmax_mse.py`, `rooted_diagrams.py`).
- Added `scripts/max_endpoint_experiment.py`, `run_max_endpoint_{smoke,pilot,full}.py`,
  `plot_max_endpoint.py`; `configs/max_endpoint_{smoke,pilot,full}.json`;
  `infra/arc_infra_config.py`; 7 new test files (58 tests); `docs/`.
- Modified upstream: `src/mlp_kprop/harmonic.py` only — `proj_coef`'s cache now
  keys on the ambient default dtype (it silently poisoned across
  `set_default_dtype` flips) and zero-coefficient asserts use `zeros_like`.

## 4. Exact commands
Local: `uv sync`; `uv run pytest -q`; smoke/pilot/full wrappers as below.
Cluster (instance `kunalc`, 1×A100-80GB, never created/deleted):

    python -m arc_infra.cli upload-code kunalc --config infra/arc_infra_config.py
    python -m arc_infra.cli blob-auth kunalc --config infra/arc_infra_config.py
    python -m arc_infra.cli run kunalc scripts/run_max_endpoint_smoke.py \
        --run-name max_endpoint_smoke --num-procs 1 --config infra/arc_infra_config.py
    python -m arc_infra.cli run kunalc scripts/run_max_endpoint_pilot.py \
        --run-name max_endpoint_pilot_depth16 --num-procs 1 --config infra/arc_infra_config.py
    python -m arc_infra.cli run kunalc scripts/run_max_endpoint_full.py \
        --run-name max_endpoint_full_relu_depth16 --num-procs 1 --config infra/arc_infra_config.py
    uv run python scripts/plot_max_endpoint.py <results_dir>

## 5. Tests
- Local (macOS, CPU): 197 passed (115 upstream + new).
- kunalc (A100): all max_endpoint/argmax tests pass on GPU. 16 failures are
  pre-existing/environmental, reproduced on a pristine upstream clone on the
  same machine: 12 `tests/symb` need SageMath, 2 need `mpiexec` launch, and
  `test_crit_init_relu[0.5|2.0]` fail order-dependently after `test_kprop.py`
  flips the global default dtype (upstream fragility, unchanged by this branch).

## 6–7. Result paths
- Smoke: `~/run/mlp_cumulant_propagation/2026-07-13/max_endpoint_smoke/results/` (kunalc),
  blob `gs://arc-ml/kunalchawla/run/mlp_cumulant_propagation/2026-07-13/max_endpoint_smoke/results/`,
  local `data/max_endpoint/smoke_kunalc/`.
- Pilot: `.../max_endpoint_pilot_depth16/results/` (kunalc + blob), local
  `data/max_endpoint/pilot_kunalc/`.
- Full: `.../max_endpoint_full_relu_depth16/results/` (kunalc + blob), local
  `data/max_endpoint/full_kunalc/` (plots under `plots/`).

## 8. Estimators implemented (all working)
E0 (product-Gaussian max via stable log-CDF Gauss-Legendre quadrature),
E1 (+C2), E2_cov (+C2^2/2), E2_k3 (+C3, rank-factorized kappa3 contraction),
E2_full (+C4 trace sector), for kprop variants k1/k2 SIMPLE, k2 AUGMENT,
k3 SIMPLE/AUGMENT (factor=True). At k_max=1 the off-diagonal covariance sector
is absent, so E1/E2_cov are exactly E0 (recorded as equivalences).

## 9. Fourth-cumulant trace sector
`K[4]` at the output is an HTensor with metric `M = W_L W_L^T`:
k3 AUGMENT keeps r=1 (kappa4 ~ Sym(C ⊗ M) = harmonic sectors R·H2 ⊕ R²·H0,
traceless H4 dropped); k3 SIMPLE and k2 AUGMENT keep r=2
(kappa4 ~ c·Sym(M ⊗ M) = pure double-trace R²·H0). Contracted as the surrogate
(1/24)Σ C_ij M_kl ∂_ijkl — exact because the derivative tensor is symmetric.

## 10. Treewidth and FLOP scaling
Max induced width over all compiled diagrams: **1** (exact orders found by
exhaustive search for every template; no heuristic fallback used in
production). 67 quotient diagrams at the full k3 level. Fitted total-FLOPs
slopes (n=128..512): product_gaussian ~n^1.9, cov (k2) ~n^2.2, k3 towers
~n^2.3-2.6 empirically over this range (kprop's asymptotic n^4 not yet
dominant; endpoint contraction is 35-60% of total). Max VE table (model bound)
6e8 entries; realized peak GPU memory < 0.2 GB (einsum contracts via GEMM).

## 11. MSE over network seeds (20 seeds, depth 16, cross-fidelity)

| method (variant, estimator) | n=128 | 181 | 256 | 362 | 512 |
|---|---|---|---|---|---|
| product_gaussian (k1,E0) | 1.41e-02 | 4.51e-03 | 4.45e-03 | 1.93e-03 | 8.63e-04 |
| pg_plus_cov1 (k2,E1) | 3.44e-03 | 2.31e-03 | 1.06e-03 | 4.04e-04 | 3.49e-04 |
| pg_plus_cov2 (k2,E2cov) | 1.86e-03 | 1.10e-03 | 5.44e-04 | 2.28e-04 | 1.72e-04 |
| pg_plus_cov2_k3 (k3s,E2k3) | 1.17e-03 | 5.01e-04 | 3.58e-04 | 1.46e-04 | 6.08e-05 |
| k4trace_simple (k3s,E2full) | 1.14e-03 | 4.93e-04 | 3.50e-04 | 1.42e-04 | 5.83e-05 |
| k4trace_augment (k3a,E2full) | 1.18e-03 | 4.70e-04 | 3.39e-04 | 1.49e-04 | 6.18e-05 |
| k2aug_trace_k3k4 (k2a,E2full) | 6.49e-03 | 1.41e-03 | 5.30e-04 | 5.18e-04 | 2.14e-04 |

## 12. Fitted MSE-vs-width slopes (bootstrap 95% CI over seeds; empirical only)
product_gaussian −1.86 [−2.44,−1.16] · cov1 −1.82 [−2.13,−1.46] ·
cov2 −1.83 [−2.10,−1.50] · cov2_k3 −2.06 [−2.68,−1.47] ·
k4trace_simple −2.08 [−2.68,−1.49] · k4trace_augment −2.03 [−2.69,−1.45] ·
k2aug_trace −2.26 [−2.71,−1.59]. These are empirical fits on n=128..512 at
depth 16; no n^{-K} theorem is claimed.

## 13. Matched-budget comparison with Monte Carlo
At equal online FLOPs, plain (or spherical) MC beats every kprop-based tower:
MSE_MC/MSE_det at n=512 is ~0.36 (cov2), ~0.02 (k3 towers) — i.e. MC is
~3x/~50x better; the gap shrinks like ~n^1.2 for the k3 towers (0.006 at n=181
→ 0.02 at n=512), extrapolating to a crossover only around n~10^4.
Exception: the cheap product_gaussian estimator (1.5e7 FLOPs at n=512) beats
matched-budget MC by ~110x. MC MSE=Var/m was validated by 32 explicit repeated
runs (ratios 0.87–1.25 ≈ 1 ± sqrt(2/32)).

## 14. Reference-noise diagnostics
Spherical Rao-Blackwellized backend (validated per-network against Gaussian
MC; max backend z=2.3 over 100 networks; spherical per-sample variance is
strictly lower). Median 2.6e8 samples/stream; every stream reached the 3e-5
SE target; noise floor (se_A·se_B ≈ 1e-9) is 4.8 orders below the smallest
reported MSE — all plotted points resolved. A-B differences are N(0,1)
consistent (max |A−B|/se = 2.28).

## 15. Diagnosis: why the deterministic estimator loses to MC here
(a) Depth-16 ReLU outputs are strongly, randomly correlated: RMS pairwise
correlation ≈ 4.5/√n (0.40 at n=128, 0.19 at n=512). The expansion around an
*independent* Gaussian therefore converges slowly: each C2 order shrinks the
correction only ~10x (|C2²/2|/|C2| ≈ 0.09–0.12, width-independent), and the
residual of E2_full is ~0.5·|C2²/2| — a truncation-dominated series.
(b) Ablation with near-exact sample cumulants (mu, Sigma from 1e7 samples):
E2_cov2 MSE at n=512 is 4.2e-5 vs 8.0e-5 with kprop cumulants — i.e. even
perfect cumulant propagation only buys ~2x; the floor is the Edgeworth
truncation, and the kappa3/kappa4-trace terms only recover that factor
(E2_full 5.9e-5 ≈ the exact-cumulant cov2 floor).
(c) Cost asymmetry: the max is a single O(1)-variance scalar costing 2n²L
FLOPs per MC sample, while the k3 towers cost 1.3–3.3e12 FLOPs at n=512 for
MSE ~6e-5. Unlike the upstream mean-vector task (n outputs, per-coordinate
MSE ~ n^{-k}), the scalar-max target gives MC maximal leverage.
Actionable follow-ups: extend the compiler to C2^3/3! and C2^4/4! (diagrams
remain treewidth ≤ 2–3, cost O(Q n^2..n^3) — likely 1–2 further orders of
magnitude given the ~10x/order empirical ratio), and/or replace the
independent base with a correlated-Gaussian max (resumming Sigma exactly).

## 16. Unresolved issues
- k_max=3 (and k2 AUGMENT) towers genuinely diverge at depth 16 for n ≤ 64
  (float64 mean ~1e16; float32 NaN) — flagged `kprop_nonfinite`; widths < 128
  excluded from the final sweep at the PI's direction.
- Rows with clamped variances (flagged `negative_variance_clamped`) carry
  large quadrature-error estimates (sigma ~1e-5 spikes are unresolvable);
  never silently repaired.
- Upstream order-dependent `crit_init` test fragility and SageMath/MPI symb
  test requirements (pre-existing).

## 17. Depth sweep (depths 2 and 4; widths 128/256/512; 4 seeds)

Run `max_endpoint_depthsweep` (results under `.../max_endpoint_depthsweep/results/depth{2,4}/`,
plots in `depth{2,4}/plots_local/`). Zero failed tasks; whole sweep ran in ~7
minutes on the A100 (vs hours at depth 16), consistent with the cost being a
depth-16 artifact.

MC/deterministic matched-budget ratio (>1 = deterministic wins), k4trace_simple:

| depth | n=128 | n=256 | n=512 |
|---|---|---|---|
| 2 | 2.6 | 40 | **248** |
| 4 | 0.42 | 0.38 | **4.8** |
| 16 (sec. 13) | 0.003 | 0.006 | 0.02 |

At depth 2 the full estimator reaches MSE 7.3e-9 at n=512 (vs matched-budget
MC 1.8e-6) with an MSE-vs-width decay ~n^-3.7 over 128->512 — two-plus orders
of magnitude better than MC and steepening with width. At depth 4 it crosses
above MC around n~512 (4-5x). At depth 16 it loses by ~50x. The mechanism
matches the diagnosis in sec. 15: the RMS pairwise output correlation at
n=512 grows with depth (0.055 at depth 2, 0.080 at depth 4, 0.19 at depth
16), and the Edgeworth expansion around the independent Gaussian converges
fast exactly when those correlations are small. Caveat: the depth-2 n=512
MSE is only ~8x above the reference noise floor (9e-10); resolving deeper
would need tighter references.
