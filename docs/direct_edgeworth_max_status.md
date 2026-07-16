# Direct dense Edgeworth max experiment — status log

Goal: correctness-first *direct dense* implementation of the Cumulant/Edgeworth
estimator for T(theta) = E_X[max_i M_theta(X)_i] (no treewidth/diagram
machinery on the estimator path), testing whether successive Edgeworth terms
improve the MSE width scaling (hypotheses: E0 ~ n^-1, E1 ~ n^-2, E2_k3 ~ n^-3).
Depths (linear layers): 2, 4, 6 per user instruction.

## Relationship to the existing max-endpoint experiment

This repo already contains a diagram/treewidth max-endpoint pipeline
(`src/mlp_kprop/max_endpoint/`, docs/max_endpoint_report.md). Per instructions,
its *quadrature and endpoint formulas* are reused directly
(`quadrature.py`, `hermite_endpoint.py`, `arc_bridge.extract_gaussian_params`,
`ground_truth.py`, `flop_accounting.py`); the new direct path materializes
dense integrated derivative tensors D2/D3/D4 by set-partition equality-pattern
assembly and contracts them with dense cumulants via plain `torch.einsum`.
The treewidth pipeline is used only as a *cross-check* (small widths).

## Files added

- `src/mlp_kprop/max_endpoint/direct_dense.py` — dense D2/D3/D4 assembly
  (set partitions, sign (-1)^(p-1), exact-pattern masking, as_strided diagonal
  embedding), dense cumulant extraction (kappa3 via `.to_tensor()`, kappa4
  kept as (core, metric) trace surrogate — exact against symmetric D4),
  direct einsum contractions (C2: 1/2; C2^2/2: 1/8; C3: 1/6; C4_trace: 1/24),
  memory guard `max_dense_bytes` (`DenseMemoryError`), analytic FLOP tally.
- `tests/test_direct_dense_derivatives.py` — D tensors vs the per-index
  reference integral (all equality patterns), vs independent nested autograd,
  FD recursion D3->D4, symmetry, sum(D1)=1, memory guard.
- `tests/test_direct_dense_contractions.py` — einsum vs nested loops (n=5);
  coefficients 1/2 and 1/8 vs closed-form bivariate Gaussian max (O(c^3)
  residual scaling); 1/6 vs synthetic skewed non-Gaussian vector (spec 11.5).
- `tests/test_direct_dense_end_to_end.py` — Psi vs MC; small ReLU MLPs
  (n=8/12/16): direct == treewidth pipeline to 1e-8 on every correction;
  MC agreement; k_max=1/2 sector handling; dense_orders=(2,) mode.
- `scripts/direct_edgeworth_max_experiment.py` — driver (reuses reference /
  MC-validation / sharding / atomic-checkpoint infra from
  `max_endpoint_experiment.py`); resumable at (width, seed, kprop_variant);
  rows per (width, seed, estimator).
- `scripts/run_direct_edgeworth_max_{smoke,pilot,full}.py`,
  `configs/direct_edgeworth_max_{smoke,pilot,full}.json` — depth loops
  {2,4} (smoke) / {2,4,6} (pilot, full) with per-depth RESULTS_DIR subdirs.
- `scripts/plot_direct_edgeworth_max.py` — the six required figures.

## Method mapping

    k1_simple  -> E0_product_gaussian          (variances only; E1/E2 = E0)
    k2_simple  -> E1_cov1, E2_cov              (needs D2 / D4)
    k3_simple  -> E2_k3, E2_k3_k4trace_simple  (kappa4 = c Sym(M x M), r=2)
    k3_augment -> E2_k3_k4trace_augment        (kappa4 = Sym(C x M), r=1)

E2_k3_k4trace is a *fourth-cumulant trace sector* correction, not a full
kappa4 correction (kprop drops the traceless H4 sector at k_max=3).
Extended widths (192-512) run k1/k2 only with dense_orders=(2,): E0/E1.

## Test status

- Baseline before changes: 198 passed (tests/symb excluded: needs MPI/Sage,
  pre-existing).
- With new tests (26 added): 224 passed locally (macOS CPU).
- Local smoke (depths 2 & 4, widths 8/12/16, 2 seeds, CPU): healthy at depth 2
  (nested ladder improves MSE; direct-vs-treewidth rel diff <= 3e-8 everywhere
  it is defined). At depth 4 and n <= 12 the k_max=3 truncated expansion
  genuinely diverges (all variances clamped, quadrature error O(1..1e2)) —
  the *known* upstream small-width divergence; rows carry
  `negative_variance_clamped` and huge `quadrature_convergence_error`, and the
  treewidth cross-check now reports `undefined_in_divergent_regime` there
  instead of a spurious FAIL.

## Dense memory plan (float64)

| n | D2 (n^2) | D3 (n^3) | D4 (n^4) |
|---|---|---|---|
| 16 | 2.0 KB | 32 KB | 0.5 MB |
| 64 | 33 KB | 2.1 MB | 134 MB |
| 128 | 131 KB | 16.8 MB | 2.15 GB |
| 192 | 295 KB | 56.6 MB | 10.9 GB |
| 256 | 524 KB | 134 MB | 34.4 GB (> 30 GB guard: refused) |

Chunked build intermediate: node_chunk * n^3 * 8 B (0.27 GB at n=128,
chunk 16). kunalc is an A100 **40 GB** (previous instance was 80 GB), so
max_dense_bytes = 30 GB; the guard refuses n >= 256 at order 4, and the
d4probe run measures the practical wall-clock/OOM ceiling at 160-224.

## Cluster log

- Commits `7878791` -> `3bc0fa7` (branch experiment/max-endpoint), upstream `6e80f7f`.
- Recreated instance `kunalc` (user killed the previous one and expressly
  authorized recreating this one instance): `launch kunalc --num-gpus 1 --auto`
  -> verda/helsinki 1x A100-40GB. Host-key fix (ssh-keyscan) + `setup` rerun.
- Runs (all on kunalc, results mirrored to
  gs://arc-ml/kunalchawla/run/mlp_cumulant_propagation/2026-07-16/...):
  - `direct_edgeworth_max_smoke` (depths 2,4; matches local smoke; healthy).
  - `direct_edgeworth_max_pilot` — killed after user re-scoped the full grid
    (own job); relaunched as `direct_edgeworth_max_pilot2` (depths 2,4,6,
    widths 16..64, 8 seeds; 0 errors at depths 2/4; 12 expected
    kprop_nonfinite divergences at depth 6 n<=24; ~35 min).
  - `direct_edgeworth_max_full` — USER GRID: depth 4, 5 seeds, widths
    128/256/512/1024 (E0/E1 only at >=256; D4 refused at 256 with recorded
    reason). 70 rows, all references at target, ~4 min.
- Final analysis, plots, and numbers: docs/direct_edgeworth_max_report.md.

## Open issues

- k_max=3 divergence at (depth 4, n <= 12) and expected worse at depth 6 —
  handled by flagging, excluded from fits, reported not hidden.
