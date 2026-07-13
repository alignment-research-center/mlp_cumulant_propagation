# Max-endpoint experiment

Estimates, for a fixed randomly initialized bias-free He ReLU MLP `M_theta`
with input/hidden/output width `n`,

    T(theta) = E_{X ~ N(0, I_n)}[ max_i M_theta(X)_i ],

deterministically, by combining ARC cumulant propagation with a
Hermite/Edgeworth expansion of the max around an independent Gaussian
reference. The MSE reported everywhere is over random network initializations
theta (cross-fidelity estimator, see below).

## Method

1. **kprop** (`mlp_kprop`) propagates cumulants to the output layer:
   mean `mu`, covariance `Sigma`, third cumulant `kappa3`, and the retained
   fourth-cumulant sector, at budget `k_max` and kind `SIMPLE`/`AUGMENT`
   (`factor=True` at `k_max=3`).
2. **Independent Gaussian reference** `G_i = mu_i + sigma_i Z_i`,
   `sigma_i^2 = Sigma_ii`. The base value is the product-Gaussian max
   `Psi = hi - int_lo^hi prod_i Phi((t-mu_i)/sigma_i) dt`, computed via
   `log_ndtr` (never a direct CDF product), Gauss-Legendre nodes on bisected
   endpoints, verified at Q vs 2Q nodes (`max_endpoint/quadrature.py`).
3. **Endpoint derivatives**: for multi-index beta with support S, |S| = p,

       d^beta Psi = (-1)^(p-1) int B(t) prod_{i in S} u_{i,beta_i}(t) dt,
       u_{i,k} = phi(a_i) He_{k-1}(a_i) / (sigma_i^k Phi(a_i)),

   with probabilists' Hermite polynomials He_k and the hazard ratio phi/Phi
   computed as `exp(log_phi - log_ndtr)` (`max_endpoint/hermite_endpoint.py`).
4. **Diagram compiler** (`max_endpoint/diagrams.py`): each operator term
   (`C2`, `C2^2/2`, `C3`, `C4_trace`) is decomposed over equality patterns of
   its derivative slots (set partitions), signed `(-1)^(p-1)`, and converted
   from injective to unrestricted contractions by Moebius inversion on the
   partition lattice. Compiled quotient diagrams are canonicalized, merged,
   and serializable to JSON. Validated against brute-force nested-loop sums at
   n <= 5 (`tests/test_max_endpoint_diagrams.py`).
5. **Variable elimination** (`max_endpoint/treewidth.py`): each diagram is
   contracted by sum-product elimination with an exact minimum-width order
   (exhaustive search; templates have <= 6 variables; min-fill fallback is
   flagged). Cost O(Q n^(w+1)); observed max induced width for all shipped
   templates is 2 (rank-decomposed kappa3 diagrams), with most at width 1.
6. **ARC bridge** (`max_endpoint/arc_bridge.py`): diagrams contract directly
   against `FactoredTensor` (kappa3 rank decomposition, rank index becomes an
   elimination variable) and the `HTensor` radial (trace) sectors of kappa4.
   Dense `.to_tensor()` is only ever allowed at n <= 16 (guarded by assert).

### Estimators

    E0_product_gaussian = Psi
    E1_cov1             = + C2 Psi          (C2 = (1/2) sum_{i!=j} Sigma_ij d_i d_j)
    E2_cov2             = + (1/2) C2^2 Psi
    E2_k3               = + C3 Psi
    E2_full             = + C4_trace Psi

### Exact interpretation of the fourth-cumulant trace sector

At the output layer, kprop's `K[4]` is an `HTensor` with radial index r and
metric `M = W_L W_L^T` (final weight Gram matrix):

- `k_max=3 AUGMENT`: r=1, core matrix `C`; `kappa4 ~ Sym(C (x) M)` — the
  harmonic projection `P_{>=1} kappa4 = R^1 H_2 (+) R^2 H_0` (traceless `H_4`
  dropped). `C4_trace = (1/24) sum_{ijkl} C_ij M_kl d_ijkl`.
- `k_max=3 SIMPLE` and `k_max=2 AUGMENT`: r=2, scalar core c;
  `kappa4 ~ c Sym(M (x) M)` — the pure double-trace sector
  `P_{>=2} kappa4 = R^2 H_0`.

Similarly `k_max=2 AUGMENT` retains `kappa3 ~ Sym(v (x) M)` (sector `R^1 H_1`).
Sector availability is recorded per row (`k3_repr`, `k4_sector`); missing
sectors are flagged in `status`, never silently dropped. `Sym` may be replaced
by the plain tensor product in every contraction because the derivative tensor
is fully symmetric; equality patterns are taken on the surrogate product form.

### kprop-variant -> method mapping (deduplicated)

- `product_gaussian`: (k_max=1 SIMPLE, E0). At k_max=1 only variances are
  tracked (vector metric), so E1/E2_cov are *identical* to E0 — recorded as
  equivalences, only E0 is emitted.
- `pg_plus_cov1` / `pg_plus_cov2`: (k_max=2 SIMPLE, E1 / E2_cov).
- `pg_plus_cov2_k3`: (k_max=3 SIMPLE, E2_k3).
- `pg_cov2_k3_k4trace_simple` / `_augment`: (k_max=3 SIMPLE / AUGMENT, E2_full).
- Secondary: (k_max=2 AUGMENT, E2_full) — trace-only kappa3/kappa4.

## Ground truth (cross-fidelity, F)

Per network: two independent reference estimates `T_ref_A`, `T_ref_B`
(independent streams); `err_cross = (T_hat - A)(T_hat - B)` satisfies
`E[err_cross | theta] = (T_hat - T)^2`. Negative values are kept (never
clipped); aggregate plots mark noise-dominated points as unresolved.

Primary backend is the **Rao-Blackwellized spherical** estimator (F1): for the
bias-free positively homogeneous network, `T = E[R] E_u[max_i M(u)_i]` with
`E[R] = sqrt(2) Gamma((n+1)/2)/Gamma(n/2)`; implemented as
`E[R] * max_i M(X)_i / |X|`. It is validated against Gaussian-input MC per
network (`backend_consistency_z`) and in tests; it is refused when biases are
present. Adaptive stopping: stream batches until `se <= target_se` or the
sample cap (stopping reason recorded); float64 accumulation; OOM backoff.

## Monte Carlo baselines (G)

Conditional MSE of m-sample MC is `Var_X(max | theta)/m`; the variance is
estimated from a dedicated Gaussian stream (and the spherical variance from
the reference streams). Validated by explicit repeated MC runs
(`mcval_*.json`: empirical/predicted ratio ~ 1 +- sqrt(2/repeats)).
Matched-budget comparisons use `m = flops_total_det / flops_per_sample`.

## FLOP conventions (H)

- `flops_kprop`: ARC `NamedFlopCounter` total (symmetric-kernel adjusted);
  `flops_kprop_raw` unadjusted.
- `flops_endpoint`: analytic (workspace ~9 Qn + 7 Qn per Hermite order;
  quadrature 2Q; VE joins (k-1)S mults + S adds per eliminated variable),
  cross-checked against the instrumented counter in
  `tests/test_max_endpoint_flops.py`. Both Q and 2Q passes are charged.
- MC: `m * (instrumented forward per sample + (n-1) max reduction
  [+ 2n+3 spherical normalization])`; RNG excluded (upstream convention).
- Diagram compilation is offline, cached, and width-independent; reference
  computation is evaluation overhead and charged to no estimator.

## Running

    uv run pytest -q                                   # all tests must pass
    uv run python scripts/run_max_endpoint_smoke.py    # widths 16,32, depth 4
    uv run python scripts/run_max_endpoint_pilot.py    # widths 16..128, depth 16
    uv run python scripts/run_max_endpoint_full.py     # widths 16..512, depth 16
    uv run python scripts/plot_max_endpoint.py <results_dir>

Results go to `$RESULTS_DIR` (or `data/max_endpoint/<run_name>`): one atomic
JSON per (width, seed, variant) task under `tasks/` (fully resumable), merged
`results.jsonl` / `per_network.csv` / `aggregate.csv` / `manifest.json`.
Seeds: `sha256(base_seed:width:net_seed:purpose)` for purposes
net/refA/refB/gauss_var/mcval. If `OMPI_COMM_WORLD_SIZE > 1`, (width, seed)
pairs are sharded across ranks, one CUDA device per rank.
