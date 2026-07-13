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

## Tests run

- `uv run pytest -q --ignore=tests/symb` (tests/symb requires an MPI runtime
  not available on this macOS dev box; the full suite including symb is run
  on kunalc).
- Status: see final report / CI notes below.

## Cluster jobs

- (to be filled as launched) smoke -> pilot -> full on instance `kunalc`,
  one GPU, RESULTS_DIR under the arc-infra run root.

## Open issues

- (tracked here as discovered)
