# TODO

- The diagram truncation in the nonlinear step (`get_vec_cond` in
  `kprop_harmonic.py`) changed in two ways:
  1. It was tightened to match equations (13)–(14) of the paper; the
     implementation previously summed strictly more terms of the Edgeworth
     expansion than the paper describes.
  2. For `kind=AUGMENT` the cap is now raised by one relative to the paper's
     equation (14): the augmented algorithm sums every diagram of squared size
     Omega(n^{-k_max}) (all leading-order contributions to the basic
     algorithm's MSE), which is still time-subleading unfactorized. (The
     paper's S.4.2 "all other parts remain the same" should be amended to
     match.)

  **All cached kprop outputs, data files, and FLOP counts (and any downstream
  figures) for `k_max >= 2` are stale and need to be refreshed.** (`k_max = 1`
  is unaffected; test goldens have already been refreshed via
  `scripts/refresh_goldens.py`.)

- Factored (factorized) cumulant propagation with `kind=AUGMENT` is now
  defined as the augmented term set minus the terms it cannot afford
  (`kprop_harmonic.factored_keeps_term`: non-hypertree top-slice diagrams,
  e.g. covariance cycles, and products of the factored top-degree cumulant's
  all-distinct block with other blocks). It is therefore NOT equivalent to
  unfactored `kind=AUGMENT`; the dropped terms have the same Theta(n^-k_max)
  squared size as the extra augmented terms that are kept. Tests compare it
  against an unfactored reference restricted to the same term set.
  **The paper needs updating** to reflect both the augmented Edgeworth cap
  (equation (14) with K+1 instead of K for the augmented algorithm, contra
  S.4.2's "all other parts remain the same") and that factorized-augmented
  computes a different estimate than (unfactorized) augmented.
