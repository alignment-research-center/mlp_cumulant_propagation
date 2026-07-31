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

- Factored (factorized) cumulant propagation with `kind=AUGMENT` currently
  raises `NotImplementedError`: the augmented diagram set includes terms with
  no O(n)-rank factorization (e.g. cycles of covariance blocks such as the
  (1,1)+(1,1)+(1,1) triangle, and top-degree-cumulant x covariance-edge
  products), so the factored implementation cannot reproduce the unfactored
  augmented output within its FLOP budget. Decide what "factorized augmented"
  should compute (e.g. only the factorable/hypertree subset of the augmented
  diagrams) and reinstate it in `factor_k3.py`/`factor_k4.py`.
