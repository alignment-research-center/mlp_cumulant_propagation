"""
Estimation of E_X[max_i M(X)_i] for a fixed MLP via cumulant propagation plus a
Hermite/Edgeworth expansion of the maximum around an independent Gaussian
reference with matched coordinate means and variances.

Modules:
- quadrature: numerically stable 1-d integrals for the product-Gaussian max.
- hermite_endpoint: endpoint unary weights u_{i,k}(t) and derivative formulas.
- diagrams: offline compiler turning cumulant differential operators into
  quotient factor-hypergraph contractions (equality patterns + Moebius inversion).
- treewidth: variable-elimination contraction engine with analytic FLOP counts.
- arc_bridge: adapters contracting diagrams directly against ARC's
  HTensor / DSTensor / FactoredTensor cumulant representations.
- estimator: top-level nested estimators E0 ... E2_full.
- ground_truth: cross-fidelity Monte Carlo references (Gaussian and spherical).
- flop_accounting: FLOP bookkeeping shared across the pipeline.
"""

from mlp_kprop.max_endpoint.quadrature import (  # noqa: F401
    QuadratureCfg,
    product_gaussian_max,
    product_gaussian_max_reference,
)
from mlp_kprop.max_endpoint.hermite_endpoint import (  # noqa: F401
    EndpointWorkspace,
    hermite_he,
)
from mlp_kprop.max_endpoint.estimator import max_endpoint_estimate  # noqa: F401
from mlp_kprop.max_endpoint.argmax import (  # noqa: F401
    argmax_endpoint_estimate,
    product_gaussian_argmax,
    product_gaussian_argmax_reference,
    project_to_simplex,
)
