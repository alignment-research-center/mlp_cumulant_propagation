"""
FLOP accounting for the max-endpoint pipeline.

Conventions (matching the rest of this experiment; see docs/max_endpoint_experiment.md):
- kprop FLOPs use ARC's NamedFlopCounter with its symmetric-kernel adjustments
  (`flops_kprop`); the unadjusted count is stored as `flops_kprop_raw`.
- Endpoint FLOPs (workspace + quadrature + variable-elimination contractions)
  use the analytic counters in treewidth.FlopTally / EndpointWorkspace,
  cross-checked against torch's instrumented counter in tests.
- MC FLOPs = m * (one instrumented MLP forward + max reduction [+ spherical
  normalization]); random-number generation is excluded, as upstream.
- Reference (ground-truth) computation is evaluation overhead and is never
  charged to any estimator.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from mlp_kprop.flop_utils import NamedFlopCounter

__all__ = ["KpropFlopRecord", "count_kprop_flops"]


@dataclass
class KpropFlopRecord:
    total: int = 0        # symmetric-kernel-adjusted (ARC plotting convention)
    raw_total: int = 0    # unadjusted
    by_name: dict[str, int] = field(default_factory=dict)


@contextmanager
def count_kprop_flops() -> Iterator[KpropFlopRecord]:
    """Instrument a kprop call: `with count_kprop_flops() as rec: mlp_kprop(...)`."""
    record = KpropFlopRecord()
    with NamedFlopCounter() as counter:
        yield record
    record.total = counter.total()
    record.raw_total = counter.raw_total()
    record.by_name = dict(counter.flop_dict())
