"""Regenerate the goldens consumed by tests/test_kprop.py::test_goldens.

Must be kept in sync with the setup in that test (seed, MLP construction, dtype).
"""
import gc
from pathlib import Path

import torch

from mlp_kprop import kprop_harmonic as kprop
from mlp_kprop.kprop_harmonic import Kind
from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GOLDENS_ROOT = Path(__file__).parent.parent / "tests" / "goldens"

n = 16
depth = 3
k_maxs = range(1, 5)

for kprop_kind, kind in {"simple": Kind.SIMPLE, "augment": Kind.AUGMENT}.items():
    goldens_dir = GOLDENS_ROOT / f"mlp_kprop_{kprop_kind}"
    K_in = {1: torch.zeros(n, device=device), 2: torch.eye(n, device=device)}

    torch.manual_seed(0)
    mlp = MLP(n, n, n, depth).to(device)

    for k_max in k_maxs:
        K_in = {k: v.clone() for k, v in K_in.items()}
        gc.collect()
        torch.cuda.empty_cache()
        K_by_layer = kprop.mlp_kprop(mlp, K_in, k_max=k_max, output_all=True, kind=kind)
        golden_path = goldens_dir / f"n{n}" / f"depth{depth}" / f"kmax{k_max}" / "K_by_layer.pt"
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(K_by_layer, golden_path)
        print(f"Wrote {golden_path}")
