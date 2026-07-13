"""Depth sweep: rerun the max-endpoint experiment at shallow depths (2 and 4
linear layers) where the Edgeworth expansion around the independent Gaussian
should converge fast, to test whether the deterministic estimators beat
matched-budget Monte Carlo. Widths 128/256/512, 4 seeds per depth."""
import os
import sys
from dataclasses import replace
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from max_endpoint_experiment import ExperimentCfg, run_experiment  # noqa: E402

DEPTHS = (2, 4)

if __name__ == "__main__":
    base = ExperimentCfg.from_json(ROOT / "configs" / "max_endpoint_depthsweep.json")
    results_root = os.environ.get("RESULTS_DIR")
    for depth in DEPTHS:
        cfg = replace(base, num_layers=depth, run_name=f"max_endpoint_depth{depth}")
        if results_root:
            # Separate subdir per depth so task checkpoints don't collide.
            os.environ["RESULTS_DIR"] = os.path.join(results_root, f"depth{depth}")
        run_experiment(cfg)
