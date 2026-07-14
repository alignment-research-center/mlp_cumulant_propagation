"""Wide-width sweep (n = 64..1024) at shallow depths 2 and 4, 10 seeds, with
per-width reference SE targets chosen so the cross-fidelity noise floor stays
below the expected MSE at every width. Purpose: reliable MSE-vs-width and
FLOPs-vs-width scaling exponents in the regime where the estimator works."""
import os
import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from max_endpoint_experiment import ExperimentCfg, run_experiment  # noqa: E402

if __name__ == "__main__":
    results_root = os.environ.get("RESULTS_DIR")
    for depth in (4, 2):  # cheap depth-4 first as a canary
        cfg = ExperimentCfg.from_json(ROOT / "configs" / f"max_endpoint_wide_d{depth}.json")
        if results_root:
            os.environ["RESULTS_DIR"] = os.path.join(results_root, f"depth{depth}")
        run_experiment(cfg)
