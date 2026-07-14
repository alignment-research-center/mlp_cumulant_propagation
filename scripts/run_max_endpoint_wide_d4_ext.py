"""Depth-4 width extension (n = 1448, 2048): tests whether the MSE-vs-width
slope measured on 64..1024 is asymptotic or still drifting."""
import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from max_endpoint_experiment import ExperimentCfg, run_experiment  # noqa: E402

if __name__ == "__main__":
    run_experiment(ExperimentCfg.from_json(ROOT / "configs" / "max_endpoint_wide_d4_ext.json"))
