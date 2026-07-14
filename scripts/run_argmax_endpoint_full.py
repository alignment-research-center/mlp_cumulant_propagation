"""Wrapper: run the full argmax-endpoint experiment at depths 2, 4, 6
(see configs/argmax_endpoint_full.json). Each depth gets its own resumable
results directory: $RESULTS_DIR/depth<d> (or data/argmax_endpoint/<run>_depth<d>
locally), because task filenames do not encode depth."""
import argparse
import json
import os
import sys
from pathlib import Path

# arc-infra exec()s this file's source with cwd at the repo root and no
# __file__; fall back to cwd in that case.
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from argmax_endpoint_experiment import ArgmaxExperimentCfg, run_experiment  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    args, _ = parser.parse_known_args()
    cfg_path = ROOT / "configs" / "argmax_endpoint_full.json"
    raw = json.loads(cfg_path.read_text())
    depths = raw.get("depths", [2, 4, 6])
    base = args.run_name or raw["run_name"]
    base_results = os.environ.get("RESULTS_DIR")
    for depth in depths:
        if base_results:
            os.environ["RESULTS_DIR"] = str(Path(base_results) / f"depth{depth}")
        cfg = ArgmaxExperimentCfg.from_json(
            cfg_path, run_name=f"{base}_depth{depth}", num_layers=depth
        )
        run_experiment(cfg)
