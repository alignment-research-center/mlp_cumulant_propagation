"""Wrapper: run the full max-endpoint experiment (see configs/max_endpoint_full.json)."""
import argparse
import sys
from pathlib import Path

# arc-infra exec()s this file's source with cwd at the repo root and no
# __file__; fall back to cwd in that case.
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from max_endpoint_experiment import ExperimentCfg, run_experiment  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    args, _ = parser.parse_known_args()
    cfg_path = ROOT / "configs" / "max_endpoint_full.json"
    overrides = {"run_name": args.run_name} if args.run_name else {}
    cfg = ExperimentCfg.from_json(cfg_path, **overrides)
    run_experiment(cfg)
