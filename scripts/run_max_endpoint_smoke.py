"""Wrapper: run the smoke max-endpoint experiment (see configs/max_endpoint_smoke.json)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from max_endpoint_experiment import ExperimentCfg, run_experiment  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    args, _ = parser.parse_known_args()
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "max_endpoint_smoke.json"
    overrides = {"run_name": args.run_name} if args.run_name else {}
    cfg = ExperimentCfg.from_json(cfg_path, **overrides)
    run_experiment(cfg)
