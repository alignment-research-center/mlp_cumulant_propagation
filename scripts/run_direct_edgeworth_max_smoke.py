"""Wrapper: smoke direct-Edgeworth-max experiment at depths 2 and 4
(widths 8/12/16, 2 seeds; see configs/direct_edgeworth_max_smoke.json)."""
import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

# arc-infra exec()s this file's source with cwd at the repo root and no
# __file__; fall back to cwd in that case.
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from direct_edgeworth_max_experiment import DirectCfg, run_experiment  # noqa: E402

DEPTHS = (2, 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    args, _ = parser.parse_known_args()
    overrides = {"run_name": args.run_name} if args.run_name else {}
    base = DirectCfg.from_json(ROOT / "configs" / "direct_edgeworth_max_smoke.json", **overrides)
    results_root = os.environ.get("RESULTS_DIR")
    for depth in DEPTHS:
        cfg = replace(base, num_layers=depth)
        if results_root:
            os.environ["RESULTS_DIR"] = os.path.join(results_root, f"depth{depth}")
        else:
            os.environ["RESULTS_DIR"] = str(
                ROOT / "data" / "direct_edgeworth_max" / base.run_name / f"depth{depth}"
            )
        run_experiment(cfg)
