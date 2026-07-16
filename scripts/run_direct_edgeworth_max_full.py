"""Wrapper: full direct-Edgeworth-max experiment at depths 2, 4 and 6
(widths 16..128 all methods + 192..512 E0/E1 only, 20 seeds;
see configs/direct_edgeworth_max_full.json)."""
import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from direct_edgeworth_max_experiment import DirectCfg, run_experiment  # noqa: E402

DEPTHS = (2, 4, 6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    args, _ = parser.parse_known_args()
    overrides = {"run_name": args.run_name} if args.run_name else {}
    base = DirectCfg.from_json(ROOT / "configs" / "direct_edgeworth_max_full.json", **overrides)
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
