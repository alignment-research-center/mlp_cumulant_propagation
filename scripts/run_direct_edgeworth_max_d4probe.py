"""Wrapper: dense-D4 practicality probe at depth 4 (widths 160..256, 2 seeds,
D4-bearing variants only). Establishes where the direct dense order-4 endpoint
becomes impractical on the available GPU and records the stopping reason
(guard refusal / OOM / runtime)."""
import argparse
import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from direct_edgeworth_max_experiment import DirectCfg, run_experiment  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    args, _ = parser.parse_known_args()
    overrides = {"run_name": args.run_name} if args.run_name else {}
    cfg = DirectCfg.from_json(ROOT / "configs" / "direct_edgeworth_max_d4probe.json", **overrides)
    run_experiment(cfg)
