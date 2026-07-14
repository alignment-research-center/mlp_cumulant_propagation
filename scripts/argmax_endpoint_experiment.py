"""
Main driver for the argmax-endpoint experiment.

For each (width n, network seed):
  1. Build the same fixed bias-free He ReLU MLP as the scalar max experiment.
  2. Draw shared evaluation blocks of Gaussian inputs and record winner counts
     c_i per block (tie policy: torch.argmax first index; ties otherwise
     ignored). Blocks are shared across every estimator (paired comparisons).
  3. For each kprop variant (k_max, kind, factor), run ARC cumulant
     propagation under a FLOP counter and evaluate all nested argmax
     estimators q_hat = grad_mu(E) (raw + simplex-projected).
  4. Score every estimator with the unbiased Brier U-statistic per block
     (float64; may be negative; never clipped) and emit one JSON row per
     (estimator, projection) into RESULTS_DIR/tasks/ (atomic; resumable);
     q vectors go to compressed NPZ next to the row files.

Monte Carlo baseline: MSE_MC(m) = (1 - ||q||^2)/m with ||q||^2 estimated by
the pooled collision U-statistic; validated by explicit repeated MC runs on a
subset (mcval tasks). Matched budgets use m = flops_total / flops_per_sample.

Sharding, seeds, merging and atomic checkpointing mirror
max_endpoint_experiment.py (whose utilities are imported, not duplicated).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from max_endpoint_experiment import (  # noqa: E402
    atomic_write_json,
    build_mlp,
    derive_seed,
    git_commit,
    mpi_rank_size,
    results_dir,
    upstream_commit,
)
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop  # noqa: E402
from mlp_kprop.max_endpoint.argmax import (  # noqa: E402
    TIE_POLICY,
    argmax_endpoint_estimate,
)
from mlp_kprop.max_endpoint.argmax_mse import (  # noqa: E402
    argmax_flops_per_sample,
    collision_probability,
    mc_mse_predicted,
    mse_unbiased,
    winner_counts,
)
from mlp_kprop.max_endpoint.flop_accounting import count_kprop_flops  # noqa: E402
from mlp_kprop.max_endpoint.quadrature import QuadratureCfg  # noqa: E402

logger = logging.getLogger("argmax_endpoint_experiment")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvaluationCfg:
    num_blocks: int = 8
    samples_per_block: int = 4096
    batch_size: int = 262_144


@dataclass
class MCValidationCfg:
    enabled: bool = False
    widths: tuple[int, ...] = (16,)
    repeats: int = 32
    samples: int = 8192


@dataclass
class ArgmaxExperimentCfg:
    run_name: str = "argmax_endpoint"
    widths: tuple[int, ...] = (16, 32)
    network_seeds: tuple[int, ...] = (0, 1)
    num_layers: int = 4
    nonlin: str = "relu"
    base_seed: int = 0
    kprop_variants: tuple[dict, ...] = (
        {"name": "k1_simple", "k_max": 1, "kind": "SIMPLE", "factor": False},
        {"name": "k2_simple", "k_max": 2, "kind": "SIMPLE", "factor": False},
        {"name": "k2_augment", "k_max": 2, "kind": "AUGMENT", "factor": False},
        {"name": "k3_simple", "k_max": 3, "kind": "SIMPLE", "factor": True},
        {"name": "k3_augment", "k_max": 3, "kind": "AUGMENT", "factor": True},
    )
    use_avg_metric: bool = True
    quadrature_nodes: int = 256
    quadrature_convergence_factor: int = 2
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)
    mc_validation: MCValidationCfg = field(default_factory=MCValidationCfg)
    check_dense_vs_factorized_max_n: int = 16
    dtype: str = "float32"

    @staticmethod
    def from_json(path: str | Path, **overrides) -> "ArgmaxExperimentCfg":
        raw = json.loads(Path(path).read_text())
        raw.update(overrides)
        raw.pop("depths", None)  # consumed by the pilot/full wrappers (one run per depth)
        ev = EvaluationCfg(**raw.pop("evaluation", {}))
        mcv = MCValidationCfg(**raw.pop("mc_validation", {}))
        raw["widths"] = tuple(raw.get("widths", (16, 32)))
        raw["network_seeds"] = tuple(raw.get("network_seeds", (0, 1)))
        raw["kprop_variants"] = (
            tuple(raw["kprop_variants"])
            if "kprop_variants" in raw
            else ArgmaxExperimentCfg().kprop_variants
        )
        return ArgmaxExperimentCfg(evaluation=ev, mc_validation=mcv, **raw)


# ---------------------------------------------------------------------------
# Per-task work
# ---------------------------------------------------------------------------

def eval_task(
    cfg: ArgmaxExperimentCfg, width: int, net_seed: int, device: torch.device, out: Path
) -> dict:
    """Shared evaluation blocks: winner counts per block + MC baseline data."""
    path = out / "tasks" / f"argmax_eval_w{width}_s{net_seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    mlp = build_mlp(cfg, width, net_seed, device)
    ev = cfg.evaluation
    blocks = []
    for b in range(ev.num_blocks):
        c = winner_counts(
            mlp,
            num_samples=ev.samples_per_block,
            seed=derive_seed(cfg.base_seed, width, net_seed, f"eval_block{b}"),
            device=device,
            batch_size=ev.batch_size,
        )
        blocks.append(c.tolist())
    pooled = torch.tensor(blocks, dtype=torch.int64).sum(dim=0)
    payload = {
        "width": width,
        "net_seed": net_seed,
        "num_blocks": ev.num_blocks,
        "samples_per_block": ev.samples_per_block,
        "total_samples": ev.num_blocks * ev.samples_per_block,
        "block_counts": blocks,
        "collision_probability_estimate": collision_probability(pooled),
        "mc_flops_per_sample": argmax_flops_per_sample(mlp),
        "tie_policy": TIE_POLICY,
        "wall_seconds": time.time() - t0,
    }
    atomic_write_json(path, payload)
    return payload


def variant_task(
    cfg: ArgmaxExperimentCfg, width: int, net_seed: int, variant: dict, ev: dict,
    device: torch.device, out: Path,
) -> list[dict]:
    """One kprop variant -> all nested argmax estimators -> scored rows."""
    vname = variant["name"]
    path = out / "tasks" / f"argmax_est_w{width}_s{net_seed}_{vname}.json"
    if path.exists():
        return json.loads(path.read_text())["rows"]

    dtype = getattr(torch, cfg.dtype)
    torch.set_default_dtype(dtype)
    mlp = build_mlp(cfg, width, net_seed, device).to(dtype)
    kind = Kind[variant["kind"]]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    k_in = {
        1: torch.zeros(width, device=device, dtype=dtype),
        2: torch.eye(width, device=device, dtype=dtype),
    }
    t0 = time.time()
    try:
        with count_kprop_flops() as krec:
            K = mlp_kprop(
                mlp, k_in, k_max=variant["k_max"], kind=kind,
                factor=variant["factor"], use_avg_metric=cfg.use_avg_metric,
            )
        wall_kprop = time.time() - t0
        quad = QuadratureCfg(
            num_nodes=cfg.quadrature_nodes,
            convergence_factor=cfg.quadrature_convergence_factor,
        )
        res = argmax_endpoint_estimate(K, quad_cfg=quad, device=device)
    except Exception as e:  # record, don't crash the sweep
        logger.exception(f"Task w{width} s{net_seed} {vname} failed")
        rows = [{
            "run_name": cfg.run_name, "width": width, "network_seed": net_seed,
            "kprop_variant": vname, "estimator_name": "ALL",
            "estimator_projection": "raw", "status": "failed",
            "warning": f"{type(e).__name__}: {e}",
        }]
        atomic_write_json(path, {"rows": rows})
        return rows

    peak_mem = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    block_counts = [torch.tensor(c, dtype=torch.int64) for c in ev["block_counts"]]
    m_block = ev["samples_per_block"]

    # q vectors -> one compressed NPZ per task.
    npz_path = out / "tasks" / f"argmax_q_w{width}_s{net_seed}_{vname}.npz"
    np.savez_compressed(
        npz_path,
        **{f"{name}__raw": q.cpu().numpy() for name, q in res.q_raw.items()},
        **{f"{name}__projected": q.cpu().numpy() for name, q in res.q_projected.items()},
    )

    rows = []
    for est_name in res.q_raw:
        for projection in ("raw", "projected"):
            q = (res.q_raw if projection == "raw" else res.q_projected)[est_name]
            block_mse = [mse_unbiased(q, c) for c in block_counts]
            bm = np.asarray(block_mse, dtype=np.float64)
            mse_mean = float(bm.mean())
            mse_se = float(bm.std(ddof=1) / math.sqrt(len(bm))) if len(bm) > 1 else float("nan")
            sim = res.simplex[est_name]
            raw_proj_dist = float(
                (res.q_raw[est_name] - res.q_projected[est_name]).norm()
            )
            rows.append({
                "run_name": cfg.run_name,
                "git_commit": git_commit(),
                "upstream_commit": upstream_commit(),
                "scalar_endpoint_state": "shared-worktree scalar max endpoint (see manifest)",
                "width": width,
                "input_dim": width,
                "hidden_dim": width,
                "output_dim": width,
                "num_layers": cfg.num_layers,
                "network_seed": net_seed,
                "net_torch_seed": derive_seed(cfg.base_seed, width, net_seed, "net"),
                "kprop_variant": vname,
                "estimator_name": est_name,
                "estimator_projection": projection,
                "k_max": variant["k_max"],
                "kprop_kind": variant["kind"],
                "factorized": variant["factor"],
                "tie_policy": TIE_POLICY,
                # simplex diagnostics always describe the raw vector
                "q_sum": sim["q_sum"],
                "q_min": sim["q_min"],
                "q_max": sim["q_max"],
                "q_l1": sim["q_l1"],
                "q_l2_sq": sim["q_l2_sq"],
                "simplex_residual": sim["simplex_residual"],
                "num_negative_coordinates": sim["num_negative_coordinates"],
                "raw_projected_l2_distance": raw_proj_dist,
                "mse_total_unbiased": mse_mean,
                "mse_total_input_se": mse_se,
                "mse_per_coordinate_unbiased": mse_mean / width,
                "mse_per_coordinate_input_se": mse_se / width,
                "block_mse": block_mse,
                "evaluation_num_blocks": ev["num_blocks"],
                "evaluation_samples_per_block": m_block,
                "evaluation_total_samples": ev["total_samples"],
                "collision_probability_estimate": ev["collision_probability_estimate"],
                "scalar_estimate": res.scalar_estimates[est_name],
                "psi": res.psi,
                "equivalences": res.equivalences,
                "k3_repr": res.k3_repr,
                "k4_sector": res.k4_sector,
                "num_clamped_var": res.num_clamped_var,
                "flops_kprop": krec.total,
                "flops_kprop_raw": krec.raw_total,
                "flops_endpoint_forward": res.flops_endpoint_forward,
                "flops_endpoint_backward": res.flops_endpoint_backward,
                "flops_endpoint_total": res.flops_endpoint_total,
                "flops_total": krec.total + res.flops_endpoint_total,
                "mc_flops_per_sample": ev["mc_flops_per_sample"],
                "wall_seconds_kprop": wall_kprop,
                "wall_seconds_endpoint": res.wall_seconds,
                "peak_gpu_memory_bytes": peak_mem,
                "quadrature_nodes": cfg.quadrature_nodes,
                "quadrature_scalar_error": res.quadrature_scalar_error.get(est_name),
                "quadrature_argmax_linf_error": res.quadrature_argmax_linf_error.get(est_name),
                "quadrature_argmax_l2_error": res.quadrature_argmax_l2_error.get(est_name),
                "scalar_treewidth": res.max_treewidth,   # same contraction graphs
                "argmax_treewidth": res.max_treewidth,   # root adds no edge
                "treewidth_exact": res.treewidth_exact,
                "num_diagrams": res.num_diagrams,
                "maximum_intermediate_table_entries": res.max_table_numel,
                "q_npz": npz_path.name,
                "status": ";".join(sorted(set(res.status))) or "ok",
                "warning": "",
            })
    atomic_write_json(path, {"rows": rows})
    return rows


def dense_check_task(
    cfg: ArgmaxExperimentCfg, width: int, net_seed: int, device: torch.device, out: Path
) -> dict:
    """Factorized vs dense .to_tensor() argmax gradients (small widths only)."""
    from mlp_kprop.harmonic import HTensor

    path = out / "tasks" / f"argmax_densecheck_w{width}_s{net_seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        mlp = build_mlp(cfg, width, net_seed, device)
        k_in = {1: torch.zeros(width, device=device), 2: torch.eye(width, device=device)}
        K = mlp_kprop(mlp, k_in, k_max=3, kind=Kind.AUGMENT, factor=True,
                      use_avg_metric=cfg.use_avg_metric)
        res_fac = argmax_endpoint_estimate(K, device=device)
        K_dense = dict(K)
        K_dense[3] = HTensor(K[3].to_tensor().to(torch.float64), r=0)
        K_dense[4] = HTensor(K[4].to_tensor().to(torch.float64), r=0)
        res_dense = argmax_endpoint_estimate(
            K_dense, device=device, dense_max_n=cfg.check_dense_vs_factorized_max_n
        )
        max_linf = 0.0
        for name in res_fac.q_raw:
            d = float((res_fac.q_raw[name] - res_dense.q_raw[name]).abs().max())
            max_linf = max(max_linf, d)
        payload = {
            "width": width, "net_seed": net_seed, "max_q_linf_diff": max_linf,
            "passed": bool(max_linf < 1e-6),
        }
    except Exception as e:  # a diverged tower must not kill the sweep
        logger.exception(f"dense check w{width} s{net_seed} failed")
        payload = {
            "width": width, "net_seed": net_seed, "max_q_linf_diff": float("nan"),
            "passed": False, "error": f"{type(e).__name__}: {e}",
        }
    atomic_write_json(path, payload)
    return payload


def mc_validation_task(
    cfg: ArgmaxExperimentCfg, width: int, net_seed: int, device: torch.device, out: Path
) -> dict:
    """Validate MSE_MC(m) = (1 - ||q||^2)/m with explicit repeated MC runs."""
    path = out / "tasks" / f"argmax_mcval_w{width}_s{net_seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    # Pure sampling task (no kprop); failures here should also never kill the
    # sweep, so the caller logs `ratio` defensively.
    mlp = build_mlp(cfg, width, net_seed, device)
    m = cfg.mc_validation.samples
    reps = cfg.mc_validation.repeats
    # Large reference for q (evaluation overhead, charged to no estimator).
    ref_counts = winner_counts(
        mlp, num_samples=4_000_000,
        seed=derive_seed(cfg.base_seed, width, net_seed, "mcval_ref"),
        device=device, batch_size=cfg.evaluation.batch_size,
    )
    q_ref = ref_counts.double() / float(ref_counts.sum())
    errs = []
    for r in range(reps):
        c = winner_counts(
            mlp, num_samples=m,
            seed=derive_seed(cfg.base_seed, width, net_seed, f"mcval{r}"),
            device=device, batch_size=cfg.evaluation.batch_size,
        )
        freq = c.double() / m
        errs.append(float(((freq - q_ref) ** 2).sum()))
    empirical = sum(errs) / reps
    predicted = mc_mse_predicted(collision_probability(ref_counts), m)
    payload = {
        "width": width, "net_seed": net_seed, "samples": m, "repeats": reps,
        "empirical_mse": empirical, "predicted_mse": predicted,
        "ratio": empirical / predicted,
        "note": "ratio ~ 1 +- O(sqrt(2/repeats)); q_ref noise adds ~(1-||q||^2)/4e6",
    }
    atomic_write_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_results(cfg: ArgmaxExperimentCfg, out: Path) -> None:
    import pandas as pd

    rows = []
    for f in sorted((out / "tasks").glob("argmax_est_*.json")):
        rows.extend(json.loads(f.read_text())["rows"])
    with open(out / "results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    flat = [
        {k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows
    ]
    df = pd.DataFrame(flat)
    df.to_csv(out / "per_network.csv", index=False)
    if len(df) and "mse_total_unbiased" in df:
        ok = df[df["status"] != "failed"]
        agg = (
            ok.groupby([
                "kprop_variant", "estimator_name", "estimator_projection",
                "k_max", "kprop_kind", "width",
            ])
            .agg(
                mse_total=("mse_total_unbiased", "mean"),
                mse_total_sem=("mse_total_unbiased", "sem"),
                mse_per_coordinate=("mse_per_coordinate_unbiased", "mean"),
                n_seeds=("network_seed", "nunique"),
                flops_total=("flops_total", "mean"),
                flops_kprop=("flops_kprop", "mean"),
                flops_endpoint_total=("flops_endpoint_total", "mean"),
                simplex_residual=("simplex_residual", "max"),
                num_negative=("num_negative_coordinates", "mean"),
                quad_linf=("quadrature_argmax_linf_error", "max"),
            )
            .reset_index()
        )
        agg.to_csv(out / "aggregate.csv", index=False)
    manifest = {
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
        "git_commit": git_commit(),
        "upstream_commit": upstream_commit(),
        "tie_policy": TIE_POLICY,
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "n_rows": len(rows),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_json(out / "manifest.json", manifest)
    logger.info(f"Merged {len(rows)} rows into {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(cfg: ArgmaxExperimentCfg) -> Path:
    out = results_dir(cfg.run_name)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    rank, size = mpi_rank_size()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | r{rank} | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out / f"run_rank{rank}.log"),
        ],
        force=True,
    )
    torch.set_grad_enabled(False)  # argmax module re-enables autograd locally
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    logger.info(f"rank {rank}/{size} device={device} out={out}")
    logger.info(f"config: {json.dumps(asdict(cfg), default=str)}")

    pairs = [(w, s) for w in sorted(cfg.widths, reverse=True) for s in cfg.network_seeds]
    my_pairs = [p for i, p in enumerate(pairs) if i % size == rank]

    for width, seed in my_pairs:
        t0 = time.time()
        ev = eval_task(cfg, width, seed, device, out)
        logger.info(
            f"[w{width} s{seed}] eval blocks={ev['num_blocks']}x{ev['samples_per_block']} "
            f"collision={ev['collision_probability_estimate']:.4f}"
        )
        for variant in cfg.kprop_variants:
            rows = variant_task(cfg, width, seed, variant, ev, device, out)
            failed = [r for r in rows if r.get("status") == "failed"]
            if failed:
                logger.error(f"[w{width} s{seed} {variant['name']}] FAILED: {failed[0]['warning']}")
            else:
                best = rows[-2]  # last raw row's projected sibling is index -1
                logger.info(
                    f"[w{width} s{seed} {variant['name']}] {best['estimator_name']} "
                    f"mse={best['mse_total_unbiased']:.3e} (se={best['mse_total_input_se']:.1e}) "
                    f"q_sum-1={best['simplex_residual']:.1e} neg={best['num_negative_coordinates']} "
                    f"flops={best['flops_total']:.3g}"
                )
        if width <= cfg.check_dense_vs_factorized_max_n:
            chk = dense_check_task(cfg, width, seed, device, out)
            logger.info(
                f"[w{width} s{seed}] dense-vs-factorized q linf {chk['max_q_linf_diff']:.2e} "
                f"passed={chk['passed']}"
            )
        if cfg.mc_validation.enabled and width in tuple(cfg.mc_validation.widths):
            try:
                v = mc_validation_task(cfg, width, seed, device, out)
                logger.info(f"[w{width} s{seed}] MC-formula validation ratio {v['ratio']:.3f}")
            except Exception:
                logger.exception(f"[w{width} s{seed}] MC validation failed (continuing)")
        logger.info(f"[w{width} s{seed}] done in {time.time() - t0:.1f}s")

    if rank == 0:
        merge_results(cfg, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    args, _unknown = parser.parse_known_args()
    overrides = {}
    if args.run_name:
        overrides["run_name"] = args.run_name
    cfg = ArgmaxExperimentCfg.from_json(args.config, **overrides)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
