"""
Main driver for the max-endpoint experiment.

For each (width n, network seed):
  1. Build a fixed bias-free He-initialized ReLU MLP (input/hidden/output = n).
  2. Compute two independent Monte Carlo references T_ref_A/T_ref_B for
     T(theta) = E_X[max_i M_theta(X)_i] (cross-fidelity: their product of
     signed errors is an unbiased estimate of the deterministic squared error).
  3. For each kprop variant (k_max, kind, factor), run ARC cumulant
     propagation under a FLOP counter and evaluate all nested endpoint
     estimators E0 .. E2_full.
  4. Emit one JSON row per estimator into RESULTS_DIR/tasks/ (atomic file per
     completed (width, seed, variant) task; the sweep is resumable).

Finally rank 0 merges rows into results.jsonl / per_network.csv / aggregate.csv
and writes manifest.json.

Sharding: if OMPI_COMM_WORLD_SIZE > 1, (width, seed) pairs are sharded across
ranks and each rank pins cuda:(rank % device_count).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlp_kprop.kprop_harmonic import Kind, mlp_kprop  # noqa: E402
from mlp_kprop.max_endpoint.estimator import max_endpoint_estimate  # noqa: E402
from mlp_kprop.max_endpoint.flop_accounting import count_kprop_flops  # noqa: E402
from mlp_kprop.max_endpoint.ground_truth import (  # noqa: E402
    mc_flops_per_sample,
    reference_estimate,
)
from mlp_kprop.max_endpoint.quadrature import QuadratureCfg  # noqa: E402
from mlp_kprop.mlp import MLP  # noqa: E402

logger = logging.getLogger("max_endpoint_experiment")

UPSTREAM_COMMIT_FILE = REPO_ROOT / "UPSTREAM_COMMIT"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ReferenceCfg:
    backend: str = "spherical"           # "spherical" | "gaussian"
    validate_backend: bool = True        # cross-check spherical vs gaussian at min width
    target_se: float = 1e-4
    min_samples: int = 1_000_000
    max_samples: int = 100_000_000
    batch_size: int = 262_144
    gaussian_var_samples: int = 2_000_000  # extra stream for Var_X(max), MC baseline


@dataclass
class MCValidationCfg:
    enabled: bool = False
    widths: tuple[int, ...] = (16,)
    repeats: int = 32
    samples: int = 8192


@dataclass
class ExperimentCfg:
    run_name: str = "max_endpoint"
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
    reference: ReferenceCfg = field(default_factory=ReferenceCfg)
    mc_validation: MCValidationCfg = field(default_factory=MCValidationCfg)
    check_dense_vs_factorized_max_n: int = 16  # widths <= this get the dense cross-check
    dtype: str = "float32"

    @staticmethod
    def from_json(path: str | Path, **overrides) -> "ExperimentCfg":
        raw = json.loads(Path(path).read_text())
        raw.update(overrides)
        ref = ReferenceCfg(**raw.pop("reference", {}))
        mcv = MCValidationCfg(**raw.pop("mc_validation", {}))
        raw["widths"] = tuple(raw.get("widths", (16, 32)))
        raw["network_seeds"] = tuple(raw.get("network_seeds", (0, 1)))
        raw["kprop_variants"] = tuple(raw["kprop_variants"]) if "kprop_variants" in raw else ExperimentCfg().kprop_variants
        return ExperimentCfg(reference=ref, mc_validation=mcv, **raw)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def derive_seed(base_seed: int, width: int, net_seed: int, purpose: str) -> int:
    """Deterministic, well-separated seed derivation."""
    h = hashlib.sha256(f"{base_seed}:{width}:{net_seed}:{purpose}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63 - 1)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    # Remote copies are rsynced without .git; a GIT_COMMIT file is shipped instead.
    marker = REPO_ROOT / "GIT_COMMIT"
    if marker.exists():
        return marker.read_text().strip()
    return "unknown"


def upstream_commit() -> str:
    if UPSTREAM_COMMIT_FILE.exists():
        return UPSTREAM_COMMIT_FILE.read_text().strip()
    return "unknown"


def results_dir(run_name: str) -> Path:
    base = os.environ.get("RESULTS_DIR")
    if base:
        return Path(base)
    return REPO_ROOT / "data" / "max_endpoint" / run_name


def mpi_rank_size() -> tuple[int, int]:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", 0)))
    size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
    return rank, size


def atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def build_mlp(cfg: ExperimentCfg, width: int, net_seed: int, device: torch.device) -> MLP:
    torch.manual_seed(derive_seed(cfg.base_seed, width, net_seed, "net"))
    mlp = MLP(
        input_dim=width,
        hidden_dim=width,
        output_dim=width,
        num_layers=cfg.num_layers,
        nonlin=cfg.nonlin,
        init_kind="he",
        # Primary experiment is bias-free (positive homogeneity for the
        # Rao-Blackwellized reference): b_mean/b_var omitted => no bias params.
    ).to(device)
    assert not mlp.has_bias(), "Primary experiment requires a bias-free network"
    return mlp


# ---------------------------------------------------------------------------
# Per-task work
# ---------------------------------------------------------------------------

def reference_task(cfg: ExperimentCfg, width: int, net_seed: int, device: torch.device, out: Path) -> dict:
    """Two independent references + gaussian per-sample variance for MC baselines."""
    path = out / "tasks" / f"ref_w{width}_s{net_seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    mlp = build_mlp(cfg, width, net_seed, device)
    r = cfg.reference
    backend = r.backend
    if backend == "spherical" and mlp.has_bias():
        backend = "gaussian"
    ref_a = reference_estimate(
        mlp, seed=derive_seed(cfg.base_seed, width, net_seed, "refA"), backend=backend,
        target_se=r.target_se, min_samples=r.min_samples, max_samples=r.max_samples,
        batch_size=r.batch_size, device=device,
    )
    ref_b = reference_estimate(
        mlp, seed=derive_seed(cfg.base_seed, width, net_seed, "refB"), backend=backend,
        target_se=r.target_se, min_samples=r.min_samples, max_samples=r.max_samples,
        batch_size=r.batch_size, device=device,
    )
    # Gaussian-input per-sample variance (for the standard MC baseline curve),
    # and a gaussian mean cross-check of the spherical backend where requested.
    gv = reference_estimate(
        mlp, seed=derive_seed(cfg.base_seed, width, net_seed, "gauss_var"), backend="gaussian",
        target_se=0.0, min_samples=r.gaussian_var_samples, max_samples=r.gaussian_var_samples,
        batch_size=r.batch_size, device=device,
    )
    payload = {
        "width": width,
        "net_seed": net_seed,
        "backend": backend,
        "ref_a": asdict(ref_a),
        "ref_b": asdict(ref_b),
        "ref_mean": 0.5 * (ref_a.mean + ref_b.mean),
        "ref_diff": ref_a.mean - ref_b.mean,
        "gaussian_check": asdict(gv),
        "mc_var_gaussian": gv.var,
        "mc_var_backend": 0.5 * (ref_a.var + ref_b.var),
        "mc_flops_per_sample_gaussian": mc_flops_per_sample(mlp, "gaussian"),
        "mc_flops_per_sample_spherical": mc_flops_per_sample(mlp, "spherical"),
        "wall_seconds": time.time() - t0,
    }
    if cfg.reference.validate_backend and backend == "spherical":
        # Consistency of the Rao-Blackwellized estimator vs plain Gaussian MC.
        z = abs(payload["ref_mean"] - gv.mean) / math.sqrt(
            0.25 * (ref_a.se**2 + ref_b.se**2) + gv.se**2
        )
        payload["backend_consistency_z"] = z
        if z > 5:
            payload["warning"] = f"spherical/gaussian reference mismatch z={z:.1f}"
    atomic_write_json(path, payload)
    return payload


def kprop_task(
    cfg: ExperimentCfg, width: int, net_seed: int, variant: dict, ref: dict,
    device: torch.device, out: Path,
) -> list[dict]:
    """Run one kprop variant + all nested endpoint estimators; emit rows."""
    vname = variant["name"]
    path = out / "tasks" / f"est_w{width}_s{net_seed}_{vname}.json"
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
    status: list[str] = []
    warning = ""
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
        res = max_endpoint_estimate(K, quad_cfg=quad, device=device)
        status += res.status
    except Exception as e:  # record, don't crash the sweep
        logger.exception(f"Task w{width} s{net_seed} {vname} failed")
        rows = [{
            "run_name": cfg.run_name, "width": width, "network_seed": net_seed,
            "kprop_variant": vname, "estimator_name": "ALL", "status": "failed",
            "warning": f"{type(e).__name__}: {e}",
        }]
        atomic_write_json(path, {"rows": rows})
        return rows

    peak_mem = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    rows = []
    for est_name, value in res.estimates.items():
        err_a = value - ref["ref_a"]["mean"]
        err_b = value - ref["ref_b"]["mean"]
        rows.append({
            "run_name": cfg.run_name,
            "git_commit": git_commit(),
            "upstream_commit": upstream_commit(),
            "width": width,
            "input_dim": width,
            "hidden_dim": width,
            "output_dim": width,
            "num_layers": cfg.num_layers,
            "network_seed": net_seed,
            "net_torch_seed": derive_seed(cfg.base_seed, width, net_seed, "net"),
            "kprop_variant": vname,
            "estimator_name": est_name,
            "k_max": variant["k_max"],
            "kprop_kind": variant["kind"],
            "factorized": variant["factor"],
            "estimate": value,
            "psi": res.psi,
            "corrections": res.corrections,
            "equivalences": res.equivalences,
            "k3_repr": res.k3_repr,
            "k4_sector": res.k4_sector,
            "num_clamped_var": res.num_clamped_var,
            "ref_a": ref["ref_a"]["mean"],
            "ref_b": ref["ref_b"]["mean"],
            "ref_mean": ref["ref_mean"],
            "ref_se_a": ref["ref_a"]["se"],
            "ref_se_b": ref["ref_b"]["se"],
            "ref_samples_a": ref["ref_a"]["num_samples"],
            "ref_samples_b": ref["ref_b"]["num_samples"],
            "ref_stopping_a": ref["ref_a"]["stopping_reason"],
            "error_cross": err_a * err_b,
            "signed_error_against_ref_mean": value - ref["ref_mean"],
            "flops_kprop": krec.total,
            "flops_kprop_raw": krec.raw_total,
            "flops_endpoint": res.flops_endpoint,
            "flops_total": krec.total + res.flops_endpoint,
            "wall_seconds_kprop": wall_kprop,
            "wall_seconds_endpoint": res.wall_seconds,
            "peak_gpu_memory_bytes": peak_mem,
            "quadrature_nodes": cfg.quadrature_nodes,
            "quadrature_error_estimate": res.quadrature_error.get(est_name),
            "max_treewidth": res.max_treewidth,
            "treewidth_exact": res.treewidth_exact,
            "num_diagrams": res.num_diagrams,
            "max_table_numel": res.max_table_numel,
            "ground_truth_backend": ref["backend"],
            "mc_var_gaussian": ref["mc_var_gaussian"],
            "mc_flops_per_sample_gaussian": ref["mc_flops_per_sample_gaussian"],
            "mc_var_spherical": ref["mc_var_backend"] if ref["backend"] == "spherical" else None,
            "mc_flops_per_sample_spherical": ref["mc_flops_per_sample_spherical"],
            "status": ";".join(sorted(set(status))) or "ok",
            "warning": warning,
        })
    atomic_write_json(path, {"rows": rows})
    return rows


def dense_check_task(cfg: ExperimentCfg, width: int, net_seed: int, device: torch.device, out: Path) -> dict:
    """Smoke check: factorized contraction vs dense .to_tensor() contraction."""
    from mlp_kprop.harmonic import HTensor

    path = out / "tasks" / f"densecheck_w{width}_s{net_seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    mlp = build_mlp(cfg, width, net_seed, device)
    k_in = {1: torch.zeros(width, device=device), 2: torch.eye(width, device=device)}
    K = mlp_kprop(mlp, k_in, k_max=3, kind=Kind.AUGMENT, factor=True, use_avg_metric=cfg.use_avg_metric)
    res_fac = max_endpoint_estimate(K, device=device)
    K_dense = dict(K)
    K_dense[3] = HTensor(K[3].to_tensor().to(torch.float64), r=0)
    K_dense[4] = HTensor(K[4].to_tensor().to(torch.float64), r=0)
    res_dense = max_endpoint_estimate(
        K_dense, device=device, dense_max_n=cfg.check_dense_vs_factorized_max_n
    )
    max_rel = 0.0
    for name in res_fac.corrections:
        a, b = res_fac.corrections[name], res_dense.corrections[name]
        max_rel = max(max_rel, abs(a - b) / max(1.0, abs(a)))
    payload = {
        "width": width, "net_seed": net_seed, "max_rel_diff": max_rel,
        "passed": bool(max_rel < 2e-5),
        "corrections_factored": res_fac.corrections,
        "corrections_dense": res_dense.corrections,
    }
    atomic_write_json(path, payload)
    return payload


def mc_validation_task(cfg: ExperimentCfg, width: int, net_seed: int, device: torch.device, out: Path) -> dict:
    """Validate MSE(m) = Var_X(max)/m with explicit repeated MC runs (spec G)."""
    path = out / "tasks" / f"mcval_w{width}_s{net_seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    mlp = build_mlp(cfg, width, net_seed, device)
    m = cfg.mc_validation.samples
    reps = cfg.mc_validation.repeats
    ref = reference_estimate(
        mlp, seed=derive_seed(cfg.base_seed, width, net_seed, "mcval_ref"),
        backend="gaussian", target_se=0.0,
        min_samples=4_000_000, max_samples=4_000_000,
        batch_size=cfg.reference.batch_size, device=device,
    )
    gen = torch.Generator(device=device)
    gen.manual_seed(derive_seed(cfg.base_seed, width, net_seed, "mcval"))
    errs = []
    for _ in range(reps):
        x = torch.randn(m, width, device=device, generator=gen)
        est = mlp(x).out.max(dim=1).values.double().mean().item()
        errs.append((est - ref.mean) ** 2)
    empirical_mse = sum(errs) / reps
    predicted = ref.var / m
    payload = {
        "width": width, "net_seed": net_seed, "samples": m, "repeats": reps,
        "empirical_mse": empirical_mse, "predicted_var_over_m": predicted,
        "ratio": empirical_mse / predicted,
        "note": "ratio ~ 1 +- O(sqrt(2/repeats)) validates MSE=Var/m; ref noise adds ref.se^2",
        "ref_se2": ref.se**2,
    }
    atomic_write_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_results(cfg: ExperimentCfg, out: Path) -> None:
    import pandas as pd

    rows = []
    for f in sorted((out / "tasks").glob("est_*.json")):
        rows.extend(json.loads(f.read_text())["rows"])
    with open(out / "results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    flat = [
        {k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows
    ]
    df = pd.DataFrame(flat)
    df.to_csv(out / "per_network.csv", index=False)
    if len(df) and "error_cross" in df:
        ok = df[df["status"] != "failed"]
        agg = (
            ok.groupby(["kprop_variant", "estimator_name", "k_max", "kprop_kind", "width"])
            .agg(
                mse_cross=("error_cross", "mean"),
                mse_cross_sem=("error_cross", "sem"),
                n_seeds=("network_seed", "nunique"),
                flops_total=("flops_total", "mean"),
                flops_kprop=("flops_kprop", "mean"),
                flops_endpoint=("flops_endpoint", "mean"),
                ref_se=("ref_se_a", "mean"),
                quad_err=("quadrature_error_estimate", "max"),
            )
            .reset_index()
        )
        agg.to_csv(out / "aggregate.csv", index=False)
    manifest = {
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
        "git_commit": git_commit(),
        "upstream_commit": upstream_commit(),
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

def run_experiment(cfg: ExperimentCfg) -> Path:
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
    torch.set_grad_enabled(False)
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
        ref = reference_task(cfg, width, seed, device, out)
        logger.info(
            f"[w{width} s{seed}] ref={ref['ref_mean']:.6f} "
            f"(A-B={ref['ref_diff']:.2e}, se={ref['ref_a']['se']:.2e}, "
            f"samples={ref['ref_a']['num_samples']})"
        )
        for variant in cfg.kprop_variants:
            rows = kprop_task(cfg, width, seed, variant, ref, device, out)
            for r in rows:
                if r.get("status") == "failed":
                    logger.error(f"[w{width} s{seed} {variant['name']}] FAILED: {r['warning']}")
                    break
            else:
                best = rows[-1]
                logger.info(
                    f"[w{width} s{seed} {variant['name']}] {best['estimator_name']}="
                    f"{best['estimate']:.6f} err={best['signed_error_against_ref_mean']:.2e} "
                    f"flops={best['flops_total']:.3g}"
                )
        if width <= cfg.check_dense_vs_factorized_max_n:
            chk = dense_check_task(cfg, width, seed, device, out)
            logger.info(f"[w{width} s{seed}] dense-vs-factorized rel diff {chk['max_rel_diff']:.2e} passed={chk['passed']}")
        if cfg.mc_validation.enabled and width in tuple(cfg.mc_validation.widths):
            v = mc_validation_task(cfg, width, seed, device, out)
            logger.info(f"[w{width} s{seed}] MC validation ratio {v['ratio']:.3f}")
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
    cfg = ExperimentCfg.from_json(args.config, **overrides)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
