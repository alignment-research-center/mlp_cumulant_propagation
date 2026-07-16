"""
Direct dense Cumulant/Edgeworth estimator experiment for

    T(theta) = E_{X ~ N(0,I_n)}[ max_i M_theta(X)_i ].

Correctness-first counterpart of scripts/max_endpoint_experiment.py: the same
kprop towers and cross-fidelity ground truth, but the endpoint corrections are
evaluated by materializing dense integrated derivative tensors D2/D3/D4
(equality-pattern assembly) and contracting them with dense cumulant tensors
via plain torch.einsum (src/mlp_kprop/max_endpoint/direct_dense.py). No
treewidth / diagram machinery is on the estimator path; at small widths every
correction is cross-checked against the existing treewidth pipeline.

Method mapping (propagation order matches Edgeworth truncation):
    k1_simple  -> E0_product_gaussian
    k2_simple  -> E1_cov1, E2_cov
    k3_simple  -> E2_k3, E2_k3_k4trace_simple  (kappa4 double-trace sector)
    k3_augment -> E2_k3_k4trace_augment        (kappa4 Sym(C x M) sector)

Extended widths (E0/E1 only) run k1/k2 with dense_orders=(2,): no order-3/4
dense tensor is ever built there.

Reuses from max_endpoint_experiment.py: config dataclasses for references and
MC validation, seed derivation, atomic checkpointing, the cross-fidelity
reference task, the MC-validation task, and MPI-style sharding. Resumable at
(width, network_seed, kprop_variant) granularity; rows carry per-estimator
fields at (width, network_seed, estimator_name) granularity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from max_endpoint_experiment import (  # noqa: E402
    MCValidationCfg,
    ReferenceCfg,
    atomic_write_json,
    build_mlp,
    derive_seed,
    git_commit,
    mc_validation_task,
    mpi_rank_size,
    reference_task,
    upstream_commit,
)

from mlp_kprop.kprop_harmonic import Kind, mlp_kprop  # noqa: E402
from mlp_kprop.max_endpoint.direct_dense import (  # noqa: E402
    DenseMemoryError,
    DirectDenseCfg,
    direct_dense_estimate,
)
from mlp_kprop.max_endpoint.estimator import max_endpoint_estimate  # noqa: E402
from mlp_kprop.max_endpoint.flop_accounting import count_kprop_flops  # noqa: E402
from mlp_kprop.max_endpoint.quadrature import QuadratureCfg  # noqa: E402

logger = logging.getLogger("direct_edgeworth_max")

# Which estimators each kprop variant "owns" (emits rows for).
VARIANT_ESTIMATORS = {
    "k1_simple": {"E0_product_gaussian": "E0_product_gaussian"},
    "k2_simple": {"E1_cov1": "E1_cov1", "E2_cov": "E2_cov"},
    "k3_simple": {"E2_k3": "E2_k3", "E2_k3_k4trace": "E2_k3_k4trace_simple"},
    "k3_augment": {"E2_k3_k4trace": "E2_k3_k4trace_augment"},
}

KPROP_VARIANTS = {
    "k1_simple": {"k_max": 1, "kind": "SIMPLE", "factor": False},
    "k2_simple": {"k_max": 2, "kind": "SIMPLE", "factor": False},
    "k3_simple": {"k_max": 3, "kind": "SIMPLE", "factor": True},
    "k3_augment": {"k_max": 3, "kind": "AUGMENT", "factor": True},
}

# Endpoint FLOP attribution: incremental parts needed by each estimator
# (cumulative up the nested ladder, within one variant's endpoint evaluation).
ESTIMATOR_FLOP_PARTS = {
    "E0_product_gaussian": ("workspace", "quadrature"),
    "E1_cov1": ("workspace", "quadrature", "D2_patterns", "C2_contraction"),
    "E2_cov": (
        "workspace", "quadrature", "D2_patterns", "C2_contraction",
        "D4_patterns", "C2sq_contraction",
    ),
    "E2_k3": (
        "workspace", "quadrature", "D2_patterns", "C2_contraction",
        "D4_patterns", "C2sq_contraction", "densify_k3", "D3_patterns",
        "C3_contraction",
    ),
    "E2_k3_k4trace": (
        "workspace", "quadrature", "D2_patterns", "C2_contraction",
        "D4_patterns", "C2sq_contraction", "densify_k3", "D3_patterns",
        "C3_contraction", "C4_contraction",
    ),
}


@dataclass
class DirectCfg:
    run_name: str = "direct_edgeworth_max"
    widths: tuple[int, ...] = (8, 12, 16)
    extended_widths: tuple[int, ...] = ()   # E0/E1 only (dense_orders=(2,))
    network_seeds: tuple[int, ...] = (0, 1)
    num_layers: int = 2                     # linear layers ("depth")
    nonlin: str = "relu"
    base_seed: int = 0
    variants: tuple[str, ...] = ("k1_simple", "k2_simple", "k3_simple", "k3_augment")
    use_avg_metric: bool = True
    quadrature_nodes: int = 128
    quadrature_convergence_factor: int = 2
    node_chunk: int = 16
    max_dense_bytes: int = 40 * 10**9
    check_vs_treewidth_max_n: int = 32
    reference: ReferenceCfg = field(default_factory=ReferenceCfg)
    mc_validation: MCValidationCfg = field(default_factory=MCValidationCfg)
    dtype: str = "float32"                  # kprop dtype; endpoint is float64

    @staticmethod
    def from_json(path: str | Path, **overrides) -> "DirectCfg":
        raw = json.loads(Path(path).read_text())
        raw.update(overrides)
        ref = ReferenceCfg(**raw.pop("reference", {}))
        mcv = MCValidationCfg(**raw.pop("mc_validation", {}))
        for key in ("widths", "extended_widths", "network_seeds", "variants"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return DirectCfg(reference=ref, mc_validation=mcv, **raw)


def results_dir(run_name: str) -> Path:
    base = os.environ.get("RESULTS_DIR")
    if base:
        return Path(base)
    return REPO_ROOT / "data" / "direct_edgeworth_max" / run_name


# ---------------------------------------------------------------------------
# Per-task work
# ---------------------------------------------------------------------------


def direct_task(
    cfg: DirectCfg,
    width: int,
    net_seed: int,
    vname: str,
    ref: dict,
    device: torch.device,
    out: Path,
    extended: bool,
) -> list[dict]:
    """One kprop variant + direct-dense endpoint; emits one row per owned
    estimator. Checkpointed per (width, seed, variant)."""
    variant = KPROP_VARIANTS[vname]
    path = out / "tasks" / f"direct_w{width}_s{net_seed}_{vname}.json"
    if path.exists():
        return json.loads(path.read_text())["rows"]

    dtype = getattr(torch, cfg.dtype)
    torch.set_default_dtype(dtype)
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    mlp = build_mlp(cfg, width, net_seed, device).to(dtype)
    kind = Kind[variant["kind"]]
    k_in = {
        1: torch.zeros(width, device=device, dtype=dtype),
        2: torch.eye(width, device=device, dtype=dtype),
    }
    dense_orders = (2,) if extended else (2, 3, 4)
    quad = QuadratureCfg(
        num_nodes=cfg.quadrature_nodes,
        convergence_factor=cfg.quadrature_convergence_factor,
    )
    ddcfg = DirectDenseCfg(
        quad=quad,
        max_dense_bytes=cfg.max_dense_bytes,
        node_chunk=cfg.node_chunk,
        dense_orders=dense_orders,
    )

    base_row = {
        "run_name": cfg.run_name,
        "git_commit": git_commit(),
        "upstream_commit": upstream_commit(),
        "width": width,
        "input_dim": width,
        "hidden_dim": width,
        "output_dim": width,
        "num_layers": cfg.num_layers,
        "hidden_layers": cfg.num_layers - 1,
        "network_seed": net_seed,
        "kprop_variant": vname,
        "kprop_kmax": variant["k_max"],
        "kprop_kind": variant["kind"],
        "factorized": variant["factor"],
        "extended_width_mode": extended,
        "quadrature_nodes": cfg.quadrature_nodes,
    }

    t0 = time.time()
    try:
        with count_kprop_flops() as krec:
            K = mlp_kprop(
                mlp, k_in, k_max=variant["k_max"], kind=kind,
                factor=variant["factor"], use_avg_metric=cfg.use_avg_metric,
            )
        wall_kprop = time.time() - t0
        res = direct_dense_estimate(K, ddcfg, device=device)
    except DenseMemoryError as e:
        rows = [dict(base_row, estimator_name="ALL", status="dense_refused", warning=str(e))]
        atomic_write_json(path, {"rows": rows})
        return rows
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        rows = [dict(base_row, estimator_name="ALL", status="oom", warning=f"CUDA OOM: {e}")]
        atomic_write_json(path, {"rows": rows})
        return rows
    except Exception as e:
        logger.exception(f"Task w{width} s{net_seed} {vname} failed")
        rows = [dict(base_row, estimator_name="ALL", status="failed",
                     warning=f"{type(e).__name__}: {e}")]
        atomic_write_json(path, {"rows": rows})
        return rows

    peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    parts = res.flops_by_part
    corr = res.corrections
    rows = []
    for est_key, method_name in VARIANT_ESTIMATORS[vname].items():
        if est_key not in res.estimates:
            continue
        value = res.estimates[est_key]
        err_a = value - ref["ref_a"]["mean"]
        err_b = value - ref["ref_b"]["mean"]
        flops_endpoint_est = sum(parts.get(p, 0) for p in ESTIMATOR_FLOP_PARTS[est_key])
        rows.append(dict(
            base_row,
            estimator_name=method_name,
            nested_level=est_key,
            estimate=value,
            base_Psi=res.psi,
            correction_C2=corr.get("C2"),
            correction_C2_squared=corr.get("C2sq_half"),
            correction_C3=corr.get("C3"),
            correction_C4_trace=corr.get("C4_trace"),
            k3_repr=res.k3_repr,
            k4_sector=res.k4_sector,
            num_clamped_var=res.num_clamped_var,
            equivalences=res.equivalences,
            reference_A=ref["ref_a"]["mean"],
            reference_B=ref["ref_b"]["mean"],
            reference_mean=ref["ref_mean"],
            reference_se_A=ref["ref_a"]["se"],
            reference_se_B=ref["ref_b"]["se"],
            reference_samples_A=ref["ref_a"]["num_samples"],
            reference_samples_B=ref["ref_b"]["num_samples"],
            squared_error_cross=err_a * err_b,
            signed_error_against_reference_mean=value - ref["ref_mean"],
            quadrature_convergence_error=res.quadrature_error.get(est_key),
            largest_dense_tensor_order=res.largest_dense_tensor_order,
            largest_dense_tensor_shape=list(res.largest_dense_tensor_shape),
            estimated_dense_bytes=res.estimated_dense_bytes,
            flops_kprop=krec.total,
            flops_kprop_raw=krec.raw_total,
            flops_endpoint=flops_endpoint_est,
            flops_endpoint_variant_total=res.flops_endpoint,
            flops_total=krec.total + flops_endpoint_est,
            flops_quadrature_weights=parts.get("workspace", 0) + parts.get("quadrature", 0),
            flops_D2=parts.get("D2_patterns", 0),
            flops_D3=parts.get("D3_patterns", 0),
            flops_D4=parts.get("D4_patterns", 0),
            flops_C2_contraction=parts.get("C2_contraction", 0),
            flops_C2_squared_contraction=parts.get("C2sq_contraction", 0),
            flops_C3_contraction=parts.get("C3_contraction", 0),
            flops_C4_trace_contraction=parts.get("C4_contraction", 0),
            flops_densify_k3=parts.get("densify_k3", 0),
            wall_seconds_kprop=wall_kprop,
            wall_seconds_endpoint=res.wall_seconds,
            peak_gpu_memory_bytes=peak_mem,
            ground_truth_backend=ref["backend"],
            mc_var_gaussian=ref["mc_var_gaussian"],
            mc_flops_per_sample_gaussian=ref["mc_flops_per_sample_gaussian"],
            mc_var_spherical=ref["mc_var_backend"] if ref["backend"] == "spherical" else None,
            mc_flops_per_sample_spherical=ref["mc_flops_per_sample_spherical"],
            status=";".join(sorted(set(res.status))) or "ok",
            warning="",
        ))
    if not rows:
        # Every estimator owned by this variant was refused (e.g. D4 over the
        # dense-memory guard). Record the precise stopping reason.
        rows = [dict(
            base_row,
            estimator_name="NONE_AVAILABLE",
            status="dense_refused",
            warning=" | ".join(res.info.get("dense_refused", {}).values()) or
                    ";".join(sorted(set(res.status))),
            flops_kprop=krec.total,
            wall_seconds_kprop=wall_kprop,
            peak_gpu_memory_bytes=peak_mem,
        )]
    atomic_write_json(path, {"rows": rows})
    return rows


def treewidth_check_task(
    cfg: DirectCfg, width: int, net_seed: int, device: torch.device, out: Path
) -> dict:
    """Cross-validate the direct dense corrections against the existing
    diagram/treewidth pipeline on the k3_simple tower."""
    path = out / "tasks" / f"twcheck_w{width}_s{net_seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    dtype = getattr(torch, cfg.dtype)
    torch.set_default_dtype(dtype)
    mlp = build_mlp(cfg, width, net_seed, device).to(dtype)
    k_in = {
        1: torch.zeros(width, device=device, dtype=dtype),
        2: torch.eye(width, device=device, dtype=dtype),
    }
    quad = QuadratureCfg(num_nodes=cfg.quadrature_nodes)
    try:
        K = mlp_kprop(mlp, k_in, k_max=3, kind=Kind.SIMPLE, factor=True,
                      use_avg_metric=cfg.use_avg_metric)
        direct = direct_dense_estimate(
            K, DirectDenseCfg(quad=quad, max_dense_bytes=cfg.max_dense_bytes,
                              node_chunk=cfg.node_chunk), device=device)
        tw = max_endpoint_estimate(K, quad_cfg=quad, device=device)
    except Exception as e:
        payload = {"width": width, "net_seed": net_seed, "max_rel_diff": None,
                   "passed": None, "status": f"{type(e).__name__}: {e}"}
        atomic_write_json(path, payload)
        return payload
    if direct.num_clamped_var > 0 or max(direct.quadrature_error.values()) > 1e-3:
        # Divergent / clamped-variance regime of the truncated cumulant
        # expansion (documented upstream): both paths integrate huge signed
        # terms with 1/sigma^k weights and the comparison is dominated by
        # catastrophic cancellation, not by implementation differences.
        payload = {
            "width": width, "net_seed": net_seed, "max_rel_diff": None,
            "passed": None,
            "status": (
                f"undefined_in_divergent_regime (clamped={direct.num_clamped_var}, "
                f"max_quad_err={max(direct.quadrature_error.values()):.3g})"
            ),
        }
        atomic_write_json(path, payload)
        return payload
    max_rel = 0.0
    for name in ("C2", "C2sq_half", "C3", "C4_trace"):
        a, b = direct.corrections[name], tw.corrections[name]
        max_rel = max(max_rel, abs(a - b) / max(1.0, abs(a)))
    payload = {
        "width": width, "net_seed": net_seed, "max_rel_diff": max_rel,
        "passed": bool(max_rel < 1e-6),
        "corrections_direct": direct.corrections,
        "corrections_treewidth": tw.corrections,
    }
    atomic_write_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_results(cfg: DirectCfg, out: Path) -> None:
    import pandas as pd

    rows = []
    for f in sorted((out / "tasks").glob("direct_*.json")):
        rows.extend(json.loads(f.read_text())["rows"])
    with open(out / "results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    df = pd.DataFrame(flat)
    df.to_csv(out / "per_network.csv", index=False)
    if len(df) and "squared_error_cross" in df:
        ok = df[~df["status"].isin(["failed", "dense_refused", "oom"])]
        agg = (
            ok.groupby(["estimator_name", "kprop_variant", "width"])
            .agg(
                mse_cross=("squared_error_cross", "mean"),
                mse_cross_sem=("squared_error_cross", "sem"),
                n_seeds=("network_seed", "nunique"),
                flops_total=("flops_total", "mean"),
                flops_kprop=("flops_kprop", "mean"),
                flops_endpoint=("flops_endpoint", "mean"),
                ref_se=("reference_se_A", "mean"),
                quad_err=("quadrature_convergence_error", "max"),
                peak_gpu_mem=("peak_gpu_memory_bytes", "max"),
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


def run_experiment(cfg: DirectCfg) -> Path:
    out = results_dir(cfg.run_name)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    rank, size = mpi_rank_size()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | r{rank} | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"run_rank{rank}.log")],
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

    # (width, seed, extended?) work items; extended widths run E0/E1 only.
    pairs = [(w, s, False) for w in sorted(cfg.widths, reverse=True) for s in cfg.network_seeds]
    pairs += [(w, s, True) for w in sorted(cfg.extended_widths, reverse=True)
              for s in cfg.network_seeds]
    my_pairs = [p for i, p in enumerate(pairs) if i % size == rank]

    try:
        for width, seed, extended in my_pairs:
            t0 = time.time()
            ref = reference_task(cfg, width, seed, device, out)
            logger.info(
                f"[w{width} s{seed}{' ext' if extended else ''}] ref={ref['ref_mean']:.6f} "
                f"(A-B={ref['ref_diff']:.2e}, se={ref['ref_a']['se']:.2e}, "
                f"samples={ref['ref_a']['num_samples']})"
            )
            variants = ("k1_simple", "k2_simple") if extended else cfg.variants
            for vname in variants:
                rows = direct_task(cfg, width, seed, vname, ref, device, out, extended)
                bad = [r for r in rows if r.get("status") in ("failed", "dense_refused", "oom")]
                if bad:
                    logger.error(f"[w{width} s{seed} {vname}] {bad[0]['status']}: {bad[0]['warning']}")
                elif rows:
                    best = rows[-1]
                    logger.info(
                        f"[w{width} s{seed} {vname}] {best['estimator_name']}="
                        f"{best['estimate']:.6f} err={best['signed_error_against_reference_mean']:.2e} "
                        f"flops={best['flops_total']:.3g} peak_mem={best['peak_gpu_memory_bytes']/1e9:.2f}GB"
                    )
            if not extended and width <= cfg.check_vs_treewidth_max_n:
                chk = treewidth_check_task(cfg, width, seed, device, out)
                if chk.get("max_rel_diff") is None:
                    logger.warning(f"[w{width} s{seed}] treewidth check skipped: {chk.get('status')}")
                else:
                    logger.info(
                        f"[w{width} s{seed}] direct-vs-treewidth rel diff "
                        f"{chk['max_rel_diff']:.2e} passed={chk['passed']}"
                    )
                    if not chk["passed"]:
                        logger.error(f"[w{width} s{seed}] TREEWIDTH CROSS-CHECK FAILED")
            if cfg.mc_validation.enabled and width in tuple(cfg.mc_validation.widths):
                try:
                    v = mc_validation_task(cfg, width, seed, device, out)
                    logger.info(f"[w{width} s{seed}] MC validation ratio {v['ratio']:.3f}")
                except Exception:
                    logger.exception(f"[w{width} s{seed}] MC validation failed")
            logger.info(f"[w{width} s{seed}] done in {time.time() - t0:.1f}s")
    finally:
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
    cfg = DirectCfg.from_json(args.config, **overrides)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
