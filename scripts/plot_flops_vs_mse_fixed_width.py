"""FLOPs-vs-MSE tradeoff at a fixed width, deterministic estimators vs Monte
Carlo sampling. Default width 128 (depth 4): the largest width at which EVERY
direct-dense estimator (E0 .. E2_k3_k4trace) ran.

x: modeled online FLOPs (log). y: MSE over network seeds (cross-fidelity,
log) with 95% bootstrap CIs. The MC tradeoff is the analytic line
MSE(m) = Var_X(max)/m at FLOPs = m * flops_per_sample (Gaussian-input and
Rao-Blackwellized spherical variants), using per-network variances measured
in the run; the analytic line was validated by explicit repeated MC runs
(mcval task files).

Usage:
  uv run python scripts/plot_flops_vs_mse_fixed_width.py [results_dir] [width]
  (defaults: data/direct_edgeworth_max/full_kunalc/depth4, 128)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

LABELS = [  # estimator -> plot label, color (cheap -> expensive)
    ("E0_product_gaussian", "E0 (product Gaussian)", "tab:gray"),
    ("E1_cov1", "E1 (+C2)", "tab:olive"),
    ("E2_cov", "E2_cov (+C2^2/2)", "tab:green"),
    ("E2_k3", "E2_k3 (+C3)", "tab:cyan"),
    ("E2_k3_k4trace_simple", "E2_k3_k4trace (SIMPLE)", "tab:blue"),
    ("E2_k3_k4trace_augment", "E2_k3_k4trace (AUGMENT)", "tab:purple"),
]


def boot_ci(vals: np.ndarray, boot: int = 2000, rng=np.random.default_rng(0)):
    m = float(vals.mean())
    if len(vals) < 2:
        return m, m, m
    idx = rng.integers(0, len(vals), size=(boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return m, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO_ROOT / "data" / "direct_edgeworth_max" / "full_kunalc" / "depth4")
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 128

    df = pd.read_csv(results_dir / "per_network.csv")
    df = df[(df["width"] == width) & (~df["status"].isin(["failed", "dense_refused", "oom"]))]
    if df.empty:
        print(f"no rows at width {width} in {results_dir}"); return
    depth = int(df["num_layers"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # MC tradeoff lines from measured per-network variances.
    ref_rows = df.drop_duplicates(subset=["network_seed"])
    for var_col, fps_col, label, color in (
        ("mc_var_gaussian", "mc_flops_per_sample_gaussian",
         "MC (Gaussian input): MSE = Var/m", "tab:red"),
        ("mc_var_spherical", "mc_flops_per_sample_spherical",
         "MC (spherical Rao-Blackwell): MSE = Var/m", "tab:orange"),
    ):
        var = float(pd.to_numeric(ref_rows[var_col], errors="coerce").mean())
        fps = float(ref_rows[fps_col].mean())
        if not np.isfinite(var) or var <= 0:
            continue
        m = np.logspace(0.5, 9.5, 60)
        ax.plot(m * fps, var / m, color=color, lw=1.5, label=label)

    # Deterministic estimator points with bootstrap CIs. The E2_* points sit
    # at nearly identical FLOPs (the D4 build dominates); stagger their labels.
    noise_floor = float((df["reference_se_A"] * df["reference_se_B"]).mean())
    offsets = {"E0_product_gaussian": (6, 4), "E1_cov1": (6, 4),
               "E2_cov": (6, 4), "E2_k3": (6, 4),
               "E2_k3_k4trace_simple": (-8, -14),
               "E2_k3_k4trace_augment": (8, -2)}
    short = {"E2_k3_k4trace_simple": "k4trace (S)", "E2_k3_k4trace_augment": "k4trace (A)"}
    for est, label, color in LABELS:
        g = df[df["estimator_name"] == est]
        if g.empty:
            continue
        mse, lo, hi = boot_ci(g["squared_error_cross"].to_numpy(float))
        f = float(g["flops_total"].mean())
        ax.errorbar([f], [mse], yerr=[[max(mse - lo, 1e-30)], [max(hi - mse, 1e-30)]],
                    marker="o", ms=7, capsize=3, lw=1.2, color=color, label=label)
        ha = "right" if offsets[est][0] < 0 else "left"
        ax.annotate(short.get(est, label.split(" ")[0]), (f, mse), fontsize=6,
                    color=color, xytext=offsets[est], textcoords="offset points", ha=ha)

    ax.axhline(noise_floor, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.annotate("reference noise floor", (ax.get_xlim()[0], noise_floor),
                fontsize=6, xytext=(6, 3), textcoords="offset points")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("online FLOPs (modeled)")
    ax.set_ylabel(rf"MSE over $\theta$ (cross-fidelity, {df['network_seed'].nunique()} seeds)")
    ax.set_title(f"n = {width}, depth {depth}: FLOPs-vs-MSE, direct dense estimators vs MC")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, loc="lower left")
    plots = results_dir / "plots"
    plots.mkdir(exist_ok=True)
    name = f"direct_edgeworth_max_flops_vs_mse_n{width}"
    for ext in ("pdf", "png"):
        fig.savefig(plots / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {plots}/{name}.pdf/.png")

    var_g = float(pd.to_numeric(ref_rows["mc_var_gaussian"], errors="coerce").mean())
    fps_g = float(ref_rows["mc_flops_per_sample_gaussian"].mean())
    print(f"\nMC (Gaussian): Var={var_g:.3g}, {fps_g:.3g} FLOPs/sample")
    for est, label, _ in LABELS:
        g = df[df["estimator_name"] == est]
        if g.empty:
            continue
        mse, lo, hi = boot_ci(g["squared_error_cross"].to_numpy(float))
        f = float(g["flops_total"].mean())
        m_matched = max(np.floor(f / fps_g), 1)
        mc_at_budget = var_g / m_matched
        mc_flops_to_match = var_g / mse * fps_g if mse > 0 else float("nan")
        print(f"{label:28s} flops {f:9.3g}  MSE {mse:9.3g} [{lo:.2g},{hi:.2g}]  "
              f"MC@same-FLOPs {mc_at_budget:9.3g}  MC-FLOPs-to-match {mc_flops_to_match:9.3g}")


if __name__ == "__main__":
    main()
