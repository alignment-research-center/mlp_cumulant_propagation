"""FLOPs-vs-MSE tradeoff at a fixed width for the treewidth-pipeline
(max-endpoint) estimators vs Monte Carlo sampling. Default width 1448
(depth 4): the largest width at which EVERY canonical method ran and
resolved above the reference noise floor (k3_augment failed at 2048).

x: modeled online FLOPs (log). y: MSE over network seeds (cross-fidelity
mean of err_cross = (T_hat - T_ref_A)(T_hat - T_ref_B), log) with 95%
bootstrap CIs. The MC tradeoff is the analytic line MSE(m) = Var_X(max)/m
at FLOPs = m * flops_per_sample (Gaussian-input and Rao-Blackwellized
spherical variants), using per-network variances measured in the run.
Methods below the noise floor are drawn as open markers pinned at the
floor, never silently clipped.

Usage:
  uv run python scripts/plot_flops_vs_mse_fixed_width_treewidth.py \
      [width] [results_dir ...]
  (defaults: 1448, data/max_endpoint/wide_d4 + data/max_endpoint/wide_d4_ext;
   plots land in the first results_dir's plots/)
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

# (kprop_variant, estimator_name) -> canonical method (same names/colors as
# plot_max_endpoint.py), plot label, color, marker. Markers differ per method
# so identity survives without color (CVD/print). Cheap -> expensive.
METHODS = [
    (("k1_simple", "E0_product_gaussian"), "product_gaussian",
     "E0 (product Gaussian)", "tab:gray", "o"),
    (("k2_simple", "E1_cov1"), "pg_plus_cov1",
     "E1 (+C2)", "tab:olive", "s"),
    (("k2_simple", "E2_cov2"), "pg_plus_cov2",
     "E2_cov2 (+C2^2/2)", "tab:green", "D"),
    (("k2_augment", "E2_full"), "pg_k2aug_trace_k3k4",
     "E2_full (k2 AUGMENT traces)", "tab:pink", "v"),
    (("k3_simple", "E2_k3"), "pg_plus_cov2_k3",
     "E2_k3 (+C3)", "tab:cyan", "^"),
    (("k3_simple", "E2_full"), "pg_cov2_k3_k4trace_simple",
     "E2_full (SIMPLE)", "tab:blue", "P"),
    (("k3_augment", "E2_full"), "pg_cov2_k3_k4trace_augment",
     "E2_full (AUGMENT)", "tab:purple", "X"),
]


def boot_ci(vals: np.ndarray, boot: int = 2000, rng=np.random.default_rng(0)):
    m = float(vals.mean())
    if len(vals) < 2:
        return m, m, m
    idx = rng.integers(0, len(vals), size=(boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return m, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 1448
    dirs = ([Path(p) for p in sys.argv[2:]] if len(sys.argv) > 2 else
            [REPO_ROOT / "data" / "max_endpoint" / "wide_d4",
             REPO_ROOT / "data" / "max_endpoint" / "wide_d4_ext"])

    df = pd.concat([pd.read_csv(d / "per_network.csv") for d in dirs],
                   ignore_index=True)
    df = df[(df["width"] == width) & (df["status"] != "failed")]
    if df.empty:
        print(f"no rows at width {width} in {[str(d) for d in dirs]}"); return
    depth = int(df["num_layers"].iloc[0])
    n_seeds = df["network_seed"].nunique()
    noise_floor = float((df["ref_se_a"] * df["ref_se_b"]).mean())

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # MC tradeoff lines from measured per-network variances.
    ref_rows = df.drop_duplicates(subset=["network_seed"])
    mc = {}
    for var_col, fps_col, label, color, ls in (
        ("mc_var_gaussian", "mc_flops_per_sample_gaussian",
         "MC (Gaussian input): MSE = Var/m", "tab:red", "-"),
        ("mc_var_spherical", "mc_flops_per_sample_spherical",
         "MC (spherical Rao-Blackwell): MSE = Var/m", "tab:orange", "--"),
    ):
        var = float(pd.to_numeric(ref_rows[var_col], errors="coerce").mean())
        fps = float(pd.to_numeric(ref_rows[fps_col], errors="coerce").mean())
        if not np.isfinite(var) or var <= 0 or not np.isfinite(fps):
            continue
        mc[var_col] = (var, fps)
        m = np.logspace(0, 7.5, 60)
        ax.plot(m * fps, var / m, color=color, lw=1.5, ls=ls, label=label)

    # Deterministic treewidth-pipeline points with bootstrap CIs. E1/E2_cov2
    # and the two k3 methods share FLOPs (same kprop build); stagger labels.
    short = {"product_gaussian": "E0", "pg_plus_cov1": "E1",
             "pg_plus_cov2": "E2_cov2", "pg_k2aug_trace_k3k4": "k2aug",
             "pg_plus_cov2_k3": "E2_k3",
             "pg_cov2_k3_k4trace_simple": "k4trace (S)",
             "pg_cov2_k3_k4trace_augment": "k4trace (A)"}
    offsets = {"pg_k2aug_trace_k3k4": (-8, 4),
               "pg_cov2_k3_k4trace_simple": (-8, -14),
               "pg_cov2_k3_k4trace_augment": (8, -4)}
    rows = []
    for key, method, label, color, marker in METHODS:
        g = df[(df["kprop_variant"] == key[0]) & (df["estimator_name"] == key[1])]
        if g.empty:
            continue
        mse, lo, hi = boot_ci(g["error_cross"].to_numpy(float))
        f = float(g["flops_total"].mean())
        resolved = mse > 0 and lo > 0 and mse > 2 * noise_floor
        rows.append((method, label, f, mse, lo, hi, resolved))
        dx, dy = offsets.get(method, (6, 5))
        ha = "right" if dx < 0 else "left"
        if resolved:
            ax.errorbar([f], [mse],
                        yerr=[[max(mse - lo, 1e-30)], [max(hi - mse, 1e-30)]],
                        marker=marker, ms=8, capsize=3, lw=1.2, color=color,
                        label=label)
            ax.annotate(short[method], (f, mse), fontsize=6, color=color,
                        xytext=(dx, dy), textcoords="offset points", ha=ha)
        else:
            ax.scatter([f], [noise_floor], marker=marker, s=55,
                       facecolors="none", edgecolors=color,
                       label=f"{label} (below noise floor)")
            ax.annotate(short[method] + "?", (f, noise_floor), fontsize=6,
                        color=color, xytext=(dx, dy), textcoords="offset points",
                        ha=ha)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(noise_floor, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.annotate("reference noise floor", (ax.get_xlim()[0], noise_floor),
                fontsize=6, xytext=(6, 3), textcoords="offset points")
    ax.set_xlabel("online FLOPs (modeled)")
    ax.set_ylabel(rf"MSE over $\theta$ (cross-fidelity, {n_seeds} seeds)")
    ax.set_title(f"n = {width}, depth {depth}: FLOPs-vs-MSE, "
                 "treewidth-pipeline estimators vs MC")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, loc="upper right")
    plots = dirs[0] / "plots"
    plots.mkdir(exist_ok=True)
    name = f"max_endpoint_flops_vs_mse_n{width}"
    for ext in ("pdf", "png"):
        fig.savefig(plots / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {plots}/{name}.pdf/.png")

    if "mc_var_gaussian" in mc:
        var_g, fps_g = mc["mc_var_gaussian"]
        print(f"\nMC (Gaussian): Var={var_g:.3g}, {fps_g:.3g} FLOPs/sample; "
              f"noise floor {noise_floor:.3g}")
        for method, label, f, mse, lo, hi, resolved in rows:
            m_matched = max(np.floor(f / fps_g), 1)
            mc_at_budget = var_g / m_matched
            mc_flops_to_match = var_g / mse * fps_g if mse > 0 else float("nan")
            flag = "" if resolved else "  [below noise floor]"
            print(f"{label:30s} flops {f:9.3g}  MSE {mse:9.3g} [{lo:.2g},{hi:.2g}]  "
                  f"MC@same-FLOPs {mc_at_budget:9.3g}  "
                  f"MC-FLOPs-to-match {mc_flops_to_match:9.3g}{flag}")


if __name__ == "__main__":
    main()
