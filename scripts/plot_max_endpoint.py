"""
Plots for the max-endpoint experiment. Reads RESULTS_DIR/results.jsonl
(written by max_endpoint_experiment.py) and produces PDF+PNG figures:

    mse_vs_flops, mse_vs_width, flops_vs_width, matched_budget_comparison,
    ground_truth_diagnostics, wall_clock (supplementary, hardware-dependent).

MSE convention: mean over network seeds of the cross-fidelity estimator
err_cross = (T_hat - T_ref_A)(T_hat - T_ref_B); 95% CIs by bootstrap over
network seeds. Non-positive or noise-dominated aggregate MSEs are drawn as
open markers pinned at the reference noise floor ("unresolved"), never
silently clipped onto the log axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# (kprop_variant, estimator_name) -> canonical method name, color
PRIMARY_METHODS = {
    ("k1_simple", "E0_product_gaussian"): ("product_gaussian", "tab:gray"),
    ("k2_simple", "E1_cov1"): ("pg_plus_cov1", "tab:olive"),
    ("k2_simple", "E2_cov2"): ("pg_plus_cov2", "tab:green"),
    ("k3_simple", "E2_k3"): ("pg_plus_cov2_k3", "tab:cyan"),
    ("k3_simple", "E2_full"): ("pg_cov2_k3_k4trace_simple", "tab:blue"),
    ("k3_augment", "E2_full"): ("pg_cov2_k3_k4trace_augment", "tab:purple"),
    ("k2_augment", "E2_full"): ("pg_k2aug_trace_k3k4", "tab:pink"),
}
BOOT = 2000
RNG = np.random.default_rng(0)


def load(results_dir: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in (results_dir / "results.jsonl").read_text().splitlines()]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows])
    df = df[df["status"] != "failed"].copy()
    keys = list(zip(df["kprop_variant"], df["estimator_name"]))
    df["method"] = [PRIMARY_METHODS.get(k, (None,))[0] for k in keys]
    df["color"] = [PRIMARY_METHODS.get(k, (None, "tab:brown"))[1] for k in keys]
    return df[df["method"].notna()].copy()


def bootstrap_mean_ci(values: np.ndarray, boot: int = BOOT) -> tuple[float, float, float]:
    """Mean and 95% bootstrap CI over network seeds."""
    m = float(values.mean())
    if len(values) < 2:
        return m, m, m
    idx = RNG.integers(0, len(values), size=(boot, len(values)))
    means = values[idx].mean(axis=1)
    return m, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, width), g in df.groupby(["method", "width"]):
        vals = g["error_cross"].to_numpy(dtype=float)
        mse, lo, hi = bootstrap_mean_ci(vals)
        noise_floor = float((g["ref_se_a"] * g["ref_se_b"]).mean())
        out.append({
            "method": method,
            "color": g["color"].iloc[0],
            "width": width,
            "mse": mse,
            "ci_lo": lo,
            "ci_hi": hi,
            "noise_floor": noise_floor,
            "resolved": bool(mse > 0 and lo > 0 and mse > 2 * noise_floor),
            "flops_total": float(g["flops_total"].mean()),
            "flops_kprop": float(g["flops_kprop"].mean()),
            "flops_endpoint": float(g["flops_endpoint"].mean()),
            "wall": float((g["wall_seconds_kprop"] + g["wall_seconds_endpoint"]).mean()),
            "mc_var_gaussian": float(g["mc_var_gaussian"].mean()),
            "mc_flops_gauss": float(g["mc_flops_per_sample_gaussian"].mean()),
            "mc_var_spherical": float(pd.to_numeric(g["mc_var_spherical"], errors="coerce").mean()),
            "mc_flops_sph": float(g["mc_flops_per_sample_spherical"].mean()),
            "n_seeds": g["network_seed"].nunique(),
        })
    return pd.DataFrame(out).sort_values(["method", "width"])


def _save(fig, results_dir: Path, name: str) -> None:
    plots = results_dir / "plots"
    plots.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(plots / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {plots}/{name}.pdf/.png")


def _plot_method_points(ax, g: pd.DataFrame, x_col: str, label: str, color: str,
                        annotate_width: bool = False):
    res = g[g["resolved"]]
    unres = g[~g["resolved"]]
    if len(res):
        yerr = np.stack([res["mse"] - res["ci_lo"], res["ci_hi"] - res["mse"]])
        ax.errorbar(res[x_col], res["mse"], yerr=np.clip(yerr, 0, None), marker="o",
                    ms=4, lw=1.2, capsize=2, label=label, color=color)
        if annotate_width:
            for _, r in res.iterrows():
                ax.annotate(f"n={int(r['width'])}", (r[x_col], r["mse"]), fontsize=5,
                            xytext=(3, 3), textcoords="offset points", color=color)
    if len(unres):
        ax.scatter(unres[x_col], unres["noise_floor"], marker="v", s=22,
                   facecolors="none", edgecolors=color)
        if annotate_width:
            for _, r in unres.iterrows():
                ax.annotate(f"n={int(r['width'])}?", (r[x_col], r["noise_floor"]),
                            fontsize=5, xytext=(3, 3), textcoords="offset points", color=color)


def plot_mse_vs_flops(agg: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, g in agg.groupby("method"):
        _plot_method_points(ax, g.sort_values("flops_total"), "flops_total", method,
                            g["color"].iloc[0], annotate_width=True)
    # MC baseline curves per width (standard and spherical), at log-spaced m.
    for kind, var_col, f_col, color in (
        ("MC", "mc_var_gaussian", "mc_flops_gauss", "tab:red"),
        ("MC-sph", "mc_var_spherical", "mc_flops_sph", "tab:orange"),
    ):
        first = True
        for width, g in agg.groupby("width"):
            var = g[var_col].iloc[0]
            fps = g[f_col].iloc[0]
            if not np.isfinite(var) or var <= 0:
                continue
            m = np.logspace(2, 11, 40)
            ax.plot(m * fps, var / m, color=color, lw=0.8, alpha=0.55,
                    label=(kind if first else None))
            first = False
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("online FLOPs"); ax.set_ylabel(r"MSE over $\theta$ (cross-fidelity)")
    ax.legend(fontsize=6); ax.grid(alpha=0.25, which="both")
    ax.set_title("max-endpoint estimators vs Monte Carlo (open = below ref noise floor)")
    _save(fig, results_dir, "mse_vs_flops")


def _fit_slope(widths: np.ndarray, mses: np.ndarray) -> float:
    x, y = np.log(widths), np.log(mses)
    return float(np.polyfit(x, y, 1)[0])


def plot_mse_vs_width(agg: pd.DataFrame, df: pd.DataFrame, results_dir: Path) -> dict:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    slopes: dict[str, dict] = {}
    for method, g in agg.groupby("method"):
        g = g.sort_values("width")
        color = g["color"].iloc[0]
        res = g[g["resolved"] & (g["mse"] > 0)]
        label = method
        if len(res) >= 3:
            slope = _fit_slope(res["width"].to_numpy(float), res["mse"].to_numpy(float))
            # bootstrap slope over seeds
            bs = []
            sub = df[df["method"] == method]
            seeds = sorted(sub["network_seed"].unique())
            for _ in range(400):
                pick = RNG.choice(seeds, size=len(seeds), replace=True)
                mses, ws = [], []
                for w in res["width"]:
                    v = np.concatenate([
                        sub[(sub["width"] == w) & (sub["network_seed"] == s)]["error_cross"].to_numpy(float)
                        for s in pick
                    ])
                    if len(v) and v.mean() > 0:
                        mses.append(v.mean()); ws.append(w)
                if len(ws) >= 3:
                    bs.append(_fit_slope(np.array(ws, float), np.array(mses)))
            lo, hi = (np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan))
            slopes[method] = {"slope": slope, "ci": [float(lo), float(hi)],
                              "widths_used": res["width"].tolist()}
            label = f"{method} (slope {slope:.2f} [{lo:.2f},{hi:.2f}])"
        _plot_method_points(ax, g, "width", label, color)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n"); ax.set_ylabel("MSE")
    ax.legend(fontsize=6); ax.grid(alpha=0.25, which="both")
    ax.set_title("MSE vs width (fits use only points above the noise floor)")
    _save(fig, results_dir, "mse_vs_width")
    (results_dir / "plots" / "mse_width_slopes.json").write_text(json.dumps(slopes, indent=1))
    return slopes


def plot_flops_vs_width(agg: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    slopes = {}
    for method, g in agg.groupby("method"):
        g = g.sort_values("width")
        if len(g) >= 3:
            s_tot = _fit_slope(g["width"].to_numpy(float), g["flops_total"].to_numpy(float))
            slopes[method] = s_tot
            label = f"{method} (~n^{s_tot:.2f})"
        else:
            label = method
        ax.plot(g["width"], g["flops_total"], marker="o", ms=4, label=label,
                color=g["color"].iloc[0])
        ax.plot(g["width"], g["flops_endpoint"], marker="x", ms=3, ls="--", lw=0.8,
                color=g["color"].iloc[0], alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n"); ax.set_ylabel("FLOPs (solid: total; dashed: endpoint only)")
    ax.legend(fontsize=6); ax.grid(alpha=0.25, which="both")
    _save(fig, results_dir, "flops_vs_width")
    (results_dir / "plots" / "flops_width_slopes.json").write_text(json.dumps(slopes, indent=1))


def plot_matched_budget(agg: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, g in agg.groupby("method"):
        g = g.sort_values("width")
        color = g["color"].iloc[0]
        for var_col, f_col, ls in (("mc_var_gaussian", "mc_flops_gauss", "-"),
                                   ("mc_var_spherical", "mc_flops_sph", "--")):
            ratios, ws = [], []
            for _, r in g.iterrows():
                if not r["resolved"] or r["mse"] <= 0 or not np.isfinite(r[var_col]):
                    continue
                m_matched = max(r["flops_total"] / r[f_col], 1.0)
                mc_mse = r[var_col] / m_matched
                ratios.append(mc_mse / r["mse"]); ws.append(r["width"])
            if ws:
                ax.plot(ws, ratios, marker="o", ms=4, ls=ls, color=color,
                        label=f"{method}{' (vs sph MC)' if ls == '--' else ''}")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n")
    ax.set_ylabel("MSE_MC(matched FLOPs) / MSE_deterministic")
    ax.legend(fontsize=5); ax.grid(alpha=0.25, which="both")
    ax.set_title("matched-budget advantage over Monte Carlo (>1 favors deterministic)")
    _save(fig, results_dir, "matched_budget_comparison")


def plot_ground_truth_diagnostics(df: pd.DataFrame, results_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ref = df.drop_duplicates(subset=["width", "network_seed"])
    axes[0].scatter(ref["width"], ref["ref_se_a"], s=8, label="se(A)")
    axes[0].scatter(ref["width"], ref["ref_se_b"], s=8, label="se(B)", alpha=0.6)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("width"); axes[0].set_ylabel("reference SE"); axes[0].legend(fontsize=6)
    diff = ref["ref_a"] - ref["ref_b"]
    se = np.sqrt(ref["ref_se_a"] ** 2 + ref["ref_se_b"] ** 2)
    axes[1].scatter(ref["width"], diff / se, s=8)
    axes[1].axhline(0, color="k", lw=0.8)
    for y in (-2, 2):
        axes[1].axhline(y, color="r", lw=0.6, ls=":")
    axes[1].set_xscale("log"); axes[1].set_xlabel("width")
    axes[1].set_ylabel("(A - B) / se  (should be ~N(0,1))")
    axes[2].scatter(ref["width"], ref["ref_samples_a"], s=8)
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("width"); axes[2].set_ylabel("reference samples (A)")
    for axx in axes:
        axx.grid(alpha=0.25, which="both")
    fig.suptitle("ground-truth diagnostics", fontsize=9)
    _save(fig, results_dir, "ground_truth_diagnostics")


def plot_wall_clock(agg: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, g in agg.groupby("method"):
        g = g.sort_values("width")
        ax.plot(g["width"], g["wall"], marker="o", ms=4, label=method, color=g["color"].iloc[0])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n"); ax.set_ylabel("wall seconds (kprop + endpoint)")
    ax.set_title("wall-clock (HARDWARE-DEPENDENT; primary axis is FLOPs)")
    ax.legend(fontsize=6); ax.grid(alpha=0.25, which="both")
    _save(fig, results_dir, "wall_clock_supplementary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()
    rdir = Path(args.results_dir)
    df = load(rdir)
    if df.empty:
        print("no plottable rows"); return
    agg = aggregate(df)
    agg.to_csv(rdir / "plots_aggregate.csv", index=False)
    plot_mse_vs_flops(agg, rdir)
    slopes = plot_mse_vs_width(agg, df, rdir)
    plot_flops_vs_width(agg, rdir)
    plot_matched_budget(agg, rdir)
    plot_ground_truth_diagnostics(df, rdir)
    plot_wall_clock(agg, rdir)
    print(json.dumps(slopes, indent=1))


if __name__ == "__main__":
    main()
