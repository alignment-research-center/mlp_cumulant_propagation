"""
Plots for the direct dense Edgeworth max experiment. Reads
<results_dir>/results.jsonl (written by direct_edgeworth_max_experiment.py)
and produces PDF+PNG figures (spec section 17):

    direct_edgeworth_max_mse_vs_width
    direct_edgeworth_max_mse_vs_flops
    direct_edgeworth_max_flops_vs_width
    direct_edgeworth_max_correction_sizes
    direct_edgeworth_max_reference_diagnostics
    direct_edgeworth_max_matched_mc

MSE convention: mean over network seeds of squared_error_cross =
(T_hat - T_ref_A)(T_hat - T_ref_B); 95% CIs by bootstrap over seeds.
Noise-dominated points (mse <= 2 * noise floor) are drawn open ("unresolved")
and excluded from fits, never silently clipped.
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

METHOD_COLORS = {
    "E0_product_gaussian": "tab:gray",
    "E1_cov1": "tab:olive",
    "E2_cov": "tab:green",
    "E2_k3": "tab:cyan",
    "E2_k3_k4trace_simple": "tab:blue",
    "E2_k3_k4trace_augment": "tab:purple",
}
BOOT = 2000
RNG = np.random.default_rng(0)
BAD_STATUS = ("failed", "dense_refused", "oom")


def load(results_dir: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in (results_dir / "results.jsonl").read_text().splitlines()]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows])
    df = df[~df["status"].isin(BAD_STATUS)].copy()
    df = df[df["estimator_name"].isin(METHOD_COLORS)].copy()
    df["method"] = df["estimator_name"]
    df["color"] = df["method"].map(METHOD_COLORS)
    return df


def bootstrap_mean_ci(values: np.ndarray, boot: int = BOOT) -> tuple[float, float, float]:
    m = float(values.mean())
    if len(values) < 2:
        return m, m, m
    idx = RNG.integers(0, len(values), size=(boot, len(values)))
    means = values[idx].mean(axis=1)
    return m, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, width), g in df.groupby(["method", "width"]):
        vals = g["squared_error_cross"].to_numpy(dtype=float)
        mse, lo, hi = bootstrap_mean_ci(vals)
        noise_floor = float((g["reference_se_A"] * g["reference_se_B"]).mean())
        out.append({
            "method": method,
            "color": g["color"].iloc[0],
            "width": width,
            "mse": mse, "ci_lo": lo, "ci_hi": hi,
            "noise_floor": noise_floor,
            "resolved": bool(mse > 0 and lo > 0 and mse > 2 * noise_floor),
            "flops_total": float(g["flops_total"].mean()),
            "flops_kprop": float(g["flops_kprop"].mean()),
            "flops_endpoint": float(g["flops_endpoint"].mean()),
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


def _plot_method_points(ax, g, x_col, label, color, annotate_width=False):
    res, unres = g[g["resolved"]], g[~g["resolved"]]
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


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def plot_mse_vs_width(agg, df, results_dir, common_widths=None) -> dict:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    slopes: dict[str, dict] = {}
    for method, g in agg.groupby("method"):
        g = g.sort_values("width")
        color = g["color"].iloc[0]
        res = g[g["resolved"] & (g["mse"] > 0) & (g["mse"] < 1.0)]
        if common_widths is not None:
            res = res[res["width"].isin(common_widths)]
        label = method
        if len(res) >= 3:
            slope = _fit_slope(res["width"].to_numpy(float), res["mse"].to_numpy(float))
            bs = []
            sub = df[df["method"] == method]
            seeds = sorted(sub["network_seed"].unique())
            for _ in range(400):
                pick = RNG.choice(seeds, size=len(seeds), replace=True)
                mses, ws = [], []
                for w in res["width"]:
                    v = np.concatenate([
                        sub[(sub["width"] == w) & (sub["network_seed"] == s)][
                            "squared_error_cross"].to_numpy(float)
                        for s in pick
                    ])
                    if len(v) and v.mean() > 0:
                        mses.append(v.mean()); ws.append(w)
                if len(ws) >= 3:
                    bs.append(_fit_slope(np.array(ws, float), np.array(mses)))
            lo, hi = (np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan))
            slopes[method] = {"slope": slope, "ci": [float(lo), float(hi)],
                              "widths_used": res["width"].tolist(),
                              "n_seeds": int(res["n_seeds"].max())}
            label = f"{method} (slope {slope:.2f} [{lo:.2f},{hi:.2f}])"
        _plot_method_points(ax, g, "width", label, color)
    for target, ls in ((-1, ":"), (-2, "-."), (-3, "--")):
        xs = np.array(sorted(agg["width"].unique()), float)
        ref_y = agg["mse"].max() * (xs / xs[0]) ** target
        ax.plot(xs, ref_y, color="k", lw=0.5, ls=ls, alpha=0.4)
        ax.annotate(f"n^{target}", (xs[-1], ref_y[-1]), fontsize=5, color="k", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n"); ax.set_ylabel(r"MSE over $\theta$ (cross-fidelity)")
    ax.legend(fontsize=6); ax.grid(alpha=0.25, which="both")
    ax.set_title("direct dense Edgeworth max: MSE vs width (open = unresolved)")
    _save(fig, results_dir, "direct_edgeworth_max_mse_vs_width")
    (results_dir / "plots" / "mse_width_slopes.json").write_text(json.dumps(slopes, indent=1))
    return slopes


def plot_mse_vs_flops(agg, results_dir) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method, g in agg.groupby("method"):
        _plot_method_points(ax, g.sort_values("flops_total"), "flops_total", method,
                            g["color"].iloc[0], annotate_width=True)
    for kind, var_col, f_col, color in (
        ("MC", "mc_var_gaussian", "mc_flops_gauss", "tab:red"),
        ("MC-spherical", "mc_var_spherical", "mc_flops_sph", "tab:orange"),
    ):
        first = True
        for width, g in agg.groupby("width"):
            var, fps = g[var_col].iloc[0], g[f_col].iloc[0]
            if not np.isfinite(var) or var <= 0:
                continue
            m = np.logspace(2, 11, 40)
            ax.plot(m * fps, var / m, color=color, lw=0.8, alpha=0.5,
                    label=(kind if first else None))
            first = False
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("online FLOPs (modeled)"); ax.set_ylabel("MSE")
    ax.legend(fontsize=6); ax.grid(alpha=0.25, which="both")
    ax.set_title("MSE vs FLOPs; red/orange: Monte Carlo Var/m per width")
    _save(fig, results_dir, "direct_edgeworth_max_mse_vs_flops")


def plot_flops_vs_width(agg, results_dir) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    slopes = {}
    for method, g in agg.groupby("method"):
        g = g.sort_values("width")
        color = g["color"].iloc[0]
        if len(g) >= 3:
            s_tot = _fit_slope(g["width"].to_numpy(float), g["flops_total"].to_numpy(float))
            s_k = _fit_slope(g["width"].to_numpy(float), g["flops_kprop"].to_numpy(float))
            s_e = _fit_slope(g["width"].to_numpy(float),
                             np.maximum(g["flops_endpoint"].to_numpy(float), 1.0))
            slopes[method] = {"total": s_tot, "kprop": s_k, "endpoint": s_e}
            label = f"{method} (total ~n^{s_tot:.2f})"
        else:
            label = method
        ax.plot(g["width"], g["flops_total"], marker="o", ms=4, label=label, color=color)
        ax.plot(g["width"], g["flops_kprop"], marker="s", ms=3, ls=":", lw=0.8,
                color=color, alpha=0.7)
        ax.plot(g["width"], g["flops_endpoint"], marker="x", ms=3, ls="--", lw=0.8,
                color=color, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n")
    ax.set_ylabel("FLOPs (solid: total; dotted: kprop; dashed: endpoint)")
    ax.legend(fontsize=6); ax.grid(alpha=0.25, which="both")
    _save(fig, results_dir, "direct_edgeworth_max_flops_vs_width")
    (results_dir / "plots" / "flops_width_slopes.json").write_text(json.dumps(slopes, indent=1))


def plot_correction_sizes(df, results_dir) -> None:
    """RMS over networks of each integrated correction term (from the most
    complete tower rows available at each width)."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    terms = [
        ("correction_C2", "C2 Psi", "tab:olive"),
        ("correction_C2_squared", "(1/2) C2^2 Psi", "tab:green"),
        ("correction_C3", "C3 Psi", "tab:cyan"),
        ("correction_C4_trace", "C4_trace Psi (simple)", "tab:blue"),
    ]
    sub = df[df["kprop_variant"] == "k3_simple"].drop_duplicates(
        subset=["width", "network_seed"])
    aug = df[df["kprop_variant"] == "k3_augment"].drop_duplicates(
        subset=["width", "network_seed"])
    for col, label, color in terms:
        vals = sub.groupby("width")[col].apply(
            lambda v: float(np.sqrt(np.mean(np.square(pd.to_numeric(v, errors="coerce").dropna())))))
        vals = vals[vals > 0]
        if len(vals):
            ax.plot(vals.index, vals.values, marker="o", ms=4, label=label, color=color)
    if len(aug):
        vals = aug.groupby("width")["correction_C4_trace"].apply(
            lambda v: float(np.sqrt(np.mean(np.square(pd.to_numeric(v, errors="coerce").dropna())))))
        vals = vals[vals > 0]
        if len(vals):
            ax.plot(vals.index, vals.values, marker="s", ms=4, ls="--",
                    label="C4_trace Psi (augment)", color="tab:purple")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n"); ax.set_ylabel("RMS correction over networks")
    ax.legend(fontsize=6); ax.grid(alpha=0.25, which="both")
    ax.set_title("Edgeworth correction sizes (k3 towers)")
    _save(fig, results_dir, "direct_edgeworth_max_correction_sizes")


def plot_reference_diagnostics(df, agg, results_dir) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    ref = df.drop_duplicates(subset=["width", "network_seed"])
    axes[0].scatter(ref["width"], ref["reference_se_A"], s=8, label="se(A)")
    axes[0].scatter(ref["width"], ref["reference_se_B"], s=8, label="se(B)", alpha=0.6)
    axes[0].set_yscale("log"); axes[0].set_ylabel("reference SE"); axes[0].legend(fontsize=6)
    diff = ref["reference_A"] - ref["reference_B"]
    se = np.sqrt(ref["reference_se_A"] ** 2 + ref["reference_se_B"] ** 2)
    axes[1].scatter(ref["width"], diff / se, s=8)
    axes[1].axhline(0, color="k", lw=0.8)
    for y in (-2, 2):
        axes[1].axhline(y, color="r", lw=0.6, ls=":")
    axes[1].set_ylabel("(A - B) / se  (~N(0,1))")
    axes[2].scatter(ref["width"], ref["reference_samples_A"], s=8)
    axes[2].set_yscale("log"); axes[2].set_ylabel("reference samples (A)")
    frac = agg.groupby("width")["resolved"].apply(lambda v: 1.0 - float(np.mean(v)))
    axes[3].plot(frac.index, frac.values, marker="o", ms=4)
    axes[3].set_ylim(-0.05, 1.05); axes[3].set_ylabel("fraction unresolved points")
    for axx in axes:
        axx.set_xscale("log"); axx.set_xlabel("width"); axx.grid(alpha=0.25, which="both")
    fig.suptitle("reference diagnostics", fontsize=9)
    _save(fig, results_dir, "direct_edgeworth_max_reference_diagnostics")


def plot_matched_mc(agg, results_dir) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method, g in agg.groupby("method"):
        g = g.sort_values("width")
        color = g["color"].iloc[0]
        for var_col, f_col, ls in (("mc_var_gaussian", "mc_flops_gauss", "-"),
                                   ("mc_var_spherical", "mc_flops_sph", "--")):
            ratios, ws = [], []
            for _, r in g.iterrows():
                if not r["resolved"] or r["mse"] <= 0 or not np.isfinite(r[var_col]):
                    continue
                m_matched = max(np.floor(r["flops_total"] / r[f_col]), 1.0)
                ratios.append((r[var_col] / m_matched) / r["mse"]); ws.append(r["width"])
            if ws:
                ax.plot(ws, ratios, marker="o", ms=4, ls=ls, color=color,
                        label=f"{method}{' (sph MC)' if ls == '--' else ''}")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n")
    ax.set_ylabel("MSE_MC(matched FLOPs) / MSE_deterministic")
    ax.legend(fontsize=5); ax.grid(alpha=0.25, which="both")
    ax.set_title("matched-budget MC comparison (>1 favors deterministic)")
    _save(fig, results_dir, "direct_edgeworth_max_matched_mc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--common-widths", type=int, nargs="*", default=None,
                        help="restrict slope fits to a shared width range")
    args = parser.parse_args()
    rdir = Path(args.results_dir)
    df = load(rdir)
    if df.empty:
        print("no plottable rows"); return
    agg = aggregate(df)
    agg.to_csv(rdir / "plots_aggregate.csv", index=False)
    slopes = plot_mse_vs_width(agg, df, rdir, common_widths=args.common_widths)
    plot_mse_vs_flops(agg, rdir)
    plot_flops_vs_width(agg, rdir)
    plot_correction_sizes(df, rdir)
    plot_reference_diagnostics(df, agg, rdir)
    plot_matched_mc(agg, rdir)
    print(json.dumps(slopes, indent=1))


if __name__ == "__main__":
    main()
