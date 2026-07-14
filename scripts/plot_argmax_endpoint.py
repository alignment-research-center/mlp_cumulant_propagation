"""
Plots for the argmax-endpoint experiment. Reads RESULTS_DIR/results.jsonl
(written by argmax_endpoint_experiment.py) and produces PDF+PNG figures:

    argmax_mse_total_vs_flops, argmax_mse_per_coordinate_vs_flops,
    argmax_mse_vs_width, argmax_flops_vs_width, argmax_matched_budget_mc,
    argmax_simplex_diagnostics, argmax_sampling_diagnostics,
    argmax_wall_clock_supplementary (hardware-dependent).

MSE convention: total Brier MSE E_theta ||q_hat - q||_2^2, estimated per
network by the unbiased winner-count U-statistic averaged over evaluation
blocks (may be negative; kept signed in CSV/JSON). 95% CIs by hierarchical
bootstrap: resample network seeds, then evaluation blocks within each seed.
Aggregates whose CI includes zero are drawn as open markers at the CI upper
bound ("unresolved"), never silently clipped onto the log axis. Width
exponents are fitted only on resolved points and are empirical (no n^{-K}
theorem is claimed for ReLU).
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

# (kprop_variant, estimator_name) -> canonical method base name, color.
# Deduplication mirrors the scalar experiment: at k_max=1 E1/E2_cov are exactly
# E0 (recorded equivalences), so each method uses the cheapest tower providing
# its sectors.
PRIMARY_METHODS = {
    ("k1_simple", "E0_product_gaussian"): ("argmax_product_gaussian", "tab:gray"),
    ("k2_simple", "E1_cov1"): ("argmax_plus_cov1", "tab:olive"),
    ("k2_simple", "E2_cov2"): ("argmax_plus_cov2", "tab:green"),
    ("k3_simple", "E2_k3"): ("argmax_plus_cov2_k3", "tab:cyan"),
    ("k3_simple", "E2_full"): ("argmax_plus_cov2_k3_k4trace_simple", "tab:blue"),
    ("k3_augment", "E2_full"): ("argmax_plus_cov2_k3_k4trace_augment", "tab:purple"),
    ("k2_augment", "E2_full"): ("argmax_k2aug_trace_k3k4", "tab:pink"),
}
BOOT = 2000
RNG = np.random.default_rng(0)


def load(results_dir: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in (results_dir / "results.jsonl").read_text().splitlines()]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows])
    df = df[df["status"] != "failed"].copy()
    keys = list(zip(df["kprop_variant"], df["estimator_name"]))
    df["method_base"] = [PRIMARY_METHODS.get(k, (None,))[0] for k in keys]
    df["color"] = [PRIMARY_METHODS.get(k, (None, "tab:brown"))[1] for k in keys]
    df = df[df["method_base"].notna()].copy()
    df["method"] = df["method_base"] + "_" + df["estimator_projection"]
    return df


def hierarchical_bootstrap_ci(
    block_lists: list[np.ndarray], boot: int = BOOT
) -> tuple[float, float, float]:
    """Mean and 95% CI of the seed-mean of block means: resample seeds, then
    blocks within each selected seed."""
    seed_means = np.array([b.mean() for b in block_lists])
    m = float(seed_means.mean())
    if len(block_lists) < 2:
        return m, m, m
    stats = np.empty(boot)
    ns = len(block_lists)
    for t in range(boot):
        pick = RNG.integers(0, ns, size=ns)
        vals = []
        for s in pick:
            b = block_lists[s]
            vals.append(b[RNG.integers(0, len(b), size=len(b))].mean())
        stats[t] = np.mean(vals)
    return m, float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def aggregate(df: pd.DataFrame, raw_rows: list[dict]) -> pd.DataFrame:
    """Aggregate per (method, width) with hierarchical bootstrap CIs.

    Rows from towers with clamped (truncation-negative) variances are
    EXCLUDED here and from fits: their raw q blows up by many orders of
    magnitude (sigma ~ 1e-5 spikes) and would render the means meaningless.
    They are flagged, not hidden: the exclusion count is printed and saved,
    and the simplex-diagnostics figure displays all rows including them.
    """
    excluded = sum(1 for r in raw_rows if r.get("num_clamped_var", 0) > 0)
    if excluded:
        print(f"NOTE: excluding {excluded} rows from clamped-variance (diverged) "
              f"towers from aggregates/fits; see argmax_simplex_diagnostics and "
              f"per-row status flags.")
    df = df[pd.to_numeric(df["num_clamped_var"], errors="coerce").fillna(0) == 0].copy()
    blocks: dict[tuple, dict[int, np.ndarray]] = {}
    for r in raw_rows:
        key = (r.get("kprop_variant"), r.get("estimator_name"))
        base = PRIMARY_METHODS.get(key, (None,))[0]
        if base is None or r.get("status") == "failed" or r.get("num_clamped_var", 0) > 0:
            continue
        method = base + "_" + r["estimator_projection"]
        bm = np.asarray(r.get("block_mse", []), dtype=float)
        if bm.size:
            blocks.setdefault((method, r["width"]), {})[r["network_seed"]] = bm
    out = []
    for (method, width), g in df.groupby(["method", "width"]):
        bl = blocks.get((method, width), {})
        block_lists = [bl[s] for s in sorted(bl)]
        if block_lists:
            mse, lo, hi = hierarchical_bootstrap_ci(block_lists)
        else:
            vals = g["mse_total_unbiased"].to_numpy(float)
            mse, lo, hi = float(vals.mean()), float(vals.min()), float(vals.max())
        out.append({
            "method": method,
            "method_base": g["method_base"].iloc[0],
            "projection": g["estimator_projection"].iloc[0],
            "color": g["color"].iloc[0],
            "width": width,
            "mse": mse,
            "ci_lo": lo,
            "ci_hi": hi,
            "resolved": bool(mse > 0 and lo > 0),
            "mse_pc": mse / width,
            "ci_lo_pc": lo / width,
            "ci_hi_pc": hi / width,
            "flops_total": float(g["flops_total"].mean()),
            "flops_kprop": float(g["flops_kprop"].mean()),
            "flops_endpoint_forward": float(g["flops_endpoint_forward"].mean()),
            "flops_endpoint_backward": float(g["flops_endpoint_backward"].mean()),
            "flops_endpoint_total": float(g["flops_endpoint_total"].mean()),
            "mc_flops_per_sample": float(g["mc_flops_per_sample"].mean()),
            "collision": float(g["collision_probability_estimate"].mean()),
            "input_se": float(pd.to_numeric(g["mse_total_input_se"], errors="coerce").mean()),
            "simplex_residual": float(g["simplex_residual"].max()),
            "q_min": float(g["q_min"].min()),
            "num_negative": float(g["num_negative_coordinates"].mean()),
            "raw_proj_dist": float(g["raw_projected_l2_distance"].mean()),
            "total_samples": float(g["evaluation_total_samples"].mean()),
            "wall": float((g["wall_seconds_kprop"] + g["wall_seconds_endpoint"]).mean()),
            "peak_mem": float(g["peak_gpu_memory_bytes"].max()),
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


def _ls(projection: str) -> str:
    return "-" if projection == "raw" else "--"


def _plot_points(ax, g, x_col, y_col, lo_col, hi_col, label, color, ls,
                 annotate_width=False):
    res = g[g["resolved"]]
    unres = g[~g["resolved"]]
    if len(res):
        yerr = np.stack([res[y_col] - res[lo_col], res[hi_col] - res[y_col]])
        ax.errorbar(res[x_col], res[y_col], yerr=np.clip(yerr, 0, None), marker="o",
                    ms=4, lw=1.2, ls=ls, capsize=2, label=label, color=color)
        if annotate_width:
            for _, r in res.iterrows():
                ax.annotate(f"n={int(r['width'])}", (r[x_col], r[y_col]), fontsize=5,
                            xytext=(3, 3), textcoords="offset points", color=color)
    if len(unres):
        y = np.maximum(unres[hi_col].to_numpy(float), 1e-300)
        ax.scatter(unres[x_col], y, marker="v", s=22, facecolors="none",
                   edgecolors=color)
        if annotate_width:
            for _, r in unres.iterrows():
                ax.annotate(f"n={int(r['width'])}?", (r[x_col], max(r[hi_col], 1e-300)),
                            fontsize=5, xytext=(3, 3), textcoords="offset points",
                            color=color)


def _mc_curves(ax, agg, per_coordinate=False):
    first = True
    for width, g in agg.groupby("width"):
        coll = g["collision"].iloc[0]
        fps = g["mc_flops_per_sample"].iloc[0]
        m = np.logspace(1, 11, 40)
        y = (1.0 - coll) / m
        if per_coordinate:
            y = y / width
        ax.plot(m * fps, y, color="tab:red", lw=0.8, alpha=0.55,
                label=("MC (1-||q||^2)/m" if first else None))
        first = False


def plot_mse_vs_flops(agg, results_dir, per_coordinate=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    y, lo, hi = ("mse_pc", "ci_lo_pc", "ci_hi_pc") if per_coordinate else ("mse", "ci_lo", "ci_hi")
    for method, g in agg.groupby("method"):
        # Label with the two separate width exponents (MSE ~ n^c at F ~ n^b);
        # a combined MSE ~ F^a law is NOT expected (both are power laws in n,
        # and the FLOP exponent changes regime, e.g. kprop's n^3 term).
        res = g[g["resolved"] & (g[y] > 0)]
        label = method
        if len(res) >= 3:
            c = _fit_slope(res["width"].to_numpy(float), res[y].to_numpy(float))
            b = _fit_slope(res["width"].to_numpy(float), res["flops_total"].to_numpy(float))
            label = f"{method} (MSE~$n^{{{c:.2f}}}$ at F~$n^{{{b:.2f}}}$)"
        _plot_points(ax, g.sort_values("flops_total"), "flops_total", y, lo, hi,
                     label, g["color"].iloc[0], _ls(g["projection"].iloc[0]),
                     annotate_width=True)
    _mc_curves(ax, agg, per_coordinate)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("online FLOPs per fixed-network prediction")
    if per_coordinate:
        ax.set_ylabel(r"$E_\theta\|\hat q - q\|_2^2 / n$")
        ax.set_title("per-coordinate Brier MSE vs FLOPs (extra 1/n from metric\n"
                     "normalization only; open = CI includes zero)")
    else:
        ax.set_ylabel(r"$E_\theta\|\hat q - q\|_2^2$")
        ax.set_title("total Brier MSE vs FLOPs (open = unresolved: CI includes zero)")
    ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.25, which="both")
    _save(fig, results_dir,
          "argmax_mse_per_coordinate_vs_flops" if per_coordinate else "argmax_mse_total_vs_flops")


def _fit_slope(widths, mses):
    return float(np.polyfit(np.log(widths), np.log(mses), 1)[0])


def plot_mse_vs_width(agg, raw_rows, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    slopes = {}
    # per-seed block MSE lists for bootstrap slope fits
    blocks: dict[tuple, dict[int, np.ndarray]] = {}
    for r in raw_rows:
        key = (r.get("kprop_variant"), r.get("estimator_name"))
        base = PRIMARY_METHODS.get(key, (None,))[0]
        if base is None or r.get("status") == "failed" or r.get("num_clamped_var", 0) > 0:
            continue
        method = base + "_" + r["estimator_projection"]
        bm = np.asarray(r.get("block_mse", []), dtype=float)
        if bm.size:
            blocks.setdefault((method, r["width"]), {})[r["network_seed"]] = bm
    for method, g in agg.groupby("method"):
        g = g.sort_values("width")
        color = g["color"].iloc[0]
        res = g[g["resolved"] & (g["mse"] > 0)]
        label = method
        if len(res) >= 3:
            ws = res["width"].to_numpy(float)
            slope = _fit_slope(ws, res["mse"].to_numpy(float))
            bs = []
            for _ in range(400):
                mses, wss = [], []
                for w in ws:
                    bl = blocks.get((method, int(w)), {})
                    seeds = sorted(bl)
                    if not seeds:
                        continue
                    pick = RNG.choice(seeds, size=len(seeds), replace=True)
                    v = np.mean([bl[s].mean() for s in pick])
                    if v > 0:
                        mses.append(v); wss.append(w)
                if len(wss) >= 3:
                    bs.append(_fit_slope(np.array(wss), np.array(mses)))
            lo, hi = (np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan))
            slopes[method] = {
                "slope_total": slope, "ci": [float(lo), float(hi)],
                # per-coordinate MSE = total/n: slope shifts by exactly -1.
                "slope_per_coordinate": slope - 1.0,
                "ci_per_coordinate": [float(lo) - 1.0, float(hi) - 1.0],
                "widths_used": res["width"].tolist(),
            }
            label = f"{method} (slope {slope:.2f} [{lo:.2f},{hi:.2f}])"
        _plot_points(ax, g, "width", "mse", "ci_lo", "ci_hi", label, color,
                     _ls(g["projection"].iloc[0]))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n"); ax.set_ylabel(r"total Brier MSE $E_\theta\|\hat q - q\|_2^2$")
    ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.25, which="both")
    ax.set_title("Brier MSE vs width (empirical fits; resolved points only)")
    _save(fig, results_dir, "argmax_mse_vs_width")
    (results_dir / "plots" / "argmax_mse_width_slopes.json").write_text(
        json.dumps(slopes, indent=1))
    return slopes


def plot_flops_vs_width(agg, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    slopes = {}
    # FLOP components are per kprop tower; take the most expensive method family
    for method, g in agg.groupby("method"):
        if g["projection"].iloc[0] != "raw":
            continue  # identical FLOPs for raw/projected
        g = g.sort_values("width")
        color = g["color"].iloc[0]
        comp = {}
        for col, ls, marker in (("flops_total", "-", "o"), ("flops_kprop", ":", "s"),
                                ("flops_endpoint_forward", "--", "x"),
                                ("flops_endpoint_backward", "-.", "+")):
            if len(g) >= 3:
                comp[col] = _fit_slope(g["width"].to_numpy(float), g[col].to_numpy(float))
            ax.plot(g["width"], g[col], marker=marker, ms=3, ls=ls, lw=0.9,
                    color=color, alpha=0.8 if col == "flops_total" else 0.45,
                    label=(f"{method} total (~n^{comp.get('flops_total', float('nan')):.2f})"
                           if col == "flops_total" else None))
        slopes[method] = comp
        # Explicit check: the argmax backward must not add a power of n.
        if "flops_endpoint_forward" in comp and "flops_endpoint_backward" in comp:
            gap = abs(comp["flops_endpoint_backward"] - comp["flops_endpoint_forward"])
            slopes[method]["backward_extra_power_of_n"] = gap
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n")
    ax.set_ylabel("FLOPs (solid total; dotted kprop; dashed endpoint fwd; dashdot bwd)")
    ax.legend(fontsize=5); ax.grid(alpha=0.25, which="both")
    ax.set_title("FLOP scaling (backward exponent must equal forward exponent)")
    _save(fig, results_dir, "argmax_flops_vs_width")
    (results_dir / "plots" / "argmax_flops_width_slopes.json").write_text(
        json.dumps(slopes, indent=1))
    return slopes


def plot_matched_budget(agg, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, g in agg.groupby("method"):
        g = g.sort_values("width")
        color = g["color"].iloc[0]
        ratios, ws = [], []
        for _, r in g.iterrows():
            if not r["resolved"] or r["mse"] <= 0:
                continue
            m_matched = max(np.floor(r["flops_total"] / r["mc_flops_per_sample"]), 1.0)
            mc_mse = (1.0 - r["collision"]) / m_matched
            ratios.append(mc_mse / r["mse"]); ws.append(r["width"])
        if ws:
            ax.plot(ws, ratios, marker="o", ms=4, ls=_ls(g["projection"].iloc[0]),
                    color=color, label=method)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n")
    ax.set_ylabel("MSE_MC(matched FLOPs) / MSE_deterministic")
    ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.25, which="both")
    ax.set_title("matched-budget Monte Carlo comparison (>1 favors deterministic)")
    _save(fig, results_dir, "argmax_matched_budget_mc")


def plot_simplex_diagnostics(df, results_dir):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    raw = df[df["estimator_projection"] == "raw"]
    for method, g in raw.groupby("method"):
        g = g.groupby("width").agg(
            sr=("simplex_residual", "max"), qmin=("q_min", "min"),
            neg=("num_negative_coordinates", "mean"),
            dist=("raw_projected_l2_distance", "mean"),
        ).reset_index()
        color = df[df["method"] == method]["color"].iloc[0]
        axes[0].plot(g["width"], np.maximum(g["sr"], 1e-18), marker="o", ms=3,
                     color=color, label=method)
        axes[1].plot(g["width"], g["qmin"], marker="o", ms=3, color=color)
        axes[2].plot(g["width"], g["neg"], marker="o", ms=3, color=color)
        axes[3].plot(g["width"], g["dist"], marker="o", ms=3, color=color)
    axes[0].set_yscale("log"); axes[0].set_ylabel("max |sum(q_raw) - 1|")
    axes[1].set_ylabel("min raw coordinate"); axes[1].axhline(0, color="k", lw=0.6)
    axes[2].set_ylabel("mean # negative coordinates")
    axes[3].set_ylabel("mean ||q_raw - q_proj||_2"); axes[3].set_yscale("log")
    for axx in axes:
        axx.set_xscale("log"); axx.set_xlabel("width n"); axx.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=4)
    fig.suptitle("raw-estimate simplex diagnostics", fontsize=9)
    _save(fig, results_dir, "argmax_simplex_diagnostics")


def plot_sampling_diagnostics(df, agg, results_dir):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    per = df.drop_duplicates(subset=["width", "network_seed"])
    axes[0].scatter(per["width"], per["collision_probability_estimate"], s=8)
    axes[0].set_ylabel("collision probability ||q||^2"); axes[0].set_yscale("log")
    axes[1].scatter(per["width"], per["evaluation_total_samples"], s=8)
    axes[1].set_ylabel("evaluation input samples"); axes[1].set_yscale("log")
    for method, g in agg.groupby("method"):
        gg = g.sort_values("width")
        axes[2].plot(gg["width"], gg["input_se"], marker="o", ms=3,
                     color=gg["color"].iloc[0], ls=_ls(gg["projection"].iloc[0]))
    axes[2].set_ylabel("block-level input SE of MSE"); axes[2].set_yscale("log")
    # Signed unbiased MSE per (method,width) plus unresolved fraction.
    for method, g in agg.groupby("method"):
        gg = g.sort_values("width")
        axes[3].plot(gg["width"], gg["mse"], marker="o", ms=3,
                     color=gg["color"].iloc[0], ls=_ls(gg["projection"].iloc[0]))
    axes[3].set_yscale("symlog", linthresh=1e-8)
    axes[3].axhline(0, color="k", lw=0.6)
    unres = 1.0 - agg["resolved"].mean()
    axes[3].set_ylabel(f"signed unbiased MSE (unresolved frac {unres:.2f})")
    for axx in axes:
        axx.set_xscale("log"); axx.set_xlabel("width n"); axx.grid(alpha=0.25, which="both")
    fig.suptitle("input-sampling diagnostics (signed estimates preserved)", fontsize=9)
    _save(fig, results_dir, "argmax_sampling_diagnostics")


def plot_wall_clock(agg, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, g in agg.groupby("method"):
        if g["projection"].iloc[0] != "raw":
            continue
        g = g.sort_values("width")
        ax.plot(g["width"], g["wall"], marker="o", ms=4, label=method,
                color=g["color"].iloc[0])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width n"); ax.set_ylabel("wall seconds (kprop + argmax endpoint)")
    ax.set_title("wall-clock (HARDWARE-DEPENDENT; primary axis is FLOPs)")
    ax.legend(fontsize=5); ax.grid(alpha=0.25, which="both")
    _save(fig, results_dir, "argmax_wall_clock_supplementary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()
    rdir = Path(args.results_dir)
    raw_rows = [json.loads(x) for x in (rdir / "results.jsonl").read_text().splitlines()]
    df = load(rdir)
    if df.empty:
        print("no plottable rows"); return
    agg = aggregate(df, raw_rows)
    agg.to_csv(rdir / "plots_aggregate.csv", index=False)
    plot_mse_vs_flops(agg, rdir, per_coordinate=False)
    plot_mse_vs_flops(agg, rdir, per_coordinate=True)
    slopes = plot_mse_vs_width(agg, raw_rows, rdir)
    fslopes = plot_flops_vs_width(agg, rdir)
    plot_matched_budget(agg, rdir)
    plot_simplex_diagnostics(df, rdir)
    plot_sampling_diagnostics(df, agg, rdir)
    plot_wall_clock(agg, rdir)
    print(json.dumps({"mse_slopes": slopes, "flop_slopes": fslopes}, indent=1))


if __name__ == "__main__":
    main()
