from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from mlp_kprop.flop_utils import Poly2D, fit_flop_polys, get_flops
from mlp_kprop.kprop_harmonic import Kind
from mlp_kprop.plotting_utils import fmt_power_law, setup_colors, wls_loglog

try:
    from form_df import (
        ALL_WIDTHS,
        MAIN_LAYER,
        MAIN_WIDTH,
        PRE_LAYERS,
        form_big_df,
        form_df,
        form_sample_flops,
        form_sample_times,
        load_df,
        save_df,
    )
except ModuleNotFoundError:  # pragma: no cover - import path depends on invocation mode
    from scripts.form_df import (
        ALL_WIDTHS,
        MAIN_LAYER,
        MAIN_WIDTH,
        PRE_LAYERS,
        form_big_df,
        form_df,
        form_sample_flops,
        form_sample_times,
        load_df,
        save_df,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_ROOT = REPO_ROOT / "plots"
K_MAXS = (1, 2, 3, 4)


@dataclass
class PlotResult:
    fig: Any
    width: float
    height: float | None = None


def _configure_plot_style(labelsize: int) -> None:
    plt.rc("font", family="serif", serif="Times")
    plt.rc("text", usetex=(shutil.which("latex") is not None))
    plt.rc("xtick", labelsize=labelsize)
    plt.rc("ytick", labelsize=labelsize)
    plt.rc("axes", labelsize=labelsize)


def _save_plot(
    result: PlotResult,
    *,
    plot_type: str,
    mlp_name: str,
    output_path: Path | None,
) -> Path:
    if result.height is None:
        result.height = result.width / 1.618

    if output_path is None:
        output_path = PLOTS_ROOT / f"{plot_type}_{mlp_name}.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.fig.set_size_inches(result.width, result.height)
    result.fig.savefig(output_path, bbox_inches="tight")
    plt.close(result.fig)
    return output_path


def _parse_layer(layer: str) -> tuple[str, int]:
    if layer.startswith("pre"):
        return "pre", int(layer[3:])
    if layer.startswith("act"):
        return "act", int(layer[3:])
    raise ValueError(f"Unknown layer format: {layer}")


def _layer_sort_key(layer: str) -> float:
    layer_type, idx = _parse_layer(layer)
    return float(idx) if layer_type == "pre" else idx + 0.5


def _main_factor_mask(df: pd.DataFrame) -> pd.Series:
    return ((df["k"] <= 2) & (~df["factor"])) | ((df["k"] > 2) & (df["factor"]))


def _filter_main(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["kprop_name"].isin(["simple", "augment"])]
    return sub[_main_factor_mask(sub)]


def _poly_label(poly: Poly2D, l_val: int, var: str = "n", zero_tol: float = 0.5) -> str:
    c = np.asarray(poly.c, dtype=float)
    dx, dy = poly.degrees

    coeffs_1d: dict[int, float] = {}
    for i in range(dx + 1):
        total = float(sum(c[i, j] * (l_val**j) for j in range(dy + 1)))
        if abs(total) > zero_tol:
            coeffs_1d[i] = total

    if not coeffs_1d:
        return "$0$"

    max_deg = max(coeffs_1d.keys())

    def fmt_monomial(i: int) -> str:
        if i == 0:
            return "1"
        if i == 1:
            return var
        return f"{var}^{i}"

    def fmt_coef(coef: float) -> str:
        rounded = round(coef)
        if abs(coef - rounded) < 0.5:
            return str(int(rounded))
        return f"{coef:.3g}"

    lead_coef = coeffs_1d[max_deg]
    lead_mon = fmt_monomial(max_deg)
    c_str = fmt_coef(abs(lead_coef))
    sign = "-" if lead_coef < 0 else ""
    if c_str == "1" and lead_mon != "1":
        leading = f"{sign}{lead_mon}"
    else:
        leading = f"{sign}{c_str}{lead_mon}"

    remaining = [i for i in sorted(coeffs_1d.keys(), reverse=True) if i < max_deg]
    if remaining:
        return f"${leading} + O({fmt_monomial(remaining[0])})$"
    return f"${leading}$"


def _plot_mse_vs_flops_main(payload: dict[str, Any], args: argparse.Namespace) -> PlotResult:
    agg = form_df(
        args.n,
        args.layer,
        mlp_name=args.mlp_name,
        kprop_names=["simple", "augment"],
        aggregate=True,
        payload=payload,
    )
    df = _filter_main(agg)
    sample_flops = form_sample_flops(df)

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    if not sample_flops.empty:
        ax.plot(sample_flops["flops"], sample_flops["mse_over_var"], color="red", label="sampling")

    simple = df[df["kprop_name"] == "simple"]
    augment = df[df["kprop_name"] == "augment"]
    if not simple.empty:
        ax.scatter(simple["flops"], simple["mse_over_var_mean"], label="kprop (simple)", s=20, color="lightskyblue", edgecolors="none")
    if not augment.empty:
        ax.scatter(augment["flops"], augment["mse_over_var_mean"], label="kprop (augment)", s=20, color="darkblue", edgecolors="none")

    if args.annotate_k:
        for _, row in df.iterrows():
            ax.annotate(
                f"k={int(row['k'])}",
                (row["flops"], row["mse_over_var_mean"]),
                fontsize=6,
                textcoords="offset points",
                xytext=(4, 2),
            )

    if not df.empty:
        ax.set_xlim(left=float(df["mlp_flops"].iloc[0]))
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("MSE / Var")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"{args.mlp_name}, layer={args.layer}, width={args.n}")
    return PlotResult(fig=fig, width=3.487)


def _plot_mse_vs_flops_pk_ablate(payload: dict[str, Any], args: argparse.Namespace) -> PlotResult:
    agg = form_df(
        args.n,
        args.layer,
        mlp_name=args.mlp_name,
        kprop_names=["simple", "augment", "pK_ablate"],
        aggregate=True,
        payload=payload,
    )
    main_df = _filter_main(agg)
    sample_flops = form_sample_flops(main_df if not main_df.empty else agg)

    ablate = agg[agg["kprop_name"] == "pK_ablate"].copy()
    if ablate.empty:
        raise ValueError(f"No pK_ablate data for mlp_name={args.mlp_name!r}")

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    if not sample_flops.empty:
        ax.plot(sample_flops["flops"], sample_flops["mse_over_var"], color="red", label="sampling")

    simple = main_df[main_df["kprop_name"] == "simple"]
    augment = main_df[main_df["kprop_name"] == "augment"]
    if not simple.empty:
        ax.scatter(simple["flops"], simple["mse_over_var_mean"], label="kprop (simple)", s=20, color="lightskyblue", edgecolors="none")
    if not augment.empty:
        ax.scatter(augment["flops"], augment["mse_over_var_mean"], label="kprop (augment)", s=20, color="darkblue", edgecolors="none")

    ax.scatter(ablate["flops"], ablate["mse_over_var_mean"], label="kprop (no pK)", s=20, color="green", edgecolors="none")

    if args.annotate_k:
        for _, row in ablate.iterrows():
            ax.annotate(
                f"k={int(row['k'])}",
                (row["flops"], row["mse_over_var_mean"]),
                fontsize=6,
                textcoords="offset points",
                xytext=(4, 2),
            )

    left = float(main_df["mlp_flops"].iloc[0]) if not main_df.empty else float(ablate["mlp_flops"].iloc[0])
    ax.set_xlim(left=left)
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("MSE / Var")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"{args.mlp_name}, pK ablate, layer={args.layer}, width={args.n}")
    return PlotResult(fig=fig, width=3.487)


def _plot_mse_vs_time(payload: dict[str, Any], args: argparse.Namespace) -> PlotResult:
    agg = form_df(
        args.n,
        args.layer,
        mlp_name=args.mlp_name,
        kprop_names=["simple", "augment"],
        aggregate=True,
        payload=payload,
    )
    df = _filter_main(agg)
    sample_times = form_sample_times(
        args.n,
        args.layer,
        mlp_name=args.mlp_name,
        payload=payload,
    )

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    ax.plot(sample_times["time"], sample_times["mse_over_var"], color="red", label="sampling")

    simple = df[df["kprop_name"] == "simple"]
    augment = df[df["kprop_name"] == "augment"]
    if not simple.empty:
        ax.scatter(simple["time"], simple["mse_over_var_mean"], label="kprop (simple)", s=20, color="lightskyblue", edgecolors="none")
    if not augment.empty:
        ax.scatter(augment["time"], augment["mse_over_var_mean"], label="kprop (augment)", s=20, color="darkblue", edgecolors="none")

    if args.annotate_k:
        for _, row in df.iterrows():
            ax.annotate(
                f"k={int(row['k'])}",
                (row["time"], row["mse_over_var_mean"]),
                fontsize=6,
                textcoords="offset points",
                xytext=(4, 2),
            )

    ax.set_xlim(left=float(sample_times["time"].min()))
    ax.set_xlabel("Wall Clock Time (s)")
    ax.set_ylabel("MSE / Var")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"{args.mlp_name}, layer={args.layer}, width={args.n}")
    return PlotResult(fig=fig, width=3.487)


def _plot_mse_vs_width(payload: dict[str, Any], args: argparse.Namespace) -> PlotResult:
    agg = form_big_df(
        args.mlp_name,
        widths=args.widths,
        layers=[args.layer],
        kprop_names=["simple"],
        aggregate=True,
        payload=payload,
    )

    df = agg[agg["kprop_name"] == "simple"].copy()
    df = df[_main_factor_mask(df)]

    baseline = agg[
        (agg["kprop_name"] == "simple")
        & (agg["factor"] == False)
        & (agg["l"] == args.layer)
    ].copy()
    baseline["sampling_mse"] = baseline["mlp_var_mean"] * baseline["mlp_flops"] / baseline["flops"]

    fig, axes = plt.subplots(1, len(K_MAXS), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    cmap, norm, _ = setup_colors(K_MAXS, "Blues")

    for ci, k in enumerate(K_MAXS):
        ax = axes[ci]
        sub = df[df["k"] == k].sort_values("n")
        if sub.empty:
            ax.set_visible(False)
            continue

        xs = sub["n"].to_numpy(dtype=float)
        ys = sub["mse_mean"].to_numpy(dtype=float)
        yerrs = sub["mse_sem"].to_numpy(dtype=float)
        color = cmap(norm(k))

        ax.errorbar(xs, ys, yerr=yerrs, color=color, lw=1.2, fmt="-", ms=0)

        c, b = wls_loglog(xs, ys, w="exp")
        if c is not None:
            label = fmt_power_law(c, b)
            xp = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 256)
            ax.plot(xp, c * xp**b, ls="--", color="black", lw=0.8, label=label)

        samp = baseline[baseline["k"] == k].sort_values("n")
        if not samp.empty:
            ax.plot(
                samp["n"],
                samp["sampling_mse"],
                ls="-",
                color="red",
                lw=0.8,
                alpha=0.5,
                label="sampling (same FLOPs)",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(left=12)
        ax.set_title(f"$k_{{\\max}}={k}$")
        ax.grid(True, which="both", alpha=0.15)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=6)
        if ci == 0:
            ax.set_ylabel("MSE")

    fig.supxlabel("width")
    fig.tight_layout()
    return PlotResult(fig=fig, width=14, height=3.0)


def _plot_mse_vs_layer(payload: dict[str, Any], args: argparse.Namespace) -> PlotResult:
    agg = form_big_df(
        args.mlp_name,
        widths=[args.n],
        layers=None,
        kprop_names=["simple"],
        aggregate=True,
        payload=payload,
    )

    df = agg[agg["kprop_name"] == "simple"].copy()
    df = df[_main_factor_mask(df)]

    if args.layer_type in ("pre", "act"):
        df = df[df["l"].str.startswith(args.layer_type)]

    if args.layer_type == "all":
        df["layer_x"] = df["l"].map(_layer_sort_key)
    else:
        df["layer_x"] = df["l"].str.extract(r"(\d+)").astype(int)

    df = df[df["layer_x"] >= 2]

    fig, ax = plt.subplots()
    cmap, norm, _ = setup_colors(K_MAXS, "Blues")

    for k in K_MAXS:
        sub = df[df["k"] == k].sort_values("layer_x")
        if sub.empty:
            continue

        xs = sub["layer_x"].to_numpy(dtype=float)
        ys = sub["mse_mean"].to_numpy(dtype=float)
        yerrs = sub["mse_sem"].to_numpy(dtype=float)
        color = cmap(norm(k))

        fit_txt = ""
        c, b = wls_loglog(xs, ys, w="unif")
        if c is not None:
            inner = fmt_power_law(c, b, varname="l", sci=True)
            fit_txt = f"\\;({inner.strip('$')})"
            xp = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 256)
            ax.plot(xp, c * xp**b, ls="--", color=color, lw=0.8, alpha=0.6)

        ax.errorbar(
            xs,
            ys,
            yerr=yerrs,
            color=color,
            lw=1.2,
            fmt="-",
            ms=0,
            label=f"$k_{{\\max}}={k}{fit_txt}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{args.layer_type} layer index")
    ax.set_ylabel("MSE")
    ax.grid(True, which="both", axis="x", alpha=0.15)
    ax.grid(True, which="major", axis="y", alpha=0.15)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return PlotResult(fig=fig, width=5, height=3.0)


def _plot_flops_vs_width(payload: dict[str, Any], args: argparse.Namespace) -> PlotResult:
    agg = form_big_df(
        args.mlp_name,
        widths=args.widths,
        layers=[args.layer],
        kprop_names=["simple"],
        aggregate=True,
        payload=payload,
    )

    df = agg[agg["kprop_name"] == "simple"].copy()
    ns = np.array(sorted(df["n"].unique()), dtype=float)
    xp = np.logspace(np.log10(ns.min()), np.log10(ns.max()), 256)

    layer_type, layer_idx = _parse_layer(args.layer)
    act_idx = layer_idx - 1 if layer_type == "pre" else layer_idx

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for ci, k in enumerate((3, 4)):
        ax = axes[ci]
        for factor, color in ((False, "C0"), (True, "C1")):
            sub = df[(df["k"] == k) & (df["factor"] == factor)].sort_values("n")
            if sub.empty:
                continue

            kprop_kw = dict(k_max=k, factor=factor, kind=Kind.SIMPLE, use_avg_metric=False)
            flops_smooth = get_flops(xp, args.layer, {}, kprop_kw)
            poly = fit_flop_polys({}, kprop_kw)["total"]
            poly_label = _poly_label(poly, l_val=act_idx)

            factor_label = "factored" if factor else "unfactored"
            ax.scatter(sub["n"], sub["flops"], color=color, s=15, zorder=3, label=factor_label)
            ax.plot(xp, flops_smooth, ls=":", color=color, lw=0.8, alpha=0.7, label=poly_label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(left=12)
        ax.set_title(f"$k_{{\\max}}={k}$, layer={args.layer}")
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(fontsize=5)
        if ci == 0:
            ax.set_ylabel("FLOPs")

    fig.supxlabel("width")
    fig.tight_layout()
    return PlotResult(fig=fig, width=7, height=3.0)


def _plot_mse_vs_flops_big(payload: dict[str, Any], args: argparse.Namespace) -> PlotResult:
    widths = tuple(args.widths)
    layers = tuple(args.layers)

    agg = form_big_df(
        args.mlp_name,
        widths=widths,
        layers=layers,
        kprop_names=None,
        aggregate=True,
        payload=payload,
    ).copy()

    agg["label"] = agg.apply(
        lambda row: f"{row['kprop_name']}{' fact.' if row['factor'] else ''}",
        axis=1,
    )

    labels = sorted(agg["label"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(labels)))
    label_colors = {label: cmap(i) for i, label in enumerate(labels)}

    fig, axes = plt.subplots(len(layers), len(widths), sharex="col", sharey="row", squeeze=False)

    for ri, layer in enumerate(layers):
        for ci, n in enumerate(widths):
            ax = axes[ri, ci]
            sub = agg[(agg["n"] == n) & (agg["l"] == layer)]
            if sub.empty:
                ax.set_visible(False)
                continue

            for label in labels:
                pts = sub[sub["label"] == label]
                if pts.empty:
                    continue
                ax.scatter(pts["flops"], pts["mse_over_var_mean"], s=6, color=label_colors[label], alpha=0.7)

            mlp_flops = float(sub["mlp_flops"].iloc[0])
            x_range = np.array([sub["flops"].min() * 0.5, sub["flops"].max() * 2.0])
            ax.plot(x_range, mlp_flops / x_range, color="red", lw=0.6, alpha=0.5)

            frontier = sub.sort_values("flops")
            frontier = frontier[frontier["mse_over_var_mean"].cummin() == frontier["mse_over_var_mean"]]
            ax.step(frontier["flops"], frontier["mse_over_var_mean"], where="post", color="green", alpha=0.6, lw=0.6)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.1)
            ax.tick_params(labelsize=4)

            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.yaxis.set_minor_formatter(NullFormatter())

            if ri == 0:
                ax.set_title(f"n={n}", fontsize=7)
            if ci == len(widths) - 1:
                ax.set_ylabel(layer, rotation=-90, labelpad=8, fontsize=6)
                ax.yaxis.set_label_position("right")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=label_colors[label], ms=5, label=label)
        for label in labels
    ]
    handles.append(Line2D([0], [0], color="red", lw=0.8, alpha=0.5, label="sampling"))
    handles.append(Line2D([0], [0], color="green", lw=0.8, alpha=0.6, label="frontier"))

    fig.legend(handles=handles, loc="lower right", ncol=3, fontsize=7, bbox_to_anchor=(0.96, 0.005), frameon=False)

    fig.text(0.005, 0.5, "MSE / Var", rotation=90, va="center", ha="left", fontsize=10)
    fig.text(0.995, 0.5, "layer", rotation=-90, va="center", ha="right", fontsize=10)
    fig.text(0.5, 0.008, "FLOPs", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.03, top=0.98, hspace=0.08, wspace=0.08)
    return PlotResult(fig=fig, width=30, height=40)


PLOTS: dict[str, Callable[[dict[str, Any], argparse.Namespace], PlotResult]] = {
    "mse_vs_flops_main": _plot_mse_vs_flops_main,
    "mse_vs_flops_pK_ablate": _plot_mse_vs_flops_pk_ablate,
    "mse_vs_width": _plot_mse_vs_width,
    "mse_vs_layer": _plot_mse_vs_layer,
    "flops_vs_width": _plot_flops_vs_width,
    "mse_vs_flops_big": _plot_mse_vs_flops_big,
    "mse_vs_time": _plot_mse_vs_time,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from formed dataframe.")
    parser.add_argument("type_of_plot", choices=sorted(PLOTS.keys()))
    parser.add_argument("mlp_name", help="MLP data key under data/<mlp_name>/")

    parser.add_argument("--n", type=int, default=MAIN_WIDTH, help=f"Width for single-plot modes (default: {MAIN_WIDTH})")
    parser.add_argument("--layer", type=str, default=MAIN_LAYER, help=f"Layer for single-plot modes (default: {MAIN_LAYER})")
    parser.add_argument("--layer-type", choices=["pre", "act", "all"], default="pre", help="For mse_vs_layer")

    parser.add_argument("--widths", type=int, nargs="*", default=list(ALL_WIDTHS), help="Widths for grid/width plots")
    parser.add_argument("--layers", type=str, nargs="*", default=list(PRE_LAYERS), help="Layers for grid/layer plots")

    parser.add_argument("--annotate-k", action="store_true", help="Annotate k values on scatter plots")
    parser.add_argument("--formed", type=Path, default=None, help="Path to formed df payload")
    parser.add_argument("--refresh-formed", action="store_true", help="Rebuild formed df before plotting")
    parser.add_argument("--output", type=Path, default=None, help="Output PDF path")
    parser.add_argument("--labelsize", type=int, default=8)
    args = parser.parse_args()

    _configure_plot_style(args.labelsize)

    if args.refresh_formed:
        save_df(args.mlp_name, output_path=args.formed)

    payload = load_df(args.mlp_name, path=args.formed, build_if_missing=True)

    plot_fn = PLOTS[args.type_of_plot]
    result = plot_fn(payload, args)
    out = _save_plot(
        result,
        plot_type=args.type_of_plot,
        mlp_name=args.mlp_name,
        output_path=args.output,
    )
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(name)s.%(funcName)s | %(message)s",
        force=True,
    )
    main()
