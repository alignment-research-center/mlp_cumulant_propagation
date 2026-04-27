from __future__ import annotations

import __main__
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from mlp_kprop.flop_utils import NamedFlopCounter, get_flops
from mlp_kprop.kprop_harmonic import Kind
from mlp_kprop.time_utils import get_kprop_time, get_sampling_time

try:
    from kprop_by_width import ALL_WIDTHS, KPropByWidthCfg, get_sample_cumulants  # noqa: F401 (torch.load unpickling)
except ModuleNotFoundError:  # pragma: no cover - import path depends on invocation mode
    from scripts.kprop_by_width import ALL_WIDTHS, KPropByWidthCfg, get_sample_cumulants  # noqa: F401

# Some legacy caches were written when kprop_by_width.py was executed as __main__.
if not hasattr(__main__, "KPropByWidthCfg"):
    __main__.KPropByWidthCfg = KPropByWidthCfg

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

PRE_LAYERS: tuple[str, ...] = tuple(f"pre{i}" for i in range(2, 16))
MAIN_WIDTH = 256
MAIN_LAYER = "pre4"
MLP_VAR_SAMPLES = 1_000_000
MLP_VAR_BATCH_SIZE = 100_000


def _parse_layer(layer: str) -> tuple[str, int]:
    if layer.startswith("pre"):
        return "pre", int(layer[3:])
    if layer.startswith("act"):
        return "act", int(layer[3:])
    raise ValueError(f"Unknown layer format: {layer}")


def _layer_sort_key(layer: str) -> float:
    layer_type, idx = _parse_layer(layer)
    return float(idx) if layer_type == "pre" else idx + 0.5


def _kind_name(kind: Kind | str) -> str:
    if isinstance(kind, Kind):
        return kind.name
    return str(kind).upper()


def _normalize_kprop_name(name: str) -> str:
    name = str(name)
    if name == "pK_ablate":
        return "pK_ablate"
    return name.lower()


def _kprop_name_from_cfg(raw_kprop_name: str, cfg: Any) -> str:
    kprop_kwargs = dict(cfg.kprop_kwargs)
    kind_name = _kind_name(kprop_kwargs.get("kind", Kind.SIMPLE))
    use_avg_metric = bool(kprop_kwargs.get("use_avg_metric", False))
    use_pK = bool(kprop_kwargs.get("use_pK", True))

    if kind_name == "BASE":
        if not use_pK:
            return "pK_ablate"
        return "base_avgmetric" if use_avg_metric else "base"

    if kind_name == "SIMPLE":
        base_name = "simple"
    elif kind_name == "AUGMENT":
        base_name = "augment"
    else:
        base_name = _normalize_kprop_name(raw_kprop_name)

    if use_avg_metric:
        base_name = f"{base_name}_avgmetric"
    if not use_pK:
        base_name = f"{base_name}_nopk"
    return base_name


def _kprop_meta_from_cfg(cfg: Any) -> dict[str, Any]:
    kprop_kwargs = dict(cfg.kprop_kwargs)
    return {
        "kprop_kind": _kind_name(kprop_kwargs.get("kind", Kind.SIMPLE)).lower(),
        "use_avg_metric": bool(kprop_kwargs.get("use_avg_metric", False)),
        "use_pK": bool(kprop_kwargs.get("use_pK", True)),
    }


def _normalize_mlp_kwargs(raw: dict, *, keep_num_layers: bool) -> dict:
    ignore = {"hidden_dim", "input_dim", "output_dim"}
    if not keep_num_layers:
        ignore = ignore | {"num_layers"}
    return {k: v for k, v in dict(raw).items() if k not in ignore}


def _merge_mlp_kwargs(cfg_by_kprop_name: dict[str, Any], *, keep_num_layers: bool) -> dict:
    merged: dict[str, Any] = {}
    values_by_key: dict[str, list[Any]] = {}
    for cfg in cfg_by_kprop_name.values():
        cfg_kwargs = _normalize_mlp_kwargs(cfg.mlp_kwargs, keep_num_layers=keep_num_layers)
        for key, val in cfg_kwargs.items():
            values_by_key.setdefault(key, []).append(val)

    for key, values in values_by_key.items():
        ref = values[0]
        if any(v != ref for v in values[1:]):
            raise ValueError(
                f"Inconsistent mlp_kwargs[{key!r}] across kprop names: {values}"
            )
        merged[key] = ref

    if "nonlin" not in merged:
        merged["nonlin"] = "relu"
    return merged


def _discover_kprop_files(mlp_name: str) -> list[Path]:
    root = DATA_ROOT / mlp_name
    if not root.exists():
        raise FileNotFoundError(f"No data directory at {root}")
    return sorted(root.glob("*/kprop_by_width.pt"))


def _load_cfg_and_metrics(
    mlp_name: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, str]]:
    files = _discover_kprop_files(mlp_name)
    if not files:
        raise FileNotFoundError(f"No kprop_by_width.pt files under {DATA_ROOT / mlp_name}")

    cfg_by_kprop_name: dict[str, Any] = {}
    errs_by_kprop_name: dict[str, pd.DataFrame] = {}
    raw_name_by_kprop_name: dict[str, str] = {}

    for path in files:
        raw_kprop_name = path.parent.name
        cfg, errs, _ = torch.load(path, weights_only=False, map_location="cpu")
        kprop_name = _kprop_name_from_cfg(raw_kprop_name, cfg)

        if kprop_name in cfg_by_kprop_name:
            prev_raw = raw_name_by_kprop_name[kprop_name]
            raise ValueError(
                f"Duplicate canonical kprop_name={kprop_name!r} from raw names {prev_raw!r} and {raw_kprop_name!r}"
            )

        cfg_by_kprop_name[kprop_name] = cfg
        errs_by_kprop_name[kprop_name] = errs
        raw_name_by_kprop_name[kprop_name] = raw_kprop_name

    return cfg_by_kprop_name, errs_by_kprop_name, raw_name_by_kprop_name


def _default_kprop_names(cfg_by_kprop_name: dict[str, Any]) -> list[str]:
    order = {
        "simple": 0,
        "simple_avgmetric": 1,
        "augment": 2,
        "augment_avgmetric": 3,
        "base": 4,
        "base_avgmetric": 5,
        "pK_ablate": 6,
    }
    return sorted(cfg_by_kprop_name.keys(), key=lambda name: (order.get(name, 999), name))


def _build_mse_df(
    cfg_by_kprop_name: dict[str, Any],
    errs_by_kprop_name: dict[str, pd.DataFrame],
    *,
    kprop_names: Iterable[str] | None,
) -> pd.DataFrame:
    if kprop_names is None:
        keep_names = set(_default_kprop_names(cfg_by_kprop_name))
    else:
        keep_names = {_normalize_kprop_name(name) for name in kprop_names}

    rows: list[pd.DataFrame] = []
    for kprop_name, errs in errs_by_kprop_name.items():
        if kprop_name not in keep_names:
            continue

        cfg = cfg_by_kprop_name[kprop_name]
        kprop_meta = _kprop_meta_from_cfg(cfg)

        part = errs.copy()
        if "err" in part.columns and "mse" not in part.columns:
            part = part.rename(columns={"err": "mse"})
        if "t" in part.columns and "seed" not in part.columns:
            part = part.rename(columns={"t": "seed"})

        required = {"n", "seed", "l", "d", "k", "mse"}
        missing = required - set(part.columns)
        if missing:
            raise KeyError(f"Missing columns in {kprop_name}: {sorted(missing)}")

        part = part[["n", "seed", "l", "d", "k", "mse"]].copy()
        part["seed"] = part["seed"].astype(int) + int(cfg.base_seed)
        part["n"] = part["n"].astype(int)
        part["k"] = part["k"].astype(int)

        # Keep only k>=1 and degree-1 cumulants for the current plot stack.
        part = part[(part["k"] >= 1) & (part["d"] == "1")].copy()

        part["kprop_name"] = kprop_name
        part["kprop_kind"] = kprop_meta["kprop_kind"]
        part["use_avg_metric"] = kprop_meta["use_avg_metric"]
        part["use_pK"] = kprop_meta["use_pK"]
        rows.append(part)

    if not rows:
        return pd.DataFrame(
            columns=[
                "n",
                "seed",
                "l",
                "d",
                "k",
                "mse",
                "kprop_name",
                "kprop_kind",
                "use_avg_metric",
                "use_pK",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _mlp_path_for_cfg(cfg: Any, n: int, seed: int) -> Path:
    return DATA_ROOT / str(cfg.sampK_cache_name) / f"seed{int(seed)}" / f"n{int(n)}" / "mlp.pt"


def _compute_mlp_vars(
    mse_df: pd.DataFrame,
    cfg_by_kprop_name: dict[str, Any],
    *,
    samples: int = MLP_VAR_SAMPLES,
    batch_size: int = MLP_VAR_BATCH_SIZE,
) -> pd.DataFrame:
    """Estimate per-layer MLP variance from explicit sampling.

    This matches the legacy plotting behavior (rather than inferring variance
    from bootstrap stderr caches).
    """
    rows: list[tuple[str, int, int, str, float]] = []
    grouped = mse_df.groupby(["n", "seed"], sort=True)
    for (n, seed), group in grouped:
        kprop_names = sorted(group["kprop_name"].unique())
        layers = sorted(group["l"].unique(), key=_layer_sort_key)

        mlp = None
        for kprop_name in kprop_names:
            cfg = cfg_by_kprop_name[kprop_name]
            mlp_path = _mlp_path_for_cfg(cfg, n=int(n), seed=int(seed))
            if not mlp_path.exists():
                continue
            mlp = torch.load(mlp_path, weights_only=False, map_location=device)
            break

        if mlp is None:
            raise ValueError(f"No MLP cache found for n={n} seed={seed}")

        K, _ = get_sample_cumulants(
            mlp,
            samples=samples,
            batch_size=batch_size,
            k_max=1,
        )
        for layer in layers:
            if layer not in K:
                continue
            try:
                var = float(K[layer].get_slice((2,)).mean().item())
            except KeyError:
                continue
            for kprop_name in kprop_names:
                rows.append((str(kprop_name), int(n), int(seed), str(layer), var))

    return pd.DataFrame(rows, columns=["kprop_name", "n", "seed", "l", "mlp_var"])


def _compute_mlp_flops(mse_df: pd.DataFrame, cfg_by_kprop_name: dict[str, Any]) -> pd.DataFrame:
    rows: list[tuple[int, str, float]] = []

    for n in sorted(mse_df["n"].unique()):
        sub = mse_df[mse_df["n"] == n]
        layers = sorted(sub["l"].unique(), key=_layer_sort_key)

        mlp = None
        for candidate in sub[["kprop_name", "seed"]].drop_duplicates().itertuples(index=False):
            cfg = cfg_by_kprop_name[str(candidate.kprop_name)]
            mlp_path = _mlp_path_for_cfg(cfg, n=int(n), seed=int(candidate.seed))
            if not mlp_path.exists():
                continue
            mlp = torch.load(mlp_path, weights_only=False, map_location=device)
            break

        if mlp is None:
            raise ValueError(f"No MLP cache found for width n={n}")

        for layer in layers:
            with NamedFlopCounter() as counter:
                x = torch.randn(128, int(n), device=device)
                mlp(x, up_to_layer=layer)
            rows.append((int(n), str(layer), float(counter.total() / 128)))

    return pd.DataFrame(rows, columns=["n", "l", "mlp_flops"])


def _kprop_kwargs_templates(cfg_by_kprop_name: dict[str, Any]) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for kprop_name, cfg in cfg_by_kprop_name.items():
        template = dict(cfg.kprop_kwargs)
        template.pop("k_max", None)
        templates[kprop_name] = template
    return templates


def _kprop_kwargs_for_name(
    kprop_name: str,
    k: int,
    factor: bool,
    *,
    template: dict[str, Any],
) -> dict[str, Any] | None:
    kind_name = _kind_name(template.get("kind", Kind.SIMPLE))

    if kind_name in ("SIMPLE", "AUGMENT") and factor and int(k) < 3:
        return None

    # Current pipeline compares base-like methods in factored mode only.
    if kind_name == "BASE" and not factor:
        return None

    kwargs = dict(template)
    kwargs["k_max"] = int(k)
    kwargs["factor"] = bool(factor)
    if "use_pK" not in kwargs:
        kwargs["use_pK"] = True
    if "use_avg_metric" not in kwargs:
        kwargs["use_avg_metric"] = False
    return kwargs


def _pick_output_d_max(cfg_by_kprop_name: dict[str, Any]) -> int | None:
    vals = sorted({cfg.output_d_max for cfg in cfg_by_kprop_name.values()})
    if not vals:
        return None
    if len(vals) > 1:
        logger.warning("Multiple output_d_max values found (%s); using %s", vals, vals[0])
    return vals[0]


def _compute_flops_and_times(
    mse_df: pd.DataFrame,
    *,
    timing_mlp_kwargs: dict,
    output_d_max: int | None,
    kprop_kwargs_templates: dict[str, dict[str, Any]],
    compute_times: bool=True,
) -> pd.DataFrame:
    settings = mse_df[["n", "l", "k", "kprop_name"]].drop_duplicates()

    rows: list[tuple[int, str, int, str, bool, float, float]] = []
    for row in settings.itertuples(index=False):
        template = kprop_kwargs_templates[str(row.kprop_name)]
        for factor in (False, True):
            kprop_kwargs = _kprop_kwargs_for_name(
                kprop_name=str(row.kprop_name),
                k=int(row.k),
                factor=factor,
                template=template,
            )
            if kprop_kwargs is None:
                continue

            try:
                flops = float(get_flops(int(row.n), str(row.l), timing_mlp_kwargs, kprop_kwargs))
            except KeyError:
                continue
            
            if compute_times:
                wall_time = float(get_kprop_time(int(row.n), str(row.l), timing_mlp_kwargs, kprop_kwargs))
            else:
                wall_time = float('nan')

            rows.append(
                (
                    int(row.n),
                    str(row.l),
                    int(row.k),
                    str(row.kprop_name),
                    bool(factor),
                    flops,
                    wall_time,
                )
            )

    return pd.DataFrame(
        rows,
        columns=["n", "l", "k", "kprop_name", "factor", "flops", "time"],
    )


def aggregate_over_seed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    stat_cols = {"seed", "mse", "mlp_var", "mse_over_var"}
    group_cols = [c for c in df.columns if c not in stat_cols]

    return (
        df.groupby(group_cols)
        .agg(
            mse_mean=("mse", "mean"),
            mse_sem=("mse", "sem"),
            mlp_var_mean=("mlp_var", "mean"),
            mlp_var_sem=("mlp_var", "sem"),
            mse_over_var_mean=("mse_over_var", "mean"),
            mse_over_var_sem=("mse_over_var", "sem"),
        )
        .reset_index()
    )


def build_raw_df(mlp_name: str, *, kprop_names: Iterable[str] | None = None, compute_times: bool=True) -> tuple[pd.DataFrame, dict[str, Any]]:
    (
        cfg_by_kprop_name,
        errs_by_kprop_name,
        raw_name_by_kprop_name,
    ) = _load_cfg_and_metrics(mlp_name)

    mse_df = _build_mse_df(cfg_by_kprop_name, errs_by_kprop_name, kprop_names=kprop_names)

    timing_mlp_kwargs = _merge_mlp_kwargs(cfg_by_kprop_name, keep_num_layers=False)
    sampling_mlp_kwargs = _merge_mlp_kwargs(cfg_by_kprop_name, keep_num_layers=True)
    output_d_max = _pick_output_d_max(cfg_by_kprop_name)
    kprop_templates = _kprop_kwargs_templates(cfg_by_kprop_name)

    mlp_var_df = _compute_mlp_vars(mse_df, cfg_by_kprop_name)
    df = mse_df.merge(mlp_var_df, on=["kprop_name", "n", "seed", "l"], how="left")
    df["mse_over_var"] = df["mse"] / df["mlp_var"]

    mlp_flops_df = _compute_mlp_flops(df, cfg_by_kprop_name)
    df = df.merge(mlp_flops_df, on=["n", "l"], how="left")

    flop_time_df = _compute_flops_and_times(
        df,
        timing_mlp_kwargs=timing_mlp_kwargs,
        output_d_max=output_d_max,
        kprop_kwargs_templates=kprop_templates,
        compute_times=compute_times,
    )
    if not compute_times:
        flop_time_df.drop(columns='time', inplace=True)
    df = df.merge(flop_time_df, on=["n", "l", "k", "kprop_name"], how="inner")

    df["mlp_name"] = mlp_name

    meta = {
        "mlp_name": mlp_name,
        "default_kprop_names": _default_kprop_names(cfg_by_kprop_name),
        "timing_mlp_kwargs": timing_mlp_kwargs,
        "sampling_mlp_kwargs": sampling_mlp_kwargs,
        "output_d_max": output_d_max,
        "kprop_names": sorted(cfg_by_kprop_name.keys()),
        "raw_name_by_kprop_name": raw_name_by_kprop_name,
        "cfg_by_kprop_name": {
            name: {
                "base_seed": int(cfg.base_seed),
                "samples": int(cfg.samples),
                "sample_k_max": int(cfg.sample_k_max),
                "output_d_max": cfg.output_d_max,
                "sampK_cache_name": str(cfg.sampK_cache_name),
                "mlp_kwargs": dict(cfg.mlp_kwargs),
                "kprop_kwargs": dict(cfg.kprop_kwargs),
            }
            for name, cfg in cfg_by_kprop_name.items()
        },
    }
    return df, meta


def _default_df_path(mlp_name: str, compute_times: bool=True) -> Path:
    if compute_times:
        return DATA_ROOT / mlp_name / "formed_df.pt"
    else:
        return DATA_ROOT / mlp_name / "formed_df_notime.pt"


def save_df(
    mlp_name: str,
    *,
    output_path: Path | None = None,
    kprop_names: Iterable[str] | None = None,
    compute_times: bool = True,
) -> Path:
    df, meta = build_raw_df(mlp_name, kprop_names=kprop_names, compute_times=compute_times)
    df_agg = aggregate_over_seed(df)

    if output_path is None:
        output_path = _default_df_path(mlp_name, compute_times=compute_times)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"df": df, "df_agg": df_agg, "meta": meta}, output_path)
    return output_path


def load_df(
    mlp_name: str,
    *,
    path: Path | None = None,
    build_if_missing: bool = False,
    compute_times: bool=True,
) -> dict[str, Any]:
    if path is None:
        path = _default_df_path(mlp_name, compute_times=compute_times)

    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(
                f"Formed df not found at {path}. Run: python scripts/form_df.py {mlp_name}"
            )
        save_df(mlp_name, output_path=path, compute_times=compute_times)

    payload = torch.load(path, weights_only=False, map_location="cpu")
    if "df" not in payload or "meta" not in payload:
        raise KeyError(f"Malformed payload at {path}; expected keys ['df', 'meta']")
    if "df_agg" not in payload:
        payload["df_agg"] = aggregate_over_seed(payload["df"])
    return payload


def _subset_df(
    df: pd.DataFrame,
    *,
    widths: Iterable[int] | None = None,
    layers: Iterable[str] | None = None,
    kprop_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    out = df
    if widths is not None:
        width_set = {int(n) for n in widths}
        out = out[out["n"].isin(width_set)]
    if layers is not None:
        layer_set = {str(l) for l in layers}
        out = out[out["l"].isin(layer_set)]
    if kprop_names is not None:
        name_set = {_normalize_kprop_name(name) for name in kprop_names}
        out = out[out["kprop_name"].isin(name_set)]
    return out


def form_big_df(
    mlp_name: str,
    *,
    widths: Iterable[int] | None = ALL_WIDTHS,
    layers: Iterable[str] | None = PRE_LAYERS,
    kprop_names: Iterable[str] | None = None,
    aggregate: bool = True,
    payload: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if payload is None:
        payload = load_df(mlp_name, build_if_missing=True)
    df = payload["df"]
    out = _subset_df(df, widths=widths, layers=layers, kprop_names=kprop_names)
    return aggregate_over_seed(out) if aggregate else out.reset_index(drop=True)


def form_df(
    n: int,
    layer: str,
    *,
    mlp_name: str = "relu",
    kprop_names: Iterable[str] | None = None,
    aggregate: bool = True,
    payload: dict[str, Any] | None = None,
) -> pd.DataFrame:
    return form_big_df(
        mlp_name,
        widths=(int(n),),
        layers=(str(layer),),
        kprop_names=kprop_names,
        aggregate=aggregate,
        payload=payload,
    )


def form_sample_flops(df: pd.DataFrame, *, x_min_scale: float = 0.8, x_max_scale: float = 1.2) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["flops", "mse_over_var"])

    mlp_flops = float(df["mlp_flops"].iloc[0])
    x = np.array([df["flops"].min() * x_min_scale, df["flops"].max() * x_max_scale])
    return pd.DataFrame({"flops": x, "mse_over_var": mlp_flops / x})


def form_sample_times(
    n: int,
    layer: str,
    *,
    mlp_name: str,
    sample_exponents: Iterable[int] = range(31),
    payload: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if payload is None:
        payload = load_df(mlp_name, build_if_missing=True)
    meta = payload["meta"]
    mlp_kwargs = dict(meta.get("sampling_mlp_kwargs", {}))

    rows = []
    for exp in sample_exponents:
        n_samples = 2 ** int(exp)
        wall_time = float(get_sampling_time(int(n), str(layer), int(n_samples), mlp_kwargs=mlp_kwargs))
        rows.append({"samples": n_samples, "time": wall_time, "mse_over_var": 1.0 / n_samples})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified dataframe for plotting.")
    parser.add_argument("mlp_name", type=str, help="Directory name under data/ (e.g. relu, sigmoid, tanh_critical).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: data/<mlp_name>/formed_df.pt).",
    )
    parser.add_argument(
        "--kprop-names",
        nargs="*",
        default=None,
        help="Optional subset of kprop names (e.g. simple augment pK_ablate).",
    )
    parser.add_argument(
        "--print-meta",
        action="store_true",
        help="Print saved metadata JSON.",
    )
    parser.add_argument(
        "--no-times",
        action="store_true",
        help="Skip wall clock time column."
    )
    args = parser.parse_args()

    out = save_df(args.mlp_name, output_path=args.output, kprop_names=args.kprop_names, compute_times=not args.no_times)
    print(f"Saved formed df to {out}")

    if args.print_meta:
        payload = torch.load(out, weights_only=False, map_location="cpu")
        print(json.dumps(payload["meta"], indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(name)s.%(funcName)s | %(message)s",
        force=True,
    )
    logging.getLogger("mlp_kprop.kprop_harmonic").setLevel(logging.DEBUG)
    logging.getLogger("mlp_kprop.factor_k3").setLevel(logging.DEBUG)
    logging.getLogger("mlp_kprop.factor_k4").setLevel(logging.DEBUG)
    main()
