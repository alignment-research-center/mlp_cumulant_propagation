from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np
import torch
from joblib import Memory

from mlp_kprop.kprop_harmonic import Kind, coerce_input, linear_kprop, nonlin_kprop
from mlp_kprop.mlp import MLP
from mlp_kprop.wick import WICK_COEF_D

logger = logging.getLogger(__name__)

_KPROP_TIME_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "kprop_time_cache"
_kprop_time_memory = Memory(_KPROP_TIME_CACHE_DIR, verbose=0)
_SAMPLING_TIME_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sampling_time_cache"
_sampling_time_memory = Memory(_SAMPLING_TIME_CACHE_DIR, verbose=0)


def _parse_layer(layer: str) -> tuple[str, int]:
    if layer.startswith("pre"):
        return "pre", int(layer[3:])
    if layer.startswith("act"):
        return "act", int(layer[3:])
    raise ValueError(f"Unknown layer format: {layer}")


def _default_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def _resolve_device(device_name: str) -> torch.device:
    """Map a human-readable device name to a ``torch.device``.

    Raises ``RuntimeError`` on cache miss when the requested GPU is not present.
    """
    if device_name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"No CUDA devices available; cannot measure times for device_name={device_name!r}"
        )
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_name(i) == device_name:
            return torch.device(f"cuda:{i}")
    available = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    raise RuntimeError(
        f"No CUDA device matching {device_name!r} found. Available: {available}"
    )


def _required_num_layers(layer: str) -> int:
    layer_type, layer_idx = _parse_layer(layer)
    if layer_type == "pre":
        return layer_idx + 1
    return layer_idx + 2


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _median_runtime(fn, repeats: int, device: torch.device) -> float:
    times: list[float] = []
    for _ in range(max(1, repeats)):
        _sync_if_needed(device)
        start = time.perf_counter()
        fn()
        _sync_if_needed(device)
        end = time.perf_counter()
        times.append(end - start)
    return float(np.median(times))


def _normalize_kind(kind: Kind | str) -> Kind:
    if isinstance(kind, Kind):
        return kind
    if isinstance(kind, str):
        return Kind[kind.upper()]
    raise TypeError(f"Unsupported kind type: {type(kind)!r}")


def _is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cannot allocate memory" in msg) or ("and less than 2147483647" in msg)


@_kprop_time_memory.cache
def _measure_segment_times_cached(
    n: int,
    mlp_kwargs: dict,
    kprop_kwargs: dict,
    output_d_max: int | None,
    repeats: int,
    device_name: str,
) -> dict[str, float]:
    device = _resolve_device(device_name)

    num_layers = 3

    mlp_kw = {k: v for k, v in mlp_kwargs.items() if k != "num_layers"}
    mlp = MLP(hidden_dim=n, num_layers=num_layers, **mlp_kw).to(device)

    k_max = int(kprop_kwargs["k_max"])
    kind = _normalize_kind(kprop_kwargs.get("kind", Kind.SIMPLE))
    use_avg_metric = bool(kprop_kwargs.get("use_avg_metric", True))
    factor = bool(kprop_kwargs.get("factor", False))
    use_pK = bool(kprop_kwargs.get("use_pK", True))

    W0 = mlp.Ws[0].weight
    b0 = mlp.Ws[0].bias
    metric0 = mlp.init_scale[0] if use_avg_metric else None
    nonlin_coef0 = WICK_COEF_D[mlp.nonlin_names[0]]

    W1 = mlp.Ws[1].weight
    b1 = mlp.Ws[1].bias
    metric_warm = mlp.init_scale[1] if use_avg_metric else None
    nonlin_coef_warm = WICK_COEF_D[mlp.nonlin_names[1]]

    # Cold starts from standard-Gaussian cumulants up to degree 2.
    K_gauss_raw = {1: torch.zeros(n, device=device), 2: torch.eye(n, device=device)}
    K_gauss = coerce_input(K_gauss_raw, k_max=k_max, kind=kind)

    @torch.no_grad()
    def run_cold_linear():
        return linear_kprop(K_gauss, W0, k_max=k_max, set_metric=metric0, bias=b0)

    @torch.no_grad()
    def run_cold_out_linear():
        return linear_kprop(K_gauss, W0, k_max=k_max, d_max=output_d_max, bias=b0)

    # Warm starts from the output of the cold first layer.
    K_after_cold_linear = run_cold_linear()

    @torch.no_grad()
    def run_cold_nonlin():
        return nonlin_kprop(
            K_after_cold_linear,
            nonlin_wick_coef=nonlin_coef0,
            k_max=k_max,
            kind=kind,
            use_pK=use_pK,
            factor=factor,
        )

    K_warm_in = run_cold_nonlin()

    @torch.no_grad()
    def run_warm_linear():
        return linear_kprop(K_warm_in, W1, k_max=k_max, set_metric=metric_warm, bias=b1)

    @torch.no_grad()
    def run_warm_out_linear():
        return linear_kprop(K_warm_in, W1, k_max=k_max, d_max=output_d_max, bias=b1)

    K_after_warm_linear = run_warm_linear()

    @torch.no_grad()
    def run_warm_nonlin():
        return nonlin_kprop(
            K_after_warm_linear,
            nonlin_wick_coef=nonlin_coef_warm,
            k_max=k_max,
            kind=kind,
            use_pK=use_pK,
            factor=factor,
        )

    return {
        "cold_linear": _median_runtime(run_cold_linear, repeats=repeats, device=device),
        "cold_nonlin": _median_runtime(run_cold_nonlin, repeats=repeats, device=device),
        "cold_out_linear": _median_runtime(run_cold_out_linear, repeats=repeats, device=device),
        "warm_linear": _median_runtime(run_warm_linear, repeats=repeats, device=device),
        "warm_nonlin": _median_runtime(run_warm_nonlin, repeats=repeats, device=device),
        "warm_out_linear": _median_runtime(run_warm_out_linear, repeats=repeats, device=device),
    }


def get_kprop_time(
    n: int,
    layer: str,
    mlp_kwargs: dict | None = None,
    kprop_kwargs: dict | None = None,
    *,
    repeats: int = 5,
    device_name: str | None = None,
) -> float:
    """Return median wall-clock time (seconds) for one `mlp_kprop` call.

    Uses cached cold/warm segment timings and composes layer time analytically.
    """
    mlp_kwargs = dict(mlp_kwargs)
    # These params don't affect time
    for key in ('init_kind', 'q_star', 'w_var', 'b_var', 'b_mean'):
        mlp_kwargs.pop(key, None)

    print(f"Measuring kprop time for n={n}, layer={layer}, mlp_kwargs={mlp_kwargs}, kprop_kwargs={kprop_kwargs}")
    if mlp_kwargs is None:
        mlp_kwargs = {}
    if kprop_kwargs is None:
        kprop_kwargs = {}
    if n > 128 and not kprop_kwargs.get("factor", False) and int(kprop_kwargs.get("k_max", -1)) >= 4:
        return float("inf")  # OOM

    mlp_kwargs = dict(mlp_kwargs)
    kprop_kwargs = dict(kprop_kwargs)

    # Width is parameterized by `n`.
    for key in ("hidden_dim", "input_dim", "output_dim"):
        if key in mlp_kwargs:
            logger.warning(
                "mlp_kwargs[%r] is ignored: width is parameterized by `n` in get_kprop_time",
                key,
            )
            mlp_kwargs.pop(key)
    if "num_layers" in mlp_kwargs:
        logger.warning(
            "mlp_kwargs['num_layers'] is ignored: get_kprop_time uses a 2-layer surrogate for cold/warm segment timing"
        )
        mlp_kwargs.pop("num_layers")

    # Layer is parameterized by `layer`.
    if "up_to_layer" in kprop_kwargs:
        logger.warning(
            "kprop_kwargs['up_to_layer'] is ignored: layer is parameterized by `layer` in get_kprop_time"
        )
        kprop_kwargs.pop("up_to_layer")

    output_d_max = kprop_kwargs.pop("output_d_max", None)

    if device_name is None:
        device_name = _default_device_name()

    try:
        segment_times = _measure_segment_times_cached(
            n=n,
            mlp_kwargs=mlp_kwargs,
            kprop_kwargs=kprop_kwargs,
            output_d_max=output_d_max,
            repeats=int(repeats),
            device_name=device_name,
        )
    except torch.cuda.OutOfMemoryError:
        logger.error(f"OOM when measuring segment times for n={n}, layer={layer}, mlp_kwargs={mlp_kwargs}, kprop_kwargs={kprop_kwargs}")
        return float("inf")

    layer_type, layer_idx = _parse_layer(layer)
    if layer_type == "act":
        # act{l}: cold (layer 0) + l warm full layers
        return float(
            segment_times["cold_linear"]
            + segment_times["cold_nonlin"]
            + layer_idx * (segment_times["warm_linear"] + segment_times["warm_nonlin"])
        )
    if layer_idx == 0:
        # pre0 is a cold output linearity (with d_max=output_d_max).
        return float(segment_times["cold_out_linear"])
    # pre{l>0}: cold full layer + (l-1) warm full layers + output linearity.
    return float(
        segment_times["cold_linear"]
        + segment_times["cold_nonlin"]
        + (layer_idx - 1) * (segment_times["warm_linear"] + segment_times["warm_nonlin"])
        + segment_times["warm_out_linear"]
    )


def _fits_single_batch(mlp: MLP, n: int, batch_size: int, device: torch.device) -> bool:
    try:
        with torch.no_grad():
            x = torch.randn(batch_size, n, device=device)
            _ = mlp(x, up_to_layer="act0")
            _sync_if_needed(device)
        return True
    except RuntimeError as exc:
        if _is_oom_error(exc):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return False
        raise

def _measure_forward_time(mlp: MLP, n: int, layer: str, batch_size: int, repeats: int, device: torch.device) -> float:
    times: list[float] = []
    for _ in range(max(1, repeats)):
        with torch.no_grad():
            x = torch.randn(batch_size, n, device=device)
            _sync_if_needed(device)
            start = time.perf_counter()
            _ = mlp(x, up_to_layer=layer)
            _sync_if_needed(device)
            end = time.perf_counter()
        times.append(end - start)
    return float(np.median(times))

@_sampling_time_memory.cache
def _measure_sampling_time(
    n: int,
    layer: str,
    n_samples: int,
    mlp_kwargs: dict,
    repeats: int,
    device_name: str,
) -> float:
    device = _resolve_device(device_name)
    target_samples = max(1, int(n_samples))

    req_num_layers = _required_num_layers(layer)
    num_layers = int(mlp_kwargs.get("num_layers", req_num_layers))
    num_layers = max(num_layers, req_num_layers)

    mlp_kw = {k: v for k, v in mlp_kwargs.items() if k != "num_layers"}
    mlp = MLP(hidden_dim=n, num_layers=num_layers, **mlp_kw).to(device)

    # If target fits as one batch, time it directly.
    if _fits_single_batch(mlp, n=n, batch_size=target_samples, device=device):
        return _measure_forward_time(
            mlp,
            n=n,
            layer=layer,
            batch_size=target_samples,
            repeats=repeats,
            device=device,
        )

    # Otherwise find the largest power-of-two batch that fits and scale linearly.
    # Recurse with n_samples=base so the base measurement itself gets cached.
    base = 1 << (target_samples.bit_length() - 1)
    while base >= 1 and not _fits_single_batch(mlp, n=n, batch_size=base, device=device):
        base //= 2
    if base < 1:
        raise RuntimeError(f"Could not fit even batch_size=1 for n={n}, layer={layer}")

    base_time = _measure_sampling_time(
        n=n,
        layer=layer,
        n_samples=base,
        mlp_kwargs=mlp_kwargs,
        repeats=repeats,
        device_name=device_name,
    )
    return float(base_time * (target_samples / base))


def get_sampling_time(
    n: int,
    layer: str,
    n_samples: int,
    mlp_kwargs: dict | None = None,
    *,
    repeats: int = 5,
    device_name: str | None = None,
) -> float:
    """Return wall-clock time (seconds) for `n_samples` forward passes to `up_to_layer=layer`.

    Measures ``pre0`` and ``pre1``, then extrapolates linearly using
    ``pre_act = pre1_time - pre0_time`` as the per-layer (linear + activation)
    increment:

    * ``preK  = K * pre_act + pre0_time``
    * ``actK  = (K + 1) * pre_act``
    """
    if mlp_kwargs is None:
        mlp_kwargs = {}
    mlp_kwargs = dict(mlp_kwargs)

    for key in ("hidden_dim", "input_dim", "output_dim"):
        if key in mlp_kwargs:
            logger.warning(
                "mlp_kwargs[%r] is ignored: width is parameterized by `n` in get_sampling_time",
                key,
            )
            mlp_kwargs.pop(key)

    if device_name is None:
        device_name = _default_device_name()

    cache_kw = dict(
        n=n,
        n_samples=int(n_samples),
        mlp_kwargs=mlp_kwargs,
        repeats=int(repeats),
        device_name=device_name,
    )

    pre0_time = _measure_sampling_time(layer="pre0", **cache_kw)
    pre1_time = _measure_sampling_time(layer="pre1", **cache_kw)
    pre_act = pre1_time - pre0_time

    layer_type, layer_idx = _parse_layer(layer)
    if layer_type == "pre":
        return float(layer_idx * pre_act + pre0_time)
    else:
        return float((layer_idx + 1) * pre_act)
