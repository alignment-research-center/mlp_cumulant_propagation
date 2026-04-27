import gc
import logging
import math
import multiprocessing as mp
import os
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm

from mlp_kprop.cumulants import *
from mlp_kprop.diagslice import *
from mlp_kprop import kprop_harmonic, kprop_ds
from mlp_kprop.kprop_harmonic import Kind as HKind
from mlp_kprop.kprop_harmonic import get_int_cond
from mlp_kprop.harmonic import *
from mlp_kprop.mlp import MLP
from mlp_kprop.partitions import *
from mlp_kprop.logging_utils import *

logger = logging.getLogger('kprop_by_width')

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

ERRS_COLUMNS = ["n", "t", "l", "d", "k", "err"]
SAMPLE_SE2_COLUMNS = ["n", "t", "l", "d", "se2"]

def get_sample_cumulants(
    mlp: MLP,
    samples: int,
    batch_size: int = 100_000,
    return_stderr: bool = False,
    batches_per_bs: int = 5,
    *,
    k_max: int,
    int_cond=None,
    d_max=None,
    include_pre: bool = True,
    include_act: bool = True,
) -> tuple[dict[DSTower], dict[DSTower]]:
    """
    Compute cumulants and stderrs of the output of the MLP via sampling.
    Stderrs are computed via bootstrapping on batches.

    Args:
        mlp: MLP instance.
        k_max: Compute cumulants with multi-index a satisfying |ceil(a/2)| <= k_max
        d_max: Compute cumulants up to degree d_max
        include_pre: If True, include cumulants for pre-activation layers (including output pre layer).
        include_act: If True, include cumulants for activation layers.
        samples: Number of samples to use for cumulant estimation.
        B: Number of bootstrap samples for stderr estimation.
    Returns:
        cumulants: dict mapping layer name to cumulant for each order
        stderrs: dict mapping layer name to per-entry standard error of cumulant for each order
    """
    logger.debug(f"samples {samples}, batch_size {batch_size}, batches_per_bs {batches_per_bs}")
    assert samples % (batch_size * batches_per_bs) == 0, (
        "samples must be a multiple of batch_size*batches_per_bs"
    )
    if int_cond is None:
        int_cond = get_int_cond(k_max)
    if d_max is None:
        d_max = 2 * k_max
    if not include_pre and not include_act:
        raise ValueError("At least one of include_pre/include_act must be True.")
    K = dict()  # l -> DSTower of cumulants generator
    K_bs = dict()  # l -> DSTower of cumulants generator for current bootstrap
    K_bs_count = defaultdict(int)    # l -> count of bootstrap batches seen so far (Welford)
    K_bs_mean = defaultdict(DSTower) # l -> Welford running mean of cumulant over bootstrap batches
    K_bs_M2 = defaultdict(DSTower)   # l -> Welford running M2 (sum of squared deviations from mean)
    n = mlp.input_dim
    num_bs = samples // (batch_size * batches_per_bs)
    batch_size = int(batch_size)
    samples = int(samples)
    num_batch = int(samples // batch_size)
    if num_bs < 20:
        logger.warning(f"Only {num_bs} bootstraps, stderr estimates may be inaccurate")

    def prime_and_send(gen, X):
        if gen is None:
            gen = DS_cumulant_gen(part_cond=int_cond, d_max=d_max)
            next(gen)
        gen.send(X)
        return gen

    def update_layer(name: str, Y: torch.Tensor, batch: int) -> None:
        logger.trace(f"Cumulant {name} batch {batch + 1}/{num_bs}")
        Y = Y.to(torch.float64)
        K[name] = prime_and_send(K.get(name), Y)
        if return_stderr:
            K_bs[name] = prime_and_send(K_bs.get(name), Y)
            if (batch + 1) % batches_per_bs == 0:
                # Compute bootstrap cumulant and reset generator
                logger.trace(f"Bootstrap {name} {(batch + 1) // batches_per_bs}/{num_bs}")
                KY_bs = finish(K_bs[name])
                K_bs[name] = None
                # Welford's online algorithm for mean and variance
                K_bs_count[name] += 1
                count = K_bs_count[name]
                if count == 1:
                    K_bs_mean[name] = KY_bs.clone()
                else:
                    delta = KY_bs - K_bs_mean[name]
                    K_bs_mean[name] += delta / count
                    delta2 = KY_bs - K_bs_mean[name]
                    K_bs_M2[name] += delta * delta2

    for batch in tqdm(
        range(num_batch), desc="sampK", disable=logger.getEffectiveLevel() > logging.DEBUG
    ):
        logger.trace(f"Processing batch {batch + 1}/{num_batch}")
        X = torch.randn(batch_size, n, device=device)
        out = mlp(X, output_acts=True)
        if include_pre:
            pre = out.pre  # (batch, layer, width)
            for l in range(pre.shape[1]):
                update_layer(f"pre{l}", pre[:, l, :], batch)
            # Include the output pre layer (final linear output).
            update_layer(f"pre{pre.shape[1]}", out.out, batch)
        if include_act:
            act = out.act  # (batch, layer, width)
            for l in range(act.shape[1]):
                update_layer(f"act{l}", act[:, l, :], batch)

    K = {l: finish(K[l]) for l in K}
    if return_stderr:
        stderrs = {
            l: {
                # Welford's M2 / (num_bs - 1) gives sample variance;
                # dividing by num_bs gives variance of the mean (i.e. SE^2)
                d: (K_bs_M2[l][d] / max(1, num_bs - 1) / num_bs).clamp(min=0.).pow(0.5)
                for d in K[l]
            }
            for l in K
        }
        stderrs = {l: DSTower(stderrs[l]) for l in stderrs}
        return K, stderrs
    else:
        return K, None

@dataclass
class KPropByWidthCfg:
    k_maxs: tuple[int]   # k_maxs to run kprop with
    ns: tuple[int] = (4, 8, 16, 32, 64, 128)  # widths to run kprop on
    sample_k_max: int = 1    # max k to get sample cumulants up to
    output_d_max: Optional[int] = 1    # max degree to output kprop cumulants up to
    base_seed: int = 0
    samples: int = 1_000_000
    batch_size: int = 10_000
    trials: int = 5
    name: str = "mKprop_by_width"
    sampK_cache_name: str = "sampK_cache"
    batches_per_bs: int = 5
    mlp_kwargs: dict = field(default_factory=dict)
    kprop_kwargs: dict = field(default_factory=dict)
    kind: str = 'harmonic'


def add_row(df, **kwargs):
    assert set(df.columns) == set(kwargs.keys())
    if len(df) == 0:
        return pd.DataFrame([kwargs])
    return pd.concat([df, pd.DataFrame(kwargs, index=[0])], ignore_index=True)


def _tuple_to_str(ds) -> str:
    return ",".join(str(x) for x in ds)


def _empty_errs_df() -> pd.DataFrame:
    return pd.DataFrame(columns=ERRS_COLUMNS)


def _empty_sample_se2_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SAMPLE_SE2_COLUMNS)


def numel(part: IntPartition, n: int):
    """Number of elements of multiplicity type part in n dimensions."""
    return math.prod(range(n, n - len(part), -1))


def _set_worker_device(device_index: int | None) -> torch.device:
    """
    Ensure the current process uses the requested CUDA device.
    """
    global device
    if device_index is None or not torch.cuda.is_available():
        return device
    worker_device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(worker_device)
    device = worker_device
    return worker_device


def _run_single_trial(
    cfg: KPropByWidthCfg, n: int, t: int, force_refresh: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    k_maxs = cfg.k_maxs
    sample_k_max = cfg.sample_k_max
    output_d_max = cfg.output_d_max
    base_seed = cfg.base_seed
    N = cfg.samples
    batch_size = cfg.batch_size
    batches_per_bs = cfg.batches_per_bs
    if cfg.kind == 'harmonic':
        kprop = kprop_harmonic
    elif cfg.kind == 'ds':
        kprop = kprop_ds
    else:
        raise ValueError(f"Unknown kind {cfg.kind}")

    errs = _empty_errs_df()
    sample_se2 = _empty_sample_se2_df()

    torch.manual_seed(base_seed + t)
    # torch.set_default_dtype(torch.float64)  # cached results were in float64
    mlp_kwargs = dict(
        input_dim=n,
        hidden_dim=n,
        output_dim=n,
        **cfg.mlp_kwargs
    )
    mlp = MLP(**mlp_kwargs).to(device, dtype=dtype)
    # torch.set_default_dtype(dtype)  # switch back to default dtype
    cache_path = f"{Path(__file__).parent}/../data/{cfg.sampK_cache_name}/seed{base_seed + t}/n{n}"
    mlp_path = f"{cache_path}/mlp.pt"
    kwargs_path = f"{cache_path}/mlp_kwargs.pt"
    if output_d_max is None:
        output_d_max = 2 * sample_k_max
    sampK_cache_path = f"{cache_path}/kmax{sample_k_max}_dmax{output_d_max}/N{N:.1g}"
    if os.path.exists(sampK_cache_path) and not force_refresh:
        logger.info(f"Loading from cache dir {sampK_cache_path}")
        mlp_cached = torch.load(mlp_path, weights_only=False, map_location=device)
        assert mlp.state_dict().keys() == mlp_cached.state_dict().keys()
        for k in mlp.state_dict().keys():
            assert torch.allclose(
                mlp.state_dict()[k].to("cpu", dtype=dtype),
                mlp_cached.state_dict()[k].to("cpu", dtype=dtype),
            )

        sampK = torch.load(f"{sampK_cache_path}/sampK.pt", weights_only=False, map_location=device)
        sampKse = torch.load(f"{sampK_cache_path}/sampKse.pt", weights_only=False, map_location=device)
        sampK = {k: v.to(device, dtype=dtype) for k, v in sampK.items()}
        if not isinstance(next(iter(sampKse.values())), DSTower):
            # Mistakenly saved sampKse as raw dicts instead of DSTowers
            sampKse = {
                k: DSTower(v).to(device, dtype=dtype) for k, v in sampKse.items()
            }  
        sampKse = {k: v.to(device, dtype=dtype) for k, v in sampKse.items()}

    else:
        logger.info(f"(Re)sampling cache {sampK_cache_path}")
        sampK, sampKse = get_sample_cumulants(
            mlp,
            samples=N,
            batch_size=batch_size,
            return_stderr=True,
            batches_per_bs=batches_per_bs,
            k_max=sample_k_max,
            d_max=output_d_max,
        )
        os.makedirs(sampK_cache_path, exist_ok=True)
        torch.save(mlp, mlp_path)
        torch.save(mlp_kwargs, kwargs_path)
        torch.save(sampK, f"{sampK_cache_path}/sampK.pt")
        torch.save(sampKse, f"{sampK_cache_path}/sampKse.pt")

    # TODO: When kind=harmonic and for d>1, these can't be directly plotted bc the kprop cumulants
    # are indexed by degree, not k. Probably fine to sum sampKse by degree?
    if sampKse is not None:
        for l in sampKse.keys():
            for d in sampKse[l].keys():
                for part in sampKse[l][d].slices:
                    sample_se2 = add_row(
                        sample_se2,
                        n=n,
                        t=t,
                        l=l,
                        d=_tuple_to_str(part),
                        se2=sampKse[l][d].get_slice(part).nan_to_num(0.).pow(2).sum().item() / numel(part, n),
                    )
    del sampKse
    
    logger.info(f"Sample cumulants done")

    K_in = {1: torch.zeros(n, device=device), 2: torch.eye(n, device=device)}
    if cfg.kind == 'ds':
        K_in = DSTower.from_tower(K_in)
    for k_max in k_maxs[::-1]:
        gc.collect()
        torch.cuda.empty_cache()
        if k_max > 0:
            K_by_layer = kprop.mlp_kprop(
                mlp, K_in, k_max=k_max, output_all=True, output_d_max=output_d_max, **cfg.kprop_kwargs
            )
        else:
            K_by_layer = {l: dict() for l in sampK.keys()}

        for l in sampK.keys():
            for d in sampK[l].keys():
                if cfg.kind == 'ds':
                    for part in sampK[l][d].slices:
                        ret = (
                            0.0
                            if d not in K_by_layer[l]
                            else K_by_layer[l][d].get_slice(part, strict=False)
                        )
                        errs = add_row(
                            errs,
                            n=n,
                            t=t,
                            l=l,
                            d=_tuple_to_str(part),
                            k=k_max,
                            err=(ret - sampK[l][d].get_slice(part)).pow(2).sum().item()
                            / numel(part, n),
                        )
                else:
                    assert cfg.kind == 'harmonic'
                    if (1,) * d not in sampK[l][d].slices:
                        logger.warning(
                            f"Using degree {d} sample cumulants as ground truth, "
                            f"but these have only been computed up to k={sample_k_max}."
                        )
                    if d not in K_by_layer[l]:
                        ret = 0.
                    else:
                        ret = K_by_layer[l][d].to_tensor()
                    errs = add_row(
                        errs,
                        n=n,
                        t=t,
                        l=l,
                        d=str(d),
                        k=k_max,
                        err=(ret - sampK[l][d].to_tensor()).pow(2).mean().item()
                    )

    return errs, sample_se2


def _parallel_worker(
    cfg: KPropByWidthCfg, force_refresh: bool, task_queue, result_queue, device_index: int
):
    try:
        _set_worker_device(device_index)
        while True:
            task = task_queue.get()
            if task is None:
                break
            n, t = task
            errs, sample_se2 = _run_single_trial(cfg, n, t, force_refresh)
            result_queue.put(("result", (errs, sample_se2)))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))
        raise
    finally:
        result_queue.put(("done", device_index))


def _run_tasks_in_parallel(
    cfg: KPropByWidthCfg,
    force_refresh: bool,
    tasks: list[tuple[int, int]],
    pbar,
    steps_per_task: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    worker_count = min(torch.cuda.device_count(), len(tasks))
    assert worker_count > 0
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(worker_count):
        task_queue.put(None)

    procs = []
    for device_index in range(worker_count):
        proc = ctx.Process(
            target=_parallel_worker,
            args=(cfg, force_refresh, task_queue, result_queue, device_index),
        )
        proc.start()
        procs.append(proc)

    errs_parts: list[pd.DataFrame] = []
    sample_se2_parts: list[pd.DataFrame] = []
    expected_results = len(tasks)
    results_received = 0
    done_workers = 0
    try:
        while results_received < expected_results or done_workers < worker_count:
            msg_type, payload = result_queue.get()
            if msg_type == "result":
                errs_parts.append(payload[0])
                sample_se2_parts.append(payload[1])
                results_received += 1
                pbar.update(steps_per_task)
            elif msg_type == "error":
                for proc in procs:
                    if proc.is_alive():
                        proc.terminate()
                raise RuntimeError(f"Parallel worker failed:\n{payload}")
            elif msg_type == "done":
                done_workers += 1
    finally:
        for proc in procs:
            proc.join()
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

    return errs_parts, sample_se2_parts


def kprop_by_width(cfg: KPropByWidthCfg, force_refresh: bool = False):
    k_maxs = cfg.k_maxs
    ns = cfg.ns
    trials = cfg.trials
    name = cfg.name

    tasks = [(n, t) for n in ns[::-1] for t in range(trials)]
    total_steps = len(tasks) * len(k_maxs)
    errs_parts: list[pd.DataFrame] = []
    sample_se2_parts: list[pd.DataFrame] = []

    if not tasks:
        return errs, sample_se2

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_parallel = available_gpus > 1 and len(tasks) > 1

    with tqdm(total=total_steps, disable=logger.getEffectiveLevel() > logging.INFO) as pbar:
        if use_parallel:
            pbar.set_description("parallel GPUs", refresh=True)
            parallel_errs, parallel_sample = _run_tasks_in_parallel(
                cfg, force_refresh, tasks, pbar, len(k_maxs)
            )
            errs_parts.extend(parallel_errs)
            sample_se2_parts.extend(parallel_sample)
        else:
            for n, t in tasks:
                pbar.set_description(f"n={n}, t={t + 1}/{trials}", refresh=True)
                errs_i, sample_se2_i = _run_single_trial(cfg, n, t, force_refresh)
                errs_parts.append(errs_i)
                sample_se2_parts.append(sample_se2_i)
                pbar.update(len(k_maxs))

    errs = pd.concat(errs_parts, ignore_index=True) if errs_parts else _empty_errs_df()
    sample_se2 = (
        pd.concat(sample_se2_parts, ignore_index=True)
        if sample_se2_parts
        else _empty_sample_se2_df()
    )

    os.makedirs(Path(f"../data/{name}").parent, exist_ok=True)
    torch.save((cfg, errs, sample_se2), f"../data/{name}.pt")
    return errs, sample_se2


def aggregate(errs=None, sample_se2=None, save_name=None):
    """
    Aggregate and rename columns in preparation for plot_mse_by_width
    """
    if save_name is not None:
        errs, sample_se2 = torch.load(f"../data/{save_name}.pt", weights_only=False)
    else:
        assert errs is not None and sample_se2 is not None
    errs = (
        errs.groupby(["n", "l", "d", "k"])
        .agg(err=("err", "mean"), sem=("err", "sem"))
        .reset_index()
    )
    sample_se2 = sample_se2.groupby(["n", "l", "d"]).agg(se2=("se2", "mean")).reset_index()
    errs = errs.rename(columns={"sem": "yerr", "err": "y", "n": "x"})  # l d k x y yerr
    sample_se2 = sample_se2.rename(columns={"se2": "y", "n": "x"})  # l d x y
    return errs, sample_se2

SAMPLES = 2**34
BATCH_SIZE = 2**19
BATCHES_PER_BS = 2**7  # 2**10 bootstraps total
# SAMPLES = 2**30
# BATCH_SIZE = 2**19
# BATCHES_PER_BS = 2**5  # 2**6 bootstraps total
ALL_WIDTHS = (8, 16, 23, 32, 45, 64, 91, 128, 181, 256)
K_MAXS = tuple(range(0, 5))

def _run_nonlin(nonlin: str):
    for augment, use_avg_metric in product([False, True], [False, True]):  # TODO: ([True, False], [False, True])  # (want to run biggest first)
        print(f'augment={augment}, use_avg_metric={use_avg_metric}')
        name = ('augment' if augment else 'simple') + ('_avgmetric' if use_avg_metric else '')
        hkind = HKind.AUGMENT if augment else HKind.SIMPLE
        cfg = KPropByWidthCfg(
            samples=SAMPLES,
            batch_size=BATCH_SIZE,
            batches_per_bs=BATCHES_PER_BS,
            k_maxs=K_MAXS,
            ns=ALL_WIDTHS,
            sample_k_max=1,
            output_d_max=1,
            sampK_cache_name=f"{nonlin}/cache/sampK_depth16",
            name=f"{nonlin}/{name}/kprop_by_width",
            kprop_kwargs=dict(
                kind=hkind,
                use_avg_metric=use_avg_metric,
                factor=True,
            ),
            mlp_kwargs=dict(num_layers=16, nonlin=nonlin),
        )
        errs, sample_se2 = kprop_by_width(cfg, force_refresh=False)

run_relu = partial(_run_nonlin, 'relu')
run_sigmoid = partial(_run_nonlin, 'sigmoid')
run_gelu = partial(_run_nonlin, 'gelu')
run_tanh = partial(_run_nonlin, 'tanh')

def run_pk_ablate():
    cfg = KPropByWidthCfg(
        samples=SAMPLES,
        batch_size=BATCH_SIZE,
        batches_per_bs=BATCHES_PER_BS,
        k_maxs=K_MAXS,   # BASE = SIMPLE on even k
        ns=ALL_WIDTHS,
        sample_k_max=1,
        output_d_max=1,
        sampK_cache_name="relu/cache/sampK_depth16",
        name=f"relu/pK_ablate/kprop_by_width",
        kprop_kwargs=dict(
            kind=HKind.BASE,
            use_avg_metric=False,
            factor=True,
            use_pK=False,
        ),
        mlp_kwargs=dict(num_layers=16),
    )
    errs, sample_se2 = kprop_by_width(cfg, force_refresh=False)

def run_lpe(layer=3, b=3.):
    factor = True
    use_avg_metric = False

    mlp_kwargs = dict(
        num_layers=layer+2,
        nonlin=['relu'] * layer + ['heaviside'],
        b_mean=[0.] * layer + [-b] + [0.],
    )
    
    for augment in [False, True]:
        print(f'augment={augment}')
        name = 'augment' if augment else 'simple'
        hkind = HKind.AUGMENT if augment else HKind.SIMPLE
        kprop_kwargs = dict(
            kind=hkind,
            use_avg_metric=use_avg_metric,
            factor=factor,
        )
        cfg = KPropByWidthCfg(
            samples=SAMPLES,
            batch_size=BATCH_SIZE,
            batches_per_bs=BATCHES_PER_BS,
            k_maxs=K_MAXS,
            ns=(256,),
            sample_k_max=2,
            output_d_max=4,
            sampK_cache_name=f"lpe/cache/sampK_layer{layer}_b{b:.1f}",
            name=f"lpe/layer{layer}_b{b:.1f}/{name}/kprop_by_width",
            mlp_kwargs=mlp_kwargs,
            kprop_kwargs=kprop_kwargs,
        )
        errs, sample_se2 = kprop_by_width(cfg, force_refresh=False)

Q_STAR = 1.
def _run_nonlin_critical(nonlin):
    for augment in [False, True]:
        print(f'augment={augment}')
        name = 'augment' if augment else 'simple'
        hkind = HKind.AUGMENT if augment else HKind.SIMPLE
        cfg = KPropByWidthCfg(
            samples=SAMPLES,
            batch_size=BATCH_SIZE,
            batches_per_bs=BATCHES_PER_BS,
            k_maxs=K_MAXS,
            ns=ALL_WIDTHS,
            sample_k_max=1,
            output_d_max=1,
            sampK_cache_name=f"{nonlin}_critical/cache/sampK_depth16",
            name=f"{nonlin}_critical/{name}/kprop_by_width",
            kprop_kwargs=dict(
                kind=hkind,
                use_avg_metric=False,
                factor=True,
            ),
            mlp_kwargs=dict(
                num_layers=16,
                nonlin=nonlin,
                init_kind='critical',
                q_star=Q_STAR,
            ),
        )
        errs, sample_se2 = kprop_by_width(cfg, force_refresh=False)

run_tanh_critical = partial(_run_nonlin_critical, 'tanh')
run_gelu_critical = partial(_run_nonlin_critical, 'gelu')

EXPERIMENTS = {
    'relu': run_relu,
    'sigmoid': run_sigmoid,
    'lpe': run_lpe,
    'lpe_shallow': lambda: run_lpe(layer=1),
    'pk_ablate': run_pk_ablate,
    'gelu': run_gelu,
    'tanh': run_tanh,
    'tanh_critical': run_tanh_critical,
    'gelu_critical': run_gelu_critical,
}

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(name)s.%(funcName)s | %(message)s",
        force=True,
    )
    logging.getLogger("kprop_by_width").setLevel(logging.DEBUG)
    logging.getLogger("mlp_kprop.factor_k4").setLevel(logging.INFO)
    logging.getLogger("mlp_kprop.kprop_harmonic").setLevel(logging.INFO)

    if len(sys.argv) == 1:
        for experiment in ['pk_ablate', 'sigmoid', 'lpe']:
            print(f"Running experiment {experiment}")
            EXPERIMENTS[experiment]()
    else:
        experiment = sys.argv[1]
        EXPERIMENTS[experiment]()
