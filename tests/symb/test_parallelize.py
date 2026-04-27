import os
import pickle
import subprocess

import blobfile as bf
from mlp_kprop.symb.kprop import abstract_kprop, reduce_treewidth_all
from mlp_kprop.symb.parallelize import is_root
from mlp_kprop.symb.parallelize import multi_map_parallel as multi_map
from mpi4py import MPI

REMOTE_BASE_PATH = "REDACTED"


def mpi_test(num_procs):
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            if num_procs > 1 and MPI.COMM_WORLD.size == 1:
                test_file = os.path.abspath(test_func.__code__.co_filename)
                test_name = test_func.__name__
                command = f"mpiexec -n {num_procs} pytest {test_file} -k {test_name}"
                result = subprocess.run(command, shell=True)
                assert result.returncode == 0, "MPI test failed"
            else:
                test_func(*args, **kwargs)

        return wrapper

    return decorator


@mpi_test(num_procs=2)
def test_multi_map():
    nums = [(1, 2), (3, 4), (5, 6), (7, 8)]
    nums = multi_map(lambda x, y: [(x, y), (x + 1, y + 2)], nums)
    nums = multi_map(lambda x, y: [(y, x)], nums)
    assert list(nums) == [
        (2, 1),
        (4, 2),
        (4, 3),
        (6, 4),
        (6, 5),
        (8, 6),
        (8, 7),
        (10, 8),
    ]


@mpi_test(num_procs=3)
def test_kprop():
    for k_max, depth, treewidth in [(2, 6, 2), (3, 3, 2)]:
        all_ks = abstract_kprop(
            depth=depth,
            k_max=k_max,
            output_all=True,
            mean_only=True,
            var_only=True,
            simplify=True,
            prune=True,
            verbose=is_root,
        )
        if is_root:
            load_path = os.path.join(
                REMOTE_BASE_PATH, f"raw_k_max_{k_max}_depth_{depth}.pkl"
            )
            print(f"Loading from {load_path}...")
            with bf.BlobFile(load_path, "rb", streaming=False) as fh:
                all_ks_cached = pickle.load(fh)
            assert all_ks == all_ks_cached
        reduced_ks = reduce_treewidth_all(
            all_ks,
            k_max=k_max,
            treewidth=treewidth,
            simplify=True,
            prune=True,
            verbose=is_root,
        )
        if is_root:
            load_path = os.path.join(
                REMOTE_BASE_PATH,
                f"reduced_k_max_{k_max}_depth_{depth}_treewidth_{treewidth}.pkl",
            )
            print(f"Loading from {load_path}...")
            with bf.BlobFile(load_path, "rb", streaming=False) as fh:
                reduced_ks_cached = pickle.load(fh)
            assert reduced_ks == reduced_ks_cached
