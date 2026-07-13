# ARC infra config for the mlp_cumulant_propagation max-endpoint experiment.
# Use with:  python -m arc_infra.cli <cmd> kunalc --config infra/arc_infra_config.py
import os

CREATE_GPU_TYPE = "*100*"

BLOB_AUTH_LIFETIME = 12 * 3600

REPO_RELPATH = "max_kprop/mlp_cumulant_propagation"

CODE_ROOT_LOCAL = os.path.expanduser("~/code")
CODE_ROOT_REMOTE = "~/code"
# Exclude .venv (macOS binaries), .git, and local data/plots.
CODE_RELPATHS = [
    f"{REPO_RELPATH}/pyproject.toml",
    f"{REPO_RELPATH}/.python-version",
    f"{REPO_RELPATH}/uv.lock",
    f"{REPO_RELPATH}/README.md",
    f"{REPO_RELPATH}/UPSTREAM_COMMIT",
    f"{REPO_RELPATH}/GIT_COMMIT",
    f"{REPO_RELPATH}/pytest.ini",
    f"{REPO_RELPATH}/src",
    f"{REPO_RELPATH}/tests",
    f"{REPO_RELPATH}/scripts",
    f"{REPO_RELPATH}/configs",
    f"{REPO_RELPATH}/docs",
]

SETUP_ROOT_REMOTE = "~/setup"
SETUP_SCRIPT = f"""#!/bin/bash
set -e
export PATH="/usr/mpi/gcc/openmpi-4.1.5rc2/bin:$PATH"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd ~/code/{REPO_RELPATH}
uv sync
uv pip install boostedblob
"""

RUN_ROOT_REMOTE = "~/run"
RUN_ROOT_BLOB = f"gs://arc-ml/{os.environ['USER']}/run"
RUN_RELPATH_FORMAT = "{package}/{date}/{run_name}"
RUN_ENV = f"""
export PATH="/usr/mpi/gcc/openmpi-4.1.5rc2/bin:$PATH"
cd ~/code/{REPO_RELPATH}
source .venv/bin/activate
"""
