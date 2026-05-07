# Cumulant propagation for wide random MLPs

This repo contains code for the paper "[Estimating the expected output of wide random MLPs more efficiently than sampling.](https://arxiv.org/abs/2605.05179)", including:
- Implementation of the cumulant propagation ("kprop") algorithm in `src/mlp_kprop`
- Scripts for running experiments and plotting in `scripts/`


## Installation
Clone this repo. Then,
- To run scripts within this package:
    ```bash
    cd /path/to/mlp_kprop
    uv sync
    uv run python ./path/to/script
    ```
- To add this package as a dependency to another package:
    ```bash
    uv add --editable /path/to/mlp_kprop
    ```

## Basic usage
In the below example, we estimate the output mean of a width 16 MLP with 3 hidden layers (see [Layer conventions](#layer-conventions)) on a standard Gaussian input.
Here `k_max` is the maximum cumulant order that is to be tracked in full (referred to as $K$ in the paper).
```python
import torch
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop
from mlp_kprop.harmonic import HTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

n = 16
num_layers = 4
k_max = 2
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=num_layers).to(device)
K_in = {1: torch.zeros(n, device=device), 2: torch.eye(n, device=device)}   # Standard gaussian: 2nd cumulant is I, all other cumulants are 0
K_out: dict[int, HTensor] = mlp_kprop(mlp, K_in, k_max=k_max)
estimated_mean = K_out[1].to_tensor()
```

### Layer conventions
The `mlp.MLP` constructor's `num_layers` refers to the number of *linear* layers.
Since there is no activation after the last linear layer, this means that `MLP(num_layers=num_layers)` has `num_layers-1` hidden layers.

Layers are labeled `pre0`, `act0`, `pre1`, `act1`, ..., `pre{num_layers-2}`,`act{num_layers-2}`, where `pre{i}` are the pre-activations and `act{i}` are the activations.
In this scheme, the input can be thought of as `act{-1}`, and the output can be thought of as `pre{num_layers-1}`.
This convention is off by one from that used in the paper. That is,
- `act{i}` is $X_{i+1}$.
- `pre{i}` is $Z_{i+1}$.

## Sweeping widths and plotting
For example, to run all variants of cumulant propagation on ReLU MLPs over a sweep of widths, run:
```bash
uv run python scripts/kprop_by_width.py relu          # Generates ./data/relu
uv run python scripts/form_df.py relu                 # Generates ./data/relu/formed_df.pt
uv run python scripts/plot.py mse_vs_flops_main relu  # Generates ./plots/mse_vs_flops_main_relu.pdf
```


## Files
- `src/mlp_kprop/`
    - `cumulants.py`: Utilities for computing sample cumulants and moving between moment/cumulant/power-cumulant representations
    - `diagslice.py`: Provides `DSTensor`, which stores a symmetric tensor as a decomposition into diagonal slices, and provides utilities for working with this decomposition.
    - `factor_k3.py`, `factor_k4.py`: Implementations of the factorized version of cumulant propagation for maximum cumulant order 3 and 4, respectively.
    - `flop_utils.py`: Utilities for counting FLOPs. Includes adjustments for operations on symmetric tensors assuming a fictitious symmetric tensor kernel.
    - `harmonic.py`: Provides `HTensor`, which stores a symmetric tensor as a decomposition into harmonics, and provides utilities for working with this decomposition.
    - `kprop_ds.py`: Deprecated version of kprop using only the diagonal slice decomposition
    - `kprop_harmonic.py`: Implementation of kprop as described in the paper. Alternates between the harmonic and diagonal slice decompositions.
    - `mlp.py`: Implementation of MLP
    - `partitions.py`: Utilities for combinatorics of partitions, vector partitions, etc.
    - `plotting_utils.py`: Plotting utilities
    - `tensor_utils.py`: Basic tensor operations
    - `time_utils.py`: Timing utilities
    - `wick.py`: Wick (i.e. Hermite) coefficients of various activation functions
    - `symb/`: Symbolic cumulant propagation. See `symb/README.md`.
- `scripts/`
    - `kprop_by_width.py`: Main experiment runner. Computes sample cumulants and kprop estimates over a sweep of MLP estimates, and records MSEs.
    - `form_df:.py`: Consumes `kprop_by_width.py` outputs and condenses into a summary DataFrame
    - `plot.py`: Consumes `form_df.py` output and plots MSEs
