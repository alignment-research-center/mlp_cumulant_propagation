import numpy as np
from typing import Optional
import torch
from matplotlib import pyplot as plt
import math
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from typing import Callable
from functools import wraps
import os

def wls_loglog(x: list[float], y: list[float], w: str='unif'):
    """
    Weighted least squares in log-log space.
    Model: log y = a + b log x  =>  y = c * x**b with c = exp(a).
    Weight ∝ x, clipped below at 1e-12.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        return None, None
    x = x[m]
    y = y[m]

    lx = np.log(x)
    ly = np.log(y)

    if isinstance(w, str):
        if w == 'unif':
            w = np.ones_like(x)
        elif w == 'last':
            w = np.zeros_like(x)
            w[-2:] = 1.
        elif w == 'x':
            w = x
        elif w == 'exp':
            w = np.clip(2**x, 1e-12, None)
    sw = np.sqrt(w)

    X = np.c_[np.ones_like(lx), lx]
    beta, *_ = np.linalg.lstsq(X * sw[:, None], ly * sw, rcond=None)
    a, b = beta
    c = math.exp(a)
    return c, b

def setup_colors(ks, color):
    """Set up colormap and normalization for k values."""
    kmin, kmax = min(ks), max(ks)
    norm = (
        Normalize(vmin=kmin - 0.5, vmax=kmax + 0.5)
        if kmin == kmax
        else Normalize(vmin=-2, vmax=kmax)
    )
    cmap = get_cmap(color)
    return cmap, norm, kmax

def fmt_power_law(c, b, varname='n', sigfig_c=3, sigfig_b=3, sci=False):
    """Format c * x^b as a LaTeX string.
    sci=False: positional notation (e.g. '0.201').
    sci=True:  scientific notation (e.g. '2.01 \\cdot 10^{-1}').
    """
    if c is None:
        return None
    if sci:
        exp = int(np.floor(np.log10(abs(c))))
        mantissa = c / 10**exp
        dec = max(sigfig_c - 1, 0)
        m_str = f'{mantissa:.{dec}f}'.rstrip('0').rstrip('.')
        c_str = f'{m_str} \\cdot 10^{{{exp}}}' if exp != 0 else m_str
    else:
        dec = max(sigfig_c - 1 - int(np.floor(np.log10(abs(c)))), 0)
        c_str = np.format_float_positional(c, precision=dec, unique=False, trim='-')
    return f'${c_str} \\cdot {varname}^{{{b:.{sigfig_b}g}}}$'


def pdf_plot(name: str, width: float = 3.487, height: float | None = None, labelsize: int = 8):
    """Decorator factory to save a plot as PDF. The decorated function creates its own figure."""
    if height is None:
        height = width / 1.618
    def decorator(plot_f: Callable):
        @wraps(plot_f)
        def wrapper(*args, **kwargs):
            plt.rc('font', family='serif', serif='Times')
            plt.rc('text', usetex=True)
            plt.rc('xtick', labelsize=labelsize)
            plt.rc('ytick', labelsize=labelsize)
            plt.rc('axes', labelsize=labelsize)

            result = plot_f(*args, **kwargs)

            plt.gcf().set_size_inches(width, height)
            os.makedirs('../plots', exist_ok=True)
            plt.savefig(f'../plots/{name}.pdf', bbox_inches='tight')
            plt.close()
            return result
        return wrapper
    return decorator