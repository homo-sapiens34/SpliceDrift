"""
SpliceDrift — Age-related splicing drift analysis using Bayesian
Beta-Binomial regression.
"""

__version__ = "0.1.0"

from splicedrift.io import load_data, load_vastdb
from splicedrift.model import fit, fit_single
from splicedrift.plotting import plot_scatter, plot_event

__all__ = [
    "load_data",
    "load_vastdb",
    "fit",
    "fit_single",
    "plot_scatter",
    "plot_event",
]
