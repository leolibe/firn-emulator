"""Diagnostics: annual sums, runoff fraction, ice slabs and tipping year."""
import numpy as np

from firn.physics import RHO_IMP


def annual_sums(daily, days_per_year=365):
    """Sum each daily flux over years. Returns a dict of arrays of length n_years."""
    n_years = len(next(iter(daily.values()))) // days_per_year
    return {k: v[: n_years * days_per_year].reshape(n_years, days_per_year).sum(axis=1)
            for k, v in daily.items()}


def runoff_fraction(annual):
    """Fraction of the annual melt that runs off (0 when there is no melt)."""
    return np.divide(annual["runoff"], annual["melt"],
                     out=np.zeros_like(annual["melt"]), where=annual["melt"] > 0)


def running_mean(x, window=5):
    """Trailing running mean (the first window-1 values use the available years)."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        j = max(0, i - window + 1)
        out[i] = (c[i + 1] - c[j]) / (i + 1 - j)
    return out


def tipping_year(frac, threshold=0.5, window=5):
    """First year index where the running mean of the runoff fraction exceeds threshold.
    Criterion fixed before running the experiments. Returns None if never reached."""
    idx = np.flatnonzero(running_mean(frac, window) > threshold)
    return int(idx[0]) if idx.size else None


def shallowest_impermeable_depth(rho, depth, max_depth=20.0):
    """Depth (m) of the first impermeable layer within max_depth, NaN if none."""
    mask = (rho >= RHO_IMP) & (depth <= max_depth)
    return depth[mask][0] if mask.any() else np.nan