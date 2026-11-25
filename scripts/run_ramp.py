"""Main physical experiment: 100-year warming ramp after a 40-year spin-up.

Produces figures/ramp.png: forcing and annual water budget, runoff fraction with the
tipping year, and the evolution of the density profile (ice slab formation).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firn.forcing import make_forcing, mean_temperature
from firn.physics import init_column, run, mass_budget_residual, RHO_IMP
from firn.diagnostics import (annual_sums, runoff_fraction, running_mean, tipping_year,
                              shallowest_impermeable_depth)

N_SPINUP, N_YEARS, T_START, T_END, SEED = 40, 100, -13.5, -8.5, 0
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

_, melt, T_surf = make_forcing(N_YEARS, T_START, T_END, n_spinup_years=N_SPINUP, seed=SEED)
col = init_column(T_init=T_START)
daily, mass, prof = run(col, melt, T_surf)
print(f"max mass budget residual: {np.max(np.abs(mass_budget_residual(daily, mass))):.2e} kg m-2")

# drop the spin-up
ann = {k: v[N_SPINUP:] for k, v in annual_sums(daily).items()}
prof = {k: v[N_SPINUP:] for k, v in prof.items()}
years = np.arange(N_YEARS)
T_mean = mean_temperature(N_YEARS, T_START, T_END, 0)
frac = runoff_fraction(ann)
y_tip = tipping_year(frac)
slab_depth = np.array([shallowest_impermeable_depth(r, d) for r, d in zip(prof["rho"], prof["depth"])])
print(f"tipping year: {y_tip}  (T_mean = {T_mean[y_tip]:.2f} °C)" if y_tip is not None
      else "no tipping")

# density on a regular depth grid for the image
z = np.linspace(0, 25, 251)
rho_img = np.array([np.interp(z, d, r) for r, d in zip(prof["rho"], prof["depth"])]).T

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
                         gridspec_kw={"height_ratios": [1, 0.8, 1.4]})
ax = axes[0]
ax.plot(years, ann["melt"], label="melt", color="tab:red")
ax.plot(years, ann["refreeze"], label="refreezing", color="tab:blue")
ax.plot(years, ann["runoff"], label="runoff", color="black")
ax.set_ylabel("kg m$^{-2}$ yr$^{-1}$")
ax.legend(loc="upper left")
ax2 = ax.twinx()
ax2.plot(years, T_mean, "--", color="grey")
ax2.set_ylabel("T_mean (°C)", color="grey")
ax.set_title("Warming ramp: annual water budget")

ax = axes[1]
ax.plot(years, frac, color="lightgrey", label="annual")
ax.plot(years, running_mean(frac), color="black", label="5-yr running mean")
ax.axhline(0.5, color="grey", ls=":")
if y_tip is not None:
    for a in axes:
        a.axvline(y_tip, color="tab:orange", lw=1.5)
ax.set_ylabel("runoff / melt")
ax.set_ylim(0, 1)
ax.legend(loc="upper left")

ax = axes[2]
im = ax.pcolormesh(years, z, rho_img, cmap="viridis", vmin=350, vmax=917, shading="auto")
ax.contour(years, z, rho_img, levels=[RHO_IMP], colors="white", linewidths=0.8)
ax.plot(years, slab_depth, ".", color="tab:orange", ms=3, label="shallowest impermeable layer")
ax.invert_yaxis()
ax.set_ylabel("Depth (m)")
ax.set_xlabel("Years since start of warming")
ax.legend(loc="lower left")
fig.colorbar(im, ax=axes[2], label="density (kg m$^{-3}$)", orientation="horizontal", pad=0.15)

fig.tight_layout()
fig.savefig(FIG_DIR / "ramp.png", dpi=150)
print(f"Figure saved in {FIG_DIR}")