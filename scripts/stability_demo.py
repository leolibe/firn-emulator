"""Stability of the explicit heat diffusion scheme.

Same uniform grid as the analytic test. The column starts at -10 °C and the surface is
suddenly set to -5 °C. We take n_steps single steps (no substepping) with
dt = factor * stable_dt, for several factors, and plot the final profiles.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firn.heat import diffuse_heat, stable_dt

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

N, DZ, KAPPA = 80, 0.5, 1e-6
dz = np.full(N, DZ)
C = np.full(N, 917.0 * 2097.0)
k = KAPPA * C
z = (np.arange(N) + 0.5) * DZ
dt_max = stable_dt(dz, k, C)
print(f"stable_dt = {dt_max / 86400:.3f} days")

factors = [0.9, 1.4, 1.6]
n_steps = 300
fig, axes = plt.subplots(1, len(factors), figsize=(10, 4), sharey=True)

for ax, f in zip(axes, factors):
    T = np.full(N, -10.0)
    for _ in range(n_steps):
        T, _ = diffuse_heat(T, dz, k, C, T_surf=-5.0, dt=f * dt_max, n_sub=1)
    print(f"factor {f}: max |T| = {np.max(np.abs(T)):.3g} °C")
    ax.plot(T, z)
    ax.set_title(f"dt = {f} x stable_dt")
    ax.set_xlabel("T (°C)")
axes[0].set_ylabel("Depth (m)")
axes[0].invert_yaxis()
fig.suptitle(f"Explicit scheme after {n_steps} steps")
fig.tight_layout()
fig.savefig(FIG_DIR / "stability_demo.png", dpi=150)
print(f"Figure saved in {FIG_DIR}")