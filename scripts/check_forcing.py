import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `firn` importable
from firn.forcing import make_forcing, DAYS_PER_YEAR

ACCUMULATION = 500.0                        # kg m-2 yr-1, value according to Lewis et al. (2019), The Cryosphere in west-greenland
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_three_years(T_mean=-10.0):
    T_air, melt, T_surf = make_forcing(n_years=3, T_start=T_mean, T_end=T_mean,
                                       n_spinup_years=0)
    t = np.arange(len(T_air)) / DAYS_PER_YEAR          # time in years

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t, T_air, lw=0.8, label="T_air")
    ax1.plot(t, T_surf, "--", lw=0.8, label="T_surf (capped at 0 °C)")
    ax1.axhline(0, color="grey", lw=0.5)
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="lower right")
    ax1.set_title(f"Daily forcing, T_mean = {T_mean} °C")

    ax2.plot(t, melt, color="tab:red", lw=0.8)
    ax2.set_ylabel("Melt (kg m$^{-2}$ day$^{-1}$)")
    ax2.set_xlabel("Time (years)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "forcing_3yr.png", dpi=150)
    plt.close(fig)


def melt_ratio(T_mean, n_years=30, seed=0):
    _, melt, _ = make_forcing(n_years=n_years, T_start=T_mean, T_end=T_mean,
                              n_spinup_years=0, seed=seed)
    return melt.sum() / n_years / ACCUMULATION


def plot_calibration():
    T_values = np.arange(-15.0, -4.5, 0.5)
    ratios = np.array([melt_ratio(T) for T in T_values])

    for T, r in zip(T_values, ratios):                  # table in the terminal
        print(f"T_mean = {T:6.1f} °C   melt/accumulation = {r:5.2f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(T_values, ratios, "o-")
    ax.axhline(1.0, color="grey", ls="--", label="melt = accumulation")
    ax.set_xlabel("Mean annual air temperature (°C)")
    ax.set_ylabel("Annual melt / accumulation")
    ax.set_title("Calibration of the warming ramp")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "melt_vs_Tmean.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_three_years()
    plot_calibration()
    print(f"Figures saved in {FIG_DIR}")