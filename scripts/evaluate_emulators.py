import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from firn.forcing import make_forcing
from firn.physics import init_column, run, mass_budget_residual, percolate_refreeze
from firn.diagnostics import annual_sums, runoff_fraction, running_mean, tipping_year
from emulators.models import Emulator

N_SPINUP, TEST_SEED = 40, 100
CLIMATES = {"cold": (60, -13.5, -10.5), "warm": (100, -13.5, -8.5)}
FIG_DIR = ROOT / "figures"


def load(head):
    model = Emulator(head)
    model.load_state_dict(torch.load(ROOT / "models" / f"emulator_{head}.pt"))
    model.eval()
    return model


def hybrid_run(n_years, T_start, T_end):
    _, melt, T_surf = make_forcing(n_years, T_start, T_end, N_SPINUP, seed=TEST_SEED)
    k = N_SPINUP * 365
    col0 = init_column(T_init=T_start)
    run(col0, melt[:k], T_surf[:k])                      # physical spin-up
    out = {}
    for name, hydro in [("physics", percolate_refreeze),
                        ("A", load("A").scan), ("B", load("B").scan)]:
        t = time.time()
        daily, mass, _ = run(col0.copy(), melt[k:], T_surf[k:], hydrology=hydro)
        out[name] = dict(ann=annual_sums(daily),
                         residual=mass_budget_residual(daily, mass),
                         seconds=time.time() - t)
    return out


def main():
    lines = ["| climate | model | max mass error, 50 yr (kg m-2) | max mass error, full run | "
             "tipping year | annual runoff RMSE (kg m-2) | annual refreezing RMSE |",
             "|---|---|---|---|---|---|---|"]
    for climate, (n_years, T0, T1) in CLIMATES.items():
        res = hybrid_run(n_years, T0, T1)
        ref = res["physics"]["ann"]
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        years = np.arange(n_years)
        for name, style in [("physics", "k-"), ("A", "C1--"), ("B", "C0-.")]:
            r = res[name]
            frac = runoff_fraction(r["ann"])
            tip = tipping_year(frac)
            err50 = np.max(np.abs(r["residual"][:50 * 365]))
            err = np.max(np.abs(r["residual"]))
            rmse_run = np.sqrt(np.mean((r["ann"]["runoff"] - ref["runoff"]) ** 2))
            rmse_ref = np.sqrt(np.mean((r["ann"]["refreeze"] - ref["refreeze"]) ** 2))
            lines.append(f"| {climate} | {name} | {err50:.1e} | {err:.1e} | {tip} | "
                         f"{rmse_run:.1f} | {rmse_ref:.1f} |")
            print(lines[-1], f"({r['seconds']:.0f} s)")
            axes[0].plot(years, r["ann"]["refreeze"], style, label=name)
            axes[1].plot(years, running_mean(frac), style, label=name)
            axes[2].semilogy(np.arange(len(r["residual"])) / 365,
                             np.abs(r["residual"]) + 1e-16, style, label=name)
        axes[0].set_ylabel("refreezing (kg m$^{-2}$ yr$^{-1}$)")
        axes[1].set_ylabel("runoff / melt (5-yr mean)")
        axes[1].axhline(0.5, color="grey", ls=":")
        axes[2].set_ylabel("|cumulative mass error| (kg m$^{-2}$)")
        axes[2].set_xlabel("Years since start of warming")
        axes[0].legend()
        axes[0].set_title(f"Hybrid model, {climate} test climate ({T0} to {T1} °C)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"emulators_{climate}.png", dpi=150)
        plt.close(fig)
    (ROOT / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()