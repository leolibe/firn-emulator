import numpy as np
from firn.forcing import make_forcing
from firn.physics import (init_column, run, mass_budget_residual,
                          percolate_refreeze, RHO_ICE)


def small_column(rho, T, w=None, dz=1.0):
    n = len(rho)
    return (np.asarray(rho, float) * dz, np.zeros(n) if w is None else np.asarray(w, float),
            np.asarray(T, float), np.full(n, dz))


def test_hydrology_conserves_mass_exactly():
    m, w, T, dz = small_column([400, 500, 600, 700], [-10, -2, -0.5, -5])
    q_in = 80.0
    m2, w2, T2, refreeze, runoff, drainage = percolate_refreeze(m, w, T, dz, q_in)
    before = m.sum() + w.sum() + q_in
    after = m2.sum() + w2.sum() + runoff + drainage
    assert abs(after - before) < 1e-10


def test_cold_layer_refreezes_everything():
    m, w, T, dz = small_column([400, 400], [-15, -15])
    _, w2, T2, refreeze, runoff, drainage = percolate_refreeze(m, w, T, dz, 10.0)
    assert np.isclose(refreeze.sum(), 10.0) and runoff == 0.0 and drainage == 0.0
    assert np.all(T2 <= 0.0) and T2[0] > -15.0          # latent heat warmed the layer


def test_impermeable_layer_causes_runoff():
    # temperate (0 °C) top layer cannot refreeze; layer below is impermeable
    m, w, T, dz = small_column([400, 850, 400], [0.0, 0.0, -10.0])
    _, _, _, refreeze, runoff, drainage = percolate_refreeze(m, w, T, dz, 100.0)
    assert runoff > 0.0 and refreeze[2] == 0.0 and drainage == 0.0


def test_full_run_closes_mass_budget_and_stays_physical():
    _, melt, T_surf = make_forcing(n_years=20, T_start=-11.0, T_end=-8.0,
                                   n_spinup_years=10, seed=1)
    col = init_column()
    daily, mass, _ = run(col, melt, T_surf)
    assert np.max(np.abs(mass_budget_residual(daily, mass))) < 1e-8
    assert np.all(col.T <= 1e-12)
    assert np.all(col.w >= 0.0) and np.all(col.rho <= RHO_ICE + 1e-9)