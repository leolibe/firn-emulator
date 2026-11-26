import numpy as np
import torch
from emulators.models import Emulator, n_params


def random_column(rng, n=40):
    rho = rng.uniform(350, 900, n)
    dz = rng.uniform(0.3, 1.5, n)
    return rho * dz, rng.uniform(0, 5, n), rng.uniform(-15, 0, n), dz


def test_same_number_of_parameters():
    assert n_params(Emulator("A")) == n_params(Emulator("B"))


def test_flux_emulator_conserves_mass_even_untrained():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = Emulator("B")                       # random weights: no learning involved
    for _ in range(20):
        m, w, T, dz = random_column(rng)
        q_in = rng.uniform(0, 30)
        m2, w2, T2, refreeze, runoff, drainage = model.scan(m, w, T, dz, q_in)
        before = m.sum() + w.sum() + q_in
        after = m2.sum() + w2.sum() + runoff + drainage
        assert abs(after - before) < 1e-10
        assert np.all(T2 <= 0.0) and np.all(w2 >= 0.0)


def test_penalty_emulator_does_not_conserve_mass_untrained():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m, w, T, dz = random_column(rng)
    m2, w2, _, _, runoff, drainage = Emulator("A").scan(m, w, T, dz, 20.0)
    assert abs(m2.sum() + w2.sum() + runoff + drainage - (m.sum() + w.sum() + 20.0)) > 1e-3