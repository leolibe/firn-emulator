import numpy as np
from firn.forcing import make_forcing


def test_shapes_and_physical_bounds():
    T_air, melt, T_surf = make_forcing(n_years=10, T_start=-12, T_end=-8,
                                       n_spinup_years=5)
    assert T_air.shape == melt.shape == T_surf.shape == (15 * 365,)
    assert np.all(melt >= 0)
    assert np.all(T_surf <= 0)


def test_reproducible():
    a = make_forcing(10, -12, -8, seed=1)[0]
    b = make_forcing(10, -12, -8, seed=1)[0]
    assert np.array_equal(a, b)