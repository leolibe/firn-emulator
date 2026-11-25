import numpy as np
from firn.heat import diffuse_heat, stable_dt


def setup_uniform(N=80, dz=0.5, kappa=1e-6):
    C = np.full(N, 917.0 * 2097.0)
    k = kappa * C
    return np.full(N, dz), k, C


def test_periodic_surface_forcing_matches_analytic():
    dz, k, C = setup_uniform()
    z = (np.arange(len(dz)) + 0.5) * dz
    omega = 2 * np.pi / (365 * 86400)
    d = np.sqrt(2 * 1e-6 / omega)
    T0, A = -10.0, 5.0
    T = np.full(len(dz), T0)
    for day in range(10 * 365):
        t = (day + 1) * 86400
        T, _ = diffuse_heat(T, dz, k, C, T0 + A * np.sin(omega * t), 86400.0)
    T_exact = T0 + A * np.exp(-z / d) * np.sin(omega * t - z / d)
    assert np.max(np.abs(T - T_exact)) < 0.1


def test_steady_state_is_uniform_surface_temperature():
    dz, k, C = setup_uniform(N=20)
    T = np.linspace(-20, -5, 20)
    for _ in range(6000):  # ~16 years, >> diffusion time L^2/kappa ~ 3 years
        T, _ = diffuse_heat(T, dz, k, C, -8.0, 86400.0)
    assert np.allclose(T, -8.0, atol=1e-3)


def test_input_not_modified():
    dz, k, C = setup_uniform(N=10)
    T = np.full(10, -10.0)
    diffuse_heat(T, dz, k, C, 0.0, 86400.0)
    assert np.all(T == -10.0)