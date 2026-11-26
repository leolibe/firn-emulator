import numpy as np

C_ICE = 2097.0      # specific heat capacity of ice (J kg-1 K-1)
RHO_ICE = 917.0     # density of ice (kg m-3)


def conductivity(rho):
    return 2.2 * (rho / RHO_ICE) ** 2


def heat_capacity(rho):
    return rho * C_ICE


def conductances(dz, k):
    h = 0.5 * (dz[:-1] + dz[1:])                                   # centre-to-centre distance
    k_int = (dz[:-1] + dz[1:]) / (dz[:-1] / k[:-1] + dz[1:] / k[1:])  # harmonic mean
    G_int = k_int / h
    G_top = k[0] / (0.5 * dz[0])
    return G_top, G_int


def stable_dt(dz, k, C):
    G_top, G_int = conductances(dz, k)
    s = np.zeros_like(dz, dtype=float)
    s[0] += G_top
    s[:-1] += G_int          # interface below each layer
    s[1:] += G_int           # interface above each layer
    return np.min(C * dz / s)


def diffuse_heat(T, dz, k, C, T_surf, dt, n_sub=None, safety=0.9):
    T = np.array(T, dtype=float, copy=True)
    G_top, G_int = conductances(dz, k)
    if n_sub is None:
        n_sub = int(np.ceil(dt / (safety * stable_dt(dz, k, C))))
    h = dt / n_sub
    capacity = C * dz                      # J m-2 K-1, heat needed to warm a layer by 1 K

    F = np.zeros(len(T) + 1)               # fluxes at the N+1 interfaces, positive downward
    for _ in range(n_sub):
        F[0] = -G_top * (T[0] - T_surf)    # surface -> layer 0
        F[1:-1] = -G_int * (T[1:] - T[:-1])
        # F[-1] stays 0: no flux at the bottom
        T += h * (F[:-1] - F[1:]) / capacity   # in from above minus out below
    return T, n_sub