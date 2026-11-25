"""Heat conduction in the firn column.

Explicit finite-volume scheme for  rho*c * dT/dt = d/dz (k dT/dz),  z positive downward.
Layer 0 is at the surface. Each layer is a cell of thickness dz[i] with temperature T[i]
at its centre.

Boundary conditions:
  - top    : Dirichlet, T = T_surf at the top of layer 0 (distance dz[0]/2 from its centre)
  - bottom : no heat flux
"""
import numpy as np

C_ICE = 2097.0      # specific heat capacity of ice (J kg-1 K-1)
RHO_ICE = 917.0     # density of ice (kg m-3)


def conductivity(rho):
    """Firn thermal conductivity (W m-1 K-1).

    Simple assumption: k = 2.2 * (rho / rho_ice)^2, i.e. pure ice (2.2) scaled down
    by the amount of air. Good enough for a toy model; documented in the README.
    """
    return 2.2 * (rho / RHO_ICE) ** 2


def heat_capacity(rho):
    """Volumetric heat capacity C = rho * c_ice (J m-3 K-1). Air is neglected."""
    return rho * C_ICE


def conductances(dz, k):
    """Thermal conductances (W m-2 K-1).

    Returns
    -------
    G_top : conductance between the surface and the centre of layer 0
    G_int : array of length N-1, conductance between the centres of layers i and i+1
    """
    h = 0.5 * (dz[:-1] + dz[1:])                                   # centre-to-centre distance
    k_int = (dz[:-1] + dz[1:]) / (dz[:-1] / k[:-1] + dz[1:] / k[1:])  # harmonic mean
    G_int = k_int / h
    G_top = k[0] / (0.5 * dz[0])
    return G_top, G_int


def stable_dt(dz, k, C):
    """Largest explicit time step (s) satisfying, for every layer i,
        dt * (sum of conductances touching layer i) / (C_i * dz_i) <= 1.

    This is a sufficient condition (no new temperature extrema are created).
    On a uniform grid it gives C dz^2 / (3k), limited by the surface layer;
    interior layers alone would allow C dz^2 / (2k).
    """
    G_top, G_int = conductances(dz, k)
    s = np.zeros_like(dz, dtype=float)
    s[0] += G_top
    s[:-1] += G_int          # interface below each layer
    s[1:] += G_int           # interface above each layer
    return np.min(C * dz / s)


def diffuse_heat(T, dz, k, C, T_surf, dt, n_sub=None, safety=0.9):
    """Advance the temperature profile by dt seconds.

    Parameters
    ----------
    T, dz, k, C : arrays of length N (°C, m, W m-1 K-1, J m-3 K-1)
    T_surf      : surface temperature (°C), held constant during dt
    dt          : total time step (s), e.g. 86400 for one day
    n_sub       : number of substeps; if None, chosen from stable_dt with a safety factor

    Returns
    -------
    T_new : updated temperatures (T itself is not modified)
    n_sub : number of substeps actually used
    """
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