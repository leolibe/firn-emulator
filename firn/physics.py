"""Simplified firn column: accumulation, surface melt, percolation, refreezing,
heat conduction and dry compaction.

Units: masses in kg m-2 (= mm water equivalent), lengths in m, temperatures in °C,
time step = one day. Layer 0 is at the surface. The number of layers N stays fixed.

State of each layer i:
    m_ice[i]  ice mass (kg m-2)
    w[i]      liquid water mass (kg m-2)
    T[i]      temperature (°C), <= 0
    dz[i]     thickness (m)          -> density rho = m_ice / dz
"""
from dataclasses import dataclass, replace

import numpy as np

from firn.heat import C_ICE, RHO_ICE, conductivity, heat_capacity, diffuse_heat

RHO_W = 1000.0          # water density (kg m-3)
L_F = 3.34e5            # latent heat of fusion (J kg-1)
RHO_SNOW = 350.0        # density of fresh snow (kg m-3)
RHO_IMP = 830.0         # impermeability threshold, ~pore close-off (kg m-3)
THETA_IRR = 0.04        # irreducible water, fraction of pore volume
DZ_MIN = 0.1            # thinner surface layers are merged (m)
ACC_DAY = 273           # day of year when the annual snow layer is added (~1 Oct)
R_GAS = 8.314           # J mol-1 K-1


@dataclass
class Column:
    m_ice: np.ndarray
    w: np.ndarray
    T: np.ndarray
    dz: np.ndarray

    @property
    def rho(self):
        return self.m_ice / self.dz

    @property
    def total_mass(self):
        return self.m_ice.sum() + self.w.sum()

    def copy(self):
        return replace(self, m_ice=self.m_ice.copy(), w=self.w.copy(),
                       T=self.T.copy(), dz=self.dz.copy())


def init_column(n_layers=40, accumulation=500.0, T_init=-13.5):
    """Uniform fresh-snow column; a spin-up replaces it with a realistic profile."""
    m = np.full(n_layers, accumulation)
    return Column(m_ice=m, w=np.zeros(n_layers), T=np.full(n_layers, T_init),
                  dz=m / RHO_SNOW)


# --------------------------------------------------------------------------------------
# Layer bookkeeping: N stays fixed. Every operation returns the mass that entered or
# left the column so that the mass budget can be closed exactly.
# --------------------------------------------------------------------------------------
def _shift_up_and_import_bottom(col):
    """Remove layer 0, shift everything up, duplicate the bottom layer.
    Returns the imported mass (the duplicated bottom layer)."""
    for name in ("m_ice", "w", "T", "dz"):
        a = getattr(col, name)
        a[:-1] = a[1:].copy()          # a[-1] is unchanged: duplicate of the old bottom
    return col.m_ice[-1] + col.w[-1]


def add_snow_layer(col, accumulation, T_surf):
    """Insert a fresh snow layer at the top and drop the bottom layer.
    Returns (mass_added, mass_exported)."""
    exported = col.m_ice[-1] + col.w[-1]
    for name in ("m_ice", "w", "T", "dz"):
        a = getattr(col, name)
        a[1:] = a[:-1].copy()
    col.m_ice[0] = accumulation
    col.w[0] = 0.0
    col.T[0] = min(T_surf, 0.0)
    col.dz[0] = accumulation / RHO_SNOW
    return accumulation, exported


def surface_melt(col, melt):
    """Melt `melt` kg m-2 of ice from the top, layer by layer (density kept constant).
    A layer that melts out entirely releases its water and is removed.
    Returns (q_surface, mass_imported)."""
    q = 0.0
    imported = 0.0
    remaining = melt
    while remaining > 0.0:
        rho0 = col.rho[0]
        take = min(remaining, col.m_ice[0])
        col.m_ice[0] -= take
        col.dz[0] = col.m_ice[0] / rho0
        q += take
        remaining -= take
        if col.m_ice[0] <= 1e-9:                 # layer gone: release its water
            q += col.w[0] + col.m_ice[0]
            imported += _shift_up_and_import_bottom(col)
    return q, imported


def merge_thin_top_layer(col):
    """Merge a too-thin surface layer into the next one (mass and heat conserved).
    Returns the imported mass (0 if nothing was merged)."""
    if col.dz[0] >= DZ_MIN:
        return 0.0
    m0, m1 = col.m_ice[0], col.m_ice[1]
    col.T[1] = (m0 * col.T[0] + m1 * col.T[1]) / (m0 + m1)   # ice heat content
    col.m_ice[1] += m0
    col.w[1] += col.w[0]
    col.dz[1] += col.dz[0]
    return _shift_up_and_import_bottom(col)


# --------------------------------------------------------------------------------------
# Hydrology: the operator the neural emulators will replace.
# --------------------------------------------------------------------------------------
def percolate_refreeze(m_ice, w, T, dz, q_in):
    """Tipping-bucket percolation with refreezing, from the top down.

    For each layer:
      1. if impermeable (rho >= RHO_IMP), the incoming water runs off;
         otherwise it is added to the layer;
      2. refreezing, limited by available water, cold content and pore space;
         the released latent heat warms the layer;
      3. water above the irreducible content drains to the layer below
         (or runs off if the layer is impermeable).

    Pure function: inputs are not modified.
    Returns m_ice, w, T, refreeze (per layer), runoff, drainage (out of the bottom).
    """
    m_ice, w, T = m_ice.copy(), w.copy(), T.copy()
    n = len(m_ice)
    refreeze = np.zeros(n)
    runoff = 0.0
    q = q_in
    for i in range(n):
        impermeable = m_ice[i] / dz[i] >= RHO_IMP
        if impermeable:
            runoff += q
            q = 0.0
        else:
            w[i] += q
            q = 0.0
        # refreezing
        cold_content = C_ICE * m_ice[i] * max(-T[i], 0.0)            # J m-2
        r = min(w[i], cold_content / L_F, max(RHO_ICE * dz[i] - m_ice[i], 0.0))
        if r > 0.0:
            m_ice[i] += r
            w[i] -= r
            refreeze[i] = r
            T[i] = -(cold_content - L_F * r) / (C_ICE * m_ice[i])
        # retention
        pore = max(dz[i] - m_ice[i] / RHO_ICE, 0.0)
        excess = max(w[i] - THETA_IRR * RHO_W * pore, 0.0)
        w[i] -= excess
        if impermeable:
            runoff += excess
        else:
            q = excess
    return m_ice, w, T, refreeze, runoff, q


# --------------------------------------------------------------------------------------
# Dry compaction (Herron & Langway, 1980)
# --------------------------------------------------------------------------------------
def compaction(col, accumulation, dt_years):
    """Herron & Langway (1980) densification of the dry firn matrix.
    Stage 1 (rho < 550): drho/dt = k0 * A * (rho_i - rho),        k0 = 11  exp(-10160/RT)
    Stage 2 (rho >= 550): drho/dt = k1 * sqrt(A) * (rho_i - rho), k1 = 575 exp(-21400/RT)
    with A in m w.e. a-1 and densities in Mg m-3. Mass is unchanged, dz shrinks."""
    A = accumulation / RHO_W
    Tk = col.T + 273.15
    rho = col.rho / 1000.0
    k0 = 11.0 * np.exp(-10160.0 / (R_GAS * Tk))
    k1 = 575.0 * np.exp(-21400.0 / (R_GAS * Tk))
    rate = np.where(rho < 0.55, k0 * A, k1 * np.sqrt(A)) * (RHO_ICE / 1000.0 - rho)
    rho_new = np.minimum(rho + rate * dt_years, RHO_ICE / 1000.0) * 1000.0
    col.dz = col.m_ice / rho_new


# --------------------------------------------------------------------------------------
# One day and a full run
# --------------------------------------------------------------------------------------
def step(col, melt, T_surf, doy, accumulation=500.0, hydrology=percolate_refreeze):
    """Advance the column by one day (in place). Returns a dict of daily fluxes.
    `hydrology` can be swapped for a neural emulator with the same signature."""
    d = dict(acc=0.0, exported=0.0, imported=0.0, melt=melt)
    if doy == ACC_DAY:
        d["acc"], d["exported"] = add_snow_layer(col, accumulation, T_surf)

    q, imp = surface_melt(col, melt)
    d["imported"] += imp

    col.m_ice, col.w, col.T, refreeze, runoff, drainage = hydrology(
        col.m_ice, col.w, col.T, col.dz, q)
    d.update(refreeze=refreeze.sum(), runoff=runoff, drainage=drainage)

    k = conductivity(col.rho)
    C = heat_capacity(col.rho)
    col.T, _ = diffuse_heat(col.T, col.dz, k, C, T_surf, 86400.0)

    compaction(col, accumulation, 1.0 / 365.0)
    d["imported"] += merge_thin_top_layer(col)
    return d


def run(col, melt, T_surf, accumulation=500.0, hydrology=percolate_refreeze):
    """Run the model over the whole forcing. Returns daily fluxes, the daily total mass
    and the end-of-September density/temperature profiles of each year."""
    n_days = len(melt)
    keys = ("acc", "exported", "imported", "melt", "refreeze", "runoff", "drainage")
    daily = {key: np.zeros(n_days) for key in keys}
    mass = np.zeros(n_days + 1)
    mass[0] = col.total_mass
    profiles = {"rho": [], "T": [], "depth": []}
    for day in range(n_days):
        doy = day % 365
        if doy == ACC_DAY:                      # snapshot just before the new layer
            profiles["rho"].append(col.rho.copy())
            profiles["T"].append(col.T.copy())
            profiles["depth"].append(np.cumsum(col.dz) - 0.5 * col.dz)
        d = step(col, melt[day], T_surf[day], doy, accumulation, hydrology)
        for key in keys:
            daily[key][day] = d[key]
        mass[day + 1] = col.total_mass
    profiles = {k: np.array(v) for k, v in profiles.items()}
    return daily, mass, profiles


def mass_budget_residual(daily, mass):
    """Cumulative difference between the change in column mass and the boundary fluxes."""
    inflow = daily["acc"] + daily["imported"]
    outflow = daily["exported"] + daily["runoff"] + daily["drainage"]
    expected = mass[0] + np.cumsum(inflow - outflow)
    return mass[1:] - expected