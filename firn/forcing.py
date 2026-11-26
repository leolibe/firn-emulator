import numpy as np

DAYS_PER_YEAR = 365          
PEAK_DAY = 196               # ~15 July, warmest day of the year. Cosinus seasonal cycle is shifted so that the peak occurs on this day.

def mean_temperature(n_years, T_start, T_end, n_spinup_years):
    spinup = np.full(n_spinup_years, T_start)
    ramp = np.linspace(T_start, T_end, n_years) # suppose a linear ramp from T_start to T_end over n_years
    return np.concatenate([spinup, ramp])


def make_forcing(n_years, T_start, T_end, n_spinup_years=40, amp=12.0, # 24C of diff between summer and winter, so amplitude is 12C
                 sigma_day=2.0, sigma_year=1.0, ddf=3.0, seed=0): # for each degree above 0C, 3mm of melt per day
    rng = np.random.default_rng(seed)
    n_total = n_spinup_years + n_years
    n_days = n_total * DAYS_PER_YEAR

    day = np.arange(n_days)
    year = day // DAYS_PER_YEAR          # year index of each day
    doy = day % DAYS_PER_YEAR            # day of year, 0..364

    T_mean = mean_temperature(n_years, T_start, T_end, n_spinup_years)[year]
    seasonal = amp * np.cos(2 * np.pi * (doy - PEAK_DAY) / DAYS_PER_YEAR)
    eta = rng.normal(0.0, sigma_year, n_total)[year]   # yearly anomaly
    eps = rng.normal(0.0, sigma_day, n_days)           # daily anomaly

    T_air = T_mean + seasonal + eta + eps
    melt = ddf * np.maximum(T_air, 0.0)
    T_surf = np.minimum(T_air, 0.0)
    return T_air, melt, T_surf
