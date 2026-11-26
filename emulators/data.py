import numpy as np

from firn.forcing import make_forcing
from firn.physics import init_column, run, percolate_refreeze, RHO_IMP


class Recorder:
    """Wraps percolate_refreeze and stores its inputs and outputs on wet days."""

    def __init__(self):
        self.records = []

    def __call__(self, m_ice, w, T, dz, q_in):
        out = percolate_refreeze(m_ice, w, T, dz, q_in)
        if q_in > 0.0 or w.sum() > 0.0:
            self.records.append((m_ice.copy(), w.copy(), T.copy(), dz.copy(), q_in,
                                 out[1].copy(), out[3].copy()))
        return out


def layer_targets(m_ice, w, dz, q_in, w_new, refreeze):
    """Recover incoming flux, runoff and downward flux of every layer from the budget
    available_i = w_i + q_i = refreeze_i + w_new_i + runoff_i + q_{i+1}.
    In the physical model only impermeable layers produce runoff, and they pass nothing down."""
    n = len(m_ice)
    q = np.zeros(n + 1)
    runoff = np.zeros(n)
    q[0] = q_in
    impermeable = m_ice / dz >= RHO_IMP
    for i in range(n):
        if impermeable[i]:
            # incoming water runs off directly; the layer's own excess also runs off
            out = w[i] - refreeze[i] - w_new[i]
            runoff[i] = q[i] + out
            q[i + 1] = 0.0
        else:
            q[i + 1] = w[i] + q[i] - refreeze[i] - w_new[i]
    q[1:] = np.maximum(q[1:], 0.0)          # remove round-off negatives
    return q[:-1], runoff, q[1:]


def generate(seeds, n_years, T_start, T_end, n_spinup_years=40):
    """Run the physical model for several seeds and return stacked per-column arrays."""
    keys = ("m_ice", "w", "T", "dz", "q_in", "refreeze", "w_new", "runoff", "q_down")
    data = {k: [] for k in keys}
    for seed in seeds:
        _, melt, T_surf = make_forcing(n_years, T_start, T_end, n_spinup_years, seed=seed)
        rec = Recorder()
        run(init_column(T_init=T_start), melt, T_surf, hydrology=rec)
        for m, w, T, dz, q0, w_new, refr in rec.records:
            q_in_layer, runoff, q_down = layer_targets(m, w, dz, q0, w_new, refr)
            for k, v in zip(keys, (m, w, T, dz, q_in_layer, refr, w_new, runoff, q_down)):
                data[k].append(v)
    return {k: np.array(v) for k, v in data.items()}