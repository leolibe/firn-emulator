import numpy as np
import torch
from torch import nn

from firn.physics import L_F
from firn.heat import C_ICE, RHO_ICE

N_FEAT = 6           # per-layer features
N_IN = 3 * N_FEAT + 2


def layer_features(m_ice, w, T, dz):
    """Normalised per-layer features (works on numpy arrays or torch tensors)."""
    lib = torch if isinstance(m_ice, torch.Tensor) else np
    rho = m_ice / dz
    cold = C_ICE * m_ice * lib.clip(-T, 0.0, None) / L_F        # refreezable mass (kg m-2)
    pore = lib.clip(RHO_ICE * dz - m_ice, 0.0, None)            # room left for ice (kg m-2)
    feats = [rho / RHO_ICE, T / 10.0, w / 10.0, dz, cold / 50.0, pore / 500.0]
    return lib.stack(feats, -1) if lib is torch else np.stack(feats, -1)


def caps(m_ice, T, dz):
    """Physical upper bounds on refreezing (kg m-2)."""
    cold = C_ICE * m_ice * torch.clamp(-T, min=0.0) / L_F
    pore = torch.clamp(RHO_ICE * dz - m_ice, min=0.0)
    return torch.minimum(cold, pore)


class Emulator(nn.Module):
    def __init__(self, head, hidden=64):
        super().__init__()
        assert head in ("A", "B")
        self.head = head
        self.mlp = nn.Sequential(nn.Linear(N_IN, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 4))

    @staticmethod
    def neighbours(f):
        """Concatenate features of layers i-1, i, i+1 (zero padding at the ends)."""
        pad = torch.zeros_like(f[..., :1, :])
        up = torch.cat([pad, f[..., :-1, :]], dim=-2)
        down = torch.cat([f[..., 1:, :], pad], dim=-2)
        return torch.cat([up, f, down], dim=-1)

    def probabilities(self, x):
        """Fractions of the available water: softmax for B (they sum to 1),
        independent sigmoids for A (they do not)."""
        out = self.mlp(x)
        return torch.softmax(out, dim=-1) if self.head == "B" else torch.sigmoid(out)

    def apply_head(self, p, avail, cap, lib=torch):
        """Turn fractions into fluxes (kg m-2): refreeze, retained water, runoff, down.
        Works with torch tensors (training) and numpy float64 arrays (inference)."""
        refreeze = lib.minimum(p[..., 0] * avail, cap)       # same physical cap for A and B
        retain = p[..., 1] * avail
        runoff = p[..., 2] * avail
        if self.head == "B":
            down = avail - refreeze - retain - runoff        # remainder: exact conservation
        else:
            down = p[..., 3] * avail                         # free: not tied to the others
        return refreeze, retain, runoff, down

    def forward_parallel(self, m_ice, w, T, dz, q_in):
        """Training mode: the true incoming flux of every layer is given (teacher forcing
        along depth), so all layers are computed at once."""
        f = self.neighbours(layer_features(m_ice, w, T, dz))
        avail = w + q_in
        x = torch.cat([f, (q_in / 10.0).unsqueeze(-1), (avail / 10.0).unsqueeze(-1)], -1)
        return self.apply_head(self.probabilities(x), avail, caps(m_ice, T, dz))

    @torch.no_grad()
    def scan(self, m_ice, w, T, dz, q_in):
        """Inference mode: top-down sweep, the water passed down by layer i is the input
        of layer i+1. Same signature and outputs as firn.physics.percolate_refreeze."""
        m_ice, w, T, dz = (np.array(a, dtype=float) for a in (m_ice, w, T, dz))
        n = len(m_ice)
        f = self.neighbours(torch.tensor(layer_features(m_ice, w, T, dz), dtype=torch.float32))
        cap = caps(*(torch.tensor(a, dtype=torch.float64) for a in (m_ice, T, dz))).numpy()
        refreeze_all = np.zeros(n)
        runoff_total = 0.0
        q = float(q_in)
        for i in range(n):
            avail = w[i] + q
            if avail <= 0.0:                    # dry layer: nothing happens
                q = 0.0
                continue
            x = torch.cat([f[i], torch.tensor([q / 10.0, avail / 10.0],
                                              dtype=torch.float32)]).unsqueeze(0)
            p = self.probabilities(x)[0].double().numpy()
            if self.head == "B":
                p = p / p.sum()                 # exact simplex in float64
            r, ret, ro, dn = self.apply_head(p, avail, cap[i], lib=np)
            dn = max(dn, 0.0)                   # B: only removes ~1e-16 round-off
            cold = C_ICE * m_ice[i] * max(-T[i], 0.0)
            m_ice[i] += r
            T[i] = -max(cold - L_F * r, 0.0) / (C_ICE * m_ice[i])
            w[i] = ret
            refreeze_all[i] = r
            runoff_total += ro
            q = dn
        return m_ice, w, T, refreeze_all, runoff_total, q


def n_params(model):
    return sum(p.numel() for p in model.parameters())