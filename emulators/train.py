from pathlib import Path

import numpy as np
import torch

from emulators.data import generate
from emulators.models import Emulator, n_params

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.npz"
MODELS = ROOT / "models"

# Cold training climates: the ramp stops at -10.5 °C, before the tipping point
# (which happens near -9.8 °C in the reference experiment).
TRAIN_SEEDS, VAL_SEEDS = list(range(1, 9)), [9]
N_YEARS, T_START, T_END = 60, -13.5, -10.5
LAMBDA_A = 1.0          # weight of the mass-balance penalty for emulator A
EPOCHS, BATCH, LR = 30, 256, 2e-3


def load_or_generate():
    if DATA.exists():
        d = np.load(DATA)
        return {k: d[k] for k in d.files}
    DATA.parent.mkdir(exist_ok=True)
    seeds = TRAIN_SEEDS + VAL_SEEDS
    parts = [generate([s], N_YEARS, T_START, T_END) for s in seeds]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d["seed"] = np.concatenate([np.full(len(p["q_in"]), s) for s, p in zip(seeds, parts)])
    np.savez_compressed(DATA, **d)
    return d


def to_tensors(d, mask):
    keys = ("m_ice", "w", "T", "dz", "q_in", "refreeze", "w_new", "runoff", "q_down")
    return [torch.tensor(d[k][mask], dtype=torch.float32) for k in keys]


def loss_fn(model, batch, lam):
    m, w, T, dz, q_in, refr, w_new, runoff, q_down = batch
    p_refr, p_ret, p_run, p_down = model.forward_parallel(m, w, T, dz, q_in)
    s = 10.0                                      # kg m-2 scale
    mse = (((p_refr - refr) / s) ** 2 + ((p_ret - w_new) / s) ** 2
           + ((p_run - runoff) / s) ** 2 + ((p_down - q_down) / s) ** 2).mean()
    budget = ((p_refr + p_ret + p_run + p_down - (w + q_in)) / s) ** 2
    return mse + lam * budget.mean(), mse, budget.mean()


def train(head, d, seed=0):
    torch.manual_seed(seed)
    tr = to_tensors(d, np.isin(d["seed"], TRAIN_SEEDS))
    va = to_tensors(d, np.isin(d["seed"], VAL_SEEDS))
    model = Emulator(head)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lam = LAMBDA_A if head == "A" else 0.0       # B conserves mass by construction
    n = len(tr[0])
    print(f"Emulator {head}: {n_params(model)} parameters, {n} training columns")
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss, _, _ = loss_fn(model, [t[idx] for t in tr], lam)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch % 5 == 4 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                _, mse, budget = loss_fn(model, va, lam)
            print(f"  epoch {epoch + 1:3d}  val mse {mse.item():.2e}  val budget {budget.item():.2e}")
    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS / f"emulator_{head}.pt")
    return model


if __name__ == "__main__":
    data = load_or_generate()
    for head in ("A", "B"):
        train(head, data)