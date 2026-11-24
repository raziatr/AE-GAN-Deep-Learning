"""Latih Vanilla AE, Sparse AE, dan VAE pada Fashion-MNIST."""
import json, os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import *
from src.models import *

EPOCHS, LR, LAM_SPARSE, ZD = 20, 1e-3, 0.01, 64
set_seed(); torch.set_num_threads(1)
d, tr, va, te = loaders(128)
logs = {}

def run(name, model, step_loss):
    set_seed()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    L = {"train_loss": [], "val_loss": [], "epoch_time": []}
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = n = 0
        for x, _ in tr:
            loss = step_loss(model, x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
        L["epoch_time"].append(time.time() - t0); L["train_loss"].append(tot / n)
        model.eval(); vt = vn = 0
        with torch.no_grad():
            for x, _ in va:
                vt += step_loss(model, x).item() * len(x); vn += len(x)
        L["val_loss"].append(vt / vn)
        print(name, ep + 1, round(L["train_loss"][-1], 5), round(L["val_loss"][-1], 5), flush=True)
    torch.save(model.state_dict(), os.path.join(CKPT, f"{name}.pt")); logs[name] = L

def ae_loss(m, x): return F.mse_loss(m(x)[0], x)
def sae_loss(m, x):
    xr, z, h = m(x); return F.mse_loss(xr, x) + LAM_SPARSE * h.abs().mean()
def vae_loss(m, x):                   
    xr, mu, lv = m(x)
    bce = F.binary_cross_entropy(xr.flatten(1), x.flatten(1), reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp()) / len(x)
    return bce + kl

run("ae", AE(ZD), ae_loss)
run("sae", SparseAE(ZD), sae_loss)
run("vae", VAE(ZD), vae_loss)
json.dump(logs, open(os.path.join(LOGS, "ae_family.json"), "w"))