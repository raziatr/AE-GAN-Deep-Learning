"""Latih DDPM (prediksi derau, L_simple) dengan U-Net ringan, T=200, EMA bobot."""
import copy, json, os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import *
from src.models import *

EPOCHS, LR = 6, 1e-3
set_seed(); torch.set_num_threads(1)
d, tr, va, te = loaders(128)
net, diff = UNet(), Diffusion(T=200)
ema = copy.deepcopy(net)
opt = torch.optim.Adam(net.parameters(), lr=LR)
L = {"train_loss": [], "val_loss": [], "epoch_time": []}
for ep in range(EPOCHS):
    net.train(); t0 = time.time(); tot = n = 0
    for x, _ in tr:
        x0 = x * 2 - 1
        t = torch.randint(0, diff.T, (len(x),)); eps = torch.randn_like(x0)
        loss = F.mse_loss(net(diff.q_sample(x0, t, eps), t), eps)       # L_simple
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()): pe.mul_(0.995).add_(p, alpha=0.005)
        tot += loss.item() * len(x); n += len(x)
    L["epoch_time"].append(time.time() - t0); L["train_loss"].append(tot / n)
    net.eval(); g = torch.Generator().manual_seed(0); vt = vn = 0
    with torch.no_grad():
        for x, _ in va:
            x0 = x * 2 - 1
            t = torch.randint(0, diff.T, (len(x),), generator=g); eps = torch.randn(x0.shape, generator=g)
            vt += F.mse_loss(net(diff.q_sample(x0, t, eps), t), eps).item() * len(x); vn += len(x)
    L["val_loss"].append(vt / vn)
    print("ddpm", ep + 1, round(L["train_loss"][-1], 5), round(L["val_loss"][-1], 5), round(L["epoch_time"][-1]), flush=True)
    torch.save(ema.state_dict(), os.path.join(CKPT, "ddpm_ema.pt"))
    json.dump({"ddpm": L}, open(os.path.join(LOGS, "ddpm.json"), "w"))