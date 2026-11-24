"""Latih Vanilla GAN (MLP) dan DCGAN: loss non-saturating, one-sided label smoothing 0.9, Adam(2e-4, beta1=0.5)."""
import json, os, sys, time
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import *
from src.models import *

ZD, LR = 100, 2e-4
set_seed(); torch.set_num_threads(1)
d, tr, va, te = loaders(128)
bce = nn.BCEWithLogitsLoss()
logs = {}

def run(name, G, D, epochs, init=False):
    set_seed()
    if init: G.apply(weights_init); D.apply(weights_init)
    oG = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    L = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": [], "epoch_time": []}
    for ep in range(epochs):
        G.train(); D.train(); t0 = time.time(); a = np_ = 0; acc = [0, 0, 0, 0]
        for x, _ in tr:
            real = x * 2 - 1; B = len(x)
            z = torch.randn(B, ZD); fake = G(z)
            lr_, lf_ = D(real), D(fake.detach())
            lossD = 0.5 * (bce(lr_, torch.full((B, 1), 0.9)) + bce(lf_, torch.zeros(B, 1)))
            oD.zero_grad(); lossD.backward(); oD.step()
            lossG = bce(D(fake), torch.ones(B, 1))
            oG.zero_grad(); lossG.backward(); oG.step()
            acc[0] += lossD.item(); acc[1] += lossG.item()
            acc[2] += torch.sigmoid(lr_).mean().item(); acc[3] += torch.sigmoid(lf_).mean().item(); np_ += 1
        for k, v in zip(["d_loss", "g_loss", "d_real", "d_fake"], acc): L[k].append(v / np_)
        L["epoch_time"].append(time.time() - t0)
        print(name, ep + 1, [round(v / np_, 3) for v in acc], round(L["epoch_time"][-1], 1), flush=True)
    torch.save(G.state_dict(), os.path.join(CKPT, f"{name}_G.pt"))
    torch.save(D.state_dict(), os.path.join(CKPT, f"{name}_D.pt")); logs[name] = L
    json.dump(logs, open(os.path.join(LOGS, f"{name}.json"), "w"))

import sys as _s
which = _s.argv[1] if len(_s.argv) > 1 else "both"
if which in ("gan", "both"): run("gan", VanillaG(ZD), VanillaD(), epochs=30)
if which in ("dcgan", "both"): run("dcgan", DCG(ZD), DCD(), epochs=8, init=True)