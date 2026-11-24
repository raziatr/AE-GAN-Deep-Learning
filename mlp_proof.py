"""Bagian 2: pembuktian batasan MLP Classifier untuk image generation (Tahap 1-5)."""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import *
from src.models import *

set_seed(); torch.set_num_threads(1)
d, tr, va, te = loaders(128)
xte, yte = d["test"]
# Tahap 1: latih classifier 784-256-64-10 (CE) sampai akurasi test >= 88%
clf = MLPClassifier(); opt = torch.optim.Adam(clf.parameters(), 1e-3)
accs = []; tl = []
for ep in range(40):
    clf.train(); tot = n = 0
    for x, y in tr:
        loss = F.cross_entropy(clf(x), y); opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(x); n += len(x)
    clf.eval()
    with torch.no_grad(): acc = (clf(xte).argmax(1) == yte).float().mean().item()
    accs.append(acc); tl.append(tot / n); print("clf", ep + 1, round(tl[-1], 4), round(acc, 4), flush=True)
    if ep >= 14 and acc >= 0.88: break
torch.save(clf.state_dict(), os.path.join(CKPT, "mlp.pt"))
for p in clf.parameters(): p.requires_grad_(False)       
W1, W2, W3 = clf.fc1.weight, clf.fc2.weight, clf.fc3.weight     

with torch.no_grad():
    # Tahap 2: proyeksi balik vektor one-hot kelas via transpos bobot (tanpa pelatihan)
    Y = torch.eye(10)
    zh = F.relu(Y @ W3); h1 = F.relu(zh @ W2); x_cls = torch.sigmoid(h1 @ W1).view(-1, 1, 28, 28)
    acc_cls = (clf(x_cls).argmax(1) == torch.arange(10)).float().mean().item()
    # Tahap 3: proyeksi balik z_MLP citra uji
    z_te = clf.features(xte)
    x_lat = torch.sigmoid(F.relu(z_te @ W2) @ W1).view(-1, 1, 28, 28)

# Tahap 4: frozen latent decoding 64 -> 256 -> 784 (MSE)
dec = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())
od = torch.optim.Adam(dec.parameters(), 1e-3); L = {"train_loss": [], "val_loss": []}
with torch.no_grad(): ztr = clf.features(d["train"][0]); zva = clf.features(d["val"][0])
from torch.utils.data import DataLoader, TensorDataset
dl = DataLoader(TensorDataset(ztr, d["train"][0].flatten(1)), 128, shuffle=True)
for ep in range(20):
    tot = n = 0
    for z, x in dl:
        loss = F.mse_loss(dec(z), x); od.zero_grad(); loss.backward(); od.step(); tot += loss.item() * len(z); n += len(z)
    with torch.no_grad(): vl = F.mse_loss(dec(zva), d["val"][0].flatten(1)).item()
    L["train_loss"].append(tot / n); L["val_loss"].append(vl); print("mlp_dec", ep + 1, round(tot / n, 5), round(vl, 5), flush=True)
torch.save(dec.state_dict(), os.path.join(CKPT, "mlp_dec.pt"))
# Tahap 5: sampling dari N(mu_z, diag(sigma_z^2)) pada ruang laten test
with torch.no_grad():
    mu, sd = z_te.mean(0), z_te.std(0)
    set_seed(); zr = mu + sd * torch.randn(1000, 64)
    x_rand = dec(zr).view(-1, 1, 28, 28)
    x_rec = dec(z_te).view(-1, 1, 28, 28)
torch.save({"x_cls": x_cls, "x_lat": x_lat, "x_rec": x_rec, "x_rand": x_rand, "z_te": z_te}, os.path.join(CKPT, "mlp_tensors.pt"))
json.dump({"test_acc": accs[-1], "epochs": len(accs), "acc_curve": accs, "train_loss": tl,
           "acc_on_class_backprojection": acc_cls, "dec": L,
           "latent_dead_frac": float((z_te.max(0).values < 1e-6).float().mean())},
          open(os.path.join(LOGS, "mlp_proof.json"), "w"))
print("test acc", accs[-1], "class-backproj acc", acc_cls)