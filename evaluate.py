"""Evaluasi akhir: metrik rekonstruksi (MSE/PSNR/SSIM), FID, latensi, cakupan mode, interpolasi, dan 5 figur wajib."""
import json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
from scipy import linalg
from skimage.metrics import structural_similarity
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import *
from src.models import *

set_seed(); torch.set_num_threads(1)
FIGS = os.path.join(RES, "figs"); os.makedirs(FIGS, exist_ok=True)
d = load_splits(); xte, yte = d["test"]
N_FID = 1000
R = {}

def ld(m, name, key=None):
    m.load_state_dict(torch.load(os.path.join(CKPT, name), map_location="cpu")); return m.eval()

ae, sae, vae = ld(AE(), "ae.pt"), ld(SparseAE(), "sae.pt"), ld(VAE(), "vae.pt")
gG, dG = ld(VanillaG(), "gan_G.pt"), ld(DCG(), "dcgan_G.pt")
unet, diff = ld(UNet(), "ddpm_ema.pt"), Diffusion(T=200)
clf = ld(MLPClassifier(), "mlp.pt")
mdec = ld(torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ReLU(), torch.nn.Linear(256, 784), torch.nn.Sigmoid()), "mlp_dec.pt")
logs = {}
mp = json.load(open(os.path.join(LOGS, "mlp_proof.json")))
for f in ["ae_family", "gan", "dcgan", "ddpm"]:
    logs.update(json.load(open(os.path.join(LOGS, f + ".json"))))

fp = os.path.join(CKPT, "fid_cnn.pt"); fcnn = FeatureCNN()
if not os.path.exists(fp):
    set_seed(); o = torch.optim.Adam(fcnn.parameters(), 1e-3); xtr, ytr = d["train"]
    for ep in range(3):
        p = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = p[i:i + 128]; l = F.cross_entropy(fcnn(xtr[idx]), ytr[idx]); o.zero_grad(); l.backward(); o.step()
    torch.save(fcnn.state_dict(), fp)
fcnn.load_state_dict(torch.load(fp)); fcnn.eval()
with torch.no_grad(): R["fid_cnn_test_acc"] = (fcnn(xte).argmax(1) == yte).float().mean().item()

@torch.no_grad()
def feats(x):
    return torch.cat([fcnn.features(x[i:i + 500]) for i in range(0, len(x), 500)]).numpy().astype(np.float64)

def fid(a, b):
    fa, fb = feats(a), feats(b)
    m1, m2 = fa.mean(0), fb.mean(0); c1, c2 = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    cs = np.real(linalg.sqrtm(c1 @ c2))
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * cs))

@torch.no_grad()
def coverage(x):
    p = F.softmax(fcnn(x), 1); pred = p.argmax(1)
    h = torch.bincount(pred, minlength=10).float() / len(x)
    ent = float(-(h[h > 0] * h[h > 0].log()).sum() / np.log(10))
    return {"class_entropy": ent, "classes_ge2pct": int((h >= 0.02).sum()), "mean_conf": float(p.max(1).values.mean()),
            "hist": h.tolist()}

def recon_metrics(xh, x):
    xh, x = xh.clamp(0, 1), x
    mse_i = ((xh - x) ** 2).flatten(1).mean(1)
    psnr = (10 * torch.log10(1 / mse_i.clamp_min(1e-10))).mean().item()
    a, b = xh.squeeze(1).numpy(), x.squeeze(1).numpy()
    ss = np.mean([structural_similarity(a[i], b[i], data_range=1.0) for i in range(len(a))])
    return {"mse": mse_i.mean().item(), "psnr": psnr, "ssim": float(ss)}

with torch.no_grad():
    rec = {"ae": ae(xte)[0], "sae": sae(xte)[0], "vae": vae.decode(vae.encode(xte)[0])}
    mt = torch.load(os.path.join(CKPT, "mlp_tensors.pt"))
    rec["mlp_frozen_dec"] = mt["x_rec"]; rec["mlp_WT_latent"] = mt["x_lat"]
R["recon"] = {k: recon_metrics(v, xte) for k, v in rec.items()}
print("recon", R["recon"], flush=True)

with torch.no_grad():
    h_ae = F.relu(ae.enc[0](xte.flatten(1))); h_sae = sae.encode_h(xte)[0]
for k, h in [("ae", h_ae), ("sae", h_sae)]:
    R.setdefault("sparsity", {})[k] = {"mean_l1": h.abs().mean().item(), "zero_frac": (h < 1e-3).float().mean().item(),
                                       "dead_units": int((h.max(0).values < 1e-3).sum()),
                                       "active_per_sample": (h > 1e-3).float().sum(1).mean().item()}

def to01(x): return ((x + 1) / 2).clamp(0, 1)
set_seed(); gens = {}
with torch.no_grad():
    gens["vae"] = vae.decode(torch.randn(N_FID, 64))
    gens["gan"] = to01(gG(torch.randn(N_FID, 100)))
    gens["dcgan"] = to01(dG(torch.randn(N_FID, 100)))
t0 = time.time(); set_seed()
cp = os.path.join(CKPT, "ddpm_samples.pt")
if os.path.exists(cp): gens["ddpm"] = torch.load(cp)
else:
    dd = []
    for i in range(0, N_FID, 100): dd.append(to01(diff.sample(unet, min(100, N_FID - i))))
    gens["ddpm"] = torch.cat(dd); torch.save(gens["ddpm"], cp)
R["ddpm_sampling_time_s_1000"] = time.time() - t0

with torch.no_grad():
    for k, m in [("ae", ae), ("sae", sae)]:
        z = m.encode(xte); set_seed()
        gens[k + "_prior"] = m.decode(z.mean(0) + z.std(0) * torch.randn(N_FID, 64))
    gens["mlp_prior"] = mt["x_rand"]
ref, ref_other = xte[:N_FID], d["train"][0][:N_FID]
R["fid"] = {k: fid(v, ref) for k, v in gens.items()}; R["fid"]["real_vs_real_floor"] = fid(ref_other, ref)
R["coverage"] = {k: coverage(v) for k, v in gens.items()}; R["coverage"]["real"] = coverage(ref)
print("fid", R["fid"], flush=True)


def lat(fn, runs=30, warm=3):
    for _ in range(warm): fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))
x64, z100, z64 = xte[:64], torch.randn(64, 100), torch.randn(64, 64)
with torch.no_grad():
    R["latency_ms"] = {"ae_recon": lat(lambda: ae(x64)), "sae_recon": lat(lambda: sae(x64)),
                       "vae_generate": lat(lambda: vae.decode(z64)), "gan_generate": lat(lambda: gG(z100)),
                       "dcgan_generate": lat(lambda: dG(z100)),
                       "ddpm_generate": lat(lambda: diff.sample(unet, 64), runs=2, warm=1)}
    R["params"] = {k: sum(p.numel() for p in m.parameters()) for k, m in
                   [("ae", ae), ("sae", sae), ("vae", vae), ("gan_G", gG), ("dcgan_G", dG), ("ddpm_unet", unet), ("mlp", clf)]}
print("latency", R["latency_ms"], flush=True)

def stab(name):
    L = logs[name]; h = len(L["d_loss"]) // 2
    dl, gl = np.array(L["d_loss"][h:]), np.array(L["g_loss"][h:])
    return {"d_mean": dl.mean(), "d_std": dl.std(), "g_mean": gl.mean(), "g_std": gl.std(), "g_max": max(L["g_loss"]),
            "d_min": min(L["d_loss"]), "d_real_final": L["d_real"][-1], "d_fake_final": L["d_fake"][-1],
            "g_cv": gl.std() / gl.mean()}
R["stability"] = {k: {a: float(b) for a, b in stab(k).items()} for k in ["gan", "dcgan"]}
R["epoch_time_s"] = {k: float(np.mean(logs[k]["epoch_time"])) for k in ["ae", "sae", "vae", "gan", "dcgan", "ddpm"]}
R["epochs"] = {k: len(logs[k]["epoch_time"]) for k in ["ae", "sae", "vae", "gan", "dcgan", "ddpm"]}
R["final_loss"] = {k: {"train": logs[k]["train_loss"][-1], "val": logs[k]["val_loss"][-1]} for k in ["ae", "sae", "vae", "ddpm"]}
R["mlp"] = {k: mp[k] for k in ["test_acc", "epochs", "acc_on_class_backprojection", "latent_dead_frac"]}
R["mlp"]["dec_final_val_mse"] = mp["dec"]["val_loss"][-1]

# figur
def grid(x, nrow, ncol):
    x = x[: nrow * ncol].squeeze(1).numpy(); out = np.ones((nrow * 30, ncol * 30))
    for i in range(nrow * ncol):
        r, c = divmod(i, ncol); out[r * 30 + 1:r * 30 + 29, c * 30 + 1:c * 30 + 29] = x[i]
    return out

def rows_fig(rows, labels, path, titles=None, w=10):
    fig, ax = plt.subplots(len(rows), 1, figsize=(w, 1.05 * len(rows) + 0.4))
    for a, r, l in zip(np.atleast_1d(ax), rows, labels):
        a.imshow(grid(r, 1, r.shape[0]), cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
        a.set_ylabel(l, rotation=0, ha="right", va="center", fontsize=8)
    if titles is not None:
        np.atleast_1d(ax)[0].set_xticks([15 + 30 * i for i in range(len(titles))]); np.atleast_1d(ax)[0].set_xticklabels(titles, fontsize=7)
        np.atleast_1d(ax)[0].xaxis.tick_top()
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()

idx10 = [int((yte == c).nonzero()[0]) for c in range(10)]
sel = xte[idx10]
with torch.no_grad():
    # Figur 1
    rows_fig([sel, ae(sel)[0], sae(sel)[0], vae.decode(vae.encode(sel)[0])],
             ["Ground Truth", "Vanilla AE", "Sparse AE", "VAE"], os.path.join(FIGS, "fig1_recon.png"), titles=CLASSES)
    # Figur 2
    fig, ax = plt.subplots(1, 4, figsize=(11, 3.1))
    for a, (k, t) in zip(ax, [("vae", "A: VAE"), ("gan", "B: Vanilla GAN"), ("dcgan", "C: DCGAN"), ("ddpm", "D: DDPM")]):
        a.imshow(grid(gens[k], 4, 4), cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=10); a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig2_samples.png"), dpi=200); plt.close()
    # Figur 3
    xc = mt["x_cls"]
    fig = plt.figure(figsize=(10, 7.4)); gs = fig.add_gridspec(7, 1, hspace=0.12)
    spec = [("(2a) $W^T$ dari kelas $y_c$", xc), ("(1) Ground Truth", sel), ("(2b) $W^T$ dari $z_{MLP}$", mt["x_lat"][idx10]),
            ("(3a) Frozen decoder $z_{MLP}$", mt["x_rec"][idx10]), ("(3b) Vanilla AE $z_{AE}$", ae(sel)[0]),
            ("(4) Prior sampling MLP", mt["x_rand"][:10]), ("(4') Prior sampling AE", gens["ae_prior"][:10])]
    for i, (l, im) in enumerate(spec):
        a = fig.add_subplot(gs[i]); a.imshow(grid(im, 1, 10), cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
        a.set_ylabel(l, rotation=0, ha="right", va="center", fontsize=8)
        if i == 0: a.set_xticks([15 + 30 * j for j in range(10)]); a.set_xticklabels(CLASSES, fontsize=7); a.xaxis.tick_top()
    plt.savefig(os.path.join(FIGS, "fig3_mlp_limits.png"), dpi=200, bbox_inches="tight"); plt.close()

# Figur 4: interpolasi
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
def pick_pair(G, dim, seed):
    set_seed(seed)
    with torch.no_grad():
        z = torch.randn(300, dim); im = to01(G(z)); p = F.softmax(fcnn(im), 1)
    pred, conf = p.argmax(1), p.max(1).values
    i = int(conf.argmax()); j = int(((pred != pred[i]) * conf).argmax())
    return z[i], z[j]
xa, xb = xte[idx10[1]][None], xte[idx10[8]][None]      # Trouser <-> Bag
with torch.no_grad():
    zs = {"Vanilla AE": (ae.encode(xa), ae.encode(xb), ae.decode), "Sparse AE": (sae.encode(xa), sae.encode(xb), sae.decode),
          "VAE": (vae.encode(xa)[0], vae.encode(xb)[0], vae.decode)}
    za, zb = pick_pair(gG, 100, 7); zs["Vanilla GAN"] = (za[None], zb[None], lambda z: to01(gG(z)))
    za, zb = pick_pair(dG, 100, 7); zs["DCGAN"] = (za[None], zb[None], lambda z: to01(dG(z)))
    rows = [torch.cat([dec((1 - a) * A + a * B) for a in alphas]) for A, B, dec in zs.values()]
rows_fig(rows, list(zs.keys()), os.path.join(FIGS, "fig4_interp.png"), titles=[f"α={a}" for a in alphas], w=6.4)

# statistik interpolasi
@torch.no_grad()
def interp_stats(kind):
    set_seed(11); steps, midconf = [], []
    for _ in range(200):
        if kind in ("ae", "sae", "vae"):
            while True:
                i, j = np.random.randint(0, len(xte), 2)
                if yte[i] != yte[j]: break
            m = {"ae": ae, "sae": sae, "vae": vae}[kind]
            enc = (lambda x: m.encode(x)[0]) if kind == "vae" else m.encode
            A, B, dec = enc(xte[i][None]), enc(xte[j][None]), m.decode
        else:
            G, dim = (gG, 100) if kind == "gan" else (dG, 100)
            A, B = torch.randn(1, dim), torch.randn(1, dim); dec = lambda z, G=G: to01(G(z))
        ims = torch.cat([dec((1 - a) * A + a * B) for a in alphas])
        steps.append(((ims[1:] - ims[:-1]) ** 2).flatten(1).mean(1).sqrt().numpy())
        midconf.append(F.softmax(fcnn(ims[2:4]), 1).max(1).values.mean().item())
    s = np.array(steps)
    return {"mean_step": float(s.mean()), "jump_ratio": float((s.max(1) / s.mean(1)).mean()), "mid_conf": float(np.mean(midconf))}
R["interp"] = {k: interp_stats(k) for k in ["ae", "sae", "vae", "gan", "dcgan"]}

# Figur 5a: kurva loss
fig, ax = plt.subplots(2, 3, figsize=(11, 5.2))
def curve(a, ys, labs, title, yl):
    for y, l in zip(ys, labs): a.plot(range(1, len(y) + 1), y, label=l, marker="o", ms=2.5, lw=1.2)
    a.set_title(title, fontsize=9); a.set_xlabel("Epoch", fontsize=8); a.set_ylabel(yl, fontsize=8); a.legend(fontsize=7); a.grid(alpha=.3)
    a.tick_params(labelsize=7)
curve(ax[0, 0], [logs["ae"]["train_loss"], logs["ae"]["val_loss"]], ["train", "val"], "Vanilla AE", "MSE")
curve(ax[0, 1], [logs["sae"]["train_loss"], logs["sae"]["val_loss"]], ["train", "val"], "Sparse AE (MSE + λ·mean|h|)", "loss total")
curve(ax[0, 2], [logs["vae"]["train_loss"], logs["vae"]["val_loss"]], ["train", "val"], "VAE (−ELBO per citra)", "nat")
curve(ax[1, 0], [logs["gan"]["d_loss"], logs["gan"]["g_loss"]], ["D loss", "G loss"], "Vanilla GAN", "BCE")
curve(ax[1, 1], [logs["dcgan"]["d_loss"], logs["dcgan"]["g_loss"]], ["D loss", "G loss"], "DCGAN", "BCE")
curve(ax[1, 2], [logs["ddpm"]["train_loss"], logs["ddpm"]["val_loss"]], ["train", "val"], "DDPM (prediksi derau)", "MSE ε")
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig5a_loss.png"), dpi=200); plt.close()
# Figur 5b: trajektori reverse diffusion
set_seed(5); x0, traj = diff.sample(unet, 4, return_traj=True, every=20)
cols = len(traj); fig, ax = plt.subplots(4, cols, figsize=(11, 4.1))
tl = ["$x_T$"] + [f"t={t}" for t in list(range(180, -1, -20))]
for r in range(4):
    for c in range(cols):
        ax[r, c].imshow(to01(traj[c][r, 0]) if c == cols - 1 else ((traj[c][r, 0] + 1) / 2).clamp(0, 1), cmap="gray", vmin=0, vmax=1); ax[r, c].axis("off")
        if r == 0: ax[r, c].set_title(tl[c], fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "fig5b_ddpm_traj.png"), dpi=200); plt.close()

def cvt(o):
    if isinstance(o, dict): return {k: cvt(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
json.dump(cvt(R), open(os.path.join(RES, "results.json"), "w"), indent=1)
print("SELESAI")