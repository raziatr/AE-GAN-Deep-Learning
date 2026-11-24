"""Arsitektur: MLP Classifier, Vanilla/Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM (U-Net ringan)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(784, 256), nn.Linear(256, 64), nn.Linear(64, 10)

    def features(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x.flatten(1))))) 

    def forward(self, x):
        return self.fc3(self.features(x))

class FeatureCNN(nn.Module):
    """Ekstraktor fitur untuk FID (pengganti Inception-v3)."""
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 32, 3, 1, 1), nn.Conv2d(32, 64, 3, 1, 1)
        self.fc, self.out = nn.Linear(64 * 7 * 7, 128), nn.Linear(128, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.out(self.features(x))

class AE(nn.Module):
    def __init__(self, zd=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, zd))
        self.dec = nn.Sequential(nn.Linear(zd, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def encode(self, x): return self.enc(x.flatten(1))
    def decode(self, z): return self.dec(z).view(-1, 1, 28, 28)
    def forward(self, x):
        z = self.encode(x); return self.decode(z), z

class SparseAE(nn.Module):
    def __init__(self, zd=64):
        super().__init__()
        self.e1, self.e2 = nn.Linear(784, 256), nn.Linear(256, zd)
        self.dec = nn.Sequential(nn.Linear(zd, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def encode_h(self, x):
        h = F.relu(self.e1(x.flatten(1))); return h, self.e2(h)
    def encode(self, x): return self.encode_h(x)[1]
    def decode(self, z): return self.dec(z).view(-1, 1, 28, 28)
    def forward(self, x):
        h, z = self.encode_h(x); return self.decode(z), z, h

class VAE(nn.Module):
    def __init__(self, zd=64):
        super().__init__()
        self.zd = zd
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.mu, self.lv = nn.Linear(256, zd), nn.Linear(256, zd)
        self.dec = nn.Sequential(nn.Linear(zd, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x.flatten(1)); return self.mu(h), self.lv(h)
    def decode(self, z): return self.dec(z).view(-1, 1, 28, 28)
    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)          
        return self.decode(z), mu, lv

class VanillaG(nn.Module):
    def __init__(self, zd=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zd, 256), nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, True),
            nn.Linear(512, 784), nn.Tanh())
    def forward(self, z): return self.net(z).view(-1, 1, 28, 28)

class VanillaD(nn.Module):                      
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, True), nn.Linear(256, 1))
    def forward(self, x): return self.net(x.flatten(1))

class DCG(nn.Module):                         
    def __init__(self, zd=100, ngf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(zd, ngf * 2, 7, 1, 0, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False), nn.Tanh())
    def forward(self, z): return self.net(z.view(z.size(0), -1, 1, 1))

class DCD(nn.Module):
    def __init__(self, ndf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),      # tanpa BN pada input
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False))
    def forward(self, x): return self.net(x).view(-1, 1)

def weights_init(m):                         
    n = m.__class__.__name__
    if "Conv" in n: nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in n:
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)

def sin_emb(t, dim=64):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)

class Block(nn.Module):
    def __init__(self, ci, co, td=128):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(ci, co, 3, 1, 1), nn.Conv2d(co, co, 3, 1, 1)
        self.n1, self.n2 = nn.GroupNorm(8, co), nn.GroupNorm(8, co)
        self.t = nn.Linear(td, co)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x))) + self.t(te)[:, :, None, None]
        return F.silu(self.n2(self.c2(h))) + self.sk(x)

class UNet(nn.Module):
    def __init__(self, ch=32, td=128):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(64, td), nn.SiLU(), nn.Linear(td, td))
        self.stem = nn.Conv2d(1, ch, 3, 1, 1)
        self.down = nn.Conv2d(ch, ch, 3, 2, 1)
        self.b1, self.b2 = Block(ch, ch, td), Block(ch, ch * 2, td)
        self.mid = Block(ch * 2, ch * 2, td)
        self.u2 = Block(ch * 3, ch, td)
        self.up = nn.ConvTranspose2d(ch, ch, 4, 2, 1)
        self.out = nn.Sequential(nn.GroupNorm(8, ch * 2), nn.SiLU(), nn.Conv2d(ch * 2, 1, 3, 1, 1))

    def forward(self, x, t):
        te = self.temb(sin_emb(t))
        s0 = self.stem(x)                                      
        d1 = self.b1(self.down(s0), te)                         
        d2 = self.b2(F.avg_pool2d(d1, 2), te)                  
        m = self.mid(d2, te)
        u = self.u2(torch.cat([F.interpolate(m, scale_factor=2), d1], 1), te)  
        return self.out(torch.cat([self.up(u), s0], 1))        

class Diffusion:
    def __init__(self, T=200, b0=5e-4, b1=0.1):
        self.T = T
        self.beta = torch.linspace(b0, b1, T)
        self.alpha = 1 - self.beta
        self.abar = torch.cumprod(self.alpha, 0)

    def q_sample(self, x0, t, eps):                             
        a = self.abar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    @torch.no_grad()
    def sample(self, net, n, return_traj=False, every=20):
        x = torch.randn(n, 1, 28, 28); traj = [x.clone()]
        for t in reversed(range(self.T)):
            eps = net(x, torch.full((n,), t, dtype=torch.long))
            a, ab, b = self.alpha[t], self.abar[t], self.beta[t]
            x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 0: x = x + b.sqrt() * torch.randn_like(x)
            if return_traj and (t % every == 0 or t == 0): traj.append(x.clone())
        return (x, traj) if return_traj else x