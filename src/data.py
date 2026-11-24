"""Load Fashion-MNIST (idx.gz), split 80/10/10 tetap, dan DataLoader."""
import gzip, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _read_images(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)

def _read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

def load_splits():
    """70.000 citra -> Train 80% / Val 10% / Test 10% (permutasi seed 42)."""
    xs = np.concatenate([_read_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")),
                         _read_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))])
    ys = np.concatenate([_read_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")),
                         _read_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))])
    perm = np.random.RandomState(SEED).permutation(len(xs))
    xs, ys = xs[perm], ys[perm]
    n = len(xs); ntr, nva = int(0.8 * n), int(0.1 * n)
    sl = {"train": slice(0, ntr), "val": slice(ntr, ntr + nva), "test": slice(ntr + nva, n)}
    return {k: (torch.from_numpy(xs[s].astype(np.float32) / 255.0),
                torch.from_numpy(ys[s].astype(np.int64))) for k, s in sl.items()}

def loaders(batch_size=128):
    d = load_splits()
    mk = lambda k, sh: DataLoader(TensorDataset(*d[k]), batch_size=batch_size, shuffle=sh, num_workers=0)
    return d, mk("train", True), mk("val", False), mk("test", False)