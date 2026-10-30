"""Shared helpers: paths, plot style, nearest-neighbour retrieval, gap closing and robustness."""
import os
import shutil

import torch
from matplotlib import pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", "data")  # raw datasets (CIFAR, ImageNet, COCO, Flickr30k, A-OKVQA)
EMB_DIR = os.environ.get("EMB_DIR", "embeddings")  # cached model embeddings
FIG_DIR = os.environ.get("FIG_DIR", "figures")  # output figures
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained (open_clip name, pretrained tag) pairs used in the zero-shot and VQA experiments.
OPENCLIP_MODELS = [("ViT-B-16", "openai"), ("ViT-L-14", "openai"), ("ViT-B-16-SigLIP", "webli")]


def set_style():
    plt.rcParams.update({
        "text.usetex": shutil.which("latex") is not None,
        "font.family": "sans-serif",
        "figure.dpi": 90,
        "figure.autolayout": True,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "font.weight": "bold",
        "font.size": 18,
        "axes.linewidth": 1.5,
        "axes.labelweight": "bold",
        "lines.linewidth": 3,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.0,
        "legend.borderpad": 0.4,
        "legend.fontsize": 16,
        "axes.edgecolor": "#303030",
        "savefig.facecolor": "white",
        "text.latex.preamble": r"\usepackage{amsfonts}",
    })


def fig_path(*parts):
    path = os.path.join(FIG_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def load_embeddings(dataset, modelname, *names):
    """Load tensors saved by the extract_*.py scripts, e.g. load_embeddings('cifar10', 'ViT-B-16-openai', 'image_embeddings').

    Embeddings are stored unnormalized and projected to the unit sphere here; other tensors (labels) are returned as is.
    """
    tensors = [torch.load(os.path.join(EMB_DIR, dataset, modelname, f"{n}.pt"), weights_only=True).to(DEVICE) for n in names]
    return [normalize(t.float()) if n.endswith("embeddings") else t for n, t in zip(names, tensors)]


def sq_dists(a, b):
    """Pairwise squared l2 distances, shape (len(a), len(b))."""
    return (a ** 2).sum(-1, keepdim=True) + (b ** 2).sum(-1, keepdim=True).T - 2 * a @ b.T


def nearest(queries, keys):
    """Index of the l2 nearest neighbour in `keys` for every query."""
    return sq_dists(queries, keys).argmin(-1)


def accuracy(ims, txts, labels):
    return 100 * (nearest(ims, txts) == labels).float().mean().item()


def robustness(ims, txts, noisy_txts, noisy_ims=None):
    """Eq. (12) for a single noise sample: % of queries whose nearest neighbour is unchanged by the noise."""
    noisy_ims = ims if noisy_ims is None else noisy_ims
    return 100 * (nearest(ims, txts) == nearest(noisy_ims, noisy_txts)).float().mean().item()


def gap(ims, txts):
    """Global gap vector pointing from the text mean to the image mean."""
    return ims.mean(0, keepdim=True) - txts.mean(0, keepdim=True)


def orthogonal_gap(g, txts):
    """Eq. (11): project g onto the orthogonal complement of the (affine) text subspace."""
    _, _, Vh = torch.linalg.svd(txts - txts.mean(0, keepdim=True), full_matrices=False)
    Vh = Vh[:-1]  # n centred points span at most n-1 directions
    return g - g @ Vh.T @ Vh


def quantize(x, minval=-3.0, maxval=3.0, levels=256):
    """Round every coordinate to the nearest of `levels` evenly spaced values in [minval, maxval]."""
    buckets = torch.linspace(minval, maxval, levels, device=x.device)
    return buckets[(x.unsqueeze(-1) - buckets).abs().argmin(-1)]


def contrastive_loss(x, y, temperature, l2=True):
    """Symmetric multi-modal contrastive loss (Eq. 1); logits are -||x_i-y_j||^2 / tau or cosine / tau."""
    logits = -sq_dists(x, y) / temperature if l2 else x @ y.T / temperature
    labels = torch.arange(x.shape[0], device=x.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels))
