"""Figure 4: gradient descent on the contrastive loss starting from two tight clusters.

  plane  (top row):    N=10 points in R^2, l2 logits, tau=0.2, 5e4 SGD steps.
  sphere (bottom row): N=1000 points on S^2 (polar-opposite clusters), cosine logits, tau=0.1, 1e7 SGD steps.

Both converge to ~zero loss while keeping a gap vector orthogonal to the two modalities.

    python fig4_toy_dynamics.py --setting plane
    python fig4_toy_dynamics.py --setting sphere            # long: 1e7 steps
"""
import argparse

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import DEVICE, contrastive_loss, fig_path, set_style

SETTINGS = {
    "plane": dict(N=10, init_std=0.01, temperature=0.2, n_iters=50_000, snapshots=[0, 500, 1000, 10_000, 50_000]),
    "sphere": dict(N=1000, init_std=0.01, temperature=0.1, n_iters=10_000_000,
                   snapshots=[0, 5000, 50_000, 500_000, 10_000_000]),
}


def init(setting, N, init_std):
    if setting == "plane":
        X, Y = torch.randn(N, 2) * init_std, torch.randn(N, 2) * init_std
        X, Y = X - X.mean(0), Y - Y.mean(0)
        X[:, -1] += 0.5
        Y[:, -1] -= 0.5
    else:
        X, Y = torch.randn(N, 3) * init_std, torch.randn(N, 3) * init_std
        X[:, 0] += 1
        Y[:, 0] -= 1
        X, Y = F.normalize(X, dim=-1), F.normalize(Y, dim=-1)
    return X, Y


def embed(setting, X, Y):
    """The embeddings the loss sees: raw coordinates on the plane, normalized ones on the sphere."""
    return (X, Y) if setting == "plane" else (F.normalize(X, dim=-1), F.normalize(Y, dim=-1))


def plot_plane(X, Y, title, path):
    plt.figure()
    for i in range(X.shape[0]):
        plt.plot([Y[i, 0], X[i, 0]], [Y[i, 1], X[i, 1]], "-k", linewidth=0.01)
    plt.scatter(Y[:, 0], Y[:, 1], label="Y", s=100, marker="o")
    plt.scatter(X[:, 0], X[:, 1], label="X", s=100, marker="x")
    both = torch.cat([X, Y])
    plt.xlim([min(both[:, 0].min().item() - 0.5, -0.5), max(0.5, both[:, 0].max().item() + 0.5)])
    plt.ylim([min(both[:, 1].min().item() - 0.5, -0.5), max(0.5, both[:, 1].max().item() + 0.5)])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_sphere(X, Y, title, path):
    ax = plt.figure().add_subplot(1, 1, 1, projection="3d")
    for Z, color, label in [(X, "blue", "X"), (Y, "red", "Y")]:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=color, label=label)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([-0.5, 0.5])
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main(args):
    cfg = SETTINGS[args.setting]
    n_iters = args.n_iters or cfg["n_iters"]
    snapshots = set(s for s in cfg["snapshots"] if s <= n_iters) | {n_iters}
    torch.manual_seed(args.seed)
    X, Y = init(args.setting, cfg["N"], cfg["init_std"])
    X, Y = torch.nn.Parameter(X.to(DEVICE)), torch.nn.Parameter(Y.to(DEVICE))
    opt = torch.optim.SGD([X, Y], lr=args.lr)
    plot = plot_plane if args.setting == "plane" else plot_sphere
    for t in tqdm(range(n_iters + 1)):
        Xe, Ye = embed(args.setting, X, Y)
        loss = contrastive_loss(Xe, Ye, cfg["temperature"], l2=args.setting == "plane")
        if t in snapshots:
            title = f"t={t} , " + r"$\mathcal{L}(X,Y)=$" + f"{loss.item():.2f}"
            plot(Xe.detach().cpu(), Ye.detach().cpu(), title, fig_path("fig4", args.setting, f"t={t}.pdf"))
        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=list(SETTINGS), required=True)
    parser.add_argument("--n_iters", type=int, default=None, help="override the number of SGD steps")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=12876)
    set_style()
    main(parser.parse_args())
