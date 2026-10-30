"""Figure 5 and Figure 6 (left): isotropic Gaussian clusters in R^2 trained with the contrastive loss.

  Fig 5 top:    initial clusters, Y coloured by S^y_i = sum_k Q^x(k, i) (reimplementation; the original was not in Python).
  Fig 5 bottom: snapshots at t = 0, 25, 100 - variance along the gap direction shrinks first (Theorem 3.1).
  Fig 6 left:   Var(S_i^x), Var(S_i^y) over training - Q^x, Q^y become approximately doubly stochastic.

    python fig5_6_variance_shrinking.py
"""
import argparse

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import DEVICE, contrastive_loss, fig_path, set_style, sq_dists


def column_sums(X, Y, temperature):
    """(S^x, S^y) with S^y_i = sum_k Q^x(k, i) and S^x_i = sum_k Q^y(i, k); both have mean 1 by construction."""
    logits = -sq_dists(X, Y) / temperature
    return logits.softmax(dim=0).sum(1), logits.softmax(dim=1).sum(0)


def plot_snapshot(X, Y, ylims, path):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c="blue", label="X")
    plt.scatter(Y[:, 0], Y[:, 1], c="red", label="Y")
    plt.xlim([-1.1, 1.1])
    if ylims is not None:
        plt.ylim(ylims)
    plt.xticks([i * 0.5 for i in range(-2, 3)])
    plt.yticks([i * 0.5 for i in range(-2, 3)])
    plt.tight_layout()
    ylims = plt.gca().get_ylim()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    return ylims


def plot_si_colouring(X, Y, temperature, path):
    s_y = column_sums(X, Y, temperature)[1]
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c="gray", marker="x", label="X")
    sc = plt.scatter(Y[:, 0], Y[:, 1], c=s_y, cmap="viridis", label="Y")
    plt.colorbar(sc, label=r"$S^y_i$")
    plt.axis("equal")
    plt.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def main(args):
    torch.manual_seed(args.seed)
    X, Y = torch.randn(args.N, 2) * args.init_std, torch.randn(args.N, 2) * args.init_std
    X, Y = X - X.mean(0), Y - Y.mean(0)
    X[:, -1] += 1.0
    Y[:, -1] -= 1.0
    plot_si_colouring(X, Y, args.temperature, fig_path("fig5", "twoClustersSi.pdf"))

    X, Y = torch.nn.Parameter(X.to(DEVICE)), torch.nn.Parameter(Y.to(DEVICE))
    opt = torch.optim.SGD([X, Y], lr=args.lr)
    all_sx, all_sy, ylims = [], [], None
    for t in tqdm(range(args.n_iters + 1)):
        with torch.no_grad():
            if t in args.snapshots:
                g = (X.mean(0) - Y.mean(0)) / (X.mean(0) - Y.mean(0)).norm()
                var_x, var_y = (X @ g).var().item(), (Y @ g).var().item()
                name = f"N={args.N}_t={t}_var_in_gap_x={var_x:.4f}_y={var_y:.4f}.pdf"
                ylims = plot_snapshot(X.cpu(), Y.cpu(), ylims, fig_path("fig5", name))
            s_x, s_y = column_sums(X, Y, args.temperature)
            all_sx.append(s_x.var().item())
            all_sy.append(s_y.var().item())
        loss = contrastive_loss(X, Y, args.temperature, l2=True)
        opt.zero_grad()
        loss.backward()
        opt.step()

    plt.figure(figsize=(10, 6))
    plt.plot(all_sx, label=r"$S_i^x$")
    plt.plot(all_sy, label=r"$S_i^y$")
    plt.yscale("log")
    plt.xlabel("Training Iteration")
    plt.ylabel(r"$\log{Var(S_i)}$")
    plt.legend()
    plt.savefig(fig_path("fig6", f"S_i_summary_{args.n_iters + 1}log.pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--init_std", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_iters", type=int, default=10_000)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[0, 25, 100])
    parser.add_argument("--seed", type=int, default=0)
    set_style()
    main(parser.parse_args())
