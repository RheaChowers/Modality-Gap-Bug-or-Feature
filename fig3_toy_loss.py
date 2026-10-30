"""Figure 3: three points per modality in R^2 with the contrastive loss and its gradient norm.

Relative alignment (true pair = nearest neighbour) gives ~zero loss and gradient even with a gap.

    python fig3_toy_loss.py
"""
import math

import torch
from matplotlib import pyplot as plt

from utils import contrastive_loss, fig_path, set_style


def line(perm=None):
    """Two parallel rows of 3 points (unit gap between them)."""
    X = torch.stack([2 * torch.linspace(0, 1, 3), torch.ones(3)])
    Y = torch.stack([2 * torch.linspace(0, 1, 3), torch.zeros(3)])
    X[0] -= X[0].mean()
    Y[0] -= Y[0].mean()
    return (X if perm is None else X[:, perm]), Y


def triangle(perm=None):
    """Vertices of an equilateral triangle, identical in both modalities (no gap)."""
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]]).T
    Y = X.clone()
    return (X if perm is None else X[:, perm]), Y


def plot(X, Y, title, name):
    X, Y = X.clone().requires_grad_(), Y.clone().requires_grad_()
    loss = contrastive_loss(X.T, Y.T, temperature=0.1)
    loss.backward()
    grad = (X.grad.norm(dim=0).mean() + Y.grad.norm(dim=0).mean()).item() / 2
    print(f"{name}: loss={loss.item():.2e}, mean grad norm={grad:.2e}")
    X, Y = X.detach(), Y.detach()
    plt.figure()
    for i in range(3):
        plt.plot([Y[0, i], X[0, i]], [Y[1, i], X[1, i]], "-k", linewidth=0.01)
    plt.scatter(Y[0], Y[1], label="Y", s=100, marker="o")
    plt.scatter(X[0], X[1], label="X", s=100, marker="x")
    plt.legend()
    plt.title(title)
    plt.savefig(fig_path("fig3", f"{name}.pdf"))
    plt.close()


if __name__ == "__main__":
    set_style()
    plot(*triangle(perm=[1, 2, 0]), r"$\mathcal{L}(X,Y) \approx 10$ , $||\nabla_x\mathcal{L}||\approx 6.7$", "non_aligned_triangle")
    plot(*line(perm=[1, 2, 0]), r"$\mathcal{L}(X,Y) > 20 $ , $||\nabla_x\mathcal{L}||>8$", "non_aligned_line")
    plot(*triangle(), r"$\mathcal{L}(X,Y) < 10^{-4}$ , $||\nabla_x\mathcal{L}||\approx 5\times 10^{-4}$", "aligned_triangle")
    plot(*line(), r"$\mathcal{L}(X,Y) < 10^{-4}$ , $||\nabla_x\mathcal{L}||\approx 2\times 10^{-4}$", "aligned_line")
