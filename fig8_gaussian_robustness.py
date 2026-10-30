"""Figure 8: accuracy and robustness under Gaussian noise on the text embeddings while closing the gap (Sec. 5.1).

The texts are translated along the gap projected orthogonally to the text subspace (Eq. 11), so the clean
accuracy is unchanged while robustness (Eq. 12, K=100 noise samples) grows as the gap is closed.

  (a) CLIP ViT-L/14 on CIFAR10, (b) CLIP ViT-B/16 on CIFAR100   -> x axis: gap norm
  (c) CLIP ViT-B/16 on A-OKVQA, (d) SigLIP on A-OKVQA           -> x axis: -alpha (0 = original gap)

    python extract_zeroshot.py --dataset cifar10 && python extract_zeroshot.py --dataset cifar100 && python extract_aokvqa.py
    python fig8_gaussian_robustness.py
"""
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import accuracy, fig_path, gap, load_embeddings, orthogonal_gap, robustness, set_style
from vqa import load_vqa, predict

NOISE_LEVELS = [0.005, 0.0225, 0.04]  # noise = level * N(0, I); the paper's legend reports these values as sigma^2
K = 100
ZERO_SHOT = [("cifar10", "ViT-L-14-openai"), ("cifar100", "ViT-B-16-openai")]
VQA = ["ViT-B-16-openai", "ViT-B-16-SigLIP-webli"]


def alpha_range(txts, ims, direction, min_alpha=-5.0, n=100):
    """From min_alpha (larger gap) up to the alpha that minimizes the gap norm, plus alpha=0."""
    search = torch.linspace(min_alpha, 2, 1000)
    gaps = torch.stack([(gap(ims, txts + a * direction)).norm() for a in search])
    alphas = torch.cat([torch.linspace(min_alpha, search[gaps.argmin()].item(), n), torch.zeros(1)])
    return alphas.sort().values


def plot_legends(ax, lines, extra):
    leg = ax.legend(handles=lines, title=r"$\sigma^2$", loc="upper right")
    ax.add_artist(leg)
    ax.legend(handles=extra, loc="lower right")
    ax.set_ylim(0, 100)
    ax.grid()
    ax.set_ylabel(r"\%")


def zero_shot(dataset, modelname):
    ims, txts, labels = load_embeddings(dataset, modelname, "image_embeddings", "text_embeddings", "classes")
    direction = orthogonal_gap(gap(ims, txts), txts)
    alphas = alpha_range(txts, ims, direction)
    gaps = torch.tensor([gap(ims, txts + a * direction).norm().item() for a in alphas])
    order = gaps.argsort()
    accs = torch.tensor([accuracy(ims, txts + a * direction, labels) for a in alphas])

    fig, ax = plt.subplots()
    lines = []
    for level in NOISE_LEVELS:
        rob = torch.zeros(len(alphas))
        for _ in tqdm(range(K), desc=f"{dataset} {modelname} {level}"):
            noise = level * torch.randn_like(txts)
            rob += torch.tensor([robustness(ims, txts + a * direction, txts + a * direction + noise) for a in alphas])
        lines.append(ax.plot(gaps[order], (rob / K)[order], label=f"{level:.3f}")[0])
    acc_line = ax.plot(gaps[order], accs[order], "--r", label="Accuracy")[0]
    og = ax.axvline(gaps[alphas == 0].item(), linestyle="--", color="k", label="Original Gap")
    plot_legends(ax, lines, [acc_line, og])
    ax.set_xlabel("Gap")
    plt.savefig(fig_path("fig8", f"{dataset}_robustness_{modelname}.pdf"))
    plt.close()


def vqa(modelname, alphas=torch.linspace(0, 1, 11)):
    d = load_vqa(modelname)
    ims, options, gaps, labels = d["ims"], d["options"], d["gaps"], d["labels"]
    accs = torch.tensor([100 * (predict(ims + a * gaps, options) == labels).float().mean().item() for a in alphas])

    fig, ax = plt.subplots()
    lines = []
    for level in NOISE_LEVELS:
        rob = torch.zeros(len(alphas))
        for _ in tqdm(range(K), desc=f"aokvqa {modelname} {level}"):
            noisy = options + level * torch.randn_like(options)
            rob += torch.tensor([100 * (predict(ims + a * gaps, options) == predict(ims + a * gaps, noisy)).float().mean().item()
                                 for a in alphas])
        lines.append(ax.plot(-alphas, rob / K, label=f"{level:.3f}")[0])
    acc_line = ax.plot(-alphas, accs, "--r", label="Acc")[0]
    og = ax.axvline(0, linestyle="--", color="k", label="Original Gap")
    plot_legends(ax, lines, [acc_line, og])
    ax.set_xlabel(r"$\alpha$")
    plt.savefig(fig_path("fig8", f"aokvqa_{modelname}.pdf"))
    plt.close()


if __name__ == "__main__":
    set_style()
    for dataset, modelname in ZERO_SHOT:
        zero_shot(dataset, modelname)
    for modelname in VQA:
        vqa(modelname)
