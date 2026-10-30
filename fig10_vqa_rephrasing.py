"""Figure 10: accuracy under rephrasing noise on A-OKVQA while closing the gap (Sec. 5.3).

One noise sample replaces each question's options by a random rephrasing of the correct answer (one of its
direct answers) and 3 random wrong answers from the rest of the dataset. Images are moved by alpha * g_i
towards their options; we report clean accuracy and accuracy on the noisy options, averaged over K=500 samples.

    python extract_aokvqa.py && python fig10_vqa_rephrasing.py
"""
import os

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import DEVICE, fig_path, set_style
from vqa import load_vqa, predict

MODELS = ["ViT-B-16-openai", "ViT-B-16-SigLIP-webli"]
ALPHAS = torch.linspace(0, 0.9, 11)
K = 500


def sample_noisy_options(d, wrong, wrong_lens, n_wrong=3):
    """(n, 1 + n_wrong, dim) options: a random rephrasing of the correct answer at index `label`, wrong ones elsewhere."""
    n = d["ims"].shape[0]
    reph_lens = d["rephrasing_lens"]
    right = d["rephrasings"][(reph_lens.cumsum(0) - reph_lens) + (torch.rand(n) * reph_lens).long()]
    wrong_idx = (wrong_lens.cumsum(0) - wrong_lens).unsqueeze(1) + (torch.rand(n, n_wrong) * wrong_lens.unsqueeze(1)).long()
    noisy = torch.cat([right.unsqueeze(1), wrong[wrong_idx]], dim=1)
    labels = d["labels"].cpu()
    rows = torch.arange(n)
    noisy[rows, 0], noisy[rows, labels] = noisy[rows, labels].clone(), right  # put the correct answer at its label
    return noisy.to(DEVICE)


def main(modelname):
    d = load_vqa(modelname)
    n = d["ims"].shape[0]
    wrong = [torch.load(os.path.join(d["dir"], "wrong_answers", f"{i}.pt"), weights_only=True) for i in tqdm(range(n), desc="loading")]
    wrong_lens, wrong = torch.tensor([w.shape[0] for w in wrong]), torch.cat(wrong)

    acc = lambda options, a: 100 * (predict(d["ims"] + a * d["gaps"], options) == d["labels"]).float().mean().item()
    clean = torch.tensor([acc(d["options"], a) for a in ALPHAS])
    noisy = torch.zeros(len(ALPHAS))
    for _ in tqdm(range(K), desc=modelname):
        options = sample_noisy_options(d, wrong, wrong_lens)
        noisy += torch.tensor([acc(options, a) for a in ALPHAS])
    noisy /= K

    plt.figure()
    plt.plot(-ALPHAS.flip(0), clean.flip(0), "--r", label="Accuracy")
    plt.plot(-ALPHAS.flip(0), noisy.flip(0), label="Rephrase Accuracy")
    plt.vlines(0, min(clean.min(), noisy.min()) - 5, max(clean.max(), noisy.max()) + 5, colors="#1f77b4", linestyles="--", label="Original Gap")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"\%")
    plt.savefig(fig_path("fig10", f"{modelname}_Avg_no_rob.pdf"))
    plt.close()


if __name__ == "__main__":
    set_style()
    for modelname in MODELS:
        main(modelname)
