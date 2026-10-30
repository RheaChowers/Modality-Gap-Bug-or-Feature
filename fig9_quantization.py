"""Figure 9: robustness to post-training quantization of the embeddings while closing the gap (Sec. 5.2).

Image embeddings are translated along the text-orthogonal gap (Eq. 11), then both modalities are quantized
to `levels` values in [-3, 3]. Robustness = % of images whose nearest text is unchanged by quantization.
Since quantization noise is not zero-mean, each curve stops where the gap in the *quantized* space is minimal.

    python extract_zeroshot.py --dataset cifar10 && python extract_zeroshot.py --dataset cifar100
    python fig9_quantization.py
"""
import torch
from matplotlib import pyplot as plt

from utils import accuracy, fig_path, gap, load_embeddings, orthogonal_gap, quantize, robustness, set_style

RUNS = [("cifar10", "ViT-L-14-openai"), ("cifar100", "ViT-B-16-openai")]
LEVELS = [32, 64, 128, 256]
ALPHAS = torch.cat([torch.arange(-3, 1.01, 0.1), torch.zeros(1)]).unique()  # moves images towards the texts


def main(dataset, modelname):
    ims, txts, labels = load_embeddings(dataset, modelname, "image_embeddings", "text_embeddings", "classes")
    direction = orthogonal_gap(-gap(ims, txts), txts)
    gaps = torch.tensor([gap(ims + a * direction, txts).norm().item() for a in ALPHAS])
    plt.figure()
    plt.plot(gaps, [accuracy(ims + a * direction, txts, labels) for a in ALPHAS], "r--", label="Accuracy")
    lowest = 100
    for level in LEVELS:
        rob, q_gaps = [], []
        for a in ALPHAS:
            q_ims, q_txts = quantize(ims + a * direction, levels=level), quantize(txts, levels=level)
            rob.append(robustness(ims + a * direction, txts, q_txts, q_ims))
            q_gaps.append(gap(q_ims, q_txts).norm().item())
        stop = int(torch.tensor(q_gaps).argmin()) + 1  # ALPHAS ascend, i.e. the gap shrinks along the curve
        plt.plot(gaps[:stop], rob[:stop], label=f"{level}")
        lowest = min(lowest, min(rob))
    plt.vlines(gaps[ALPHAS == 0].item(), lowest, 100, linestyles="--", colors="#4976AF", label="Original Gap")
    plt.legend(title="Intervals", loc="lower right")
    plt.xlabel("Gap")
    plt.ylabel(r"\%")
    plt.grid()
    plt.savefig(fig_path("fig9", f"{modelname}_{dataset}.pdf"))
    plt.close()


if __name__ == "__main__":
    set_style()
    for dataset, modelname in RUNS:
        main(dataset, modelname)
