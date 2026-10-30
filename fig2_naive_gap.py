"""Figure 2: naively moving the text embeddings by alpha * g (then re-normalizing) has an inconsistent effect.

  (a) zero-shot accuracy on ImageNet,  (b) image-to-text R@1 on MS-COCO.

    python extract_zeroshot.py --dataset imagenet && python extract_coco.py
    python fig2_naive_gap.py
"""
import torch
from matplotlib import pyplot as plt

from utils import fig_path, gap, load_embeddings, nearest, set_style

IMAGENET_MODELS = {"ViT-B-16-SigLIP-webli": "SigLIP", "ViT-B-16-openai": "CLIP (ViT-B16)", "ViT-L-14-openai": "CLIP (ViT-L14)"}
COCO_MODELS = {"metaclip": "MetaCLIP (ViT-L14)", "clip": "CLIP (ViT-L14)", "siglip": "SigLIP"}
ALPHAS = torch.linspace(-1, 1, 21)


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def zero_shot_accuracy(ims, txts, labels):
    return 100 * (nearest(normalize(ims), normalize(txts)) == labels).float().mean().item()


def recall_at_1(ims, txts, captions_per_image):
    """An image is correct if its nearest caption is one of its own (captions are stored contiguously)."""
    end = captions_per_image.cumsum(0)
    nn = nearest(normalize(ims), normalize(txts))
    return 100 * ((nn >= end - captions_per_image) & (nn < end)).float().mean().item()


def sweep(models, dataset, metric, label_name):
    plt.figure()
    for modelname, pretty in models.items():
        ims, txts, labels = load_embeddings(dataset, modelname, "image_embeddings", "text_embeddings", label_name)
        g = gap(ims, txts)
        plt.plot(ALPHAS, [metric(ims, txts + a * g, labels) for a in ALPHAS], label=pretty)
    plt.axvline(0, linestyle="--", color="gray", label="Original Gap")
    plt.axvline(1, linestyle=":", color="gray", label="No Gap")
    plt.xlabel("Alpha")
    plt.legend(loc="lower left")


if __name__ == "__main__":
    set_style()
    sweep(IMAGENET_MODELS, "imagenet", zero_shot_accuracy, "classes")
    plt.ylabel("Accuracy")
    plt.savefig(fig_path("fig2", "imagenet_naive_accuracy_alpha_normalized.pdf"))
    plt.close()

    sweep(COCO_MODELS, "coco", recall_at_1, "labels")
    plt.ylabel("R@1")
    plt.savefig(fig_path("fig2", "mscoco_normalized.pdf"))
    plt.close()
