"""Figure 1a: CLIP (ViT-L/14) embeddings of MS-COCO val projected onto their first 3 principal components.

    python extract_coco.py --models clip && python fig1_coco_pca.py
"""
import torch
from matplotlib import pyplot as plt

from utils import fig_path, load_embeddings, set_style

if __name__ == "__main__":
    set_style()
    ims, txts = load_embeddings("coco", "clip", "image_embeddings", "text_embeddings")
    _, _, comps = torch.pca_lowrank(torch.cat([ims, txts]), q=3)
    ims, txts = (ims @ comps).cpu(), (txts @ comps).cpu()
    ax = plt.figure().add_subplot(111, projection="3d")
    ax.scatter(ims[:, 0], ims[:, 1], ims[:, 2], c="blue", s=0.3, label="Image Embeddings")
    ax.scatter(txts[:, 0], txts[:, 1], txts[:, 2], c="orange", s=0.3, label="Text Embeddings")
    ax.view_init(elev=25, azim=75)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    plt.legend(loc="lower right")
    plt.savefig(fig_path("fig1", "clip3d_75.pdf"))
