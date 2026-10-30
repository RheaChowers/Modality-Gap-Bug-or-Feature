"""Figure 6 (right): at random initialization CLIP embeds images and texts in two far-apart, tight clusters.

KDE of ||g||^2 and of the within-modality distances to the mean, over 50 random initializations of
CLIP ViT-B-32 on 5000 random Flickr30k image-caption pairs.

Requires $DATA_DIR/flickr30k/results.csv ('|'-separated: image_name| comment_number| comment) and images/.

    python fig6_clip_init_kde.py
"""
import argparse
import os

import open_clip
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import DATA_DIR, DEVICE, fig_path, set_style


class Flickr30k(Dataset):
    def __init__(self, transform, tokenizer, root=os.path.join(DATA_DIR, "flickr30k")):
        self.annotations = pd.read_csv(os.path.join(root, "results.csv"), sep="|")
        self.annotations.loc[19999, " comment"] = "A dog runs across the grass ."  # missing caption in the csv
        self.root, self.transform, self.tokenizer = root, transform, tokenizer

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image = Image.open(os.path.join(self.root, "images", row["image_name"])).convert("RGB")
        return self.transform(image), self.tokenizer(row[" comment"]).squeeze()


def reset_weights(module):
    for layer in module.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        else:
            reset_weights(layer)


def init_text_tower(model):
    """CLIP's text-tower initialization (open_clip's TextTransformer.init_parameters)."""
    nn.init.normal_(model.token_embedding.weight, std=0.02)
    nn.init.normal_(model.positional_embedding, std=0.01)
    width, layers = model.transformer.width, model.transformer.layers
    proj_std, attn_std, fc_std = width ** -0.5 * (2 * layers) ** -0.5, width ** -0.5, (2 * width) ** -0.5
    for block in model.transformer.resblocks:
        nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    nn.init.normal_(model.text_projection, std=width ** -0.5)


@torch.no_grad()
def main(args):
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=None)
    model = model.eval().to(DEVICE)
    loader = DataLoader(Flickr30k(preprocess, open_clip.get_tokenizer(args.model)), batch_size=10, shuffle=True, num_workers=8)
    gap_norms, dists_x, dists_y = [], [], []
    for _ in tqdm(range(args.n_inits)):
        reset_weights(model.visual)
        init_text_tower(model)
        ims, txts, n = [], [], 0
        for images, captions in loader:
            ims.append(model.encode_image(images.to(DEVICE)))
            txts.append(model.encode_text(captions.to(DEVICE)))
            n += images.shape[0]
            if n >= args.n_samples:
                break
        ims, txts = torch.cat(ims), torch.cat(txts)
        ims, txts = ims / ims.norm(dim=-1, keepdim=True), txts / txts.norm(dim=-1, keepdim=True)
        mu_x, mu_y = ims.mean(0, keepdim=True), txts.mean(0, keepdim=True)
        gap_norms.append((mu_y - mu_x).norm().item() ** 2)
        dists_x.append(((ims - mu_x).norm(dim=-1) ** 2).cpu())
        dists_y.append(((txts - mu_y).norm(dim=-1) ** 2).cpu())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(torch.cat(dists_x).numpy(), label=r"$||y_i-\mu_y||^2$", ax=ax, fill=True, alpha=0.08)
    sns.kdeplot(torch.cat(dists_y).numpy(), label=r"$||x_i-\mu_x||^2$", ax=ax, fill=True, alpha=0.08)
    sns.kdeplot(gap_norms, label=r"$||\vec{g}||^2$", ax=ax, fill=True, alpha=0.08)
    ax.set_xlabel(r"$|| \cdot ||^2$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_path("fig6", f"{args.model}_kdeplot.pdf"), bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--n_inits", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=5000)
    set_style()
    main(parser.parse_args())
