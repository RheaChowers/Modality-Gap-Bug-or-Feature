"""Embed CIFAR10 / CIFAR100 / ImageNet (val) images and class prompts with open_clip models.

Saves image_embeddings.pt, text_embeddings.pt, classes.pt (raw, unnormalized) to
$EMB_DIR/<dataset>/<model>-<pretrained>/.

    python extract_zeroshot.py --dataset cifar10
"""
import argparse
import os
import re

import open_clip
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from classnames import CIFAR10, CIFAR100, IMAGENET
from utils import DATA_DIR, DEVICE, EMB_DIR, OPENCLIP_MODELS, normalize

TEMPLATES = {"cifar10": "a photo of a {}", "cifar100": "a candid {} digital photo", "imagenet": "a photo of a {}"}


def get_dataset(name, transform):
    if name == "cifar10":
        return datasets.CIFAR10(os.path.join(DATA_DIR, "cifar10"), train=False, download=True, transform=transform), CIFAR10
    if name == "cifar100":
        return datasets.CIFAR100(os.path.join(DATA_DIR, "cifar100"), train=False, download=True, transform=transform), CIFAR100
    # ImageNet val in the standard ImageFolder layout (val/<wnid>/*.JPEG); sorted wnids match the OpenAI class order.
    return datasets.ImageFolder(os.path.join(DATA_DIR, "imagenet", "val"), transform=transform), IMAGENET


def make_prompts(dataset, template, classnames):
    prompts = [template.format(c) for c in classnames]
    if dataset == "cifar10":  # "a photo of an airplane", "a photo of an automobile"
        prompts = [re.sub(r"\ba ([aeiou])", r"an \1", p) for p in prompts]
    return prompts


@torch.no_grad()
def main(args):
    for modelname, pretrained in OPENCLIP_MODELS:
        print("running", modelname, pretrained, "on", args.dataset)
        outdir = os.path.join(EMB_DIR, args.dataset, f"{modelname}-{pretrained}")
        os.makedirs(outdir, exist_ok=True)
        model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=pretrained)
        model = model.eval().to(DEVICE)
        tokenizer = open_clip.get_tokenizer(modelname)
        dataset, classnames = get_dataset(args.dataset, preprocess)

        prompts = make_prompts(args.dataset, args.template or TEMPLATES[args.dataset], classnames)
        text = model.encode_text(tokenizer(prompts).to(DEVICE))

        images, labels = [], []
        for x, y in tqdm(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)):
            f = model.encode_image(x.to(DEVICE))
            images.append(f.cpu())
            labels.append(y)
        images, labels = torch.cat(images), torch.cat(labels)
        print("zero-shot accuracy:", ((normalize(images) @ normalize(text.cpu()).T).argmax(1) == labels).float().mean().item())

        torch.save(text.cpu(), os.path.join(outdir, "text_embeddings.pt"))
        torch.save(images, os.path.join(outdir, "image_embeddings.pt"))
        torch.save(labels, os.path.join(outdir, "classes.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], required=True)
    parser.add_argument("--template", default=None, help="prompt template, defaults to the one used in the paper")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args())
