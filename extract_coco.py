"""Embed the MS-COCO val2017 images and all of their captions with Hugging Face CLIP-like models.

Saves raw (unnormalized) image_embeddings.pt, text_embeddings.pt (captions of image i are contiguous) and
labels.pt (number of captions per image) to $EMB_DIR/coco/<model>/.

    python extract_coco.py
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from utils import DATA_DIR, DEVICE, EMB_DIR

MODELS = {
    "clip": "openai/clip-vit-large-patch14",
    "metaclip": "facebook/metaclip-l14-400m",
    "siglip": "google/siglip-so400m-patch16-256-i18n",
}


@torch.no_grad()
def main(args):
    coco = os.path.join(DATA_DIR, "coco")
    dataset = CocoCaptions(os.path.join(coco, "val2017"), os.path.join(coco, "annotations", "captions_val2017.json"))
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: tuple(zip(*b)))
    for name in args.models:
        print("running", name)
        model = AutoModel.from_pretrained(MODELS[name]).eval().to(DEVICE)
        processor = AutoProcessor.from_pretrained(MODELS[name])
        padding = "max_length" if name == "siglip" else True  # SigLIP was trained with max_length padding
        images, texts, lens = [], [], []
        for ims, caps in tqdm(loader):
            pix = processor(images=list(ims), return_tensors="pt")["pixel_values"].to(DEVICE)
            images.append(model.get_image_features(pixel_values=pix).cpu())
            flat = [c for cs in caps for c in cs]
            tok = processor(text=flat, return_tensors="pt", padding=padding, truncation=True).to(DEVICE)
            texts.append(model.get_text_features(**tok).cpu())
            lens += [len(cs) for cs in caps]
        outdir = os.path.join(EMB_DIR, "coco", name)
        os.makedirs(outdir, exist_ok=True)
        images, texts = torch.cat(images), torch.cat(texts)
        torch.save(images, os.path.join(outdir, "image_embeddings.pt"))
        torch.save(texts, os.path.join(outdir, "text_embeddings.pt"))
        torch.save(torch.tensor(lens), os.path.join(outdir, "labels.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--batch_size", type=int, default=32)
    main(parser.parse_args())
