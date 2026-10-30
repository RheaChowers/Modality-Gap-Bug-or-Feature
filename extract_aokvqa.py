"""Embed A-OKVQA (val) for CLIP-style multiple-choice VQA (texts are "<question> <answer>").

Per model, saves to $EMB_DIR/aokvqa/<model>-<pretrained>/:
  image_embeddings.pt      one embedding per question image
  answers.pt, answer_lens.pt, labels.pt
                           the multiple-choice options (concatenated), #options and index of the correct one
  rephrasings.pt, rephrasing_lens.pt
                           the distinct direct answers (= rephrasings of the correct answer)
  wrong_answers/<i>.pt     every answer in the dataset that is neither an option nor a direct answer of question i

Requires A-OKVQA in $DATA_DIR/aokvqa (aokvqa_v1p0_val.json) and COCO val2017 images in $DATA_DIR/coco.

    python extract_aokvqa.py
"""
import json
import os

import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from utils import DATA_DIR, DEVICE, EMB_DIR, OPENCLIP_MODELS


@torch.no_grad()
def encode_text(model, tokenizer, texts, batch_size=256):
    out = []
    for i in range(0, len(texts), batch_size):
        f = model.encode_text(tokenizer(texts[i:i + batch_size]).to(DEVICE))
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


@torch.no_grad()
def main(split="val"):
    with open(os.path.join(DATA_DIR, "aokvqa", f"aokvqa_v1p0_{split}.json")) as f:
        examples = json.load(f)
    all_answers = set().union(*[set(e["choices"] + e["direct_answers"]) for e in examples]) - {""}

    for modelname, pretrained in OPENCLIP_MODELS:
        print("running", modelname, pretrained)
        outdir = os.path.join(EMB_DIR, "aokvqa", f"{modelname}-{pretrained}")
        os.makedirs(os.path.join(outdir, "wrong_answers"), exist_ok=True)
        model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=pretrained)
        model = model.eval().to(DEVICE)
        tokenizer = open_clip.get_tokenizer(modelname)

        images, answers, rephrasings, labels, answer_lens, rephrasing_lens = [], [], [], [], [], []
        for i, e in enumerate(tqdm(examples)):
            q = e["question"] + " "
            options = [q + a for a in e["choices"]]
            direct = [q + a for a in set(e["direct_answers"])]
            wrong = [q + a for a in all_answers.difference(e["choices"], e["direct_answers"])]

            answers.append(encode_text(model, tokenizer, options))
            rephrasings.append(encode_text(model, tokenizer, direct))
            torch.save(encode_text(model, tokenizer, wrong), os.path.join(outdir, "wrong_answers", f"{i}.pt"))
            labels.append(e["correct_choice_idx"])
            answer_lens.append(len(options))
            rephrasing_lens.append(len(direct))

            image = Image.open(os.path.join(DATA_DIR, "coco", f"{split}2017", f"{e['image_id']:012}.jpg")).convert("RGB")
            f = model.encode_image(preprocess(image).unsqueeze(0).to(DEVICE))
            images.append((f / f.norm(dim=-1, keepdim=True)).cpu())

        torch.save(torch.cat(images), os.path.join(outdir, "image_embeddings.pt"))
        torch.save(torch.cat(answers), os.path.join(outdir, "answers.pt"))
        torch.save(torch.cat(rephrasings), os.path.join(outdir, "rephrasings.pt"))
        torch.save(torch.tensor(labels), os.path.join(outdir, "labels.pt"))
        torch.save(torch.tensor(answer_lens), os.path.join(outdir, "answer_lens.pt"))
        torch.save(torch.tensor(rephrasing_lens), os.path.join(outdir, "rephrasing_lens.pt"))


if __name__ == "__main__":
    main()
