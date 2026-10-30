"""A-OKVQA multiple-choice VQA with CLIP-style models (shared by fig8 and fig10).

Each question has its own small text modality (its answer options). The gap of question i is
mean(options_i) - image_i, projected to be orthogonal to the options' affine subspace (Eq. 11).
"""
import os

import torch

from utils import DEVICE, EMB_DIR, normalize

PAD = 1000.0  # padding value for questions with fewer options; never a nearest neighbour


def load_vqa(modelname):
    d = os.path.join(EMB_DIR, "aokvqa", modelname)
    load = lambda n: torch.load(os.path.join(d, f"{n}.pt"), weights_only=True)
    answers, answer_lens, labels = normalize(load("answers").float()), load("answer_lens"), load("labels")
    ims = normalize(load("image_embeddings").float())
    n, dim = ims.shape

    options = torch.full((n, int(answer_lens.max()), dim), PAD)
    gaps = torch.zeros(n, dim)
    start = 0
    for i, l in enumerate(answer_lens.tolist()):
        cur = answers[start:start + l]
        _, _, Vh = torch.linalg.svd(cur - cur.mean(0, keepdim=True), full_matrices=False)
        Vh = Vh[:-1]
        g = cur.mean(0) - ims[i]
        gaps[i] = g - g @ Vh.T @ Vh
        options[i, :l] = cur
        start += l
    return dict(ims=ims.to(DEVICE), options=options.to(DEVICE), gaps=gaps.to(DEVICE), labels=labels.to(DEVICE),
                rephrasings=normalize(load("rephrasings").float()), rephrasing_lens=load("rephrasing_lens"), dir=d)


def predict(ims, options):
    """Index of the option closest (l2) to each image. ims: (n, d), options: (n, k, d)."""
    return (options - ims.unsqueeze(1)).norm(dim=-1).argmin(-1)
