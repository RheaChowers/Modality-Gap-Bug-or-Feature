# Is the Modality Gap a Bug or a Feature? A Robustness Perspective

Official implementation of [**"Is the Modality Gap a Bug or a Feature? A Robustness Perspective"**](https://openaccess.thecvf.com/content/CVPR2026/papers/Chowers_Is_the_Modality_Gap_a_Bug_or_a_Feature_A_CVPR_2026_paper.pdf) by Rhea Chowers, Oshri Naparstek, Udi Barzelay, and Yair Weiss.

🌟 **Highlight paper at CVPR 2026**

## Setup

```bash
pip install -r requirements.txt
```

All scripts are run from the repository root. Paths are controlled by three environment variables (defaults in parentheses):

| Variable | Default | Contents |
|---|---|---|
| `DATA_DIR` | `./data` | raw datasets |
| `EMB_DIR` | `./embeddings` | cached model embeddings |
| `FIG_DIR` | `./figures` | output figures (one subfolder per figure) |


## Datasets

CIFAR10/100 are downloaded automatically. Place the others under `$DATA_DIR` as follows:

```
data/
  imagenet/val/<wnid>/*.JPEG              # ILSVRC2012 validation set in ImageFolder layout
  coco/val2017/*.jpg                      # MS-COCO 2017 val images
  coco/annotations/captions_val2017.json
  aokvqa/aokvqa_v1p0_val.json             # https://github.com/allenai/aokvqa (uses the COCO val2017 images)
  flickr30k/results.csv                   # '|'-separated: image_name| comment_number| comment
  flickr30k/images/*.jpg
```

## Creating the embeddings

The experiments in Sections 1 and 5 operate on precomputed embeddings. The `extract_*.py` scripts encode each dataset
and save the **raw, unnormalized** model outputs; every analysis script L2-normalizes them when loading
(`utils.load_embeddings`, `vqa.load_vqa`), so embeddings produced elsewhere can be dropped in as long as they follow
the layout below.

```bash
python extract_zeroshot.py --dataset cifar10    # Figs 8, 9
python extract_zeroshot.py --dataset cifar100   # Figs 8, 9
python extract_zeroshot.py --dataset imagenet   # Fig 2a
python extract_coco.py                          # Figs 1a, 2b
python extract_aokvqa.py                        # Figs 8, 10
```

| Script | Models | Data | Text inputs |
|---|---|---|---|
| `extract_zeroshot.py` | open_clip `ViT-B-16` (openai), `ViT-L-14` (openai), `ViT-B-16-SigLIP` (webli) | test / val split | one prompt per class: `a photo of a {}` (CIFAR10, ImageNet), `a candid {} digital photo` (CIFAR100); override with `--template` |
| `extract_coco.py` | Hugging Face `openai/clip-vit-large-patch14`, `facebook/metaclip-l14-400m`, `google/siglip-so400m-patch16-256-i18n` | COCO val2017 | all captions of every image |
| `extract_aokvqa.py` | same open_clip models as above | A-OKVQA val | `"<question> <answer>"` for the options, the direct answers and all other answers in the dataset |

The resulting layout (`<model>` is `ViT-B-16-openai`, `ViT-L-14-openai` or `ViT-B-16-SigLIP-webli`):

```
embeddings/
  cifar10/<model>/     image_embeddings.pt  (N, d)   text_embeddings.pt  (C, d)   classes.pt  (N,)
  cifar100/<model>/    same
  imagenet/<model>/    same
  coco/{clip,metaclip,siglip}/
                       image_embeddings.pt  (N, d)
                       text_embeddings.pt   (M, d)   captions of image i stored contiguously
                       labels.pt            (N,)     number of captions of each image
  aokvqa/<model>/      image_embeddings.pt  (Q, d)
                       answers.pt           (sum of answer_lens, d)   multiple-choice options, concatenated
                       answer_lens.pt (Q,)  labels.pt (Q,)            #options, index of the correct option
                       rephrasings.pt       (sum of rephrasing_lens, d)   distinct direct answers
                       rephrasing_lens.pt   (Q,)
                       wrong_answers/<i>.pt (W_i, d)  answers that are neither an option nor a direct answer of question i
```

`extract_aokvqa.py` encodes every answer in the dataset once per question for the wrong-answer pool, so it is the slowest
step and needs tens of GB of disk per model.

## Reproducing the figures

| Figure | Script | Needs |
|---|---|---|
| 1a — PCA of CLIP's COCO embeddings | `fig1_coco_pca.py` | `coco/clip` |
| 2 — naive gap shifting on ImageNet / COCO | `fig2_naive_gap.py` | `imagenet`, `coco` |
| 3 — loss and gradient with / without a gap | `fig3_toy_loss.py` | – |
| 4 — toy dynamics in the plane and on the sphere | `fig4_toy_dynamics.py --setting {plane,sphere}` | – |
| 5, 6 (left) — variance shrinks along the gap, Var(S_i) | `fig5_6_variance_shrinking.py` | – |
| 6 (right) — CLIP at random initialization | `fig6_clip_init_kde.py` | Flickr30k images |
| 8 — robustness under Gaussian noise | `fig8_gaussian_robustness.py` | `cifar10`, `cifar100`, `aokvqa` |
| 9 — robustness to quantization | `fig9_quantization.py` | `cifar10`, `cifar100` |
| 10 — rephrasing on A-OKVQA | `fig10_vqa_rephrasing.py` | `aokvqa` |

The gap-closing algorithm (Sec. 4, Eq. 11) is `utils.orthogonal_gap`; robustness (Eq. 12) is `utils.robustness`.

## Notes

- The Gaussian noise in Fig 8 is $\sigma \cdot \mathcal{N}(0, I)$ with $\sigma$ in {0.005, 0.0225, 0.04}. 
- The top panel of Fig 5 (initial clusters coloured by $S^y_i$) is a reimplementation; the original figure was not produced with this code.


## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Chowers_2026_CVPR,
    author    = {Chowers, Rhea and Naparstek, Oshri and Barzelay, Udi and Weiss, Yair},
    title     = {Is the Modality Gap a Bug or a Feature? A Robustness Perspective},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {30288--30298}
}
```
