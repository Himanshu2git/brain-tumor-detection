# Brain Tumor MRI Classification

Four-class classification of brain MRI slices (glioma, meningioma, no tumor,
pituitary) using Xception, loaded from `timm` as `legacy_xception.tf_in1k`. The
model is trained in two configurations: a frozen backbone with only the 4-class
linear head trainable, and a full fine-tune of all 20,815,148 parameters.

The evaluation covers held-out test accuracy, external validation on an
independent dataset with verified provenance, calibration in and out of
distribution, robustness sweeps over photometric and geometric perturbations,
and Grad-CAM localization measured against expert segmentation masks.

`docs/study_guide.md` contains the full method write-up and the reasoning behind
each design choice. This file covers setup and reproduction only.

---

## Results

All figures below are copied from the reference table in `docs/study_guide.md`.

| Measurement | Value |
|---|---|
| Training images | 5,600 (1,400/class) |
| Test images | 1,600 (400/class) |
| Train/val split | 4,480 / 1,120, stratified, seed 42 |
| Model parameters | 20,815,148 |
| Frozen trainable | 8,196 |
| Frozen validation | 0.8804 (epoch 15) |
| Fine-tuned validation | 0.9920 (epoch 4) |
| Frozen test accuracy | 0.8494 (macro F1 0.8456) |
| Fine-tuned test accuracy | 0.9487 (macro F1 0.9479) |
| Glioma recall / precision | 0.8275 / 0.9940 |
| External accuracy (4-class) | 0.8983 |
| External accuracy (3-class) | 0.9339 |
| External glioma recall | 0.9980 |
| False "no tumor" predictions | 269 of 6,056 |
| ECE in-distribution | 0.0325 |
| ECE external | 0.0227 |
| Grad-CAM lift | 4.66x (median 4.49x) |
| Grad-CAM IoU | 0.1357 |
| Linked mask images | 949 |
| Noise: accuracy at sigma=0.02 | 0.7963 |
| Noise: accuracy at sigma=0.05 | 0.2669 (confidence 0.9806) |
| Duplicate groups | 153, none crossing splits |
| Slices per patient (source) | 13.2 across 233 patients |
| Peak VRAM, batch 16 | 4.24 GB |

Reference environment: Python 3.12.14, PyTorch 2.11.0+cu128, timm 1.0.29,
scikit-learn 1.9.0, on an NVIDIA RTX 4050 Laptop GPU (6 GB), Intel i5-13420H,
under WSL2.

---

## Environment

Python 3.12 in a `uv`-managed virtual environment. All scripts hard-code
`device = "cuda"`; there is no CPU fallback path, so a CUDA GPU is required.

```bash
uv venv --python 3.12
source .venv/bin/activate
```

`torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` are local CUDA 12.8
builds and are not on PyPI, so `requirements.txt` must be installed against the
PyTorch cu128 index:

```bash
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

That single command is sufficient. `requirements.txt` is a full
`uv pip freeze` of the reference environment, so it pins every package the
scripts import directly -- including `timm` (`model.py`), `scikit-learn`
(`dataset.py`, `evaluate.py`, `external_eval.py`), `matplotlib`
(`calibration.py`, `robustness.py`, `visualize_cam.py`), and `h5py`
(`cheng.py`, `match.py`) -- along with their transitive dependencies. It also
pins tooling that was present in the environment but is not imported by any
script, such as `kaggle` and `jupytext`.

The first run of `model.py` or `train.py` downloads the pretrained
`legacy_xception.tf_in1k` weights through `timm`, which requires network access.

---

## Data

No data is committed; `data/` is listed in `.gitignore`. Four datasets must be
downloaded and placed under `data/` before the scripts will run. The paths below
are the exact ones the code expects, and the scripts do not accept path
arguments, so the layout has to match.

### 1. Primary dataset (training and test)

The four-class brain tumor MRI collection published on Kaggle by Masoud
Nickparvar (`masoudnickparvar/brain-tumor-mri-dataset`), in the revision that
carries 5,600 training and 1,600 test images balanced at 1,400 and 400 per
class. Earlier releases of the same collection had 5,712 unbalanced training
images and will not reproduce these numbers.

Read by `dataset.py` via `torchvision.datasets.ImageFolder`, which fixes the
class index order alphabetically: glioma 0, meningioma 1, notumor 2,
pituitary 3.

```
data/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
data/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
```

### 2. BRISC 2025 (segmentation masks)

BRISC 2025, the expert-annotated brain tumor segmentation and classification
dataset of Fateh et al., arXiv:2506.14318. Used only by `link_masks.py`, which
reads `manifest.csv` and links our test images to BRISC masks by exact SHA-256
digest. Both the manifest and the extracted `brisc2025/` tree are required:

```
data/brisc/manifest.csv
data/brisc/brisc2025/segmentation_task/{train,test}/images/*.jpg
data/brisc/brisc2025/segmentation_task/{train,test}/masks/*.png
```

`link_masks.py` derives each mask path from its image path by replacing
`/images/` with `/masks/` and changing the suffix to `.png`, so the two trees
must stay side by side with matching filenames.

### 3. Bangladesh dataset (external validation)

The "Brain Cancer - MRI dataset" originating from Bangladeshi hospitals,
distributed on Mendeley Data and mirrored on Kaggle. It has three folders and
6,056 images: `brain_glioma` (2,004), `brain_menin` (2,004), `brain_tumor`
(2,048). Read by `external_eval.py`:

```
data/ext1/Brain_Cancer raw MRI data/Brain_Cancer/brain_glioma/*.jpg
data/ext1/Brain_Cancer raw MRI data/Brain_Cancer/brain_menin/*.jpg
data/ext1/Brain_Cancer raw MRI data/Brain_Cancer/brain_tumor/*.jpg
```

The directory name contains a space and must be preserved verbatim. The folder
names do not match our class names, so `external_eval.py` maps them explicitly:
`brain_glioma` to index 0, `brain_menin` to index 1, and `brain_tumor` to
index 3 (pituitary). The dataset contains no "no tumor" class, which is why the
4-class and 3-class accuracies are reported separately.

### 4. Cheng figshare dataset (provenance check)

The figshare "brain tumor dataset" of Cheng et al. (DOI
10.6084/m9.figshare.1512427), 3,064 `.mat` records over 233 patients. Required
by `cheng.py` and `match.py` for the provenance-matching attempt. It downloads
as four zip archives; extract all of them into a single flat directory, keeping
the numeric filenames:

```
data/cheng/mat/1.mat ... data/cheng/mat/3064.mat
```

`cheng.py` sorts these by `int(path.stem)`, so any non-numeric filename in that
directory will raise. Note that this is a fourth dataset beyond the three named
in the project description; the provenance scripts do not run without it.

---

## Running the pipeline

Only `train.py` defines an argparse interface. Every other script takes no
command-line arguments and reads its inputs from hard-coded paths, so each is
run bare. The order below matters because later steps consume artifacts written
by earlier ones.

Optional sanity checks. `dataset.py` prints split sizes and the tensor value
range; `model.py` prints the parameter counts and peak VRAM for a batch of 4.

```bash
python dataset.py
python model.py
```

### Training

`train.py` accepts exactly four flags: `--finetune` (a store-true switch
selecting the fine-tuned configuration), `--batch-size` (default 32), `--lr`
(default 1e-3), and `--out` (default `best.pt`). Epoch count (30), early-stopping
patience (5), dataloader workers (4), and the checkpoint directory
(`checkpoints/`) are module constants and are not configurable from the command
line.

`--out` must be set explicitly, because `evaluate.py` looks for the two
checkpoints under fixed names, `frozen_baseline.pt` and `finetuned.pt`, which
are not the argparse default.

```bash
python train.py --batch-size 32 --lr 1e-3 --out frozen_baseline.pt
python train.py --finetune --batch-size 16 --lr 1e-4 --out finetuned.pt
```

The two runs differ in batch size and learning rate. The frozen run trains only
the 8,196-parameter head, so batch 32 fits comfortably; fine-tuning at batch 32
allocates 8.19 GB, above the 6 GB card, so batch 16 is used. The learning rate
drops to 1e-4 for fine-tuning to avoid overwriting the pretrained backbone.

Both runs stop early. The frozen run's best epoch was 15, the fine-tuned run's
was 4. Only the best-validation-loss checkpoint is kept.

### Evaluation

`evaluate.py` scores both checkpoints on the test set and writes
`results/probs_frozen.npy`, `results/probs_finetuned.npy`, `results/labels.npy`,
and `results/test_metrics.json`.

```bash
python evaluate.py
```

### Grad-CAM figures

`visualize_cam.py` loads `checkpoints/finetuned.pt` and the probability arrays
written by `evaluate.py`, and writes two panels to `results/gradcam/`: four
correctly classified gliomas and four gliomas misclassified as meningioma.

```bash
python visualize_cam.py
```

`gradcam.py` is a module, not a script. It hooks the `act4` layer and sets
`inplace = False` on it, which is required for the backward hook to return
correct gradients.

### Provenance

`cheng.py` run directly prints the class counts, patient count, and mean slices
per patient for the figshare records. `match.py` compares 32x32 normalized
intensity signatures of all 7,200 primary-dataset images against all 3,064
figshare records on the GPU, and writes `results/matches.json`.

```bash
python cheng.py
python match.py
```

This match failed: the median best-match correlation was 0.724 against a noise
floor of approximately 0.75, so patient identifiers could not be recovered.

`match.py` collects its images with the glob `data/*/*/*.jpg`, filtered to
exclude paths containing "cheng". Under the layout documented above this
resolves to `data/Training` and `data/Testing` only, because the BRISC and
Bangladesh images sit one directory deeper. Flattening either of those trees
would silently pull them into the comparison.

### External validation

`external_eval.py` evaluates `checkpoints/finetuned.pt` on the Bangladesh set,
reports accuracy both over all four output classes and restricted to the three
classes present, and writes `results/probs_external.npy` and
`results/labels_external.npy`.

```bash
python external_eval.py
```

### Mask linking and Grad-CAM validation

`link_masks.py` hashes every test image and matches it against the BRISC
manifest, writing `results/mask_links.json`. `cam_vs_mask.py` then computes
Grad-CAM for each linked non-`notumor` image and measures IoU, CAM mass inside
the mask, lift over uniform attention, and normalized centre-of-mass distance,
writing `results/cam_vs_mask.json`.

```bash
python link_masks.py
python cam_vs_mask.py
```

`cam_vs_mask.py` binarizes the heatmap at a fixed threshold of 0.5 for the IoU
figure. Lift and centre distance use the continuous heatmap.

### Calibration

`calibration.py` reads the in-distribution and external probability arrays,
computes ECE and MCE over 15 equal-width confidence bins, and writes
`results/calibration.json` and `results/reliability.png`. It requires both
`evaluate.py` and `external_eval.py` to have run.

```bash
python calibration.py
```

### Robustness

`robustness.py` sweeps five perturbations over the test set using the fine-tuned
checkpoint: brightness, contrast, rotation, translation, and additive Gaussian
noise. Perturbations are applied to the already-normalized [-1, 1] tensor. For
each level it records accuracy, the fraction of predictions that changed
relative to the unperturbed baseline, and mean confidence, writing
`results/robustness.json` and `results/robustness.png`.

```bash
python robustness.py
```

Rotation and translation are marked in the output as perturbations already
present in the training augmentation.

---

## Determinism

The pipeline is deterministic. Running `train.py` twice with the same arguments
produces identical per-epoch training and validation losses and accuracies to
four decimal places.

Three seeds establish this. `dataset.py` passes `random_state=42` to
`train_test_split`, so the stratified 80/20 split is fixed. `train.py` calls
`torch.manual_seed(42)` before the model is built, which fixes the random
initialization of the classifier head, the shuffle order of the training
loader, and the per-worker augmentation seeds derived from it.
`robustness.py` calls `torch.manual_seed(42)` as well, which fixes the Gaussian
noise sweep.

One qualification: this is an observed property of these runs on this hardware,
not a guarantee enforced by the code. Neither
`torch.backends.cudnn.deterministic` nor `torch.use_deterministic_algorithms` is
set, so a different GPU, driver, or cuDNN version could select a
non-deterministic convolution backward kernel and produce small divergences.
Setting those flags in `train.py` would make the guarantee explicit at some cost
in throughput.

---

## Preprocessing

Images are resized to 299x299 and normalized with mean and standard deviation
0.5 on all three channels, mapping pixels to [-1, 1]. This matches the
preprocessing of the TensorFlow implementation the `legacy_xception.tf_in1k`
weights were ported from, and is deliberately not the ImageNet normalization
that the initial commit used. Training adds a random horizontal flip and a
random rotation up to 10 degrees; the validation, test, and external transforms
add nothing.

---

## Limitations

Stated here because the numbers above should not be read without them.

Patient-level leakage cannot be excluded. The largest source component averages
13.2 slices per patient across 233 patients, and adjacent slices from one patient
are visually near-identical but byte-distinct, so an image-level random split can
place them on both sides. The attempt to recover patient identifiers by matching
back to the figshare originals failed.

Under additive Gaussian noise the model degrades sharply while confidence
recovers: accuracy is 0.2669 at sigma=0.05 with mean confidence 0.9806, and
0.9967 confidence at sigma=0.1 where accuracy is at chance. The highest
confidence in the sweep coincides with the lowest accuracy, with no signal that
the trustworthy range has been left.

The 949 mask-linked images are not a representative subset. They contain 4 of
the 69 glioma errors (1.6%), while the 147 test gliomas outside the annotation
contain 65 of them (44%), so the localization figures are measured on an easier
subset than the test set as a whole.

This is a research prototype. It has not been prospectively evaluated and is not
clinically usable.
