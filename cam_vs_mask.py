import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import build_datasets, eval_tf
from model import build_model
from gradcam import GradCAM

RESULTS = Path("results")
THRESH = 0.5


def mask_metrics(cam, mask):
    """cam: (H,W) in [0,1].  mask: (H,W) binary."""
    m = mask > 0
    if m.sum() == 0:
        return None

    binary = cam >= THRESH
    inter = (binary & m).sum()
    union = (binary | m).sum()
    iou = inter / union if union else 0.0

    # fraction of total CAM mass falling inside the tumour
    coverage = cam[m].sum() / (cam.sum() + 1e-8)
    # what that would be if attention were uniform
    chance = m.sum() / m.size

    # centre-of-mass distance, normalised by image diagonal
    ys, xs = np.mgrid[0:cam.shape[0], 0:cam.shape[1]]
    w = cam / (cam.sum() + 1e-8)
    cam_c = np.array([(w * ys).sum(), (w * xs).sum()])
    mask_c = np.array([ys[m].mean(), xs[m].mean()])
    diag = np.hypot(*cam.shape)
    dist = np.linalg.norm(cam_c - mask_c) / diag

    return {"iou": float(iou), "coverage": float(coverage),
            "chance": float(chance), "lift": float(coverage / chance),
            "centre_dist": float(dist),
            "tumour_frac": float(m.mean())}


def main():
    device = "cuda"
    links = json.load(open(RESULTS / "mask_links.json"))
    _, _, test_ds, classes = build_datasets()

    model = build_model(freeze_backbone=False)
    model.load_state_dict(torch.load("checkpoints/finetuned.pt",
                                     map_location=device))
    model = model.to(device)
    cam_fn = GradCAM(model, "act4")

    out = []
    for k, l in enumerate(links):
        if l["true_class"] == "notumor":
            continue
        x, _ = test_ds[l["test_index"]]
        cam, _, _ = cam_fn(x.unsqueeze(0).to(device))
        cam = cam[0]

        mask = np.asarray(Image.open(l["brisc_mask"]).convert("L")
                          .resize((cam.shape[1], cam.shape[0]), Image.NEAREST))
        m = mask_metrics(cam, mask)
        if m is None:
            continue
        out.append({**m, "correct": l["correct"], "true_class": l["true_class"],
                    "pred_class": l["pred_class"], "plane": l["plane"],
                    "confidence": l["confidence"]})

        if (k + 1) % 200 == 0:
            print(f"  processed {k+1}/{len(links)}")

    with open(RESULTS / "cam_vs_mask.json", "w") as f:
        json.dump(out, f, indent=2)

    a = {k: np.array([o[k] for o in out]) for k in
         ["iou", "coverage", "chance", "lift", "centre_dist", "tumour_frac"]}
    corr = np.array([o["correct"] for o in out])

    print(f"\nn = {len(out)}  ({corr.sum()} correct, {(~corr).sum()} wrong)\n")
    print(f"tumour occupies       {a['tumour_frac'].mean()*100:.2f}% of image")
    print(f"CAM mass inside mask  {a['coverage'].mean()*100:.2f}%")
    print(f"  -> lift over chance {a['lift'].mean():.2f}x  "
          f"(median {np.median(a['lift']):.2f}x)")
    print(f"centre distance       {a['centre_dist'].mean():.4f} "
          f"(median {np.median(a['centre_dist']):.4f})")
    print(f"IoU                   {a['iou'].mean():.4f} "
          f"(median {np.median(a['iou']):.4f})")

    print("\nby class:")
    for c in sorted(set(o["true_class"] for o in out)):
        m = np.array([o["true_class"] == c for o in out])
        print(f"  {c:12} n={m.sum():4d}  lift {a['lift'][m].mean():5.2f}x  "
              f"dist {a['centre_dist'][m].mean():.4f}  "
              f"IoU {a['iou'][m].mean():.4f}")


if __name__ == "__main__":
    main()