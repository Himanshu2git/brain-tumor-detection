import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from dataset import build_datasets
from model import build_model

RESULTS = Path("results")
BATCH = 32

# perturbations applied to the already-normalised [-1,1] tensor
def brightness(x, d):
    return (x + d).clamp(-1, 1)

def contrast(x, f):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) * f
            + x.mean(dim=(1, 2, 3), keepdim=True)).clamp(-1, 1)

def rotate(x, deg):
    return TF.rotate(x, deg)

def shift(x, px):
    return TF.affine(x, angle=0, translate=[px, px], scale=1.0, shear=[0.0, 0.0])

def noise(x, sigma):
    g = torch.randn_like(x) * sigma
    return (x + g).clamp(-1, 1)


SWEEPS = {
    "brightness":  (brightness, [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4], False),
    "contrast":    (contrast,   [0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0],   False),
    "rotation":    (rotate,     [-30, -15, -5, 0, 5, 15, 30],           True),
    "shift":       (shift,      [-30, -15, -5, 0, 5, 15, 30],           True),
    "noise":       (noise,      [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2], False),
   }


@torch.no_grad()
def run(model, loader, device, fn, level):
    preds, labels, confs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if fn is not None:
            x = fn(x, level)
        p = torch.softmax(model(x), dim=1)
        preds.append(p.argmax(1).cpu())
        confs.append(p.max(1).values.cpu())
        labels.append(y)
    return (torch.cat(preds).numpy(), torch.cat(labels).numpy(),
            torch.cat(confs).numpy())


def main():
    device = "cuda"
    torch.manual_seed(42)
    RESULTS.mkdir(exist_ok=True)

    _, _, test_ds, classes = build_datasets()
    loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = build_model(freeze_backbone=False)
    model.load_state_dict(torch.load("checkpoints/finetuned.pt",
                                     map_location=device))
    model = model.to(device).eval()

    base_preds, labels, base_conf = run(model, loader, device, None, None)
    base_acc = (base_preds == labels).mean()
    print(f"baseline accuracy: {base_acc:.4f}\n")

    out = {}
    for name, (fn, levels, trained_on) in SWEEPS.items():
        rows = []
        tag = " (in training augmentation)" if trained_on else ""
        print(f"=== {name}{tag} ===")
        for lv in levels:
            preds, _, conf = run(model, loader, device, fn, lv)
            acc = (preds == labels).mean()
            flip = (preds != base_preds).mean()
            rows.append({"level": lv, "accuracy": float(acc),
                         "flip_rate": float(flip),
                         "mean_conf": float(conf.mean())})
            print(f"  {lv:>7}   acc {acc:.4f}   flip {flip:.4f}   "
                  f"conf {conf.mean():.4f}")
        out[name] = {"trained_on": trained_on, "levels": rows}
        print()

    with open(RESULTS / "robustness.json", "w") as f:
        json.dump({"baseline_accuracy": float(base_acc), "sweeps": out}, f,
                  indent=2)

    fig, axes = plt.subplots(1, len(out), figsize=(4 * len(out), 3.6))
    for ax, (name, d) in zip(axes, out.items()):
        xs = [r["level"] for r in d["levels"]]
        ax.plot(xs, [r["accuracy"] for r in d["levels"]], "o-", label="accuracy")
        ax.plot(xs, [r["flip_rate"] for r in d["levels"]], "s--", label="flip rate")
        ax.axhline(base_acc, color="k", lw=0.8, alpha=0.4)
        ax.set_title(name + ("*" if d["trained_on"] else ""), fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("* = perturbation present in training augmentation", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness.png", dpi=150, bbox_inches="tight")
    print("saved results/robustness.png")


if __name__ == "__main__":
    main()