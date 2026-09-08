from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import build_datasets, XCEPTION_MEAN, XCEPTION_STD
from model import build_model
from gradcam import GradCAM

CKPT = Path("checkpoints/finetuned.pt")
OUT = Path("results/gradcam")
N_PER_GROUP = 4


def denormalize(t):
    mean = torch.tensor(XCEPTION_MEAN).view(3, 1, 1)
    std = torch.tensor(XCEPTION_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    device = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)

    _, _, test_ds, classes = build_datasets()

    model = build_model(freeze_backbone=False)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model = model.to(device)
    cam_fn = GradCAM(model, "act4")

    probs = np.load("results/probs_finetuned.npy")
    labels = np.load("results/labels.npy")
    preds = probs.argmax(1)

    # glioma correctly classified, and glioma called meningioma
    gi, mi = classes.index("glioma"), classes.index("meningioma")
    correct_idx = np.where((labels == gi) & (preds == gi))[0][:N_PER_GROUP]
    wrong_idx = np.where((labels == gi) & (preds == mi))[0][:N_PER_GROUP]

    groups = [("correct_glioma", correct_idx), ("glioma_as_meningioma", wrong_idx)]

    for title, idxs in groups:
        fig, axes = plt.subplots(2, len(idxs), figsize=(3 * len(idxs), 6))
        for col, i in enumerate(idxs):
            x, y = test_ds[int(i)]
            xb = x.unsqueeze(0).to(device)
            cam, logits, pred_idx = cam_fn(xb)
            conf = torch.softmax(logits, 1)[0, pred_idx[0]].item()

            img = denormalize(x)
            axes[0, col].imshow(img)
            axes[0, col].set_title(f"true: {classes[y]}", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(img)
            axes[1, col].imshow(cam[0], cmap="jet", alpha=0.45)
            axes[1, col].set_title(
                f"pred: {classes[pred_idx[0]]} ({conf:.2f})", fontsize=9)
            axes[1, col].axis("off")

        fig.suptitle(title.replace("_", " "), fontsize=12)
        fig.tight_layout()
        path = OUT / f"{title}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}  ({len(idxs)} examples)")


if __name__ == "__main__":
    main()
    