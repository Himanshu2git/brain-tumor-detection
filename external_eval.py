import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataset import eval_tf
from model import build_model

EXT_ROOT = Path("data/ext1/Brain_Cancer raw MRI data/Brain_Cancer")
RESULTS = Path("results")

# our model's class indices (alphabetical from ImageFolder)
OUR_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Bangladesh dataset folder -> our class index
FOLDER_TO_IDX = {
    "brain_glioma": 0,      # glioma
    "brain_menin": 1,       # meningioma
    "brain_tumor": 3,       # pituitary
}


class ExternalDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = []
        for folder, idx in FOLDER_TO_IDX.items():
            for p in sorted((root / folder).glob("*.jpg")):
                self.samples.append((p, idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        probs.append(torch.softmax(model(x), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    device = "cuda"
    RESULTS.mkdir(exist_ok=True)

    ds = ExternalDataset(EXT_ROOT, eval_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"external set: {len(ds)} images\n")

    model = build_model(freeze_backbone=False)
    model.load_state_dict(torch.load("checkpoints/finetuned.pt",
                                     map_location=device))
    model = model.to(device)

    probs, labels = predict(model, loader, device)
    preds = probs.argmax(1)

    print("=== all 4 output classes allowed ===")
    print(f"accuracy: {accuracy_score(labels, preds):.4f}\n")
    print(classification_report(labels, preds, labels=[0, 1, 3],
                                target_names=["glioma", "meningioma", "pituitary"],
                                digits=4, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    print("confusion matrix (rows = true, cols = predicted):")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in OUR_CLASSES))
    for i, c in enumerate(OUR_CLASSES):
        print(f"{c:>12}" + "".join(f"{v:>12}" for v in cm[i]))

    # restricted: force a choice among the three classes that exist here
    probs3 = probs[:, [0, 1, 3]]
    preds3 = np.array([0, 1, 3])[probs3.argmax(1)]
    print(f"\n=== restricted to 3 present classes ===")
    print(f"accuracy: {accuracy_score(labels, preds3):.4f}")

    np.save(RESULTS / "probs_external.npy", probs)
    np.save(RESULTS / "labels_external.npy", labels)

    conf = probs.max(1)
    correct = preds == labels
    print(f"\nmean confidence when correct:   {conf[correct].mean():.4f}")
    print(f"mean confidence when incorrect: {conf[~correct].mean():.4f}")


if __name__ == "__main__":
    main()