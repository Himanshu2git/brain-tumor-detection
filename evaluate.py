import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

from dataset import build_datasets
from model import build_model

BATCH_SIZE = 32
NUM_WORKERS = 4
CKPT = Path("checkpoints")
RESULTS = Path("results")

DEFAULT_CKPTS = {
    "frozen": CKPT / "frozen_baseline.pt",
    "finetuned": CKPT / "finetuned.pt",
}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        probs = torch.softmax(model(x), dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def parse_args():
    p = argparse.ArgumentParser()
    for name, default in DEFAULT_CKPTS.items():
        p.add_argument(f"--{name}", type=Path, default=default,
                       help=f"checkpoint for the {name} model (default: {default})")
    args = p.parse_args()

    models = {name: getattr(args, name) for name in DEFAULT_CKPTS}
    for name, path in models.items():
        if not path.is_file():
            raise SystemExit(f"checkpoint not found: {path}\n"
                             f"pass --{name} PATH, or train it with "
                             f"train.py --out {path.name}")
    return models


def main():
    models = parse_args()
    device = "cuda"
    RESULTS.mkdir(exist_ok=True)

    _, _, test_ds, classes = build_datasets()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"test set: {len(test_ds)} images, classes {classes}\n")

    summary = {}

    for name, ckpt_path in models.items():
        model = build_model(freeze_backbone=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)

        probs, labels = predict(model, test_loader, device)
        preds = probs.argmax(1)

        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, target_names=classes,
                                       digits=4, output_dict=True)

        print(f"=== {name} ({ckpt_path}) ===")
        print(f"test accuracy: {acc:.4f}\n")
        print(classification_report(labels, preds, target_names=classes, digits=4))
        print("confusion matrix (rows = true, cols = predicted):")
        print(f"{'':>12}" + "".join(f"{c:>12}" for c in classes))
        for i, c in enumerate(classes):
            print(f"{c:>12}" + "".join(f"{v:>12}" for v in cm[i]))
        print()

        np.save(RESULTS / f"probs_{name}.npy", probs)
        np.save(RESULTS / f"labels.npy", labels)
        summary[name] = {"accuracy": float(acc),
                         "macro_f1": report["macro avg"]["f1-score"],
                         "confusion_matrix": cm.tolist(),
                         "per_class": {c: report[c] for c in classes}}

    with open(RESULTS / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== comparison ===")
    print(f"{'model':>12}{'accuracy':>12}{'macro F1':>12}")
    for name, s in summary.items():
        print(f"{name:>12}{s['accuracy']:>12.4f}{s['macro_f1']:>12.4f}")


if __name__ == "__main__":
    main()
    