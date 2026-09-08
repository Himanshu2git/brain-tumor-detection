import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dataset import build_datasets

BRISC = Path("data/brisc")
SEG_ROOT = BRISC / "brisc2025" / "segmentation_task"
RESULTS = Path("results")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RESULTS.mkdir(exist_ok=True)

    # hash every BRISC segmentation image, keep its mask path
    rows = list(csv.DictReader(open(BRISC / "manifest.csv")))
    seg_imgs = [r for r in rows
                if r["task"] == "segmentation" and r["is_mask"] == "False"]

    by_hash = {}
    for r in seg_imgs:
        rel = r["relative_path"].replace("\\", "/")
        img_path = BRISC / "brisc2025" / rel
        mask_path = Path(str(img_path).replace("/images/", "/masks/")
                         ).with_suffix(".png")
        by_hash[r["sha256"]] = {
            "brisc_image": str(img_path),
            "brisc_mask": str(mask_path),
            "plane": r["plane_label"],
            "brisc_class": r["tumor_label"],
        }

    # our test set, in the exact order evaluate.py used
    _, _, test_ds, classes = build_datasets()
    probs = np.load(RESULTS / "probs_finetuned.npy")
    labels = np.load(RESULTS / "labels.npy")
    preds = probs.argmax(1)

    links = []
    for i, (path, label) in enumerate(test_ds.samples):
        h = sha256(Path(path))
        if h not in by_hash:
            continue
        b = by_hash[h]
        if not Path(b["brisc_mask"]).exists():
            continue
        links.append({
            "test_index": i,
            "path": path,
            "true_class": classes[labels[i]],
            "pred_class": classes[preds[i]],
            "correct": bool(preds[i] == labels[i]),
            "confidence": float(probs[i].max()),
            **b,
        })

    with open(RESULTS / "mask_links.json", "w") as f:
        json.dump(links, f, indent=2)

    print(f"linked {len(links)} test images to masks\n")

    # sanity: our label vs BRISC label must agree
    disagree = [l for l in links
                if l["true_class"].replace("notumor", "no_tumor")
                != l["brisc_class"]]
    print(f"label disagreements: {len(disagree)}")

    print("\nby class and outcome:")
    grid = defaultdict(Counter)
    for l in links:
        grid[l["true_class"]]["correct" if l["correct"] else "wrong"] += 1
    for c in sorted(grid):
        g = grid[c]
        print(f"  {c:12} correct {g['correct']:4d}  wrong {g['wrong']:4d}")

    print("\nby plane:", dict(Counter(l["plane"] for l in links)))


if __name__ == "__main__":
    main()