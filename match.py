import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from cheng import CHENG_DIR, CHENG_LABELS, to_uint8

NICK_ROOT = Path("data")
SIG = 32
RESULTS = Path("results")


def bbox_crop(arr, thresh=15):
    """Crop to the bounding box of non-background pixels."""
    mask = arr > thresh
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return arr
    return arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def signature(arr):
    """Crop to brain, downsample, zero-mean unit-variance normalize."""
    arr = bbox_crop(np.asarray(arr))
    img = Image.fromarray(arr).convert("L").resize((SIG, SIG), Image.BILINEAR)
    v = np.asarray(img, dtype=np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def build_cheng_signatures():
    files = sorted(CHENG_DIR.glob("*.mat"), key=lambda p: int(p.stem))
    sigs, meta = [], []
    for p in files:
        with h5py.File(p, "r") as f:
            d = f["cjdata"]
            img = to_uint8(np.array(d["image"]).astype(np.float32))
            pid = "".join(chr(c) for c in np.array(d["PID"]).ravel())
            label = int(np.array(d["label"]).ravel()[0])
        sigs.append(signature(img))
        meta.append({"file": p.name, "pid": pid,
                     "class": CHENG_LABELS[label]})
    return np.stack(sigs), meta


def build_nick_signatures():
    paths = sorted(NICK_ROOT.glob("*/*/*.jpg"))
    paths = [p for p in paths if "cheng" not in str(p)]
    sigs, meta = [], []
    for p in paths:
        arr = np.asarray(Image.open(p).convert("L"))
        sigs.append(signature(arr))
        meta.append({"path": str(p), "split": p.parts[1], "class": p.parts[2]})
    return np.stack(sigs), meta


def main():
    RESULTS.mkdir(exist_ok=True)

    print("building Cheng signatures...")
    csig, cmeta = build_cheng_signatures()
    print(f"  {csig.shape[0]} records")

    print("building Nickparvar signatures...")
    nsig, nmeta = build_nick_signatures()
    print(f"  {nsig.shape[0]} images")

    # correlation = dot product of normalized signatures
    C = torch.from_numpy(csig).cuda()
    N = torch.from_numpy(nsig).cuda()
    corr = N @ C.T                                    # (n_nick, n_cheng)

    top2 = corr.topk(2, dim=1)
    best_score = top2.values[:, 0].cpu().numpy()
    second_score = top2.values[:, 1].cpu().numpy()
    best_idx = top2.indices[:, 0].cpu().numpy()
    margin = best_score - second_score

    matches = []
    for i, m in enumerate(nmeta):
        c = cmeta[best_idx[i]]
        matches.append({**m,
                        "cheng_file": c["file"],
                        "cheng_pid": c["pid"],
                        "cheng_class": c["class"],
                        "score": float(best_score[i]),
                        "margin": float(margin[i]),
                        "class_agrees": c["class"] == m["class"]})

    with open(RESULTS / "matches.json", "w") as f:
        json.dump(matches, f, indent=2)

    print("\n=== match quality ===")
    for t in [0.99, 0.98, 0.95, 0.90]:
        n = int((best_score >= t).sum())
        print(f"  score >= {t}: {n:5d} ({n/len(matches)*100:.1f}%)")

    print(f"\n  median score:  {np.median(best_score):.4f}")
    print(f"  median margin: {np.median(margin):.4f}")

    agree = sum(m["class_agrees"] for m in matches)
    print(f"\n  class agreement: {agree}/{len(matches)} "
          f"({agree/len(matches)*100:.1f}%)")

    notumor = sum(1 for m in matches if m["class"] == "notumor")
    print(f"  (notumor images: {notumor} — these come from Br35H, "
          f"not Cheng, so should NOT match)")


if __name__ == "__main__":
    main()