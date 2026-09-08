from pathlib import Path

import h5py
import numpy as np

CHENG_DIR = Path("data/cheng/mat")

# Cheng label convention differs from our alphabetical ImageFolder ordering
CHENG_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def load_record(path):
    with h5py.File(path, "r") as f:
        d = f["cjdata"]
        pid = "".join(chr(c) for c in np.array(d["PID"]).ravel())
        label = int(np.array(d["label"]).ravel()[0])
        image = np.array(d["image"]).astype(np.float32)
        mask = np.array(d["tumorMask"]).astype(np.uint8)

    return {
        "file": path.name,
        "pid": pid,
        "label": label,
        "class": CHENG_LABELS[label],
        "image": image,
        "mask": mask,
    }


def to_uint8(image):
    """Normalize raw MRI intensities to 0-255, as the JPEG version would be."""
    lo, hi = image.min(), image.max()
    return ((image - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


if __name__ == "__main__":
    files = sorted(CHENG_DIR.glob("*.mat"), key=lambda p: int(p.stem))
    print(f"found {len(files)} files")

    from collections import Counter
    classes, pids = Counter(), set()
    for p in files:
        with h5py.File(p, "r") as f:
            d = f["cjdata"]
            classes[CHENG_LABELS[int(np.array(d["label"]).ravel()[0])]] += 1
            pids.add("".join(chr(c) for c in np.array(d["PID"]).ravel()))

    print("class counts:", dict(classes))
    print("unique patients:", len(pids))
    print("mean slices per patient:", round(len(files) / len(pids), 1))