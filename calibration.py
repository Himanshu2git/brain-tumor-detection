import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
N_BINS = 15


def calibration_stats(probs, labels, n_bins=N_BINS):
    conf = probs.max(1)
    preds = probs.argmax(1)
    correct = (preds == labels).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1], right=False)

    rows, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": b, "n": 0, "conf": None, "acc": None})
            continue
        c, a = conf[m].mean(), correct[m].mean()
        gap = abs(c - a)
        ece += n / len(conf) * gap
        mce = max(mce, gap)
        rows.append({"bin": b, "n": n, "conf": float(c), "acc": float(a),
                     "gap": float(gap)})

    return {"ece": float(ece), "mce": float(mce),
            "accuracy": float(correct.mean()),
            "mean_confidence": float(conf.mean()),
            "bins": rows}


def reliability_plot(stats_by_name, path):
    fig, axes = plt.subplots(1, len(stats_by_name),
                             figsize=(5 * len(stats_by_name), 4.5))
    if len(stats_by_name) == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, stats_by_name.items()):
        xs = [b["conf"] for b in s["bins"] if b["n"] > 0]
        ys = [b["acc"] for b in s["bins"] if b["n"] > 0]
        ns = [b["n"] for b in s["bins"] if b["n"] > 0]

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.scatter(xs, ys, s=np.sqrt(ns) * 4, alpha=0.75, zorder=3)
        ax.plot(xs, ys, lw=1, alpha=0.6)

        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("mean confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{name}\nECE {s['ece']:.4f}   acc {s['accuracy']:.4f}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    sets = {
        "in-distribution (test)": ("probs_finetuned.npy", "labels.npy"),
        "external (Bangladesh)": ("probs_external.npy", "labels_external.npy"),
    }

    out = {}
    for name, (pf, lf) in sets.items():
        probs = np.load(RESULTS / pf)
        labels = np.load(RESULTS / lf)
        s = calibration_stats(probs, labels)
        out[name] = s

        print(f"=== {name} ===")
        print(f"  n:               {len(labels)}")
        print(f"  accuracy:        {s['accuracy']:.4f}")
        print(f"  mean confidence: {s['mean_confidence']:.4f}")
        print(f"  ECE:             {s['ece']:.4f}")
        print(f"  MCE:             {s['mce']:.4f}")
        print(f"  gap (conf-acc):  {s['mean_confidence']-s['accuracy']:+.4f}\n")

    reliability_plot(out, RESULTS / "reliability.png")
    with open(RESULTS / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/reliability.png")


if __name__ == "__main__":
    main()