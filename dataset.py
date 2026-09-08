from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("data")
IMG_SIZE = 299
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_datasets():
    base = datasets.ImageFolder(DATA_ROOT / "Training")
    labels = [s[1] for s in base.samples]
    idx_train, idx_val = train_test_split(
        range(len(labels)), test_size=0.2, stratify=labels, random_state=SEED
    )

    train_full = datasets.ImageFolder(DATA_ROOT / "Training", transform=train_tf)
    val_full = datasets.ImageFolder(DATA_ROOT / "Training", transform=eval_tf)

    train_ds = Subset(train_full, idx_train)
    val_ds = Subset(val_full, idx_val)
    test_ds = datasets.ImageFolder(DATA_ROOT / "Testing", transform=eval_tf)

    return train_ds, val_ds, test_ds, base.classes


if __name__ == "__main__":
    tr, va, te, classes = build_datasets()
    print("classes:", classes)
    print("train:", len(tr), "val:", len(va), "test:", len(te))
    x, y = tr[0]
    print("tensor shape:", x.shape, "dtype:", x.dtype)
    print("value range:", round(x.min().item(), 3), "to", round(x.max().item(), 3))
