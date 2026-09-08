import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_datasets
from model import build_model

MAX_EPOCHS = 30
PATIENCE = 5
NUM_WORKERS = 4
CKPT = Path("checkpoints")


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

    return total_loss / seen, correct / seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="best.pt")
    args = p.parse_args()

    torch.manual_seed(42)
    device = "cuda"
    CKPT.mkdir(exist_ok=True)

    train_ds, val_ds, _, classes = build_datasets()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(freeze_backbone=not args.finetune).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr
    )

    trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"mode: {'finetune' if args.finetune else 'frozen'}  "
          f"batch {args.batch_size}  lr {args.lr}  trainable {trainable:,}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        dt = time.time() - t0

        marker = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CKPT / args.out)
            marker = "  <- saved"
        else:
            epochs_without_improvement += 1

        print(f"epoch {epoch:02d}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}  "
              f"({dt:.0f}s){marker}")

        if epochs_without_improvement >= PATIENCE:
            print(f"early stop: no improvement for {PATIENCE} epochs")
            break

    print(f"best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()