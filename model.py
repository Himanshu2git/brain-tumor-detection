import timm
import torch
import torch.nn as nn

MODEL_NAME = "legacy_xception.tf_in1k"
NUM_CLASSES = 4


def build_model(freeze_backbone=True):
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model


def trainable_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    m = build_model(freeze_backbone=True)
    total, trainable = trainable_summary(m)
    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"classifier: {m.get_classifier()}")

    m = m.cuda().eval()
    with torch.no_grad():
        out = m(torch.randn(4, 3, 299, 299, device="cuda"))
    print("output shape:", out.shape)
    print("VRAM used (MB):", round(torch.cuda.max_memory_allocated() / 1e6, 1))
