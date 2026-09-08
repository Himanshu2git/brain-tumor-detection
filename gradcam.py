import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer_name="act4"):
        self.model = model.eval()
        self.activations = None
        self.gradients = None

        layer = dict(model.named_modules())[target_layer_name]
        if hasattr(layer, "inplace"):
            layer.inplace = False
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient) 
    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)

        self.model.zero_grad()
        score = logits[range(len(class_idx)), class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)

        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        cam_min = flat.min(1)[0].view(-1, 1, 1)
        cam_max = flat.max(1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.cpu().numpy(), logits.detach(), class_idx
        