# src/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.backbone.blocks[-1][-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
        print("GradCAM hooked into: backbone.blocks[-1][-1]")

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        action_probs, _ = self.model(input_tensor)

        if class_idx is None:
            class_idx = action_probs.argmax().item()

        self.model.zero_grad()
        action_probs[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam)
        cam     = cam.cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = (cam * 255).astype(np.uint8)
        cam = cv2.resize(cam, (224, 224))

        return cam
