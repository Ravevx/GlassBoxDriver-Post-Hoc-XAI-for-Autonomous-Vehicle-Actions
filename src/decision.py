import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import timm

ACTIONS = ["Go Straight", "Brake", "Accelerate", "Turn Left", "Turn Right"]


class DrivingCNN(nn.Module):
    def __init__(self, num_actions=5):
        super().__init__()

        # ── EfficientNet-B0 backbone (pretrained on ImageNet) ──
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,       # remove classification head
            global_pool='avg'    # output: (batch, 1280)
        )

        self.action_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_actions)
        )

        self.steering_head = nn.Linear(1280, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return F.softmax(self.action_head(feat), dim=-1), \
               torch.tanh(self.steering_head(feat)) * 30.0


# Global model instance
_model = None


def get_model():
    global _model
    if _model is None:
        _model = DrivingCNN()
        _model.eval()
        try:
            _model.load_state_dict(torch.load("models/driving_cnn.pth",
                                               map_location="cpu",
                                               weights_only=False))
            print("Loaded pretrained weights.")
        except Exception as e:
            print(f"No weights found ({e}). Using random init.")
    return _model


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def predict_action(frame, boxes):
    model  = get_model()
    tensor = transform(frame).unsqueeze(0)

    with torch.no_grad():
        action_probs, steering = model(tensor)
        print(f"Action probs: {dict(zip(ACTIONS, action_probs[0].tolist()))}")

    action_idx  = torch.argmax(action_probs, dim=-1).item()
    action_name = ACTIONS[action_idx]
    conf        = action_probs[0][action_idx].item()

    return action_name, round(steering.item(), 2), round(conf, 3)
