import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torchvision import models

ACTIONS = ["Go Straight", "Brake", "Accelerate", "Turn Left", "Turn Right"]

class DrivingCNN(nn.Module):
    def __init__(self, num_actions=5):
        super().__init__()
        base = models.resnet18(pretrained=True)
        # Replace final layer
        base.fc = nn.Identity()
        self.backbone = base
        self.action_head = nn.Linear(512, num_actions)
        self.steering_head = nn.Linear(512, 1)

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
        # Load weights if exist
        try:
            _model.load_state_dict(torch.load("models/driving_cnn.pth", map_location="cpu"))
            print("Loaded pretrained weights.")
        except:
            print("No weights found. Using random init.")
    return _model

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_action(frame, boxes):
    model = get_model()
    tensor = transform(frame).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        action_probs, steering = model(tensor)
        # Add after action_probs:
        print(f"Action probs: {dict(zip(ACTIONS, action_probs[0].tolist()))}")

    action_idx = torch.argmax(action_probs, dim=-1).item()
    action_name = ACTIONS[action_idx]
    conf = action_probs[0][action_idx].item()

    return action_name, round(steering.item(), 2), round(conf, 3)
