import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.decision import get_model, transform, ACTIONS

def get_heatmap(frame, action_name, use_plusplus=False):
    model = get_model()
    target_layer = [model.backbone.layer4[-1]]

    CAMClass = GradCAMPlusPlus if use_plusplus else GradCAM
    cam = CAMClass(model=model, target_layers=target_layer)

    tensor = transform(frame).unsqueeze(0)
    action_idx = ACTIONS.index(action_name) if action_name in ACTIONS else 0
    targets = [ClassifierOutputTarget(action_idx)]

    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.resize(rgb_frame, (224, 224))
    rgb_normalized = rgb_frame.astype(np.float32) / 255.0

    overlay = show_cam_on_image(rgb_normalized, grayscale_cam, use_rgb=True)
    return grayscale_cam, overlay

def compute_concentration(grayscale_cam):
    flat = grayscale_cam.flatten()
    top_20 = np.percentile(flat, 80)
    concentration = float(np.mean(flat >= top_20))
    return round(concentration, 4)

def compute_faithfulness(frame, action_name, original_conf, grayscale_cam):
    model = get_model()
    masked_frame = frame.copy()

    # Mask the top activated region
    cam_resized = cv2.resize(grayscale_cam, (frame.shape[1], frame.shape[0]))
    mask = cam_resized > np.percentile(cam_resized, 80)
    masked_frame[mask] = 0

    tensor = transform(masked_frame).unsqueeze(0)
    with torch.no_grad():
        action_probs, _ = model(tensor)

    action_idx = ACTIONS.index(action_name) if action_name in ACTIONS else 0
    masked_conf = action_probs[0][action_idx].item()

    faithfulness = original_conf - masked_conf
    return round(faithfulness, 4)

def compute_agreement(frame, action_name):
    cam_gc, _ = get_heatmap(frame, action_name, use_plusplus=False)
    cam_gcpp, _ = get_heatmap(frame, action_name, use_plusplus=True)

    # Cosine similarity between two heatmaps
    flat1 = cam_gc.flatten()
    flat2 = cam_gcpp.flatten()
    agreement = float(np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-8))
    return round(agreement, 4)

def generate_xai(frame, action_name, original_conf):
    grayscale_cam, overlay = get_heatmap(frame, action_name)
    concentration = compute_concentration(grayscale_cam)
    faithfulness = compute_faithfulness(frame, action_name, original_conf, grayscale_cam)
    agreement = compute_agreement(frame, action_name)
    trust = round(np.mean([concentration, faithfulness, agreement]), 4)

    return overlay, {
        'concentration': concentration,
        'faithfulness': faithfulness,
        'agreement': agreement,
        'trust': trust
    }
