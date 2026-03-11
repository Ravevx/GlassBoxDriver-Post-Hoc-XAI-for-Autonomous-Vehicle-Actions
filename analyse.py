# analyse.py — Full post-hoc XAI audit on a recorded video

import cv2
import torch
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from src.decision  import DrivingCNN, ACTIONS
from src.gradcam   import GradCAM
from src.flagging  import flag_uncertain_frames
from datetime      import datetime

# ── Config ───────────────────────────────────────────────
# Accept video path from command line (Streamlit) or use default
VIDEO_PATH        = sys.argv[1] if len(sys.argv) > 1 \
                    else r"J:\Agent My Learning\Other\XAI Driving\data\test\formulacam.mp4"
CONFIDENCE_THRESH = 0.6
TRUST_THRESH      = 0.5
SESSION_ID        = datetime.now().strftime("%Y%m%d_%H%M%S")
# ─────────────────────────────────────────────────────────

ACTION_STEERING = {
    "Go Straight" :   0,
    "Brake"       :   0,
    "Accelerate"  :   0,
    "Turn Left"   :  35,
    "Turn Right"  : -35,
}

ACTION_COLORS_MPL = {
    "Go Straight" : "#4CAF50",
    "Brake"       : "#F44336",
    "Accelerate"  : "#2196F3",
    "Turn Left"   : "#2196F3",
    "Turn Right"  : "#FF5722",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DrivingCNN().to(device)
model.load_state_dict(torch.load("models/driving_cnn.pth",
                                  map_location=device, weights_only=False))
model.eval()
gradcam = GradCAM(model)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def compute_trust_score(probs, heatmap):
    entropy       = -np.sum(probs * np.log(probs + 1e-8)) / np.log(len(probs))
    concentration = float(np.max(heatmap) - np.mean(heatmap))
    confidence    = float(np.max(probs))
    trust         = round((confidence + concentration + (1 - entropy)) / 3, 4)
    return max(0.0, min(1.0, trust))


def draw_steering_overlay(angle_deg, action, confidence, trust,
                           width=220, height=160):
    color_hex = ACTION_COLORS_MPL.get(action, "#4CAF50")

    fig, ax = plt.subplots(figsize=(width / 100, height / 100),
                            subplot_kw=dict(polar=False))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.35, 1.25)
    ax.axis('off')

    theta_bg = np.linspace(np.radians(180), np.radians(0), 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg),
            color='#444444', linewidth=10, alpha=0.5,
            solid_capstyle='round')

    needle_angle = np.radians(90 - angle_deg)
    if angle_deg >= 0:
        arc_theta = np.linspace(np.radians(90), needle_angle, 100)
    else:
        arc_theta = np.linspace(needle_angle, np.radians(90), 100)
    ax.plot(np.cos(arc_theta), np.sin(arc_theta),
            color=color_hex, linewidth=10, alpha=0.85,
            solid_capstyle='round')

    nx = 0.82 * np.cos(needle_angle)
    ny = 0.82 * np.sin(needle_angle)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color_hex,
                                lw=2.0, mutation_scale=18))
    ax.plot(0, 0, 'o', color=color_hex, markersize=7)

    ax.text(-1.3, -0.05, "R", fontsize=9,  color='#FF5722', fontweight='bold')
    ax.text( 1.15, -0.05, "L", fontsize=9, color='#2196F3', fontweight='bold')

    if angle_deg == 0:
        dir_txt = "STRAIGHT"
    elif angle_deg > 0:
        dir_txt = f"{abs(angle_deg)}deg LEFT"
    else:
        dir_txt = f"{abs(angle_deg)}deg RIGHT"
    ax.text(0, -0.28, dir_txt,
            ha='center', fontsize=8, fontweight='bold', color=color_hex)

    ax.text(0, 1.15, f"{action}  {confidence*100:.0f}%",
            ha='center', fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=color_hex, alpha=0.9))

    trust_color = '#4CAF50' if trust > TRUST_THRESH else '#F44336'
    ax.text(0, -0.02, f"Trust {trust:.2f}",
            ha='center', fontsize=7, color=trust_color)

    plt.tight_layout(pad=0.1)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100,
                bbox_inches='tight', facecolor='#111111')
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert('RGB')
    pil_img = pil_img.resize((width, height), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def stamp_overlay(frame_bgr, overlay_bgr, x=10, y=10):
    h, w   = overlay_bgr.shape[:2]
    fh, fw = frame_bgr.shape[:2]
    x2, y2 = min(x + w, fw), min(y + h, fh)
    ow, oh  = x2 - x, y2 - y
    roi     = frame_bgr[y:y2, x:x2].astype(np.float32)
    ovl     = overlay_bgr[:oh, :ow].astype(np.float32)
    blended = cv2.addWeighted(roi, 0.15, ovl, 0.85, 0)
    frame_bgr[y:y2, x:x2] = blended.astype(np.uint8)
    return frame_bgr


def run_audit():
    print(f"Running XAI Audit on: {VIDEO_PATH}")
    print(f"Session ID: {SESSION_ID}")

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        return

    cap          = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("output", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)
    out_path = f"output/audit_{SESSION_ID}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(out_path, fourcc, fps, (640, 480))

    results      = []
    log_rows     = []
    last_heatmap = np.zeros((224, 224))

    for frame_id in tqdm(range(total_frames), desc="Analysing"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(cv2.resize(frame, (640, 480)),
                                  cv2.COLOR_BGR2RGB)
        tensor    = transform(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            action_probs, _ = model(tensor)
            probs            = torch.softmax(action_probs, dim=1)[0].cpu().numpy()
            action_idx       = int(np.argmax(probs))
            confidence       = float(probs[action_idx])

        if frame_id % 5 == 0:
            last_heatmap = gradcam.generate(tensor, action_idx)
        heatmap = last_heatmap

        trust  = compute_trust_score(probs, heatmap)
        action = ACTIONS[action_idx]
        angle  = ACTION_STEERING[action]

        annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        overlay   = draw_steering_overlay(angle, action, confidence,
                                           trust, width=220, height=160)
        annotated = stamp_overlay(annotated, overlay, x=10, y=10)
        out.write(annotated)

        results.append({
            'frame_id'   : frame_id,
            'image'      : annotated,
            'action'     : action,
            'confidence' : confidence,
            'trust_score': trust,
            'heatmap'    : heatmap
        })
        log_rows.append({
            'frame_id'  : frame_id,
            'action'    : action,
            'confidence': round(confidence, 4),
            'trust'     : trust,
            'timestamp' : round(frame_id / fps, 3)
        })

    cap.release()
    out.release()

    log_path = f"logs/session_{SESSION_ID}.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nAudit complete!")
    print(f"Annotated video - {out_path}")
    print(f"Session log     - {log_path}")

    flag_uncertain_frames(results, SESSION_ID)


if __name__ == "__main__":
    run_audit()
