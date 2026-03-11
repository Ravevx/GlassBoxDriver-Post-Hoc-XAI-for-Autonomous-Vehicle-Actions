# screen_ai.py — GlassBoxDriver Live Screen Guidance with Steering Arc
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from mss import mss
from torchvision import transforms
from src.decision import DrivingCNN, ACTIONS

ACTION_STEERING = {
    "Go Straight" :   0,
    "Brake"       :   0,
    "Accelerate"  :   0,
    "Turn Left"   :  35,
    "Turn Right"  : -35,
}

ACTION_COLORS = {
    "Go Straight" : (76,  175, 80),   # green
    "Brake"       : (244, 67,  54),   # red
    "Accelerate"  : (33,  150, 245),  # blue
    "Turn Left"   : (33,  150, 245),  # blue
    "Turn Right"  : (255, 87,  34),   # orange
}

ACTION_COLORS_HEX = {
    "Go Straight" : "#4CAF50",
    "Brake"       : "#F44336",
    "Accelerate"  : "#2196F3",
    "Turn Left"   : "#2196F3",
    "Turn Right"  : "#FF5722",
}

print("Loading GlassBoxDriver model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = DrivingCNN().to(device)
model.load_state_dict(torch.load("models/driving_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

SCREEN_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080}
CONFIDENCE_THRESHOLD = 0.45   # lowered — was 0.6, too strict

sct         = mss()
frame_count = 0
last_action      = ACTIONS[0]
last_confidence  = 0.0
last_probs       = np.ones(len(ACTIONS)) / len(ACTIONS)
last_arc_img     = None        # cache rendered arc so we dont redraw every frame


def render_arc(angle_deg, action, confidence, all_probs, width=300, height=420):
    """Render steering arc + prob bars as BGR numpy image."""
    color_hex = ACTION_COLORS_HEX.get(action, "#4CAF50")

    fig = plt.figure(figsize=(width / 100, height / 100), facecolor='#111111')

    # ── Top: steering arc ────────────────────────────────────
    ax = fig.add_axes([0.05, 0.45, 0.90, 0.50])   # [left, bottom, w, h]
    ax.set_facecolor('#111111')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.35, 1.25)
    ax.axis('off')

    # Background arc
    theta_bg = np.linspace(np.radians(180), np.radians(0), 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg),
            color='#444444', linewidth=10, alpha=0.5, solid_capstyle='round')

    # Colored arc
    needle_angle = np.radians(90 - angle_deg)
    if angle_deg >= 0:
        arc_theta = np.linspace(np.radians(90), needle_angle, 100)
    else:
        arc_theta = np.linspace(needle_angle, np.radians(90), 100)
    ax.plot(np.cos(arc_theta), np.sin(arc_theta),
            color=color_hex, linewidth=10, alpha=0.85, solid_capstyle='round')

    # Needle
    nx = 0.82 * np.cos(needle_angle)
    ny = 0.82 * np.sin(needle_angle)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color_hex,
                                lw=2.0, mutation_scale=18))
    ax.plot(0, 0, 'o', color=color_hex, markersize=7)

    # R / L labels
    ax.text(-1.3, -0.05, "R", fontsize=9,  color='#FF5722', fontweight='bold')
    ax.text( 1.15, -0.05, "L", fontsize=9, color='#2196F3', fontweight='bold')

    # Degree label
    if angle_deg == 0:
        dir_txt = "STRAIGHT"
    elif angle_deg > 0:
        dir_txt = f"{abs(angle_deg)}deg LEFT"
    else:
        dir_txt = f"{abs(angle_deg)}deg RIGHT"
    ax.text(0, -0.28, dir_txt,
            ha='center', fontsize=8, fontweight='bold', color=color_hex)

    # Action badge
    ax.text(0, 1.15, f"{action}  {confidence*100:.0f}%",
            ha='center', fontsize=9, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color_hex, alpha=0.9))

    # ── Bottom: probability bars ──────────────────────────────
    ax2 = fig.add_axes([0.10, 0.02, 0.85, 0.40])
    ax2.set_facecolor('#111111')

    colors = ['#4CAF50' if ACTIONS[i] == action else '#555555'
              for i in range(len(ACTIONS))]
    bars = ax2.barh(ACTIONS, all_probs, color=colors, height=0.6)
    ax2.set_xlim(0, 1)
    ax2.set_facecolor('#111111')
    ax2.tick_params(colors='white', labelsize=7)
    ax2.spines[:].set_visible(False)
    ax2.xaxis.set_visible(False)

    # Percentage labels on bars
    for bar, prob in zip(bars, all_probs):
        ax2.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                 f"{prob*100:.0f}%", va='center', fontsize=7,
                 color='white', fontweight='bold')

    # Convert to BGR numpy
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100,
                bbox_inches='tight', facecolor='#111111')
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert('RGB')
    pil_img = pil_img.resize((width, height), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def stamp(frame, overlay, x=20, y=20):
    """Paste overlay onto frame with slight transparency."""
    h, w   = overlay.shape[:2]
    fh, fw = frame.shape[:2]
    x2, y2 = min(x + w, fw), min(y + h, fh)
    ow, oh = x2 - x, y2 - y
    roi     = frame[y:y2, x:x2].astype(np.float32)
    ovl     = overlay[:oh, :ow].astype(np.float32)
    frame[y:y2, x:x2] = cv2.addWeighted(roi, 0.1, ovl, 0.9, 0).astype(np.uint8)
    return frame


print("GlassBoxDriver Live Guidance READY!")
print("Press Q to quit")

while True:
    screenshot = sct.grab(SCREEN_REGION)
    frame      = np.array(screenshot)
    frame      = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame_count += 1

    # ── Run inference every 3 frames ─────────────────────────
    if frame_count % 3 == 0:
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            action_probs, _ = model(input_tensor)
            probs            = torch.softmax(action_probs, dim=1)[0].cpu().numpy()
            action_idx       = int(np.argmax(probs))
            last_action      = ACTIONS[action_idx]
            last_confidence  = float(probs[action_idx])
            last_probs       = probs

    # ── Redraw arc every 6 frames (not every frame — saves CPU) ──
    if frame_count % 6 == 0 or last_arc_img is None:
        angle        = ACTION_STEERING[last_action]
        last_arc_img = render_arc(angle, last_action,
                                   last_confidence, last_probs,
                                   width=280, height=400)

    # ── Stamp arc onto frame ──────────────────────────────────
    frame = stamp(frame, last_arc_img, x=20, y=20)

    cv2.imshow("GlassBoxDriver - Live Guidance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Session complete!")
