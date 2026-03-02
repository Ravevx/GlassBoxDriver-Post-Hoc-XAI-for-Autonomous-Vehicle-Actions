import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from PIL import Image

from src.detect import get_detections
from src.decision import predict_action
from src.xai import generate_xai

st.set_page_config(page_title="VideoDrive-XAI", layout="wide")

st.title("🚗 VideoDrive-XAI")
st.markdown("Upload dashcam video → YOLO detections → Driving decisions → **Grad-CAM XAI + Annotated Export**")

uploaded        = st.sidebar.file_uploader("Upload Dashcam Video (.mp4)", type=["mp4", "avi", "mov"])
frame_skip      = st.sidebar.slider("Analyse every N frames", 1, 30, 10)
trust_threshold = st.sidebar.slider("⚠️ Risk alert below trust", 0.0, 1.0, 0.60)

ACTION_COLORS = {
    "Go Straight": (0, 255, 0),
    "Brake":       (0, 0, 255),
    "Accelerate":  (0, 165, 255),
    "Turn Left":   (0, 255, 255),
    "Turn Right":  (0, 255, 255),
}

def draw_steering_arrow(frame, steering_deg):
    """
    Draws a steering wheel arrow in bottom-right corner.
    steering_deg: -30 (full left) to +30 (full right), 0 = straight
    """
    h, w = frame.shape[:2]

    # Arrow base position (bottom right area)
    cx, cy = w - 80, h - 80
    radius = 55
    arrow_len = 45

    # Draw circle (steering wheel)
    cv2.circle(frame, (cx, cy), radius, (80, 80, 80), 3)
    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

    # Convert steering angle to arrow direction
    # 0 deg = straight up (-90 in cv2 coords)
    # positive = right, negative = left
    angle_rad = np.radians(-90 + steering_deg * 2)  # scale for visibility
    end_x = int(cx + arrow_len * np.cos(angle_rad))
    end_y = int(cy + arrow_len * np.sin(angle_rad))

    # Color based on direction
    if abs(steering_deg) < 3:
        arrow_color = (0, 255, 0)    # Green = straight
    elif steering_deg > 0:
        arrow_color = (0, 255, 255)  # Yellow = right
    else:
        arrow_color = (0, 255, 255)  # Yellow = left

    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y),
                    arrow_color, 3, tipLength=0.35)

    # Degree label
    label = f"{steering_deg:+.1f}°"
    cv2.putText(frame, label, (cx - 22, cy + radius + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Direction text
    if abs(steering_deg) < 3:
        direction = "STRAIGHT"
    elif steering_deg > 0:
        direction = "RIGHT"
    else:
        direction = "LEFT"

    cv2.putText(frame, direction, (cx - 28, cy + radius + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, arrow_color, 1)

    return frame


def draw_overlay(frame, boxes, labels, confs, action, steering, trust, metrics):
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw detection boxes
    for (x1, y1, x2, y2), label, conf in zip(boxes, labels, confs):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw steering arrow on the frame itself (top right)
    out = draw_steering_arrow(out, steering)

    # Bottom HUD bar
    bar_h = 110
    bar   = np.zeros((bar_h, w, 3), dtype=np.uint8)

    color      = ACTION_COLORS.get(action, (255, 255, 255))
    risk_label = "SAFE" if trust >= trust_threshold else "RISKY"
    risk_color = (0, 200, 0) if trust >= trust_threshold else (0, 0, 255)

    # Action with colored badge
    cv2.rectangle(bar, (8, 6), (200, 34), color, -1)
    cv2.putText(bar, f"  {action}", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Trust score
    cv2.putText(bar, f"Trust: {trust:.3f}", (220, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    # Risk badge
    cv2.rectangle(bar, (370, 6), (460, 34), risk_color, -1)
    cv2.putText(bar, risk_label, (378, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # XAI metrics row
    cv2.putText(bar, f"Concentration: {metrics['concentration']:.3f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    cv2.putText(bar, f"Faithfulness: {metrics['faithfulness']:.3f}", (230, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    cv2.putText(bar, f"Agreement: {metrics['agreement']:.3f}", (450, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # Objects detected
    obj_text = ', '.join(set(labels)) if labels else 'No objects'
    cv2.putText(bar, f"Detected: {obj_text}", (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 220, 150), 1)

    return np.vstack([out, bar])


if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded.read())
    video_path = tfile.name

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info(f"Video: {total_frames} frames @ {fps:.1f} FPS — Analysing every {frame_skip} frames")

    out_path = "output_explained.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out_h    = orig_h + 110
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, out_h))

    results        = []
    heatmap_images = []
    progress       = st.progress(0)
    status         = st.empty()

    # Cached last result for non-analysed frames
    last_action   = "Go Straight"
    last_steering = 0.0
    last_trust    = 1.0
    last_metrics  = {'concentration': 0.0, 'faithfulness': 0.0, 'agreement': 0.0}
    last_boxes, last_confs, last_labels = [], [], []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            status.text(f"Analysing frame {frame_idx}/{total_frames}...")

            boxes, confs, labels       = get_detections(frame)
            action, steering, action_conf = predict_action(frame, boxes)
            overlay_img, metrics       = generate_xai(frame, action, action_conf)
            trust = round(np.mean([
                metrics['concentration'],
                metrics['faithfulness'],
                metrics['agreement']
            ]), 4)

            last_action, last_steering = action, steering
            last_trust, last_metrics   = trust, metrics
            last_boxes, last_confs, last_labels = boxes, confs, labels

            results.append({
                'Frame':            frame_idx,
                'Time (s)':         round(frame_idx / fps, 2),
                'Objects Detected': len(boxes),
                'Detected Labels':  ', '.join(set(labels)) if labels else 'None',
                'Action':           action,
                'Steering (°)':     steering,
                'Action Conf':      action_conf,
                'Concentration':    metrics['concentration'],
                'Faithfulness':     metrics['faithfulness'],
                'Agreement':        metrics['agreement'],
                'Trust Score':      trust,
                'Risk':             '🚨 RISKY' if trust < trust_threshold else '✅ Safe'
            })

            heatmap_images.append((frame_idx, overlay_img))

        annotated = draw_overlay(
            frame,
            last_boxes, last_labels, last_confs,
            last_action, last_steering, last_trust, last_metrics
        )
        writer.write(annotated)
        progress.progress(min(frame_idx / total_frames, 1.0))
        frame_idx += 1

    cap.release()
    writer.release()
    status.text("✅ Done!")

    df = pd.DataFrame(results)

    st.subheader("📊 Decision Log")
    st.dataframe(df, use_container_width=True)

    st.subheader("🔥 Grad-CAM Heatmaps")
    cols = st.columns(3)
    for i, (fidx, hmap) in enumerate(heatmap_images[:9]):
        with cols[i % 3]:
            row = df[df['Frame'] == fidx].iloc[0]
            st.image(hmap,
                     caption=f"Frame {fidx} | {row['Action']} | Trust: {row['Trust Score']}",
                     width=300)

    risky = df[df['Risk'] == '🚨 RISKY']
    if not risky.empty:
        st.error(f"🚨 {len(risky)} risky decisions detected!")
        st.dataframe(risky[['Frame', 'Time (s)', 'Action', 'Trust Score']])
    else:
        st.success("✅ All decisions above trust threshold!")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download CSV",
                           df.to_csv(index=False),
                           "drivedrive_results.csv")
    with col2:
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download Annotated Video",
                               f, "explained_drive.mp4", "video/mp4")

    st.subheader("🎬 Annotated Output Video")
    st.video(out_path)

st.markdown("---")
st.markdown("**Dataset**: nuScenes Mini | **Model**: ResNet18 | **XAI**: Grad-CAM++")
