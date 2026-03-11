# app.py — GlassBoxDriver Streamlit App (Full System)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="GlassBoxDriver",
    page_icon="🚗",
    layout="wide"
)

st.sidebar.image("https://img.icons8.com/emoji/96/automobile.png", width=80)
st.sidebar.title("GlassBoxDriver")
st.sidebar.caption("XAI for Autonomous Vehicle Decision Making")

# ── Navigation ────────────────────────────────────────────
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🎥 Run Audit",
    "🔍 Review Flags",
    "🔁 Feedback Retrain",
    "📊 Session Logs"
])
st.sidebar.divider()
st.sidebar.info("**Model:** ResNet18\n\n**Data:** nuScenes Mini\n\n**XAI:** Grad-CAM")

# ── Steering angle per action ─────────────────────────────
ACTION_STEERING = {
    "Go Straight" :   0,
    "Brake"       :   0,
    "Accelerate"  :   0,
    "Turn Left"   :  35,
    "Turn Right"  : -35,
}

ACTION_COLORS = {
    "Go Straight" : "#4CAF50",
    "Brake"       : "#F44336",
    "Accelerate"  : "#2196F3",
    "Turn Left"   : "#2196F3",
    "Turn Right"  : "#FF5722",
}

def draw_steering(angle_deg, action, confidence):
    """Draw a steering arc with needle showing direction."""
    fig, ax = plt.subplots(figsize=(4, 2.8), subplot_kw=dict(polar=False))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.25, 1.25)
    ax.axis('off')

    # Background arc
    theta = np.linspace(np.radians(180), np.radians(0), 200)
    ax.plot(np.cos(theta), np.sin(theta), color='#555555',
            linewidth=14, alpha=0.25, solid_capstyle='round')

    color = ACTION_COLORS.get(action, "#4CAF50")

    # Needle angle: 90° = straight, positive = left, negative = right
    needle_angle = np.radians(90 - angle_deg)

    # Colored arc from straight to needle
    if angle_deg >= 0:
        arc_theta = np.linspace(np.radians(90), needle_angle, 100)
    else:
        arc_theta = np.linspace(needle_angle, np.radians(90), 100)

    ax.plot(np.cos(arc_theta), np.sin(arc_theta),
            color=color, linewidth=14, alpha=0.75,
            solid_capstyle='round')

    # Needle arrow
    nx = 0.85 * np.cos(needle_angle)
    ny = 0.85 * np.sin(needle_angle)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.5, mutation_scale=22))

    # Center dot
    ax.plot(0, 0, 'o', color=color, markersize=9)

    # R / L labels
    ax.text(-1.3, -0.05, "R", fontsize=12, color='#FF5722', fontweight='bold')
    ax.text( 1.15, -0.05, "L", fontsize=12, color='#2196F3', fontweight='bold')

    # Degree label
    if angle_deg == 0:
        dir_label = "STRAIGHT"
    elif angle_deg > 0:
        dir_label = f"{abs(angle_deg)}° LEFT"
    else:
        dir_label = f"{abs(angle_deg)}° RIGHT"

    ax.text(0, -0.18, dir_label,
            ha='center', fontsize=11, fontweight='bold', color=color)

    # Action + confidence badge
    ax.text(0, 1.15, f"{action}  ({confidence*100:.1f}%)",
            ha='center', fontsize=11, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=color, alpha=0.88))

    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    plt.tight_layout()
    return fig

def run_single_image_inference(frame_bgr):
    """Run model inference on a single BGR frame."""
    import torch
    import cv2
    from torchvision import transforms
    from src.decision import DrivingCNN, ACTIONS

    infer_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # ✅ No augmentations — match exactly what model was trained on
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    @st.cache_resource
    def load_model():
        model = DrivingCNN()
        model.load_state_dict(torch.load("models/driving_cnn.pth",
                                          map_location="cpu"))
        model.eval()
        return model

    model  = load_model()
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = infer_transform(rgb).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(tensor)
        probs     = torch.softmax(logits, dim=1)[0]

    idx    = probs.argmax().item()
    return ACTIONS[idx], probs[idx].item(), probs.numpy(), ACTIONS

# ── Pages ─────────────────────────────────────────────────

if page == "🏠 Home":
    st.title("🚗 GlassBoxDriver")
    st.subheader("Post-Hoc XAI Analysis for Autonomous Vehicle Decision Making")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model",    "ResNet18")
    col2.metric("Classes",  "5 Actions")
    col3.metric("XAI",      "Grad-CAM")
    col4.metric("Pipeline", "Closed-Loop")

    st.divider()
    st.markdown("""
    ## How It Works
    
    ```
    1. 🎥  AUDIT     →  AI analyses your dashcam video frame by frame
    2. ⚠️  FLAG      →  Uncertain decisions flagged automatically  
    3. 👤  REVIEW    →  You correct the AI's mistakes
    4. 🔁  RETRAIN   →  Model learns from corrections
    5. 🔄  REPEAT    →  Continuously improving AI
    ```
    
    ## System Architecture
    - **Model:** ResNet18 pretrained on ImageNet, fine-tuned on nuScenes
    - **XAI:** Grad-CAM heatmaps show where AI looks
    - **Trust Score:** Entropy + Concentration + Confidence
    - **Feedback Loop:** Human-in-the-loop active learning
    """)

elif page == "🎥 Run Audit":
    st.title("🎥 Run XAI Audit")
    st.caption("Analyse a dashcam/game recording with full Grad-CAM explanation")

    # ── Tab layout: Video Audit | Single Image Test ──────────
    tab1, tab2 = st.tabs(["📹 Video Audit", "🖼️ Single Image Test"])

    # ────────────────────────────────────────────────────────
    # TAB 1 — full video audit
    # ────────────────────────────────────────────────────────
    with tab1:
        video_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])

        col1, col2 = st.columns(2)
        conf_thresh  = col1.slider("Confidence Threshold", 0.3, 0.9, 0.6)
        trust_thresh = col2.slider("Trust Score Threshold", 0.2, 0.8, 0.5)

        if video_file and st.button("🔍 Start Audit", type="primary"):
            import os
            os.makedirs("data/video", exist_ok=True)
            video_path = f"data/video/{video_file.name}"
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            st.info(f"Saved video to: {video_path}")  # ← confirm path

            with st.spinner("Running Grad-CAM analysis on all frames..."):
                import subprocess
                result = subprocess.run(
                    ["python", "analyse.py", video_path],  # ← fixed
                    capture_output=True, text=True
                )
                st.code(result.stdout)
                if result.returncode == 0:
                    st.success("Audit complete! Check Review Flags tab.")
                else:
                    st.error(result.stderr)


    # ────────────────────────────────────────────────────────
    # TAB 2 — single image test with steering arc
    # ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Quick Image Prediction")
        st.caption("Upload any driving image to instantly see prediction + steering direction")

        uploaded = st.file_uploader("Upload a driving image",
                                    type=['jpg', 'jpeg', 'png'],
                                    key="single_img")

        if uploaded:
            import cv2

            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col_img, col_result = st.columns([2, 1])

            with col_img:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         caption="Input Frame", use_column_width=True)

            with col_result:
                with st.spinner("Running inference..."):
                    action, conf, all_probs, ACTIONS_LIST = \
                        run_single_image_inference(frame)

                angle = ACTION_STEERING[action]

                # ── Steering arc ──
                fig = draw_steering(angle, action, conf)
                st.pyplot(fig)
                plt.close()

                st.divider()

                # ── All class probabilities ──
                st.markdown("#### Class Probabilities")
                for a, p in zip(ACTIONS_LIST, all_probs):
                    color_label = "🟢" if a == action else "⚪"
                    st.progress(float(p),
                                text=f"{color_label} {a}: {p*100:.1f}%")

elif page == "🔍 Review Flags":
    st.title("🔍 Review Flagged Frames")
    st.caption("Correct AI mistakes → approve for retraining")

    import os
    import pandas as pd
    import shutil
    from src.decision import ACTIONS

    LOG_DIR     = "logs"
    REVIEW_DIR  = "data/flagged/review"
    APPROVE_DIR = "data/flagged/approved"

    csv_files = [f for f in os.listdir(LOG_DIR)
                 if f.startswith("flagged_")] if os.path.exists(LOG_DIR) else []

    if not csv_files:
        st.warning("No flagged frames yet. Run an audit first.")
    else:
        selected = st.selectbox("Select session:", csv_files)
        df = pd.read_csv(os.path.join(LOG_DIR, selected))
        st.markdown(f"**{len(df)} frames** need review")

        for _, row in df.iterrows():
            img_path  = os.path.join(REVIEW_DIR, row['filename'])
            heat_path = img_path.replace('.jpg', '_heatmap.jpg')
            if not os.path.exists(img_path):
                continue

            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                st.image(img_path, caption=f"Frame #{row['frame_id']}")
            with col2:
                if os.path.exists(heat_path):
                    st.image(heat_path, caption="Heatmap")
            with col3:
                st.markdown(f"**AI Said:** `{row['action']}`")
                st.markdown(f"**Confidence:** `{row['confidence']:.0%}`")
                st.markdown(f"**Trust:** `{row['trust_score']}`")
                label = st.selectbox("Correct label:", ["⏭️ Skip"] + ACTIONS,
                                     key=f"l_{row['frame_id']}")
                if st.button("✅ Approve", key=f"b_{row['frame_id']}"):
                    if label != "⏭️ Skip":
                        dst_dir = os.path.join(APPROVE_DIR, label)
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.copy(img_path, os.path.join(dst_dir, row['filename']))
                        st.success(f"Saved to {label}!")
            st.divider()

elif page == "🔁 Feedback Retrain":
    st.title("🔁 Feedback Retrain")
    st.caption("Merge approved corrections into training data and retrain model")

    import os
    from src.decision import ACTIONS

    APPROVE_DIR = "data/flagged/approved"

    st.subheader("Approved Frames Ready for Training")
    total = 0
    for action in ACTIONS:
        path  = os.path.join(APPROVE_DIR, action)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        st.metric(action, f"{count} frames")
        total += count

    st.divider()
    st.metric("Total New Frames", total)

    if total == 0:
        st.warning("No approved frames yet. Go to Review Flags tab first.")
    else:
        if st.button("🔁 Start Feedback Retrain", type="primary"):
            with st.spinner("Retraining model..."):
                import subprocess
                result = subprocess.run(["python", "-c",
                    "from src.feedback import run_feedback_cycle; run_feedback_cycle()"],
                    capture_output=True, text=True)
                st.code(result.stdout)
                if result.returncode == 0:
                    st.success("✅ Retraining complete! Model updated.")
                else:
                    st.error(result.stderr)

elif page == "📊 Session Logs":
    st.title("📊 Session Logs")
    st.caption("View all past audit sessions")

    import os
    import pandas as pd
    import plotly.express as px

    LOG_DIR   = "logs"
    csv_files = [f for f in os.listdir(LOG_DIR)
                 if f.startswith("session_")] if os.path.exists(LOG_DIR) else []

    if not csv_files:
        st.warning("No sessions yet. Run an audit first.")
    else:
        selected = st.selectbox("Select session:", csv_files)
        df = pd.read_csv(os.path.join(LOG_DIR, selected))

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Frames", len(df))
        col2.metric("Avg Confidence", f"{df['confidence'].mean():.0%}")
        col3.metric("Avg Trust", f"{df['trust'].mean():.2f}")

        st.divider()

        fig1 = px.pie(df, names='action', title='Action Distribution')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(df, x='timestamp', y='trust',
                       title='Trust Score Over Time',
                       color_discrete_sequence=['#00CC96'])
        fig2.add_hline(y=0.5, line_dash="dash", line_color="red",
                       annotation_text="Threshold")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("View Raw Log"):
            st.dataframe(df, use_container_width=True)
