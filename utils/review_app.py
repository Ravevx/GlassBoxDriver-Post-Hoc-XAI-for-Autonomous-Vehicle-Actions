# review_app.py — Human review UI for flagged frames
import streamlit as st
import os
import cv2
import pandas as pd
import shutil
from src.decision import ACTIONS

st.set_page_config(page_title="GlassBoxDriver — Review", layout="wide")
st.title("🔍 GlassBoxDriver — Flagged Frame Review")
st.caption("Review uncertain AI decisions. Correct labels → used for retraining.")

REVIEW_DIR  = "data/flagged/review"
APPROVE_DIR = "data/flagged/approved"
LOG_DIR     = "logs"

# Load all flagged CSVs
csv_files = [f for f in os.listdir(LOG_DIR) if f.startswith("flagged_")] if os.path.exists(LOG_DIR) else []

if not csv_files:
    st.warning("No flagged frames yet. Run `python analyse.py` first.")
    st.stop()

selected_log = st.selectbox("Select session to review:", csv_files)
df = pd.read_csv(os.path.join(LOG_DIR, selected_log))

st.markdown(f"**{len(df)} flagged frames** in this session")

# Progress
approved_count = len([f for f in os.listdir(APPROVE_DIR)
                      if os.path.isdir(os.path.join(APPROVE_DIR, f))]) \
                 if os.path.exists(APPROVE_DIR) else 0

st.progress(min(approved_count / max(len(df), 1), 1.0),
            text=f"Reviewed: {approved_count}/{len(df)}")

st.divider()

for _, row in df.iterrows():
    img_path  = os.path.join(REVIEW_DIR, row['filename'])
    heat_path = img_path.replace('.jpg', '_heatmap.jpg')

    if not os.path.exists(img_path):
        continue

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        st.image(img_path, caption=f"Frame #{row['frame_id']}", use_column_width=True)

    with col2:
        if os.path.exists(heat_path):
            st.image(heat_path, caption="Grad-CAM Heatmap", use_column_width=True)

    with col3:
        st.markdown(f"**AI Said:** `{row['action']}`")
        st.markdown(f"**Confidence:** `{row['confidence']:.0%}`")
        trust_color = "🟢" if row['trust_score'] > 0.5 else "🔴"
        st.markdown(f"**Trust Score:** {trust_color} `{row['trust_score']}`")
        st.markdown(f"**Flag Reason:** `{row['flag_reason']}`")

        correct_label = st.selectbox(
            "Correct label:",
            ["⏭️ Skip"] + ACTIONS,
            key=f"label_{row['frame_id']}"
        )

        if st.button(f"✅ Approve Frame #{row['frame_id']}",
                     key=f"btn_{row['frame_id']}"):
            if correct_label == "⏭️ Skip":
                st.warning("Select a label first!")
            else:
                save_dir = os.path.join(APPROVE_DIR, correct_label)
                os.makedirs(save_dir, exist_ok=True)
                dst = os.path.join(save_dir, row['filename'])
                shutil.copy(img_path, dst)
                st.success(f"✅ Saved to {correct_label}!")

    st.divider()
