# 🚗 VideoDrive-XAI
## Explainable End-to-End Autonomous Driving

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-blue.svg)](https://streamlit.io)

**Video → Driving Decisions → XAI Trust Scores**

### Quick Demo
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Input**: Dashcam video  
**Output**: Explained decisions with heatmaps + risk alerts

### Architecture
```
[Video] → YOLO Detection → DrivingNet Decision → GradCAM++ XAI → Trust Score
```

**Dataset**: nuScenes Interact (40min annotated driving videos)

### Key Features
- Temporal Grad-CAM across frames
- Action faithfulness ablation tests
- Live Streamlit dashboard
- Risky decision alerts (<0.7 trust)

**🚀 Live Demo**: [streamlit.app/video-drive-xai](#)
