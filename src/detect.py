from ultralytics import YOLO
import numpy as np

_model = None

DRIVING_CLASSES = {
    0: "person",
    2: "car",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign"
}

def load_model():
    global _model
    if _model is None:
        _model = YOLO("yolov10n.pt")  # Auto-downloads on first run
    return _model

def get_detections(frame):
    model = load_model()
    results = model(frame, verbose=False)

    boxes = []
    confs = []
    labels = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            if cls in DRIVING_CLASSES:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                boxes.append([x1, y1, x2, y2])
                confs.append(conf)
                labels.append(DRIVING_CLASSES[cls])

    return boxes, confs, labels
