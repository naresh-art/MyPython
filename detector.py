# detector.py
from ultralytics import YOLO
from typing import List, Dict
from io import BytesIO
from PIL import Image
import numpy as np

# Load YOLOv8 nano model (smallest one).
# On first run it will download yolov8n.pt (~6 MB).
model = YOLO("yolov8n.pt")  # you can change to yolov8s.pt, etc.


def run_detection(image_bytes: bytes, conf_threshold: float = 0.3) -> List[Dict]:
    """
    Run YOLOv8 on the given image bytes.
    Returns list of {label, score, x, y, width, height} in pixel coords.
    """

    # Load image with PIL
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w0, h0 = pil_img.size

    # Run model (Ultralytics handles preprocessing)
    results = model(pil_img, imgsz=640, conf=conf_threshold)
    if not results:
        return []

    r = results[0]
    detections = []

    # r.boxes contains all predicted boxes
    # Each box has: xyxy, conf, cls
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = r.names[cls_id]  # class name
        score = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        detections.append({
            "label": label,
            "score": score,
            "x": float(x1),
            "y": float(y1),
            "width": float(x2 - x1),
            "height": float(y2 - y1),
        })

    return detections
