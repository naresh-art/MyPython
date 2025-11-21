# detector.py
from ultralytics import YOLO
from typing import List, Dict
from io import BytesIO
from PIL import Image

# Load YOLOv8 nano model (smallest). On first run it downloads yolov8n.pt (~6MB).
model = YOLO("yolov8s.pt")  # COCO-pretrained: person, chair, couch, tv, etc.


def run_detection(image_bytes: bytes, conf_threshold: float = 0.3) -> List[Dict]:
    """
    Run YOLOv8 on the given image bytes.
    Returns list of {label, score, x, y, width, height} in pixel coords.
    """

    # Load image with PIL
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Run model (Ultralytics handles preprocessing & resizing)
    # imgsz=640 is typical; adjust if needed
    results = model(pil_img, imgsz=640, conf=conf_threshold)
    if not results:
        return []

    r = results[0]
    detections: List[Dict] = []

    # r.boxes contains all predicted boxes
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = r.names[cls_id]        # class name
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
