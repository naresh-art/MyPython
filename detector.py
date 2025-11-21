# detector.py
from ultralytics import YOLO
from typing import List, Dict
from io import BytesIO
from PIL import Image

# Slightly larger model than nano for better accuracy
model = YOLO("yolov8s.pt")  # COCO-pretrained: bed, chair, couch, person, etc.


def run_detection(image_bytes: bytes, conf_threshold: float = 0.15) -> List[Dict]:
    """
    Run YOLOv8 and return ALL detections (no dedupe).
    We:
      - run with lower conf_threshold to catch more objects
      - clip bounding boxes so they stay inside the image.
    """
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = pil_img.size

    results = model(
        pil_img,
        imgsz=800,          # larger input → more detail
        conf=conf_threshold,
        max_det=300         # up to 300 boxes
    )
    if not results:
        return []

    r = results[0]
    detections: List[Dict] = []

    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = r.names[cls_id]
        score = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Clip to image bounds so nothing goes "outside"
        x1 = max(0.0, min(float(x1), img_w))
        x2 = max(0.0, min(float(x2), img_w))
        y1 = max(0.0, min(float(y1), img_h))
        y2 = max(0.0, min(float(y2), img_h))

        if x2 <= x1 or y2 <= y1:
            # invalid box after clipping
            continue

        detections.append({
            "label": label,
            "score": score,
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        })

    return detections
