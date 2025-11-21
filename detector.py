# detector.py
from ultralytics import YOLO
from typing import List, Dict
from io import BytesIO
from PIL import Image

# Bigger model: better accuracy than yolov8n
model = YOLO("yolov8s.pt")


def run_detection(image_bytes: bytes, conf_threshold: float = 0.2) -> List[Dict]:
    """
    Run YOLOv8 on the given image bytes.
    Returns list of {label, score, x, y, width, height} in pixel coords.
    - Clips boxes to stay inside the image
    - Removes duplicate labels (keeps highest-confidence box per label)
    """

    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = pil_img.size

    # Slightly higher resolution + lower conf → more detections
    results = model(
        pil_img,
        imgsz=800,          # was 640
        conf=conf_threshold,
        max_det=300         # allow up to 300 boxes if needed
    )
    if not results:
        return []

    r = results[0]
    all_detections: List[Dict] = []

    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = r.names[cls_id]
        score = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # 🔒 1) Clip coordinates so they stay inside the image
        x1 = max(0.0, min(float(x1), img_w))
        x2 = max(0.0, min(float(x2), img_w))
        y1 = max(0.0, min(float(y1), img_h))
        y2 = max(0.0, min(float(y2), img_h))

        if x2 <= x1 or y2 <= y1:
            continue  # invalid box after clipping

        all_detections.append({
            "label": label,
            "score": score,
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        })

    # 🔁 2) Remove duplicate labels (keep the highest-confidence one)
    #     If you want every instance (multiple chairs), remove this block.
    best_by_label: Dict[str, Dict] = {}
    for det in all_detections:
        lbl = det["label"]
        if lbl not in best_by_label or det["score"] > best_by_label[lbl]["score"]:
            best_by_label[lbl] = det

    # Final list: unique labels only
    return list(best_by_label.values())
