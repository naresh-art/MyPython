# model_loader.py
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as F

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """
    Load processor + SegFormer model (called once on startup).
    """
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return processor, model


def run_detection(image_bytes, processor, model):
    """
    Run semantic segmentation and return a list of detections:
    [
      { "label": "wall", "score": 0.42, "extra": "area_px=12345" },
      ...
    ]
    score = percentage of image area covered by that class (0..1).
    """
    # 1) Load image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # 2) Preprocess
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, num_labels, h, w]

    # 3) Upsample to original image size
    height, width = image.size[1], image.size[0]  # PIL: (W, H)
    logits_upsampled = F.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    # 4) Argmax to get class per pixel
    pred = logits_upsampled.argmax(dim=1)[0].cpu().numpy()  # (H, W)

    unique_ids, counts = np.unique(pred, return_counts=True)
    total_pixels = float(pred.size)

    # id2label mapping from config
    id2label_raw = model.config.id2label
    id2label = {int(k): v for k, v in id2label_raw.items()} if isinstance(id2label_raw, dict) else id2label_raw

    detections = []

    for class_id, cnt in zip(unique_ids, counts):
        class_id_int = int(class_id)
        label = id2label.get(class_id_int, f"class_{class_id_int}")
        area_ratio = cnt / total_pixels  # 0..1

        # Optional: skip tiny regions (noise)
        if area_ratio < 0.01:  # <1% of image
            continue

        detections.append(
            {
                "label": label,
                "score": float(round(area_ratio, 4)),  # e.g. 0.2543
                "extra": f"pixels={int(cnt)}",
            }
        )

    # Sort biggest area first
    detections.sort(key=lambda d: d["score"], reverse=True)

    return detections
