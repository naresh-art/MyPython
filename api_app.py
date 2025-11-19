# api_app.py

import base64
from io import BytesIO
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# ---------- Your existing model config ----------
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load model once ----------
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE).eval()


# ---------- Pydantic models ----------
class DetectRequest(BaseModel):
    imageBase64: str


class DetectResponse(BaseModel):
    labels: List[str]
    processedImageBase64: Optional[str] = None


# ---------- Helper: segmentation (adapted from your code) ----------
def segment_image_pil(img_pil: Image.Image):
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    with torch.no_grad():
        inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        up = torch.nn.functional.interpolate(
            outputs.logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        pred = up.argmax(dim=1)[0].cpu().numpy()

    return img_bgr, pred, model.config.id2label


# ---------- FastAPI app ----------
app = FastAPI()


@app.post("/api/detect", response_model=DetectResponse)
def detect_objects(req: DetectRequest):
    # 1) Decode base64
    try:
        img_bytes = base64.b64decode(req.imageBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # 2) Open image
    try:
        img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to open image")

    # 3) Run SegFormer segmentation
    img_bgr, pred, id2label = segment_image_pil(img_pil)

    # 4) Collect detected labels (similar to your Streamlit logic)
    detected_ids = np.unique(pred)
    h, w = pred.shape
    min_pixels = max(500, int(0.001 * h * w))

    labels_set = set()
    for cid in detected_ids:
        cid_int = int(cid)
        label = id2label.get(cid_int)
        if not label:
            continue
        if label.lower() == "background":
            continue

        mask = (pred == cid_int)
        if mask.sum() < min_pixels:
            continue

        labels_set.add(label)

    labels = sorted(labels_set)

    # 5) For now, just return original image as processedImageBase64
    processed_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return DetectResponse(
        labels=labels,
        processedImageBase64=processed_b64
    )
