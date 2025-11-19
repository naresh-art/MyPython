# api_app.py  -- FastAPI backend with SegFormer detection

from typing import List, Optional
from io import BytesIO
import base64

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# ----------------- Config -----------------
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once at startup (important for speed)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE).eval()


class DetectRequest(BaseModel):
    imageBase64: str


class DetectResponse(BaseModel):
    labels: List[str]
    # you can later return recolored image here
    processedImageBase64: Optional[str] = None


app = FastAPI()


def segment_image_pil(img_pil: Image.Image):
    """Run SegFormer and return prediction mask + label mapping."""
    img_np = np.array(img_pil)  # RGB
    h, w = img_np.shape[:2]

    with torch.no_grad():
        inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)

        # Upsample logits to original image size
        up = torch.nn.functional.interpolate(
            outputs.logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        pred = up.argmax(dim=1)[0].cpu().numpy()  # (H, W)

    return pred, model.config.id2label


@app.post("/api/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
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

    # 3) Run segmentation
    pred, id2label = segment_image_pil(img_pil)

    # 4) Collect detected labels (similar to your Streamlit logic)
    detected_ids = np.unique(pred)
    h, w = pred.shape
    min_pixels = max(500, int(0.001 * h * w))  # ignore tiny areas

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

    # For now just return ORIGINAL image again (no recolor yet)
    processed_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return DetectResponse(
        labels=labels,
        processedImageBase64=processed_b64
    )
