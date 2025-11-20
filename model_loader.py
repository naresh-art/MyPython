# model_loader.py
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import cv2
from io import BytesIO

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return processor, model


def run_detection(image_bytes, processor, model):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu().numpy()[0]

    seg_map = logits.argmax(axis=0)

    detected_classes = np.unique(seg_map).tolist()

    # map IDs to label names (simplified)
    id_to_label = processor.meta["id2label"]

    result = [
        {
            "label": id_to_label[class_id],
            "score": 1.0  # SegFormer does not give score per mask, so static
        }
        for class_id in detected_classes
    ]

    return result
