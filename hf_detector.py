# hf_detector.py
import os
import json
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
HF_TOKEN = os.environ.get("HF_TOKEN")


def run_detection(image_bytes: bytes):
    """
    Send image_bytes to Hugging Face DETR API and return a list of detections:
    [
      {"label": "chair", "score": 0.98, "extra": "{\"xmin\":..., \"ymin\":..., ...}"},
      ...
    ]
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    resp = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=60)
    resp.raise_for_status()

    preds = resp.json()  # list of dicts

    detections = []
    for p in preds:
        label = p.get("label")
        score = float(p.get("score", 0))
        box = p.get("box")  # {xmin, ymin, xmax, ymax}

        detections.append(
            {
                "label": label,
                "score": score,
                "extra": json.dumps(box) if box is not None else None,
            }
        )

    # Sort by score descending
    detections.sort(key=lambda d: d["score"], reverse=True)

    return detections
