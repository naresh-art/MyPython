from fastapi import FastAPI, UploadFile, File
from typing import List
from model_loader import load_model, run_detection

app = FastAPI()

processor, model = load_model()

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    # call your segmentation / detection logic
    detections = run_detection(image_bytes, processor, model)
    # detections should be a LIST of dicts:
    # [ { "label": "wall", "score": 0.95, "mask": "...optional..." }, ... ]
    return {"detections": detections}
