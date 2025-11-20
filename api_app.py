from fastapi import FastAPI, UploadFile, File
from model_loader import load_model, run_detection

app = FastAPI()

# Load model once at startup
processor, model = load_model()


@app.get("/")
def root():
    # Health endpoint
    return {"status": "ok"}


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data with a 'file' field (image),
    runs segmentation, and returns detected classes.
    """
    image_bytes = await file.read()
    detections = run_detection(image_bytes, processor, model)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "detections": detections,
    }
