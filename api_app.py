from fastapi import FastAPI, UploadFile, File

app = FastAPI()


@app.get("/")
def root():
    # Simple health endpoint
    return {"status": "ok"}


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read file bytes (just to prove upload works)
    content = await file.read()

    # Return dummy detections for now
    return {
        "filename": file.filename,
        "size": len(content),
        "content_type": file.content_type,
        "detections": [
            {"label": "wall", "score": 0.95},
            {"label": "floor", "score": 0.90},
            {"label": "window", "score": 0.85},
        ],
    }
