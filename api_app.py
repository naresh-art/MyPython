# api_app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from detector import run_detection

app = FastAPI(
    title="YOLOv8 Object Detection API",
    version="1.0.0"
)

# Optional: CORS for browser / Salesforce
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict to your SF domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, runs YOLOv8,
    and returns detections as JSON.
    """
    try:
        image_bytes = await file.read()
        detections = run_detection(image_bytes)

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "detections": detections
        }

    except Exception as e:
        # Any Python error shows up here
        raise HTTPException(status_code=500, detail=str(e))
