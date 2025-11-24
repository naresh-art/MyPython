# api_app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from detector import run_detection

app = FastAPI(
    title="YOLOv8 Object Detection API",
    version="1.0.0"
)

# CORS so that browser / Salesforce (if ever used directly) can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict to your Salesforce domain
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
        if file.content_type is None or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {file.content_type}"
            )

        image_bytes = await file.read()
        detections = run_detection(image_bytes)

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "detections": detections
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
