from fastapi import FastAPI, UploadFile, File, HTTPException
from hf_detector import run_detection

app = FastAPI()


@app.get("/")
def root():
    # health check
    return {"status": "ok"}


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        detections = run_detection(image_bytes)

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "detections": detections,
        }

    except Exception as e:
        # Show error in logs & send 500 to caller
        raise HTTPException(status_code=500, detail=str(e))
