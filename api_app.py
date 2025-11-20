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
    content = await file.read()
    return {
        "filename": file.filename,
        "size": len(content),
        "content_type": file.content_type,
    }
