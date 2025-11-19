# api_app.py  (temporary simple version)

from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

class DetectRequest(BaseModel):
    imageBase64: str

class DetectResponse(BaseModel):
    labels: List[str]
    processedImageBase64: Optional[str] = None

app = FastAPI()

@app.post("/api/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    # Just return a dummy label with the length of the string
    img_len = len(req.imageBase64 or "")
    label = f"received_base64_length_{img_len}"

    # No processed image for now
    return DetectResponse(
        labels=[label],
        processedImageBase64=None
    )
