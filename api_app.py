# api_app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64

from segmentor import segmentor, get_label_name

app = FastAPI(
    title="Roomvo-style Interior API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    """
    Accepts image, runs segmentation, returns:
    - session_id
    - list of classes present (id + label)
    """
    try:
        image_bytes = await file.read()
        session_id, pil_img, seg_map, unique_classes = segmentor.segment(image_bytes)

        regions = []
        for cid in unique_classes:
            label_name = get_label_name(int(cid))
            regions.append({
                "class_id": int(cid),
                "label": label_name
            })

        return {
            "session_id": session_id,
            "width": pil_img.size[0],
            "height": pil_img.size[1],
            "regions": regions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RecolorRequest(BaseModel):
    session_id: str
    class_id: int
    color_hex: str  # "#RRGGBB"


@app.post("/recolor")
async def recolor_region(req: RecolorRequest):
    """
    Recolors one class_id in a session image with color_hex.
    Returns base64 PNG of updated image.
    """
    try:
        png_bytes = segmentor.recolor(req.session_id, req.class_id, req.color_hex)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {
            "image_base64": b64,
            "session_id": req.session_id,
            "class_id": req.class_id,
            "color_hex": req.color_hex
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
