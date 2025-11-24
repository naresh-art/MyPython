# segmentor.py
import numpy as np
from PIL import Image
from io import BytesIO
import uuid
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as F

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# In-memory store: session_id -> { "image": PIL.Image, "seg_map": np.ndarray }
SESSION_STORE = {}

# Very small label map for ADE20K (approx – adjust as needed)
# Unknown IDs will be shown as "class_<id>"
ADE20K_LABELS = {
    0: "background",
    1: "wall",
    2: "building",
    3: "sky",
    4: "floor",
    5: "tree",
    6: "ceiling",
    7: "road",
    8: "bed",
    9: "window",
    10: "door",
    11: "table",
    12: "curtain",
    13: "chair",
    14: "sofa",
    15: "shelf",
    16: "cabinet",
    17: "painting",
    18: "desk",
    # ... you can extend this map later if you want more names
}


def get_label_name(class_id: int) -> str:
    return ADE20K_LABELS.get(class_id, f"class_{class_id}")


class Segmentor:
    def __init__(self):
        print("Loading SegFormer model...")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        print("SegFormer loaded.")

    def segment(self, image_bytes: bytes):
        """
        Returns:
        - session_id
        - seg_map (H,W) numpy array of class ids
        - unique_classes: sorted list of class ids present
        """
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = pil_img.size

        inputs = self.processor(images=pil_img, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch, num_classes, h, w]

        # Resize logits to original image size
        logits = F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )

        seg = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)  # (H, W)

        # Create session
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = {
            "image": pil_img,
            "seg_map": seg
        }

        unique_classes = sorted(list(np.unique(seg)))
        return session_id, pil_img, seg, unique_classes

    def recolor(self, session_id: str, class_id: int, color_hex: str, alpha: float = 0.6) -> bytes:
        """
        Recolor all pixels in seg_map equal to class_id with color_hex.
        Returns PNG bytes of the updated image.
        """
        if session_id not in SESSION_STORE:
            raise ValueError("Session not found.")

        entry = SESSION_STORE[session_id]
        pil_img: Image.Image = entry["image"]
        seg_map: np.ndarray = entry["seg_map"]

        img_np = np.array(pil_img).astype(np.float32)  # H,W,3

        # Hex to RGB
        hex_clean = color_hex.lstrip("#")
        r = int(hex_clean[0:2], 16)
        g = int(hex_clean[2:4], 16)
        b = int(hex_clean[4:6], 16)
        color_vec = np.array([r, g, b], dtype=np.float32)

        mask = (seg_map == class_id)  # H,W boolean
        if not np.any(mask):
            # nothing to recolor – just return original image
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()

        mask3 = np.stack([mask] * 3, axis=-1)  # H,W,3

        # Blend color on masked region
        img_np[mask3] = (1 - alpha) * img_np[mask3] + alpha * color_vec

        out_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

        # Update stored image so next recolor starts from updated one
        entry["image"] = out_img

        buf = BytesIO()
        out_img.save(buf, format="PNG")
        return buf.getvalue()


# Global segmentor instance
segmentor = Segmentor()
