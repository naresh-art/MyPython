import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from io import BytesIO
import os

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Map object keywords to folders with alternate designs
# Update these paths to match your actual folder structure
ALT_IMAGE_MAP = {
    "tile": "assets/tiles",
    "floor": "assets/floor",
    "wall": "assets/wall",
    # add more mappings if you want, e.g.
    # "sofa": "assets/sofa",
    # "curtain": "assets/curtains",
}


@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return processor, model


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def soft_mask(mask_u8: np.ndarray, feather_px: int = 9) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    if feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather_px)
    return (m.astype(np.float32) / 255.0).clip(0, 1)


def recolor_mask(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    target_hex: str,
    strength: float = 0.9,
    feather_px: int = 9
) -> np.ndarray:
    alpha = soft_mask(mask_u8, feather_px=feather_px) * strength
    if alpha.max() <= 0:
        return img_bgr.copy()

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    tgt_bgr = np.uint8([[hex_to_bgr(target_hex)]])
    _, at, bt = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    lab[:, :, 1] = lab[:, :, 1] * (1.0 - alpha) + at * alpha
    lab[:, :, 2] = lab[:, :, 2] * (1.0 - alpha) + bt * alpha

    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def segment_image(img_pil, processor, model):
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    with torch.no_grad():
        inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        up = torch.nn.functional.interpolate(
            outputs.logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        pred = up.argmax(dim=1)[0].cpu().numpy()
    return img_bgr, pred, model.config.id2label


def get_alt_dir_for_object(obj_label: str):
    """Return folder path for alternate designs based on object label."""
    lower = obj_label.lower()
    for key, folder in ALT_IMAGE_MAP.items():
        if key in lower:
            return folder
    return None


def load_alt_images(folder: str):
    """Load all images inside the given folder as (name, PIL.Image)."""
    if not folder or not os.path.isdir(folder):
        return []

    images = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")):
            path = os.path.join(folder, fname)
            try:
                pil_img = Image.open(path).convert("RGB")
                images.append((fname, pil_img))
            except Exception:
                continue
    return images


def dominant_hex_from_pil(pil_img: Image.Image) -> str:
    """Compute a simple average color and return as hex."""
    arr = np.array(pil_img)
    if arr.ndim == 3:
        # (H, W, 3) -> (N, 3)
        mean = arr.reshape(-1, 3).mean(axis=0)
        r, g, b = mean.astype(int)
        return f"#{r:02X}{g:02X}{b:02X}"
    return "#FFFFFF"


# ================== STREAMLIT UI ==================
st.title("🎨 Recolor Any Object (150 ADE20K classes)")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"]
)

if uploaded_file:
    # A small "id" for this session of this image
    file_tag = f"{uploaded_file.name}_{uploaded_file.size}"

    img_pil = Image.open(uploaded_file).convert("RGB")

    strength = st.slider(
        "Recolor Strength",
        0.0, 1.0, 0.9, 0.05,
        key=f"strength_{file_tag}"
    )

    processor, model = load_model()

    with st.spinner("Segmenting image..."):
        img_bgr, pred, id2label = segment_image(img_pil, processor, model)

    # ================== DETECT OBJECTS IN IMAGE ==================
    detected_ids = np.unique(pred)  # unique class IDs in the mask

    # Build label -> list of class IDs (filter out background & tiny areas)
    class_to_ids = {}
    h, w = pred.shape
    min_pixels = max(500, int(0.001 * h * w))  # ignore very tiny regions

    for cid in detected_ids:
        cid_int = int(cid)
        label = id2label.get(cid_int)
        if not label:
            continue
        if label.lower() == "background":
            continue

        mask = (pred == cid_int)
        if mask.sum() < min_pixels:
            continue

        class_to_ids.setdefault(label, []).append(cid_int)

    detected_objects = sorted(class_to_ids.keys())

    st.subheader("✅ Objects detected in this image:")
    if detected_objects:
        st.write(", ".join(detected_objects))
    else:
        st.write("No significant objects detected (except background).")

    # ================== RECOLOR UI ==================
    st.subheader("🎨 Choose objects to recolor")

    if detected_objects:
        selected_objects = st.multiselect(
            "Select objects",
            detected_objects,
            key=f"objects_{file_tag}"
        )

        recolored = img_bgr.copy()

        # Apply recolor per selected object
        for obj in selected_objects:
            st.markdown(f"---\n#### 🎨 Settings for **{obj}**")

            # ---------- Alternate designs for this object ----------
            alt_dir = get_alt_dir_for_object(obj)
            alt_images = load_alt_images(alt_dir)
            chosen_tile_color = None

            if alt_images:
                st.caption(f"Alternate {obj} designs from `{alt_dir}`")
                # Radio to choose which alt design to use
                idx = st.radio(
                    f"Choose a design for {obj}",
                    options=list(range(len(alt_images))),
                    format_func=lambda i: alt_images[i][0],
                    horizontal=True,
                    key=f"alt_{file_tag}_{obj}"
                )

                # Show small thumbnails
                cols = st.columns(min(4, len(alt_images)))
                for i, (name, pil_img) in enumerate(alt_images):
                    with cols[i % len(cols)]:
                        st.image(pil_img, caption=name, use_column_width=True)

                # Use dominant color from selected design
                _, chosen_pil = alt_images[idx]
                chosen_tile_color = dominant_hex_from_pil(chosen_pil)
                st.write(f"Suggested color from selected design: **{chosen_tile_color}**")
            else:
                st.caption(f"No alternate designs configured for **{obj}**.")

            # ---------- Color picker ----------
            default_color = chosen_tile_color or "#FF0000"
            color = st.color_picker(
                f"Pick color for {obj}",
                default_color,
                key=f"color_{file_tag}_{obj}"
            )

            obj_ids = class_to_ids.get(obj, [])
            if not obj_ids:
                continue

            mask = np.isin(pred, obj_ids).astype(np.uint8)
            recolored = recolor_mask(recolored, mask, color, strength=strength)

        # ================== SHOW RESULTS ==================
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img_pil, use_column_width=True)

        with col2:
            st.subheader("Recolored")
            recolored_rgb = cv2.cvtColor(recolored, cv2.COLOR_BGR2RGB)
            st.image(recolored_rgb, use_column_width=True)

            # ---------- DOWNLOAD BUTTON ----------
            buf = BytesIO()
            Image.fromarray(recolored_rgb).save(buf, format="PNG")
            buf.seek(0)

            st.download_button(
                label="💾 Download recolored image",
                data=buf,
                file_name="recolored_image.png",
                mime="image/png"
            )
    else:
        st.info("No objects available to recolor for this image.")
        