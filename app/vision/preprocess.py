import io
from PIL import Image, ImageEnhance

def preprocess_to_png_bytes(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    # Basic upscale (helps small text)
    scale = 2
    img = img.resize((img.size[0]*scale, img.size[1]*scale))

    # Mild contrast boost
    img = ImageEnhance.Contrast(img).enhance(1.15)

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()
