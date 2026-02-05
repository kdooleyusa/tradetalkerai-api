from __future__ import annotations

import io
from PIL import Image, ImageEnhance

def preprocess_to_png_bytes(raw: bytes) -> bytes:
    """
    Simple pre-processing for chart screenshots:
    - convert to RGB
    - upscale (helps small labels)
    - mild contrast boost
    """
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    scale = 2
    img = img.resize((img.size[0] * scale, img.size[1] * scale))

    img = ImageEnhance.Contrast(img).enhance(1.15)

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()
