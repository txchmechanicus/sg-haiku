from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

MAX_IMAGE_DIMENSION = 2000
MAX_IMAGE_BYTES = int(4.5 * 1024 * 1024)
JPEG_QUALITY_STEPS = (80, 60, 40, 20)
_SNIFF_BYTES = 16


def sniff_image_mime_type(path: Path) -> str | None:
    with path.open("rb") as file:
        header = file.read(_SNIFF_BYTES)
    return _detect_image_mime_type(header)


def _detect_image_mime_type(header: bytes) -> str | None:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header[:2] == b"BM":
        return "image/bmp"
    return None


def read_image_content(path: Path, mime_type: str) -> tuple[str, str]:
    """Returns (base64_data, final_mime_type), resizing/recompressing to fit size limits."""
    raw = path.read_bytes()

    if mime_type == "image/gif":
        # Animated GIF resizing is out of scope; pass through as-is.
        return base64.b64encode(raw).decode("ascii"), mime_type

    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception:
        return base64.b64encode(raw).decode("ascii"), mime_type

    fits_dimensions = image.width <= MAX_IMAGE_DIMENSION and image.height <= MAX_IMAGE_DIMENSION
    if fits_dimensions and len(raw) <= MAX_IMAGE_BYTES:
        return base64.b64encode(raw).decode("ascii"), mime_type

    if not fits_dimensions:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    encoded = raw
    for quality in JPEG_QUALITY_STEPS:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        encoded = buffer.getvalue()
        if len(encoded) <= MAX_IMAGE_BYTES:
            break

    return base64.b64encode(encoded).decode("ascii"), "image/jpeg"
