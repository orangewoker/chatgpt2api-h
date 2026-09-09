"""Server-side output upscaling for image generation responses.

The ChatGPT image endpoint can return a smaller bitmap even when a larger
canvas size was requested.  This module provides a dependency-light fallback
using Pillow's high-quality Lanczos resampler.  It is not an AI super-
resolution model, but it makes the requested 2K/4K output dimensions
deterministic without requiring an external GPU service.
"""

from __future__ import annotations

import re
from io import BytesIO

from PIL import Image, ImageOps

MAX_OUTPUT_DIMENSION = 4096
MAX_OUTPUT_PIXELS = MAX_OUTPUT_DIMENSION * MAX_OUTPUT_DIMENSION
_SIZE_RE = re.compile(r"^(\d{2,5})\s*[x×]\s*(\d{2,5})$", re.IGNORECASE)


def _target_size(size: object) -> tuple[int, int] | None:
    value = str(size or "").strip().lower()
    if value in {"4k", "uhd"}:
        return 3840, 2160
    match = _SIZE_RE.fullmatch(value)
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    scale = min(1.0, MAX_OUTPUT_DIMENSION / max(width, height))
    width, height = round(width * scale), round(height * scale)
    if width * height > MAX_OUTPUT_PIXELS:
        scale = (MAX_OUTPUT_PIXELS / (width * height)) ** 0.5
        width, height = round(width * scale), round(height * scale)
    return max(1, width), max(1, height)


def upscale_image_bytes(data: bytes, size: object) -> bytes:
    """Resize an output image to the requested dimensions when it is smaller.

    Images that already meet or exceed both requested dimensions are returned
    unchanged.  Resized results are encoded as PNG so callers can safely
    return them as ``b64_json`` regardless of the upstream image format.
    """

    target = _target_size(size)
    if not target or not data:
        return data
    try:
        with Image.open(BytesIO(data)) as source:
            source_size = source.size
            if source_size[0] >= target[0] and source_size[1] >= target[1]:
                return data
            image = ImageOps.exif_transpose(source)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            resized = image.resize(target, Image.Resampling.LANCZOS)
            output = BytesIO()
            resized.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except Exception:
        # An unexpected/non-raster payload should not make an otherwise valid
        # generation request fail; return the upstream bytes unchanged.
        return data

