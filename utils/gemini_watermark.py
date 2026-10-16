"""Gemini / Imagen generated image watermark removal utility."""

from __future__ import annotations

import io
from PIL import Image, ImageFilter


def remove_gemini_sparkle(image_bytes: bytes) -> bytes:
    """Detect and repair the Sparkle star watermark in the bottom-right of Gemini images.

    If image cannot be processed, returns original bytes as fallback.
    """
    if not image_bytes:
        return image_bytes

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            format_name = img.format or "PNG"
            has_alpha = "A" in img.getbands()
            working_img = img.convert("RGBA" if has_alpha else "RGB")
            width, height = working_img.size

            # The Google Sparkle watermark is placed in the bottom-right corner.
            # Calculate dynamic bounding box based on image resolution.
            # For 1024x1024, watermark region is approximately within a 64x64 ~ 88x88 box.
            box_w = max(48, min(120, width // 10))
            box_h = max(48, min(120, height // 10))

            margin_right = max(8, width // 64)
            margin_bottom = max(8, height // 64)

            x1 = max(0, width - box_w - margin_right)
            y1 = max(0, height - box_h - margin_bottom)
            x2 = min(width, width - margin_right // 2)
            y2 = min(height, height - margin_bottom // 2)

            box = (x1, y1, x2, y2)

            # Crop watermark region
            patch = working_img.crop(box)

            # Perform median filtering + slight blur for seamless inpainting of star lines
            smoothed_patch = patch.filter(ImageFilter.MedianFilter(size=5))
            smoothed_patch = smoothed_patch.filter(ImageFilter.GaussianBlur(radius=0.8))

            # Paste cleaned patch back into image
            working_img.paste(smoothed_patch, box)

            output = io.BytesIO()
            if format_name.upper() == "JPEG" or not has_alpha:
                working_img.convert("RGB").save(output, format="PNG", optimize=True)
            else:
                working_img.save(output, format="PNG", optimize=True)

            return output.getvalue()
    except Exception:
        # Graceful fallback to raw bytes
        return image_bytes
