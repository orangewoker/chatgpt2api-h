from __future__ import annotations

from io import BytesIO

from PIL import Image

from services.image_upscale import upscale_image_bytes


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), (32, 96, 160))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_upscale_4k_returns_requested_dimensions() -> None:
    result = upscale_image_bytes(_png_bytes(1024, 576), "3840x2160")
    with Image.open(BytesIO(result)) as image:
        assert image.size == (3840, 2160)


def test_upscale_does_not_downscale_existing_image() -> None:
    source = _png_bytes(3840, 2160)
    assert upscale_image_bytes(source, "1024x1024") == source

