from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from services.account_service import account_service
from services.image_providers.gemini_web import (
    GeminiWebImageError,
    GeminiWebImageProvider,
    _extract_google_image_urls,
    _parse_cookie_input,
)
from services.protocol.conversation import (
    ConversationRequest,
    ImageGenerationError,
    _generate_single_gemini_image,
    _generate_single_image,
)
from utils.gemini_watermark import remove_gemini_sparkle
from utils.helper import is_gemini_image_model, GEMINI_IMAGE_MODEL


def test_gemini_model_detection():
    assert is_gemini_image_model("gemini-image-2.5-flash") is True
    assert is_gemini_image_model(GEMINI_IMAGE_MODEL) is True
    assert is_gemini_image_model("gpt-image-2") is False
    assert is_gemini_image_model("agnes-image-2.1-flash") is False


def test_parse_cookie_input_string():
    cookie_str = "__Secure-1PSID=abc12345; __Secure-1PSIDTS=ts999; SID=sid123"
    cookies, snlm0e = _parse_cookie_input(cookie_str)
    assert cookies["__Secure-1PSID"] == "abc12345"
    assert cookies["__Secure-1PSIDTS"] == "ts999"
    assert cookies["SID"] == "sid123"
    assert snlm0e == ""


def test_parse_cookie_input_json():
    json_str = '{"cookies": {"__Secure-1PSID": "test_psid"}, "snlm0e": "at_token_123"}'
    cookies, snlm0e = _parse_cookie_input(json_str)
    assert cookies["__Secure-1PSID"] == "test_psid"
    assert snlm0e == "at_token_123"


def test_extract_google_image_urls():
    payload = r'some data ["https://lh3.googleusercontent.com/ggms/AF1QipNabc12345=s1024", "https://googleusercontent.com/image_test_456"]'
    urls = _extract_google_image_urls(payload)
    assert len(urls) == 2
    assert "https://lh3.googleusercontent.com/ggms/AF1QipNabc12345=s1024" in urls
    assert "https://googleusercontent.com/image_test_456" in urls


def test_remove_gemini_sparkle_watermark():
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    cleaned = remove_gemini_sparkle(raw_bytes)
    assert isinstance(cleaned, bytes)
    assert len(cleaned) > 0

    with Image.open(io.BytesIO(cleaned)) as cleaned_img:
        assert cleaned_img.size == (200, 200)


def test_generate_single_gemini_image_success():
    fake_token = "__Secure-1PSID=test_cookie"
    test_img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    buf = io.BytesIO()
    test_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    with patch.object(account_service, "get_available_access_token", return_value=fake_token), \
         patch.object(account_service, "get_account", return_value={"access_token": fake_token, "provider": "gemini"}), \
         patch.object(account_service, "mark_image_result") as mock_mark, \
         patch.object(account_service, "release_image_slot") as mock_release, \
         patch.object(GeminiWebImageProvider, "generate", return_value={"data": [{"b64_json": b64_str}]}):

        req = ConversationRequest(model=GEMINI_IMAGE_MODEL, prompt="a cute cat")
        outputs = _generate_single_image(req, 1, 1)

        assert len(outputs) == 1
        assert outputs[0].kind == "result"
        assert len(outputs[0].data) == 1
        mock_mark.assert_called_once_with(fake_token, True, release_slot=False)
        mock_release.assert_called_once_with(fake_token)
