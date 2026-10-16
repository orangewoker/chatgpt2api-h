from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock, PropertyMock, patch
import pytest
from PIL import Image

from services.account_service import account_service
from services.image_providers.gemini_web import (
    GeminiWebImageError,
    GeminiWebImageProvider,
    _determine_aspect_ratio,
    _extract_google_image_urls,
    _extract_gemini_full_size_context,
    _high_res_image_url,
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


def test_determine_aspect_ratio():
    assert _determine_aspect_ratio("1024x1024", None) == "1:1"
    assert _determine_aspect_ratio("1792x1024", None) == "16:9"
    assert _determine_aspect_ratio("1024x1792", None) == "9:16"
    assert _determine_aspect_ratio("1536x1024", None) == "3:2"
    assert _determine_aspect_ratio("1024x1536", None) == "2:3"
    assert _determine_aspect_ratio("1024x1024", "16:9") == "16:9"


def test_high_res_image_url():
    raw_url = "https://lh3.googleusercontent.com/gg-dl/AF1QipNabc12345"
    assert _high_res_image_url(raw_url) == "https://lh3.googleusercontent.com/gg-dl/AF1QipNabc12345=s0"
    preview_url = "https://lh3.googleusercontent.com/gg-dl/AF1QipNabc12345=s512"
    assert _high_res_image_url(preview_url) == "https://lh3.googleusercontent.com/gg-dl/AF1QipNabc12345=s0"


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
    assert len(urls) == 1
    assert "https://lh3.googleusercontent.com/ggms/AF1QipNabc12345=s1024" in urls
    assert "https://googleusercontent.com/image_test_456" not in urls


def test_extract_google_image_urls_decodes_escaped_slashes():
    payload = r'"https:\/\/lh3.googleusercontent.com\/gg-dl\/AF1QipNabc12345\\u003ds1024"'
    assert _extract_google_image_urls(payload) == ["https://lh3.googleusercontent.com/gg-dl/AF1QipNabc12345=s1024"]


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


@pytest.fixture
def isolated_accounts(tmp_path, monkeypatch):
    import api.accounts as accounts_api
    import services.protocol.conversation as conversation
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend

    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    monkeypatch.setattr(accounts_api, "account_service", service)
    monkeypatch.setattr(conversation, "account_service", service)
    return service


def test_google_accounts_are_isolated_from_openai(isolated_accounts):
    service = isolated_accounts
    token = "__Secure-1PSID=fixture; __Secure-1PSIDTS=old"
    service.add_accounts([token, "chatgpt-fixture"])
    service.update_account("chatgpt-fixture", {"quota": 3})
    with patch("services.openai_backend_api.OpenAIBackendAPI") as backend:
        assert service.refresh_accounts([token])["errors"] == []
        assert service.refresh_access_token(token, force=True) == token
        backend.assert_not_called()
    assert service._list_ready_candidate_tokens() == ["chatgpt-fixture"]
    assert service._list_ready_candidate_tokens(provider="gemini") == [token]
    assert service.get_text_access_token() == "chatgpt-fixture"


def test_cookie_rotation_preserves_aliases_status_and_slots(isolated_accounts):
    service = isolated_accounts
    old = "__Secure-1PSID=fixture; __Secure-1PSIDTS=old"
    service.add_accounts([old])
    assert service.get_available_access_token(provider="gemini") == old
    service.update_account(old, {"status": "异常"})
    new = service.rotate_gemini_cookies(old, {"__Secure-1PSIDTS": "rotated"})
    assert new != old
    assert service.get_account(old)["status"] == "异常"
    assert service.get_account(new)["access_token"] == new
    assert service._image_inflight[new] == 1
    service.release_image_slot(old)
    assert not service._image_inflight
    reloaded = type(service)(service.storage)
    assert reloaded.get_account(new)["access_token"] == new


def test_gemini_only_catalog_does_not_advertise_chatgpt(isolated_accounts):
    from services.model_service import ModelCatalogService
    from services.protocol import openai_v1_models

    service = isolated_accounts
    service.add_accounts(["__Secure-1PSID=fixture"])
    assert ModelCatalogService(service)._active_accounts_by_type() == {}
    with patch.object(openai_v1_models, "account_service", service), patch.object(
        openai_v1_models.model_catalog_service, "list_models", return_value={"data": []}
    ):
        ids = {item["id"] for item in openai_v1_models.list_models()["data"]}
    assert "gemini-image" in ids
    assert "gpt-image-2" not in ids
    assert not is_gemini_image_model("fake-gemini-image-unknown")


@pytest.mark.parametrize("code,status", [("rate_limit", 429), ("token_invalid", 401), ("invalid_prompt", 400)])
def test_gemini_errors_keep_status_and_release_slots(isolated_accounts, code, status):
    from services.config import config
    service = isolated_accounts
    token = "__Secure-1PSID=fixture"
    service.add_accounts([token])
    with patch.object(type(config), "auto_remove_rate_limited_accounts", new_callable=PropertyMock, return_value=True), patch.object(
        GeminiWebImageProvider, "generate", side_effect=GeminiWebImageError("fixture error", status_code=status, code=code)
    ):
        with pytest.raises(ImageGenerationError) as error:
            _generate_single_image(ConversationRequest(model="gemini-image", prompt="test"), 1, 1)
    assert error.value.status_code == status
    assert not service._image_inflight
    account = service.get_account(token)
    assert account is not None
    assert account["fail"] == 1
    if status == 429:
        assert account["status"] == "限流"
        assert not service._list_ready_candidate_tokens(provider="gemini")
        service.update_account(token, {"restore_at": "2000-01-01T00:00:00+00:00"})
        assert service._list_ready_candidate_tokens(provider="gemini") == [token]


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_import_and_generation_api(isolated_accounts, monkeypatch, stream):
    import api.accounts as accounts_api
    import api.ai as ai_api
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from services.log_service import LoggedCall

    monkeypatch.setattr(accounts_api, "require_admin", lambda _: None)
    monkeypatch.setattr(ai_api, "require_identity", lambda _: {})
    monkeypatch.setattr(ai_api, "check_request", lambda _: None)
    monkeypatch.setattr(LoggedCall, "log", lambda *args, **kwargs: None)
    app = FastAPI()
    app.include_router(accounts_api.create_router())
    app.include_router(ai_api.create_router())
    client = TestClient(app)
    token = "__Secure-1PSID=fixture; __Secure-1PSIDTS=fixture"
    with patch("services.openai_backend_api.OpenAIBackendAPI") as backend:
        response = client.post("/api/accounts", json={"accounts": [{"access_token": token, "provider": "gemini"}]})
        assert response.status_code == 200
        assert response.json()["errors"] == []
        backend.assert_not_called()
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(image, "PNG")
    encoded = base64.b64encode(image.getvalue()).decode()
    with patch.object(GeminiWebImageProvider, "generate", return_value={"data": [{"b64_json": encoded}]}) as generate:
        response = client.post("/v1/images/generations", json={"model": "gemini-image", "prompt": "red square", "stream": stream})
        assert response.status_code == 200, response.text
        assert encoded in response.text
        generate.assert_called_once()
    assert isolated_accounts.get_account(token)["success"] == 1
    assert not isolated_accounts._image_inflight
    response = client.post("/v1/images/edits", json={"model": "gemini-image", "prompt": "edit", "images": [{"image_url": f"data:image/png;base64,{encoded}"}]})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "unsupported_operation"


def test_provider_download_validates_image_and_uses_master_url():
    provider = GeminiWebImageProvider("__Secure-1PSID=fixture")
    provider.snlm0e = "fixture-at"
    session = MagicMock()
    provider._session = session
    session.post.return_value.status_code = 200
    session.post.return_value.text = 'https://lh3.googleusercontent.com/gg-dl/fixture=s512'
    raw = io.BytesIO()
    Image.new("RGB", (1024, 1024), "red").save(raw, "PNG")
    session.get.return_value.status_code = 200
    session.get.return_value.headers = {"content-length": str(len(raw.getvalue()))}
    session.get.return_value.content = raw.getvalue()
    result = provider.generate("red square")
    assert result["data"][0]["b64_json"]
    assert session.get.call_args.args[0].endswith("=s0")
    assert session.get.call_args.kwargs["allow_redirects"] is True
    session.get.return_value.close.assert_called_once()
    session.get.return_value.content = b"<html>not an image</html>"
    with pytest.raises(GeminiWebImageError, match="valid generated image"):
        provider.generate("red square")


def test_extract_gemini_full_size_context_from_old_stream_shape():
    candidate = ["rc_fixture"] + [None] * 12
    candidate[12] = [None] * 8
    candidate[12][7] = [[[None, None, None, [None, None, None, "https://lh3.googleusercontent.com/gg-dl/preview=s512", None, "download-token"]]]]
    inner = [None, ["conversation-id", "response-id"], None, None, [candidate]]
    payload = [["wrb.fr", "StreamGenerate", json.dumps(inner)]]
    context = _extract_gemini_full_size_context(")]}\'\n" + json.dumps(payload))
    assert context == (
        "https://lh3.googleusercontent.com/gg-dl/preview=s512",
        "download-token",
        '["conversation-id", "response-id", "rc_fixture"]',
    )


def test_fetch_full_size_url_parses_c8o8fe_frame():
    provider = GeminiWebImageProvider("__Secure-1PSID=fixture")
    provider.snlm0e = "fixture-at"
    session = MagicMock()
    provider._session = session
    session.post.return_value.status_code = 200
    session.post.return_value.text = ")]}'\\n[[[\"wrb.fr\",\"c8o8Fe\",\"[\\\"https://lh3.googleusercontent.com/full=s2048\\\"]\"]]]"
    assert provider._fetch_full_size_url("download-token", "red square", '["cid", "rid", "rcid"]') == "https://lh3.googleusercontent.com/full=s2048"
