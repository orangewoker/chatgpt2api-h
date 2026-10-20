import base64
import json
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.ai as ai
import services.gemini_chat as gemini
from services.protocol.conversation import ImageGenerationError, text_backend
from services.protocol import openai_v1_models


def catalog_body():
    models = []
    for name, number in [("3.8 Flash", 1), ("4 Argon", 8), ("3.5 Flash-Lite", 6), ("3.1 Pro", 3), ("4 Flash", 9)]:
        item = [None] * 20
        item[0], item[11], item[17], item[19] = f"model-{number}", name, number, name
        models.append(item)
    body = [None] * 18
    body[15:18] = [models, [8], []]
    return body


def frame(text):
    body = [None] * 5
    body[4] = [["candidate", [text]]]
    return json.dumps([["wrb.fr", None, json.dumps(body)]])


@pytest.fixture
def provider():
    p = gemini.GeminiChatProvider("__Secure-1PSID=test")
    p._session = Mock()
    p._ensure_snlm0e = Mock(return_value="at")
    p.selected_catalog = gemini.model_catalog(catalog_body())
    p._upload_images = Mock(return_value=[[["upload-id"], "image.png"]])
    p._session.post.return_value.status_code = 200
    p._session.post.return_value.iter_lines.return_value = iter(["that is xssi", "123", frame("O"), frame("OK")])
    return p


def test_catalog_exact_versions():
    assert set(gemini.model_catalog(catalog_body())) == gemini.CHAT_MODELS
    assert len(gemini.CHAT_MODELS) == 6
    assert gemini.model_catalog([]) == {}


@pytest.mark.parametrize("model", sorted(gemini.CHAT_MODELS))
def test_model_headers_and_tiers(provider, model):
    assert list(provider.stream_chat(model, [{"role": "user", "content": "hello"}])) == ["O", "K"]
    kwargs = provider.session.post.call_args.kwargs
    payload = json.loads(json.loads(kwargs["data"]["f.req"])[1])
    header = json.loads(kwargs["headers"][gemini.MODEL_HEADER])
    effort = gemini.EFFORTS[model.rsplit("-", 1)[1]]
    number = 1 if "flash" in model else 8
    assert payload[79:81] == [number, effort]
    assert header[-3:-1] == [number, effort]
    assert header[4] == f"model-{number}"
    assert kwargs["stream"] is True
    provider._ensure_snlm0e.assert_called_once()
    provider.session.post.return_value.close.assert_called_once()


def test_history_and_images(provider):
    messages = [{"role": "system", "content": "Be brief"}, {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                 {"type": "image", "data": b"png", "mime": "image/png"}]}]
    list(provider.stream_chat("gemini-3.8-flash-low", messages))
    payload = json.loads(json.loads(provider.session.post.call_args.kwargs["data"]["f.req"])[1])
    assert "Be brief" in payload[0][0] and "Previous answer" in payload[0][0]
    assert '[Image 1]' in payload[0][0]
    assert payload[0][3] == [[["upload-id"], "image.png"]]
    provider._upload_images.assert_called_once_with([(b"png", "image_1.png", "image/png")])


def test_single_turn_is_verbatim(provider):
    list(provider.stream_chat("gemini-3.8-flash-low", [{"role": "user", "content": "你好"}]))
    payload = json.loads(json.loads(provider.session.post.call_args.kwargs["data"]["f.req"])[1])
    assert payload[0][0] == "你好"


def test_optional_tools_do_not_select_another_provider():
    body = {"model": "gemini-4.0-argon-low", "tools": [{"type": "web_search"}], "tool_choice": "auto"}
    clean = gemini.validate_chat_body(body)
    assert "tools" not in clean and "tools" in body


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "test"}}])
def test_required_tools_rejected(choice):
    with pytest.raises(ImageGenerationError, match="required"):
        gemini.validate_chat_body({"model": "gemini-4.0-argon-low", "tools": [{"type": "function"}], "tool_choice": choice})


def test_native_history_only_sends_latest_turn(provider):
    provider.conversation_metadata = ["c_123", "r_123", "rc_123", None, None, None, None, None, None, ""]
    list(provider.stream_chat("gemini-3.8-flash-low", [
        {"role": "user", "content": "Remember 42"}, {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "What number?"}]))
    payload = json.loads(json.loads(provider.session.post.call_args.kwargs["data"]["f.req"])[1])
    assert payload[0][0] == "What number?"
    assert payload[2][:3] == ["c_123", "r_123", "rc_123"]


def test_conversation_scope_and_single_use():
    account = {"access_token": "test-one"}
    messages = [{"role": "user", "content": "Hello"}]
    metadata = ["c_123", "r_123", "rc_123"]
    gemini.remember_conversation(account, "gemini-3.8-flash-low", messages, "Hi", metadata)
    followup = messages + [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Continue"}]
    assert gemini.take_conversation({"access_token": "test-two"}, "gemini-3.8-flash-low", followup) is None
    assert gemini.take_conversation(account, "gemini-4.0-argon-low", followup) is None
    assert gemini.take_conversation(account, "gemini-3.8-flash-low", followup) == metadata
    assert gemini.take_conversation(account, "gemini-3.8-flash-low", followup) is None


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_upstream_status_and_close(provider, status):
    provider.session.post.return_value.status_code = status
    with pytest.raises(gemini.GeminiWebImageError) as exc:
        list(provider.stream_chat("gemini-3.8-flash-low", []))
    assert exc.value.status_code == (401 if status == 403 else status)
    provider.session.post.return_value.close.assert_called_once()


def test_rpc_quota_error(provider):
    provider.session.post.return_value.iter_lines.return_value = iter([json.dumps([
        ["wrb.fr", None, None, None, None, [None, None, [[None, [1037]]]]]])])
    with pytest.raises(gemini.GeminiWebImageError) as exc:
        list(provider.stream_chat("gemini-3.8-flash-low", []))
    assert exc.value.status_code == 429


def test_empty_and_revised_stream(provider):
    for texts in [[], [frame("one"), frame("two")]]:
        provider.session.post.return_value.iter_lines.return_value = iter(texts)
        with pytest.raises(gemini.GeminiWebImageError):
            list(provider.stream_chat("gemini-3.8-flash-low", []))


@pytest.mark.parametrize("body", [
    {"model": "gemini-4.0-argon-auto"},
    {"model": "gemini-4.0-argon-low", "reasoning_effort": "high"},
    {"model": "gemini-4.0-argon-low", "reasoning": {"effort": "unknown"}},
    {"model": "gemini-4.0-argon-low", "tools": [{"type": "image_generation"}]},
    {"model": "gemini-4.0-argon-low", "modalities": ["image"]},
    {"model": "gemini-4.0-argon-low", "previous_response_id": "resp-old"},
])
def test_invalid_requests(body):
    with pytest.raises(ImageGenerationError) as exc:
        gemini.validate_chat_body(body)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_http_text_vision(endpoint, stream):
    app = FastAPI()
    app.include_router(ai.create_router())
    image = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    body = {"model": "gemini-3.8-flash-low", "stream": stream}
    if endpoint.endswith("completions"):
        body["messages"] = [{"role": "user", "content": [
            {"type": "text", "text": "What color?"}, {"type": "image_url", "image_url": {"url": image}}]}]
    else:
        body["input"] = [{"role": "user", "content": [
            {"type": "input_text", "text": "What color?"}, {"type": "input_image", "image_url": image}]}]
    calls = []
    def fake(model, messages, effort):
        calls.append(messages)
        yield "Red"
        yield " square"
    with patch.object(gemini, "stream_account_chat", fake), \
         patch.object(ai.LoggedCall, "log"), \
         patch("services.protocol.conversation.account_service.get_text_access_token", side_effect=AssertionError("GPT routing")):
        response = TestClient(app).post(endpoint, json=body, headers={"Authorization": "Bearer chatgpt2api"})
    assert response.status_code == 200, response.text
    assert "square" in response.text
    assert any(p.get("type") == "image" for m in calls[0] for p in (m["content"] if isinstance(m["content"], list) else []))
    if stream:
        assert "text/event-stream" in response.headers["content-type"]


def test_provider_isolation_and_no_image_slot():
    with patch("services.protocol.conversation.account_service.get_text_access_token", side_effect=AssertionError):
        assert text_backend("gemini-3.8-flash-low") is None
    with patch.object(gemini, "available_accounts", return_value=[]):
        with pytest.raises(ImageGenerationError, match="No active Gemini"):
            list(gemini.stream_account_chat("gemini-3.8-flash-low", []))


def test_catalog_published():
    with patch.object(openai_v1_models.model_catalog_service, "list_models", return_value={"data": []}), \
         patch.object(openai_v1_models.account_service, "list_accounts", return_value=[{"provider": "gemini"}]), \
         patch.object(gemini, "available_chat_models", return_value=gemini.CHAT_MODELS):
        ids = {m["id"] for m in openai_v1_models.list_models()["data"]}
    assert gemini.CHAT_MODELS <= ids


def test_account_filter():
    accounts = [{"provider": provider, "status": status, "access_token": "token"}
                for provider in ["chatgpt", "gemini"] for status in ["正常", "禁用", "异常"]]
    with patch("services.account_service.account_service.list_accounts", return_value=accounts):
        assert gemini.available_accounts() == [{"provider": "gemini", "status": "正常", "access_token": "token"}]


def test_no_model_downgrade(provider):
    provider.selected_catalog = {"gemini-3.8-flash-low": ("id", 1, 2, 12)}
    with pytest.raises(gemini.GeminiWebImageError, match="unavailable"):
        list(provider.stream_chat("gemini-4.0-argon-high", []))
    provider.session.post.assert_not_called()


def test_network_exception_is_redacted():
    with patch.object(gemini, "available_accounts", return_value=[{"access_token": "secret-cookie"}]), \
         patch.object(gemini, "cached_catalog", side_effect=RuntimeError("secret-cookie in URL")):
        with pytest.raises(ImageGenerationError) as exc:
            list(gemini.stream_account_chat("gemini-3.8-flash-low", []))
    assert "secret-cookie" not in str(exc.value)


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
def test_http_stream_failure_is_not_success(endpoint):
    app = FastAPI()
    app.include_router(ai.create_router())
    body = {"model": "gemini-4.0-argon-high", "stream": True,
            "messages": [{"role": "user", "content": "failure test"}], "input": "failure test"}
    def fail(*args):
        yield "partial"
        raise ImageGenerationError("Gemini rate limited", status_code=429, code="rate_limit")
    with patch.object(gemini, "stream_account_chat", fail), patch.object(ai.LoggedCall, "log"):
        response = TestClient(app).post(endpoint, json=body, headers={"Authorization": "Bearer chatgpt2api"})
    assert "rate_limit" in response.text
    assert '"finish_reason": "stop"' not in response.text
    assert "response.completed" not in response.text
