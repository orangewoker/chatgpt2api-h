from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from services.image_providers.gemini_web import (
    GeminiWebImageError,
    GeminiWebImageProvider,
    _extract_gemini_full_size_context,
)


IMAGE_ID = "http://googleusercontent.com/image_generation_content/1_806"
PREVIEW = "https://lh3.googleusercontent.com/gg-dl/fixture=s1024-rj"
BASE = "https://lh3.googleusercontent.com/gg-dl/full-fixture"
HOP = "https://work.fife.usercontent.google.com/rd-gg-dl/fixture=s0-d-I?alr=yes"
MASTER = "https://lh3.googleusercontent.com/rd-gg-dl/master-fixture=s0-d-I?alr=yes"


def stream_response(*, sparse=False, image_id=IMAGE_ID):
    image = [[None, None, None, [None, None, "alt", PREVIEW, None, "fixture-token"]], [image_id, 0]]
    rich = {"8": [[image]]} if sparse else [None] * 7 + [[image]]
    candidate = ["rc_fixture"] + [None] * 11 + [rich]
    inner = [None, ["c_fixture", "r_fixture"], None, None, [candidate]]
    return ")]}'\n123\n" + json.dumps([["wrb.fr", "StreamGenerate", json.dumps(inner)]])


def response(content=b"", *, text=None, status=200, headers=None):
    r = MagicMock()
    r.content = text.encode() if text is not None else content
    r.text = text if text is not None else ""
    r.status_code = status
    r.headers = headers or {}
    return r


def image_bytes(size=(2048, 2048)):
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, "PNG")
    return buf.getvalue()


def provider():
    p = GeminiWebImageProvider("__Secure-1PSID=fixture")
    p.snlm0e = "fixture-at"
    p._push_id = "feeds/fixture"
    p._session = MagicMock()
    return p


@pytest.mark.parametrize("sparse", [False, True])
def test_context_keeps_actual_nonzero_image_id(sparse):
    context = _extract_gemini_full_size_context(stream_response(sparse=sparse))
    assert context == (PREVIEW, "fixture-token", '["c_fixture", "r_fixture", "rc_fixture"]', IMAGE_ID)
    assert _extract_gemini_full_size_context(stream_response(image_id="")) is None


def test_rpc_uses_actual_image_id_and_prefixless_source_path():
    p = provider()
    p._session_id = "-12345"
    p._session.post.return_value = response(text=")]}'\n" + json.dumps([["wrb.fr", "c8o8Fe", json.dumps([BASE])]]))
    assert p._fetch_full_size_url("fixture-token", "prompt", '["c_fixture", "r_fixture", "rc_fixture"]', image_id=IMAGE_ID) == BASE
    call = p._session.post.call_args
    payload = json.loads(json.loads(call.kwargs["data"]["f.req"])[0][0][1])
    assert payload[0][1] == [IMAGE_ID, 0]
    assert payload[1][:3] == ["r_fixture", "rc_fixture", "c_fixture"]
    assert call.kwargs["params"]["source-path"] == "/app/fixture"
    assert call.kwargs["params"]["f.sid"] == "-12345"


@pytest.mark.parametrize("size", [(2048, 2048), (1536, 2752), (4096, 4096)])
def test_two_text_body_hops_return_exact_master_bytes(size):
    p = provider()
    raw = image_bytes(size)
    hops = [response(text=HOP), response(text=MASTER), response(raw, headers={"content-type": "image/png"})]
    p._session.get.side_effect = hops
    assert p._download_full_size(BASE) == raw
    assert [c.args[0] for c in p._session.get.call_args_list] == [BASE + "=d-I?alr=yes", HOP, MASTER]
    assert all(c.kwargs["allow_redirects"] is False for c in p._session.get.call_args_list)
    for r in hops:
        r.close.assert_called_once()


def test_http_and_text_redirects_can_be_mixed():
    p = provider()
    raw = image_bytes()
    p._session.get.side_effect = [response(status=302, headers={"location": HOP}), response(text=json.dumps(MASTER)), response(raw)]
    assert p._download_full_size(BASE) == raw


@pytest.mark.parametrize("url", [
    "http://lh3.googleusercontent.com/gg/test", "https://127.0.0.1/gg/test",
    "https://lh3.googleusercontent.com.attacker.test/gg/test", "https://accounts.google.com/gg/test",
    "https://lh3.googleusercontent.com:8443/gg/test", "https://lh3.googleusercontent.com/profile/test",
])
def test_download_never_sends_cookie_session_to_unknown_hosts(url):
    p = provider()
    with pytest.raises(GeminiWebImageError) as exc:
        p._download_full_size(url)
    assert exc.value.code == "full_size_download_failed"
    p._session.get.assert_not_called()


@pytest.mark.parametrize("http_redirect", [False, True])
def test_untrusted_redirect_target_is_not_fetched(http_redirect):
    p = provider()
    target = "https://attacker.test/steal"
    p._session.get.return_value = response(status=302, headers={"location": target}) if http_redirect else response(text=target)
    with pytest.raises(GeminiWebImageError):
        p._download_full_size(BASE)
    assert p._session.get.call_count == 1


def test_download_loop_is_bounded():
    p = provider()
    p._session.get.return_value = response(text=HOP)
    with pytest.raises(GeminiWebImageError):
        p._download_full_size(BASE)
    assert p._session.get.call_count == 2


@pytest.mark.parametrize("context_available", [False, True])
def test_failed_full_size_never_falls_back_to_1k_preview(context_available):
    p = provider()
    p._session.post.return_value = response(text=stream_response(image_id=IMAGE_ID if context_available else ""))
    with patch.object(p, "_fetch_full_size_url", return_value=None):
        with pytest.raises(GeminiWebImageError) as exc:
            p.generate("test")
    assert exc.value.code == "full_size_unavailable"
    p._session.get.assert_not_called()


def test_upload_reference_is_attached_to_generation_and_master_is_unmodified():
    p = provider()
    raw = image_bytes()
    encoded = base64.b64encode(raw).decode()
    upload = response(text="/contrib_service/ttl_1d/fixture")
    p._session.post.side_effect = [upload, response(text=stream_response())]
    p._session.get.side_effect = [response(text=HOP), response(text=MASTER), response(raw)]
    with patch.object(p, "_fetch_full_size_url", return_value=BASE) as rpc, patch("services.image_providers.gemini_web.CurlMime") as mime:
        result = p.generate("Make the circle blue", size="4096x4096", images=[encoded])
    assert base64.b64decode(result["data"][0]["b64_json"]) == raw
    upload_call, generate_call = p._session.post.call_args_list
    assert upload_call.args[0] == "https://content-push.googleapis.com/upload"
    assert upload_call.kwargs["headers"] == {"X-Tenant-Id": "bard-storage", "Push-ID": "feeds/fixture"}
    multipart = mime.return_value
    assert multipart.addpart.call_args.kwargs["data"] == raw
    multipart.close.assert_called_once()
    req = json.loads(json.loads(generate_call.kwargs["data"]["f.req"])[1])
    assert req[0][3] == [[["/contrib_service/ttl_1d/fixture"], "reference-1.png"]]
    assert rpc.call_args.kwargs["image_id"] == IMAGE_ID
    assert rpc.call_args.args[1] == req[0][0]


@pytest.mark.parametrize("images", [["invalid base64!"], [base64.b64encode(b"not an image").decode()], ["x"] * 11])
def test_invalid_references_fail_before_upload(images):
    p = provider()
    with pytest.raises(GeminiWebImageError) as exc:
        p._upload_images(images)
    assert exc.value.code == "invalid_image"
    p._session.post.assert_not_called()


def test_failed_upload_does_not_start_generation():
    p = provider()
    p._session.post.return_value = response(text="unexpected", status=500)
    with pytest.raises(GeminiWebImageError) as exc:
        p.generate("edit", images=[base64.b64encode(image_bytes()).decode()])
    assert exc.value.code == "image_upload_failed"
    assert p._session.post.call_count == 1


def test_multiple_references_keep_distinct_attachment_ids():
    p = provider()
    p._session.post.side_effect = [response(text="/contrib_service/ttl_1d/one"), response(text="/contrib_service/ttl_1d/two")]
    raw = image_bytes()
    result = p._upload_images([(raw, "first.png", "image/png"), (raw, "second.png", "image/png")])
    assert result == [[["/contrib_service/ttl_1d/one"], "first.png"], [["/contrib_service/ttl_1d/two"], "second.png"]]


def test_full_size_context_does_not_depend_on_legacy_preview_path():
    p = provider()
    p._session.post.return_value = response(text=stream_response().replace("/gg-dl/fixture", "/gg/fixture"))
    raw = image_bytes()
    with patch.object(p, "_fetch_full_size_url", return_value=BASE), patch.object(p, "_download_full_size", return_value=raw):
        result = p.generate("test")
    assert base64.b64decode(result["data"][0]["b64_json"]) == raw


def test_page_parses_signed_session_and_upload_push_id():
    p = provider()
    p.snlm0e = ""
    p._session.get.return_value = response(text='{"SNlM0e":"fixture-at","cfb2h":"build","FdrFJe":"-234","qKIAYe":"feeds/test"}')
    p._ensure_snlm0e()
    assert p._session_id == "-234"
    assert p._push_id == "feeds/test"
