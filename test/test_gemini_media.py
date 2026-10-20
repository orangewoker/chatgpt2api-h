import base64
import json
from unittest.mock import patch

import pytest

from services.gemini_media import decode_media, normalize_gemini_messages
from services.protocol.conversation import ImageGenerationError
from services.protocol import openai_v1_chat_complete as chat, openai_v1_response as responses


VIDEO = b"\x00\x00\x00\x18ftypmp42test"
URL = "data:video/mp4;base64," + base64.b64encode(VIDEO).decode()


@pytest.mark.parametrize("part", [
    {"type": "video_url", "video_url": {"url": URL}},
    {"type": "input_video", "video_url": URL},
    {"type": "input_file", "file_data": URL},
    {"type": "file", "file": {"file_data": URL}},
])
def test_decode_video(part):
    result = decode_media(part)
    assert result["data"] == VIDEO
    assert result["mime"] == "video/mp4"


@pytest.mark.parametrize("part", [
    {"type": "input_audio", "input_audio": {"format": "wav", "data": "AA=="}},
    {"type": "input_audio", "input_audio": {"format": "mp3", "data": "AA=="}},
    {"type": "input_file", "file_id": "file-id"},
    {"type": "video_url", "video_url": "https://example.com/video.mp4"},
    {"type": "video_url", "video_url": "data:video/mp4;base64,!!!"},
    {"type": "video_url", "video_url": "data:video/mp4;base64,AA=="},
])
def test_reject_unverified_or_invalid_media(part):
    with pytest.raises(ImageGenerationError) as error:
        decode_media(part)
    assert error.value.status_code == 400


def test_attachment_limits():
    with pytest.raises(ImageGenerationError, match="at most 10"):
        normalize_gemini_messages([{"role": "user", "content": [
            {"type": "video_url", "video_url": URL}] * 11}])


@pytest.mark.parametrize("module", [chat, responses])
@pytest.mark.parametrize("stream", [False, True])
def test_video_protocol_routing(module, stream):
    body = {"model": "gemini-4.0-argon-low", "stream": stream}
    if module is chat:
        body["messages"] = [{"role": "user", "content": [{"type": "text", "text": "Describe"},
                             {"type": "video_url", "video_url": URL}]}]
    else:
        body["input"] = [{"type": "input_text", "text": "Describe"},
                         {"type": "input_file", "file_data": URL}]
    captured = []
    def fake(model, messages, effort):
        captured.extend(messages)
        yield "Red then blue"
    with patch("services.gemini_chat.stream_account_chat", fake):
        result = module.handle(body)
        value = list(result) if stream else result
    assert "Red then blue" in json.dumps(value)
    assert any(part.get("type") == "media" and part["data"] == VIDEO
               for message in captured for part in message["content"] if isinstance(part, dict))
