"""Bounded inline video input; reject audio until its native path is verified."""
from __future__ import annotations

import base64
import binascii

MEDIA_TYPES = {"input_audio", "audio_url", "input_video", "video_url", "input_file", "file"}
MIMES = {"wav": "audio/wav", "mp3": "audio/mpeg", "mp4": "video/mp4", "webm": "video/webm"}
MAX_FILE_BYTES = 20 * 1024 * 1024


def invalid(message):
    from services.protocol.conversation import ImageGenerationError
    raise ImageGenerationError(message, status_code=400, error_type="invalid_request_error", code="invalid_media")


def decode_media(part: dict) -> dict:
    kind = part.get("type")
    source, mime = "", ""
    if kind == "input_audio":
        audio = part.get("input_audio") or {}
        if not isinstance(audio, dict):
            invalid("input_audio must contain data and format")
        source = audio.get("data") or ""
        mime = MIMES.get(str(audio.get("format") or "").lower(), "")
    elif kind in {"audio_url", "video_url", "input_video"}:
        value = part.get(kind) or part.get("video_url") or part.get("url") or ""
        source = value.get("url", "") if isinstance(value, dict) else value
    else:
        file = part.get("file") if kind == "file" else part
        if not isinstance(file, dict) or file.get("file_id"):
            invalid("Gemini media requires inline file_data; file_id references are not supported")
        source = file.get("file_data") or ""
    if not isinstance(source, str):
        invalid("Gemini media must be a Base64 string or data URL")
    if source.startswith("data:"):
        header, sep, source = source.partition(",")
        if not sep or not header.endswith(";base64"):
            invalid("Media data URL must use Base64 encoding")
        mime = header[5:].split(";", 1)[0].lower()
    elif source.startswith(("http:", "https:")):
        invalid("Remote audio/video URLs are not supported yet; send an inline Base64 data URL")
    mime = {"audio/x-wav": "audio/wav", "audio/mp3": "audio/mpeg"}.get(mime, mime)
    if mime.startswith("audio/"):
        invalid("Native audio understanding is not yet supported by this Gemini Web adapter; audio uploads did not pass transcription verification. Use text transcripts or supported image/video inputs")
    if mime not in MIMES.values():
        invalid("Supported media formats: MP4/WebM video; native audio is not yet available")
    if len(source) > (MAX_FILE_BYTES + 2) // 3 * 4:
        invalid("Gemini media exceeds the 20MB per-file limit")
    try:
        data = base64.b64decode(source, validate=True)
    except (ValueError, binascii.Error):
        invalid("Invalid media Base64")
    if not data or len(data) > MAX_FILE_BYTES:
        invalid("Gemini media is empty or exceeds 20MB")
    valid = False
    if mime == "video/mp4":
        valid = len(data) >= 12 and data[4:8] == b"ftyp"
    elif mime == "video/webm":
        valid = data.startswith(b"\x1a\x45\xdf\xa3")
    if not valid:
        invalid("Media bytes do not match the declared format")
    extension = next(ext for ext, value in MIMES.items() if value == mime)
    return {"type": "media", "data": data, "mime": mime, "filename": f"attachment.{extension}"}


def normalize_gemini_messages(messages: list) -> list:
    from services.protocol.conversation import normalize_messages
    source = [message for message in messages if isinstance(message, dict)]
    normalized = normalize_messages(source)
    if not source:
        return normalized
    total_size, count = 0, 0
    for raw, target in zip(source, normalized[-len(source):]):
        parts = raw.get("content")
        if not isinstance(parts, list):
            continue
        media = [decode_media(part) for part in parts if isinstance(part, dict) and part.get("type") in MEDIA_TYPES]
        if media and raw.get("role", "user") != "user":
            invalid("Audio/video attachments are supported only in user messages")
        content = target.get("content", "")
        if media:
            target["content"] = (content if isinstance(content, list) else [{"type": "text", "text": content}]) + media
        for part in target.get("content", []) if isinstance(target.get("content"), list) else []:
            if part.get("type") in {"image", "media"}:
                count += 1
                total_size += len(part["data"])
    if count > 10 or total_size > 50 * 1024 * 1024:
        invalid("Gemini supports at most 10 attachments and 50MB total per request")
    return normalized
