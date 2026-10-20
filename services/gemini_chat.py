"""Gemini Web text/vision adapter. Image generation stays in image_providers."""
from __future__ import annotations

import json
import re
import uuid
import hashlib
import threading
import time
from typing import Iterator
from curl_cffi import CurlMime

from services.image_providers.gemini_web import (
    GEMINI_STREAM_GENERATE_ENDPOINT, GeminiWebImageError, GeminiWebImageProvider,
)

EFFORTS = {"low": 1, "medium": 4, "high": 2}
MODEL_BASES = ("gemini-3.8-flash", "gemini-4.0-argon")
CHAT_MODELS = {f"{base}-{effort}" for base in MODEL_BASES for effort in EFFORTS}
MODEL_HEADER = "x-goog-ext-525001261-jspb"
_catalog_cache: dict[str, tuple[float, dict]] = {}
_cooldowns: dict[str, float] = {}
_conversations: dict[str, tuple[float, list]] = {}
_lock = threading.RLock()


def conversation_key(account: dict, model: str, messages: list) -> str:
    def encode(value):
        if isinstance(value, (bytes, bytearray)):
            return {"sha256": hashlib.sha256(value).hexdigest()}
        raise TypeError("Unsupported conversation value")
    data = json.dumps([_account_key(account), model, messages], ensure_ascii=False,
                      sort_keys=True, default=encode).encode()
    return hashlib.sha256(data).hexdigest()


def take_conversation(account: dict, model: str, messages: list) -> list | None:
    if not messages or messages[-1].get("role") != "user":
        return None
    key = conversation_key(account, model, messages[:-1])
    with _lock:
        entry = _conversations.pop(key, None)
    return list(entry[1]) if entry and entry[0] > time.monotonic() else None


def remember_conversation(account: dict, model: str, messages: list, text: str, metadata: list) -> None:
    if len(metadata) < 3 or not all(isinstance(x, str) and x for x in metadata[:3]):
        return
    key = conversation_key(account, model, messages + [{"role": "assistant", "content": text}])
    with _lock:
        for expired in [k for k, (until, _) in _conversations.items() if until <= time.monotonic()]:
            _conversations.pop(expired, None)
        if len(_conversations) >= 256:
            _conversations.pop(next(iter(_conversations)))
        _conversations[key] = (time.monotonic() + 1800, list(metadata))


def validate_chat_body(body: dict) -> dict:
    """Validate before other providers/tools get a chance to route this request."""
    model = str(body.get("model") or "")
    if not model.startswith(("gemini-3.8-", "gemini-4.0-")):
        return body
    from services.protocol.conversation import ImageGenerationError
    def invalid(message):
        raise ImageGenerationError(message, status_code=400, error_type="invalid_request_error", code="invalid_request")
    if not is_gemini_chat(model):
        invalid("Unknown Gemini chat model; choose a low, medium or high model ID")
    if "image" in (body.get("modalities") or []):
        invalid("modalities=image requests image OUTPUT, not vision input. For image understanding use image_url/input_image with modalities omitted or [text]")
    tools = body.get("tools") or []
    if any(isinstance(tool, dict) and tool.get("type") == "image_generation" for tool in tools):
        invalid("image_generation requests image OUTPUT. Remove this tool for vision input, or use an image generation model")
    if body.get("tools") or body.get("functions"):
        choice = body.get("tool_choice", "auto")
        legacy_choice = body.get("function_call", "auto")
        if choice not in ("auto", "none", None) or legacy_choice not in ("auto", "none", None):
            invalid("Gemini chat cannot execute required tool/function calls; use tool_choice=none for text or image understanding")
        body = dict(body)
        for key in ("tools", "functions", "tool_choice", "function_call"):
            body.pop(key, None)
    if body.get("previous_response_id"):
        invalid("Gemini chat requires full conversation history in input; previous_response_id is not supported")
    reasoning = body.get("reasoning")
    values = [body.get("thinking_effort"), body.get("reasoning_effort"),
              reasoning.get("effort") if isinstance(reasoning, dict) else None]
    if any(value is not None and str(value).strip().lower() != model.rsplit("-", 1)[1] for value in values):
        invalid("Reasoning effort conflicts with the selected Gemini model tier")
    return body


def _account_key(account: dict) -> str:
    # Cache keys never retain credentials or expose them through diagnostics.
    stable = {key: value for key, value in account.items()
              if key not in {"last_used_at", "updated_at", "image_inflight", "quota"}}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()


def available_accounts():
    from services.account_service import account_service
    return sorted((a for a in account_service.list_accounts()
                   if account_service.account_provider(a) == "gemini"
                   and a.get("status") not in {"禁用", "异常"} and a.get("access_token")),
                  key=lambda a: str(a.get("last_used_at") or ""))


def cached_catalog(account: dict) -> dict:
    key = _account_key(account)
    with _lock:
        cached = _catalog_cache.get(key)
        if cached and cached[0] > time.monotonic():
            return cached[1]
    with GeminiChatProvider(account["access_token"], account=account, timeout=20) as provider:
        catalog = provider.discover_models()
    with _lock:
        if len(_catalog_cache) >= 128:
            _catalog_cache.clear()
        _catalog_cache[key] = (time.monotonic() + 300, catalog)
    return catalog


def available_chat_models() -> set[str]:
    models = set()
    for account in available_accounts():
        try:
            models.update(cached_catalog(account))
        except Exception:
            # A failed catalog must never advertise guessed models.
            continue
    return models


def stream_account_chat(model: str, messages: list, effort: str = "") -> Iterator[str]:
    from services.account_service import account_service
    from services.protocol.conversation import ImageGenerationError
    last_error = None
    accounts = available_accounts()
    if messages and messages[-1].get("role") == "user":
        with _lock:
            accounts.sort(key=lambda a: not (conversation_key(a, model, messages[:-1]) in _conversations))
    for account in accounts:
        key = _account_key(account)
        with _lock:
            if _cooldowns.get(key, 0) > time.monotonic():
                continue
        emitted = False
        try:
            catalog = cached_catalog(account)
            if model not in catalog:
                continue
            with GeminiChatProvider(account["access_token"], account=account) as provider:
                provider.selected_catalog = catalog
                provider.conversation_metadata = take_conversation(account, model, messages)
                response_parts = []
                for delta in provider.stream_chat(model, messages, effort):
                    emitted = True
                    response_parts.append(delta)
                    yield delta
                remember_conversation(account, model, messages, "".join(response_parts),
                                      getattr(provider, "response_metadata", []))
                account_service.mark_text_used(provider.token)
            return
        except Exception as exc:
            status = getattr(exc, "status_code", 502)
            code = getattr(exc, "code", "upstream_error")
            # Never echo HTTP exceptions (which may contain URLs or credentials).
            message = str(exc) if isinstance(exc, GeminiWebImageError) else "Gemini conversation connection failed"
            last_error = ImageGenerationError(message, status_code=status, code=code,
                                              error_type="invalid_request_error" if status == 400 else "server_error")
            if status in {401, 429}:
                with _lock:
                    for expired in [k for k, until in _cooldowns.items() if until <= time.monotonic()]:
                        _cooldowns.pop(expired, None)
                    _cooldowns[key] = time.monotonic() + 60
                    _catalog_cache.pop(key, None)
            if emitted or status == 400:
                raise last_error from None
    if last_error:
        raise last_error from None
    raise ImageGenerationError("No active Gemini account offers the selected model",
                               status_code=503, code="model_not_available")


def is_gemini_chat(model: str) -> bool:
    return model in CHAT_MODELS


def rpc_frames(lines) -> Iterator[list]:
    """The transport uses XSSI and length lines followed by JSON frame lines."""
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        try:
            frame = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(frame, list):
            for item in frame:
                if isinstance(item, list) and item:
                    yield item


def model_catalog(body: list) -> dict:
    if len(body) < 18 or not isinstance(body[15], list):
        return {}
    tiers, caps = body[16] or [], body[17] or []
    capacity, field = (1, 13) if 21 in tiers else (2, 13) if 22 in tiers else (
        (4, 12) if 115 in caps else (3, 12) if 16 in tiers or 106 in caps else
        (2, 12) if 8 in tiers or 19 in caps else (1, 12)
    )
    result = {}
    for item in body[15]:
        if not isinstance(item, list) or len(item) < 18:
            continue
        name = str(item[19] if len(item) > 19 and item[19] else item[11]).lower().strip()
        name = re.sub(r"^gemini\s+", "", name)
        base = {"3.8 flash": MODEL_BASES[0], "4 argon": MODEL_BASES[1],
                "4.0 argon": MODEL_BASES[1]}.get(name)
        if base and isinstance(item[0], str) and isinstance(item[17], int):
            for effort in EFFORTS:
                result[f"{base}-{effort}"] = (item[0], item[17], capacity, field)
    return result


class GeminiChatProvider(GeminiWebImageProvider):
    def _upload_media(self, media: list[tuple]) -> list:
        if not self._push_id:
            raise GeminiWebImageError("Gemini file upload session is unavailable", code="media_upload_failed")
        attachments = []
        for data, filename, mime in media:
            multipart = CurlMime()
            try:
                multipart.addpart(name="file", content_type=mime, filename=filename, data=data)
                response = self.session.post(
                    "https://content-push.googleapis.com/upload",
                    headers={"X-Tenant-Id": "bard-storage", "Push-ID": self._push_id},
                    multipart=multipart, timeout=self.timeout, allow_redirects=False,
                )
                try:
                    self.check_status(response.status_code)
                    file_id = response.text.strip()
                    if not file_id.startswith("/contrib_service/") or len(file_id) > 4096:
                        raise GeminiWebImageError("Gemini media upload failed", code="media_upload_failed")
                    attachments.append([[file_id], filename])
                finally:
                    response.close()
            finally:
                multipart.close()
        return attachments

    def discover_models(self) -> dict:
        self._ensure_snlm0e()
        response = self.session.post(
            "https://gemini.google.com/_/BardChatUi/data/batchexecute",
            params={"rpcids": "otAQ7b", "source-path": "/app", "rt": "c"},
            data={"at": self.snlm0e, "f.req": json.dumps([[["otAQ7b", "[]", None, "generic"]]])},
            timeout=self.timeout,
        )
        try:
            self.check_status(response.status_code)
            for frame in rpc_frames(response.text.splitlines()):
                if len(frame) > 2 and frame[:2] == ["wrb.fr", "otAQ7b"] and frame[2]:
                    return model_catalog(json.loads(frame[2]))
            raise GeminiWebImageError("Gemini model catalog unavailable", code="model_catalog_unavailable")
        finally:
            response.close()

    @staticmethod
    def check_status(status: int) -> None:
        if status != 200:
            code = "token_invalid" if status in {401, 403} else "rate_limit" if status == 429 else "upstream_error"
            raise GeminiWebImageError(f"Gemini returned HTTP {status}",
                                      status_code=401 if status == 403 else status, code=code)

    def stream_chat(self, model: str, messages: list, effort: str = "") -> Iterator[str]:
        if model not in CHAT_MODELS:
            raise GeminiWebImageError("Unsupported Gemini chat model", status_code=400, code="model_not_found")
        level = model.rsplit("-", 1)[1]
        if effort and effort != level:
            raise GeminiWebImageError("Reasoning effort conflicts with the selected model tier",
                                      status_code=400, code="invalid_reasoning_effort")
        self._ensure_snlm0e()
        catalog = getattr(self, "selected_catalog", None) or self.discover_models()
        if model not in catalog:
            raise GeminiWebImageError("Selected Gemini model is unavailable to this account",
                                      status_code=400, code="model_not_available")
        model_id, number, capacity, capacity_field = catalog[model]
        metadata = getattr(self, "conversation_metadata", None)
        # On an exact history match, continue the original Web conversation and
        # send only the new turn. Unknown/imported histories use readable text.
        active_messages = messages[-1:] if metadata else messages
        history, images, media = [], [], []
        for message in active_messages:
            parts = message.get("content", [])
            if isinstance(parts, str):
                parts = [{"type": "text", "text": parts}]
            content = []
            for part in parts:
                if part.get("type") == "text":
                    content.append({"text": part.get("text", "")})
                elif part.get("type") == "image":
                    images.append((part["data"], f"image_{len(images)+1}.png", part.get("mime", "image/png")))
                    content.append({"attached_image": len(images)})
                elif part.get("type") == "media":
                    filename = f"media_{len(media)+1}." + part["filename"].rsplit(".", 1)[-1]
                    media.append((part["data"], filename, part["mime"]))
                    content.append({"attached_file": filename})
            history.append({"role": message.get("role", "user"), "content": content})
        if len(history) == 1 and history[0]["role"] == "user":
            # Send ordinary single-turn messages verbatim, as the Web UI does.
            prompt = "".join(part["text"] for part in history[0]["content"] if "text" in part)
        else:
            labels = {"system": "System", "developer": "Developer", "user": "User", "assistant": "Assistant"}
            turns = []
            for turn in history:
                text = "\n".join(part["text"] if "text" in part else
                                 f"[Image {part['attached_image']}]" if "attached_image" in part else
                                 f"[File {part['attached_file']}]" for part in turn["content"])
                turns.append(f"{labels.get(turn['role'], turn['role'])}: {text}")
            prompt = "\n\n".join(turns) + "\n\nAssistant:"
        attachments = self._upload_images(images) if images else []
        if media:
            attachments.extend(self._upload_media(media))
        attachments = attachments or None
        payload = [None] * 81
        payload[0] = [prompt, 0, None, attachments, None, None, 0]
        payload[1] = ["en"]
        payload[2] = list(metadata) if metadata else ["", "", "", None, None, None, None, None, None, ""]
        self.response_metadata = list(payload[2])
        for index, value in {6: [1], 7: 1, 10: 1, 11: 0, 17: [[0]], 18: 0,
                             27: 1, 30: [4], 41: [1], 53: 0, 61: [], 68: 2,
                             79: number, 80: EFFORTS[level]}.items():
            payload[index] = value
        request_id = str(uuid.uuid4()).upper()
        payload[59] = request_id
        header = [1, None, None, None, model_id, None, None, 0, [4, 5, 6, 8], None, None]
        header += [None, capacity] if capacity_field == 13 else [capacity]
        header += [None, None, number, EFFORTS[level], str(uuid.uuid4())]
        params = {"hl": "en", "rt": "c"}
        if self._build_label:
            params["bl"] = self._build_label
        if self._session_id:
            params["f.sid"] = self._session_id
        response = self.session.post(
            GEMINI_STREAM_GENERATE_ENDPOINT, params=params,
            headers={MODEL_HEADER: json.dumps(header), "x-goog-ext-73010989-jspb": "[0]",
                     "x-goog-ext-73010990-jspb": "[0,0,0]",
                     "x-goog-ext-525005358-jspb": json.dumps([request_id, 1])},
            data={"at": self.snlm0e, "f.req": json.dumps([None, json.dumps(payload)])},
            timeout=self.timeout, stream=True,
        )
        text = ""
        try:
            self.check_status(response.status_code)
            for frame in rpc_frames(response.iter_lines()):
                try:
                    error_code = frame[5][2][0][1][0]
                except (IndexError, TypeError):
                    error_code = None
                if error_code:
                    status = 429 if error_code in {1037, 1060} else 400 if error_code in {1050, 1052} else 502
                    raise GeminiWebImageError("Gemini model quota exceeded" if status == 429 else
                                              "Gemini rejected the selected model or request",
                                              status_code=status, code="rate_limit" if status == 429 else "upstream_error")
                if frame[0] == "er":
                    raise GeminiWebImageError("Gemini rejected the conversation request", code="upstream_error")
                if len(frame) < 3 or frame[0] != "wrb.fr" or not frame[2]:
                    continue
                body = json.loads(frame[2])
                if len(body) > 1 and isinstance(body[1], list):
                    for index, value in enumerate(body[1][:10]):
                        if index < len(self.response_metadata) and value is not None:
                            self.response_metadata[index] = value
                candidates = body[4] if len(body) > 4 else None
                if not candidates:
                    continue
                candidate = candidates[0]
                if candidate and isinstance(candidate[0], str):
                    self.response_metadata[2] = candidate[0]
                current = candidate[1][0] if len(candidate) > 1 and candidate[1] else ""
                if not isinstance(current, str) or not current:
                    continue
                if not current.startswith(text):
                    raise GeminiWebImageError("Gemini revised an already streamed response", code="stream_revision")
                delta, text = current[len(text):], current
                if delta:
                    yield delta
            if not text:
                raise GeminiWebImageError("Gemini returned no assistant text", code="empty_response")
        finally:
            response.close()
