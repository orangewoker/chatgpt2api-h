"""PSD output via the existing ChatGPT web editable-file workflow."""
from __future__ import annotations

import base64
import re
import time

from services.editable_file_task_service import EDITABLE_FILE_ROOT, _editable_access_token, _file_url
from services.account_service import account_service
from services.openai_backend_api import OpenAIBackendAPI, EDITABLE_FILE_MODEL, EDITABLE_FILE_TIMEOUT_SECS, EDITABLE_FILE_POLL_INTERVAL_SECS
from utils.helper import new_uuid, normalize_image_output, is_web_image_model


def validate_artifact(data: bytes, fmt: str) -> None:
    if fmt != "psd" or len(data) < 26 or data[:6] != b"8BPS\x00\x01":
        raise ValueError("Upstream did not return PSD")


def generate_image_file(body: dict) -> dict:
    fmt, _, _ = normalize_image_output(body.get("output_format"), body.get("frame_count"), body.get("frame_duration_ms"))
    if fmt != "psd":
        raise ValueError("Expected PSD")
    if not is_web_image_model(str(body.get("model") or "gpt-image-2")):
        raise ValueError("PSD output requires a ChatGPT web image model")
    if body.get("stream"):
        raise ValueError("PSD output does not support streaming; use image tasks")
    response_format = body.get("response_format") or "b64_json"
    if response_format not in {"url", "b64_json"}:
        raise ValueError("response_format must be url or b64_json")
    n = int(body.get("n") or 1)
    if not 1 <= n <= 10:
        raise ValueError("n must be between 1 and 10")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    instruction = "生成真正可编辑的分层 PSD，背景、主体、文字等分别放在命名图层中，不要仅将平面图装入 PSD。"
    prompt += f"\n{instruction}尺寸：{body.get('size') or 'auto'}。只提供最终 .{fmt} 文件下载链接，不生成 ZIP，不只展示代码。"
    callback = body.get("progress_callback")
    if callback:
        callback("getting_account")
    token = _editable_access_token()
    backend = OpenAIBackendAPI(token)
    mime = "image/vnd.adobe.photoshop"
    items = []
    try:
        if callback:
            callback("image_stream_resolve_start")
            callback("generating")
        for _ in range(n):
            result = backend._export_editable_file_zip(
                [], prompt, EDITABLE_FILE_ROOT / fmt / new_uuid(),
                primary_label=fmt, primary_suffixes=(f".{fmt}",),
                primary_mime_types={mime}, primary_mime_keywords=(fmt,),
                primary_default_extension=f".{fmt}",
                export_file_re=re.compile(r"(?:sandbox:)?(/(?:mnt/data|workspace)/[^\s\"'\)\]]+\." + fmt + r")", re.I),
                timeout_secs=EDITABLE_FILE_TIMEOUT_SECS,
                poll_interval_secs=EDITABLE_FILE_POLL_INTERVAL_SECS,
                model=EDITABLE_FILE_MODEL, include_zip=False,
            )
            raw = result.primary_path.read_bytes()
            try:
                validate_artifact(raw, fmt)
            except Exception:
                result.primary_path.unlink(missing_ok=True)
                raise
            item = {"url": _file_url(result.primary_path, str(body.get("base_url") or "")), "revised_prompt": str(body.get("prompt")), "mime_type": mime}
            if response_format == "b64_json":
                item["b64_json"] = base64.b64encode(raw).decode("ascii")
            items.append(item)
        account_service.mark_text_used(token)
        return {"created": int(time.time()), "data": items, "_account_email": (account_service.get_account(token) or {}).get("email", "")}
    finally:
        backend.close()
