from __future__ import annotations

import base64
import binascii
import ipaddress
import time
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

from curl_cffi import requests
from PIL import Image

from services.proxy_service import proxy_settings
from utils.helper import AGNES_IMAGE_MODEL

AGNES_IMAGE_ENDPOINT = "https://apihub.agnes-ai.com/v1/images/generations"
AGNES_REQUEST_TIMEOUT_SECONDS = 360
AGNES_DOWNLOAD_TIMEOUT_SECONDS = 120
MAX_AGNES_IMAGE_BYTES = 50 * 1024 * 1024
MAX_AGNES_IMAGE_PIXELS = 100_000_000
MAX_AGNES_RESULTS = 4
MAX_AGNES_TOTAL_IMAGE_BYTES = 100 * 1024 * 1024

SUPPORTED_TIERS = {"1K", "2K", "3K", "4K"}
SUPPORTED_RATIOS = {"1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2", "21:9"}

# The web image composer sends exact display sizes. Preserve its intended tier
# and ratio when translating those values to Agnes' native size controls.
COMPOSER_SIZE_MAP: dict[str, tuple[str, str]] = {
    "1024x1024": ("1K", "1:1"),
    "1024x1536": ("1K", "2:3"),
    "1536x1024": ("1K", "3:2"),
    "1024x1365": ("1K", "3:4"),
    "1365x1024": ("1K", "4:3"),
    "1088x1920": ("1K", "9:16"),
    "1920x1088": ("1K", "16:9"),
    "2048x2048": ("2K", "1:1"),
    "2560x1440": ("2K", "16:9"),
    "1440x2560": ("2K", "9:16"),
    "3840x2160": ("4K", "16:9"),
    "2160x3840": ("4K", "9:16"),
}


class AgnesImageError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        error_type: str = "server_error",
        code: str = "upstream_error",
        retryable: bool = False,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


def _compact_base64(value: object) -> str:
    text = "".join(str(value or "").split())
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    if not text:
        return ""
    if len(text) > ((MAX_AGNES_IMAGE_BYTES + 2) // 3) * 4 + 8:
        raise AgnesImageError(
            "Agnes image exceeds the 50MB limit",
            code="image_too_large",
        )
    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AgnesImageError(
            "Agnes returned invalid Base64 image data",
            code="invalid_upstream_response",
        ) from exc
    if len(decoded) > MAX_AGNES_IMAGE_BYTES:
        raise AgnesImageError(
            "Agnes image exceeds the 50MB limit",
            code="image_too_large",
        )
    return text


def _validate_result_image(encoded: str) -> str:
    compact = _compact_base64(encoded)
    if not compact:
        return ""
    try:
        with Image.open(BytesIO(base64.b64decode(compact))) as image:
            if int(image.width) * int(image.height) > MAX_AGNES_IMAGE_PIXELS:
                raise AgnesImageError(
                    "Agnes image exceeds the pixel limit",
                    code="image_too_large",
                )
            image.verify()
    except AgnesImageError:
        raise
    except Exception as exc:
        raise AgnesImageError(
            "Agnes returned unsupported or corrupt image data",
            code="invalid_upstream_response",
        ) from exc
    return compact


def _image_data_uri(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("data:image/") and "," in text:
        _header, encoded = text.split(",", 1)
    else:
        encoded = text
    encoded = _compact_base64(encoded)
    try:
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            mime_type = Image.MIME.get(image.format, "image/png")
            if int(image.width) * int(image.height) > MAX_AGNES_IMAGE_PIXELS:
                raise ValueError("image exceeds pixel limit")
            image.verify()
    except Exception as exc:
        raise AgnesImageError(
            "input image is not a supported image",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_image",
        ) from exc
    return f"data:{mime_type};base64,{encoded}"


def _error_details(response: Any) -> tuple[str, str]:
    payload: object = None
    try:
        payload = response.json()
    except Exception:
        payload = None

    message = ""
    code = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("detail") or "").strip()
            code = str(error.get("code") or error.get("type") or "").strip()
        elif error:
            message = str(error).strip()
        message = message or str(payload.get("message") or payload.get("detail") or "").strip()
        code = code or str(payload.get("code") or "").strip()
    if not message:
        message = str(getattr(response, "text", "") or "").strip()
    return (message[:500] or "Agnes image request failed", code[:100])


class AgnesImageProvider:
    def __init__(
        self,
        api_key: str,
        *,
        account: dict[str, Any] | None = None,
        session: Any = None,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        if not self.api_key:
            raise ValueError("Agnes API key is required")
        self.account = dict(account or {})
        self._owns_session = session is None
        self.session = session or requests.Session(
            **proxy_settings.build_session_kwargs(
                account=self.account,
                upstream=True,
                verify=True,
            )
        )

    def close(self) -> None:
        if not self._owns_session:
            return
        try:
            self.session.close()
        except Exception:
            pass

    def __enter__(self) -> "AgnesImageProvider":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def normalize_size(size: object, ratio: object = None) -> tuple[str, str | None]:
        size_text = str(size or "").strip()
        ratio_text = str(ratio or "").strip()
        if ratio_text.lower() == "auto":
            ratio_text = ""
        if ratio_text and ratio_text not in SUPPORTED_RATIOS:
            raise AgnesImageError(
                f"unsupported Agnes image ratio: {ratio_text}",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_ratio",
            )
        normalized_ratio = ratio_text or None
        if not size_text or size_text.lower() == "auto":
            return "1K", normalized_ratio
        tier = size_text.upper()
        if tier in SUPPORTED_TIERS:
            return tier, normalized_ratio
        if size_text in COMPOSER_SIZE_MAP:
            mapped_tier, mapped_ratio = COMPOSER_SIZE_MAP[size_text]
            if normalized_ratio and normalized_ratio != mapped_ratio:
                raise AgnesImageError(
                    f"image size {size_text} conflicts with ratio {normalized_ratio}",
                    status_code=400,
                    error_type="invalid_request_error",
                    code="invalid_ratio",
                )
            return mapped_tier, normalized_ratio or mapped_ratio
        # Agnes accepts legacy exact sizes and normalizes unsupported values.
        return size_text, normalized_ratio

    @classmethod
    def build_payload(
        cls,
        *,
        prompt: str,
        size: object = None,
        ratio: object = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            raise AgnesImageError(
                "prompt is required",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_prompt",
            )
        normalized_size, normalized_ratio = cls.normalize_size(size, ratio)
        payload: dict[str, Any] = {
            "model": AGNES_IMAGE_MODEL,
            "prompt": prompt_text,
            "size": normalized_size,
        }
        if normalized_ratio:
            payload["ratio"] = normalized_ratio
        if images:
            payload["extra_body"] = {
                "image": [_image_data_uri(image) for image in images],
                "response_format": "b64_json",
            }
        else:
            # Agnes documents return_base64 for text-to-image, but the live
            # 2.1 Flash endpoint can still return a URL unless the OpenAI-style
            # response format is also made explicit.
            payload["return_base64"] = True
            payload["extra_body"] = {"response_format": "b64_json"}
        return payload

    def _raise_http_error(self, response: Any) -> None:
        status = int(getattr(response, "status_code", 0) or 0)
        message, upstream_code = _error_details(response)
        if self.api_key:
            message = message.replace(self.api_key, "[redacted]")
        headers = getattr(response, "headers", {}) or {}
        retry_after_seconds: int | None = None
        try:
            raw_retry_after = int(headers.get("Retry-After") or 0)
            if raw_retry_after > 0:
                retry_after_seconds = raw_retry_after
        except (TypeError, ValueError):
            retry_after_seconds = None
        if status in {401, 403}:
            raise AgnesImageError(
                message,
                status_code=status,
                error_type="authentication_error",
                code=upstream_code or "invalid_api_key",
            )
        if status in {402, 429}:
            raise AgnesImageError(
                message,
                status_code=status,
                error_type="rate_limit_error",
                code=upstream_code or "rate_limit_exceeded",
                retryable=True,
                retry_after_seconds=retry_after_seconds,
            )
        if 400 <= status < 500:
            raise AgnesImageError(
                message,
                status_code=status,
                error_type="invalid_request_error",
                code=upstream_code or "invalid_request",
            )
        raise AgnesImageError(
            message,
            status_code=502,
            code=upstream_code or "upstream_error",
            retryable=True,
        )

    def _download_result(self, url: str) -> str:
        parsed = urlparse(url)
        hostname = str(parsed.hostname or "").strip().lower()
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise AgnesImageError(
                "Agnes returned an invalid image URL",
                code="invalid_upstream_response",
            )
        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
            or hostname.endswith(".internal")
        ):
            raise AgnesImageError(
                "Agnes returned an unsafe image URL",
                code="invalid_upstream_response",
            )
        try:
            if not ipaddress.ip_address(hostname).is_global:
                raise AgnesImageError(
                    "Agnes returned an unsafe image URL",
                    code="invalid_upstream_response",
                )
        except ValueError:
            pass
        try:
            response = self.session.get(
                url,
                # Agnes' CDN can negotiate a content encoding that libcurl
                # cannot decode while streaming (CURLE_BAD_CONTENT_ENCODING).
                # Requesting the identity representation keeps the bounded
                # streaming download below reliable without buffering the
                # entire response in curl_cffi first.
                headers={
                    "Accept": "image/*,*/*;q=0.8",
                    "Accept-Encoding": "identity",
                },
                timeout=AGNES_DOWNLOAD_TIMEOUT_SECONDS,
                allow_redirects=False,
                # curl_cffi otherwise applies its own default
                # "gzip, deflate, br" value even when a raw header is set.
                accept_encoding="identity",
                stream=True,
            )
        except Exception as exc:
            raise AgnesImageError(
                "Unable to download the image returned by Agnes",
                code="upstream_download_error",
                retryable=True,
            ) from exc
        try:
            if not 200 <= int(getattr(response, "status_code", 0) or 0) < 300:
                raise AgnesImageError(
                    "Unable to download the image returned by Agnes",
                    code="upstream_download_error",
                    retryable=True,
                )
            headers = getattr(response, "headers", {}) or {}
            content_type = str(headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if content_type and not (
                content_type.startswith("image/")
                or content_type in {"application/octet-stream", "binary/octet-stream"}
            ):
                raise AgnesImageError(
                    "Agnes returned a non-image download",
                    code="invalid_upstream_response",
                )
            try:
                content_length = int(headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                content_length = 0
            if content_length > MAX_AGNES_IMAGE_BYTES:
                raise AgnesImageError(
                    "Agnes image exceeds the 50MB limit",
                    code="image_too_large",
                )
            content = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > MAX_AGNES_IMAGE_BYTES:
                    raise AgnesImageError(
                        "Agnes image exceeds the 50MB limit",
                        code="image_too_large",
                    )
        finally:
            try:
                response.close()
            except Exception:
                pass
        if not content:
            raise AgnesImageError(
                "Agnes returned an empty image",
                code="invalid_upstream_response",
            )
        return _validate_result_image(base64.b64encode(content).decode("ascii"))

    def generate(
        self,
        *,
        prompt: str,
        size: object = None,
        ratio: object = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = self.build_payload(prompt=prompt, size=size, ratio=ratio, images=images)
        try:
            response = self.session.post(
                AGNES_IMAGE_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
                timeout=AGNES_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            error_text = str(exc or "").lower()
            error_code = int(getattr(exc, "code", 0) or 0)
            if error_code == 28 or "timed out" in error_text or "timeout" in error_text:
                raise AgnesImageError(
                    "Agnes image request timed out",
                    status_code=504,
                    code="upstream_timeout",
                    retryable=True,
                ) from exc
            raise AgnesImageError(
                "Unable to connect to Agnes image service",
                code="upstream_connection_error",
                retryable=True,
            ) from exc

        if not 200 <= int(getattr(response, "status_code", 0) or 0) < 300:
            self._raise_http_error(response)
        try:
            result = response.json()
        except Exception as exc:
            raise AgnesImageError(
                "Agnes returned invalid JSON",
                code="invalid_upstream_response",
            ) from exc
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise AgnesImageError(
                "Agnes returned no image results",
                code="invalid_upstream_response",
            )
        if len(result["data"]) > MAX_AGNES_RESULTS:
            raise AgnesImageError(
                "Agnes returned too many image results",
                code="invalid_upstream_response",
            )

        data: list[dict[str, Any]] = []
        total_image_bytes = 0
        for item in result["data"]:
            if not isinstance(item, dict):
                continue
            b64_json = _compact_base64(item.get("b64_json"))
            url = str(item.get("url") or "").strip()
            if not b64_json and url:
                if url.startswith("data:image/"):
                    b64_json = _compact_base64(url)
                else:
                    b64_json = self._download_result(url)
            if not b64_json:
                continue
            b64_json = _validate_result_image(b64_json)
            total_image_bytes += len(base64.b64decode(b64_json))
            if total_image_bytes > MAX_AGNES_TOTAL_IMAGE_BYTES:
                raise AgnesImageError(
                    "Agnes image response exceeds the total size limit",
                    code="image_too_large",
                )
            data.append({
                "b64_json": b64_json,
                "revised_prompt": str(item.get("revised_prompt") or prompt).strip() or prompt,
            })
        if not data:
            raise AgnesImageError(
                "Agnes completed without generating an image",
                code="no_image_generated",
            )
        try:
            created = int(result.get("created") or 0)
        except (TypeError, ValueError):
            created = 0
        return {"created": created or int(time.time()), "data": data}
