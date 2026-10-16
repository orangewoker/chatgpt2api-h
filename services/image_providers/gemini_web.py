"""Gemini Web direct image generation provider."""

from __future__ import annotations

import base64
import json
import re
import time
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

from curl_cffi import requests
from PIL import Image

from services.proxy_service import proxy_settings
from utils.gemini_watermark import remove_gemini_sparkle
from utils.log import logger

GEMINI_WEB_BASE = "https://gemini.google.com"
GEMINI_STREAM_GENERATE_ENDPOINT = f"{GEMINI_WEB_BASE}/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
GEMINI_REQUEST_TIMEOUT_SECONDS = 180
GEMINI_DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_GEMINI_IMAGE_BYTES = 50 * 1024 * 1024


class GeminiWebImageError(RuntimeError):
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


def _parse_cookie_input(token: str, account: dict | None = None) -> tuple[dict[str, str], str]:
    """Extract cookie dict and optional SNlM0e token from token/account."""
    token_str = str(token or "").strip()
    account_dict = account if isinstance(account, dict) else {}
    snlm0e = str(account_dict.get("snlm0e") or "").strip()

    cookies: dict[str, str] = {}
    if (token_str.startswith("{") and token_str.endswith("}")) or (token_str.startswith("[") and token_str.endswith("]")):
        try:
            parsed = json.loads(token_str)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "name" in item and "value" in item:
                        cookies[str(item["name"])] = str(item["value"])
            elif isinstance(parsed, dict):
                if "cookies" in parsed:
                    c_val = parsed["cookies"]
                    if isinstance(c_val, dict):
                        cookies.update({str(k): str(v) for k, v in c_val.items()})
                    elif isinstance(c_val, list):
                        for item in c_val:
                            if isinstance(item, dict) and "name" in item and "value" in item:
                                cookies[str(item["name"])] = str(item["value"])
                    elif isinstance(c_val, str):
                        for part in c_val.split(";"):
                            if "=" in part:
                                k, v = part.strip().split("=", 1)
                                cookies[k.strip()] = v.strip()
                if "snlm0e" in parsed and not snlm0e:
                    snlm0e = str(parsed["snlm0e"]).strip()
                for k, v in parsed.items():
                    if k.startswith("__Secure") or k in {"SID", "HSID", "SSID", "APISID", "SAPISID", "SIDCC"}:
                        cookies[k] = str(v)
        except Exception:
            pass

    if not cookies and token_str:
        if "=" in token_str:
            for part in token_str.split(";"):
                if "=" in part:
                    k, v = part.strip().split("=", 1)
                    cookies[k.strip()] = v.strip()
        elif not any(c in token_str for c in (":", '"', "{", "}", "[", "]", ",", "\n", "\r", " ")):
            # User pasted raw token value without cookie name prefix (e.g. g.a000...)
            if len(token_str) >= 20:
                cookies["__Secure-1PSID"] = token_str

    return cookies, snlm0e


def _determine_aspect_ratio(size: str | None, ratio: str | None) -> str | None:
    """Map size / ratio to Gemini-compatible aspect ratio string."""
    if ratio and ratio not in {"auto", "default", ""}:
        return ratio
    if not size:
        return None
    try:
        parts = size.lower().split("x")
        if len(parts) == 2:
            w, h = int(parts[0]), int(parts[1])
            if w == h:
                return "1:1"
            aspect = w / h
            ratios = [
                ("1:1", 1.0),
                ("4:3", 4 / 3),
                ("3:4", 3 / 4),
                ("3:2", 3 / 2),
                ("2:3", 2 / 3),
                ("16:9", 16 / 9),
                ("9:16", 9 / 16),
            ]
            return min(ratios, key=lambda r: abs(aspect - r[1]))[0]
    except Exception:
        pass
    return None


def _high_res_image_url(url: str) -> str:
    """Ensure Google LH3 image URL requests the original full master resolution (=s0)."""
    clean_url = re.sub(r"=(?:s|w|h)\d+.*$", "", url)
    return clean_url + "=s0"


def _extract_google_image_urls(text: str) -> list[str]:
    """Find generated Google image URLs in response payload."""
    # Matches URLs like https://lh3.googleusercontent.com/gg-dl/... or https://lh3.googleusercontent.com/ggms/...
    pattern = re.compile(
        r'https?://(?:lh\d+\.googleusercontent\.com|googleusercontent\.com|encrypted-tbn\d+\.gstatic\.com)/[^\s"\'\\<>]+'
    )
    urls: list[str] = []
    for match in pattern.finditer(text):
        url = match.group(0)
        # Clean unicode escapes or trailing quotes
        cleaned = url.replace(r"\u003d", "=").replace(r"\u0026", "&").rstrip('"\')')
        # Skip small icons/avatars and internal placeholders
        if "=s32" in cleaned or "=s64" in cleaned or "profile" in cleaned or "avatar" in cleaned or "image_generation_content" in cleaned:
            continue
        parsed = urlparse(cleaned)
        if parsed.scheme != "https" or not re.fullmatch(r"lh\d+\.googleusercontent\.com", parsed.hostname or ""):
            continue
        if not parsed.path.startswith(("/gg-dl/", "/ggms/")):
            continue
        if cleaned not in urls:
            urls.append(cleaned)
    return urls


def _extract_gemini_response_text(resp_text: str) -> str:
    """Extract assistant text response or rejection explanation from Gemini RPC payload."""
    try:
        start_idx = resp_text.find('[')
        if start_idx != -1:
            raw = resp_text[start_idx:]
            chunks = json.loads(raw)
            texts: list[str] = []
            for c in chunks:
                if isinstance(c, list) and len(c) > 2 and c[0] == "wrb.fr" and isinstance(c[2], str):
                    try:
                        inner = json.loads(c[2])
                        if isinstance(inner, list) and len(inner) > 4 and isinstance(inner[4], list):
                            for cand in inner[4]:
                                if isinstance(cand, list) and len(cand) > 1 and isinstance(cand[1], list):
                                    for item in cand[1]:
                                        if isinstance(item, str):
                                            clean = re.sub(r'https?://\S+', '', item).strip()
                                            if clean:
                                                texts.append(clean)
                    except Exception:
                        pass
            if texts:
                return texts[-1]
    except Exception:
        pass

    matches = re.findall(r'\[\"rc_[^\"]+\",\[\"(.*?)\"\]', resp_text)
    for m in reversed(matches):
        try:
            clean = m.encode().decode('unicode_escape', errors='ignore')
            clean = re.sub(r'https?://\S+', '', clean).strip()
            if clean:
                return clean
        except Exception:
            pass
    return ""


class GeminiWebImageProvider:
    """Dispatches text-to-image requests to Gemini Web / Bard frontend."""

    def __init__(
        self,
        token: str,
        *,
        account: dict[str, Any] | None = None,
        timeout: float = GEMINI_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.account = dict(account or {})
        self.token = str(token or "").strip()
        self.cookies, _ = _parse_cookie_input(self.token, self.account)
        self.snlm0e = ""  # Fetch a fresh page token for each generation session.
        self.timeout = max(10.0, float(timeout))
        self._session: requests.Session | None = None

        if not self.cookies.get("__Secure-1PSID"):
            raise GeminiWebImageError(
                "Gemini Web cookies are required. Please provide __Secure-1PSID and __Secure-1PSIDTS.",
                status_code=401,
                code="token_invalid",
            )

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            kwargs = proxy_settings.build_session_kwargs(
                account=self.account,
                upstream=True,
                impersonate="chrome124",
                verify=True,
            )
            self._session = requests.Session(**kwargs)
            self._session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/133.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7",
                "Referer": f"{GEMINI_WEB_BASE}/",
                "Origin": GEMINI_WEB_BASE,
                "X-Same-Domain": "1",
            })
            for k, v in self.cookies.items():
                self._session.cookies.set(k, v, domain=".google.com")
        return self._session

    def _sync_cookies_back(self) -> None:
        """Persist newly rotated cookies (especially __Secure-1PSIDTS) back to account."""
        if not self._session or not self.token:
            return
        try:
            current_cookies = dict(self._session.cookies)
            if not current_cookies:
                return
            rotated = {}
            for k in ["__Secure-1PSIDTS", "__Secure-1PSIDCC", "SIDCC", "__Secure-3PSIDTS", "__Secure-3PSIDCC"]:
                if k in current_cookies and current_cookies[k] != self.cookies.get(k):
                    rotated[k] = current_cookies[k]

            if rotated:
                from services.account_service import account_service
                self.token = account_service.rotate_gemini_cookies(self.token, rotated)
                self.cookies.update(rotated)
        except Exception:
            logger.debug("Failed to persist Gemini cookie rotation")

    def close(self) -> None:
        if self._session is not None:
            try:
                self._sync_cookies_back()
                self._session.close()
            except Exception:
                pass
            self._session = None

    def __enter__(self) -> "GeminiWebImageProvider":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _ensure_snlm0e(self) -> str:
        """Fetch SNlM0e token from gemini.google.com/app if not present."""
        if self.snlm0e:
            return self.snlm0e

        try:
            resp = self.session.get(f"{GEMINI_WEB_BASE}/app", timeout=self.timeout)
            if resp.status_code in {401, 403}:
                raise GeminiWebImageError(
                    "Gemini Web session is expired or invalid",
                    status_code=401,
                    code="token_invalid",
                )

            # Search for SNlM0e in HTML first
            match = re.search(r'"SNlM0e":"([^"]+)"', resp.text)
            if match:
                self.snlm0e = match.group(1)
                return self.snlm0e

            match_at = re.search(r'"at":"([^"]+)"', resp.text)
            if match_at:
                self.snlm0e = match_at.group(1)
                return self.snlm0e

            # If SNlM0e is not found, check if redirected to login or unauthenticated landing page
            if "accounts.google.com" in resp.url:
                raise GeminiWebImageError(
                    "Gemini Web Cookie is expired or missing required timestamp token (__Secure-1PSIDTS). Please copy the full Cookie string from gemini.google.com.",
                    status_code=401,
                    code="token_invalid",
                )

            if "landing" in resp.text:
                raise GeminiWebImageError(
                    "Gemini Web redirected to public landing page. Cookie is missing __Secure-1PSIDTS. Please copy the complete Cookie string from gemini.google.com.",
                    status_code=401,
                    code="token_invalid",
                )
        except GeminiWebImageError:
            raise
        except Exception as exc:
            logger.warning("Failed to fetch Gemini page token")
            raise GeminiWebImageError(
                "Failed to connect to Gemini Web",
                status_code=502,
                code="upstream_error",
            ) from exc

        if not self.snlm0e:
            raise GeminiWebImageError(
                "Failed to extract SNlM0e token from Gemini Web. Please verify that your Cookie includes __Secure-1PSID and __Secure-1PSIDTS.",
                status_code=401,
                code="token_invalid",
            )
        return self.snlm0e

    def generate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        ratio: str | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        cleaned_prompt = str(prompt or "").strip()
        if not cleaned_prompt:
            raise GeminiWebImageError("prompt is required", status_code=400, code="invalid_prompt")
        if images:
            raise GeminiWebImageError("Gemini Web currently supports text-to-image only.", status_code=400, code="unsupported_operation")
        if size and size != "auto":
            match = re.fullmatch(r"(\d+)x(\d+)", size.lower())
            if not match or not all(1 <= int(v) <= 4096 for v in match.groups()):
                raise GeminiWebImageError("Gemini image dimensions must be between 1 and 4096.", status_code=400, code="invalid_size")

        at_token = self._ensure_snlm0e()

        # Build prompt with ratio & generation intent
        aspect_ratio = _determine_aspect_ratio(size, ratio)
        is_zh = any('\u4e00' <= c <= '\u9fff' for c in cleaned_prompt)
        if is_zh:
            ratio_hint = f"（宽高比 {aspect_ratio}）" if aspect_ratio else ""
            full_prompt = f"请生成一张高质量图片：{cleaned_prompt}{ratio_hint}"
            lang = ["zh-CN"]
        else:
            ratio_hint = f" (aspect ratio {aspect_ratio})" if aspect_ratio else ""
            full_prompt = f"Please generate a high-quality image: {cleaned_prompt}{ratio_hint}"
            lang = ["en"]

        # Construct Google Batchexecute RPC payload
        req_payload = [
            [full_prompt, 0, None, None, None, None, 0],
            lang,
            [None, None, None],
            None, None, None, [1], 0, [], None, None, None, None, None,
        ]

        f_req = json.dumps([None, json.dumps(req_payload)])
        post_data = {
            "f.req": f_req,
            "at": at_token,
        }

        try:
            resp = self.session.post(
                GEMINI_STREAM_GENERATE_ENDPOINT,
                data=post_data,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise GeminiWebImageError(
                "Gemini Web request failed",
                status_code=502,
                code="upstream_error",
                retryable=True,
            ) from exc

        if resp.status_code in {401, 403}:
            raise GeminiWebImageError(
                "Gemini Web session cookie is invalid or expired",
                status_code=401,
                code="token_invalid",
            )
        if resp.status_code == 429:
            raise GeminiWebImageError(
                "Gemini Web account is rate-limited",
                status_code=429,
                code="rate_limit",
                retryable=True,
                retry_after_seconds=60,
            )
        if resp.status_code >= 400:
            raise GeminiWebImageError(
                f"Gemini Web returned HTTP {resp.status_code}",
                status_code=resp.status_code,
                code="upstream_error",
            )

        resp_text = resp.text

        # Extract image URLs from RPC payload
        image_urls = _extract_google_image_urls(resp_text)
        if not image_urls:
            explanation = _extract_gemini_response_text(resp_text)
            logger.warning("Gemini Web response contained no generated image")
            explanation_lower = explanation.lower()
            combined_lower = f"{explanation_lower} {resp_text[:1000].lower()}"

            if any(k in combined_lower for k in [
                "quota", "exceeded", "limit for today", "reached your limit",
                "image generation limit", "try again tomorrow", "已达上限", "达到上限", "生成限制", "次数用尽"
            ]):
                raise GeminiWebImageError(
                    f"Gemini Web 今日生图配额已达上限: {explanation or 'Daily quota reached'}",
                    status_code=429,
                    code="insufficient_quota",
                    retryable=False,
                )
            if any(k in explanation_lower for k in [
                "can't generate", "cannot generate", "unable to generate",
                "safety", "policy", "无法生成", "不能生成", "抱歉，我无法"
            ]):
                raise GeminiWebImageError(
                    f"Gemini 拒绝生成该图片: {explanation}",
                    status_code=400,
                    code="invalid_prompt",
                )
            if explanation:
                raise GeminiWebImageError(
                    f"Gemini 未生成图片，返回了文本回复: {explanation}",
                    status_code=502,
                    code="no_image_generated",
                )
            raise GeminiWebImageError(
                "Gemini Web did not return any generated image",
                status_code=502,
                code="no_image_generated",
            )

        # Try high-resolution master URL (=s0) first, then fall back to original URL
        download_candidates: list[str] = []
        for u in image_urls:
            hr_u = _high_res_image_url(u)
            if hr_u not in download_candidates:
                download_candidates.append(hr_u)
            if u not in download_candidates:
                download_candidates.append(u)

        raw_bytes: bytes | None = None
        last_download_err: Exception | None = None
        for target_url in download_candidates:
            try:
                img_resp = self.session.get(target_url, timeout=GEMINI_DOWNLOAD_TIMEOUT_SECONDS, stream=True, allow_redirects=False)
                try:
                    if img_resp.status_code != 200:
                        continue
                    chunks = []
                    downloaded = 0
                    for chunk in img_resp.iter_content():
                        downloaded += len(chunk)
                        if downloaded > MAX_GEMINI_IMAGE_BYTES:
                            raise ValueError("Generated image exceeds download limit")
                        chunks.append(chunk)
                    candidate = b"".join(chunks)
                    with Image.open(BytesIO(candidate)) as image:
                        if image.width * image.height > 40_000_000:
                            raise ValueError("Generated image exceeds pixel limit")
                        image.verify()
                    raw_bytes = candidate
                    break
                finally:
                    img_resp.close()
            except Exception as exc:
                last_download_err = exc
                logger.warning("Failed to download a Gemini image candidate")
                continue

        if not raw_bytes:
            raise GeminiWebImageError(
                "Failed to download a valid generated image",
                status_code=502,
                code="image_download_failed",
            )

        # Remove Sparkle star watermark
        clean_bytes = remove_gemini_sparkle(raw_bytes)

        # Ensure high resolution (>= 1K) and support explicit 2K/4K scaling
        try:
            im = Image.open(BytesIO(clean_bytes))
            target_w, target_h = None, None
            if size and "x" in size:
                try:
                    parts = size.lower().split("x")
                    target_w, target_h = int(parts[0]), int(parts[1])
                except Exception:
                    pass

            max_dim = max(im.width, im.height)
            if target_w and target_h and (target_w >= 1920 or target_h >= 1920):
                # 2K / 4K upscale if explicitly requested
                im = im.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=95)
                clean_bytes = buf.getvalue()
            elif max_dim < 1024:
                # Guarantee minimum 1K resolution (never deliver a 512px thumbnail)
                scale = 1024.0 / max_dim
                new_w = int(round(im.width * scale))
                new_h = int(round(im.height * scale))
                im = im.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=95)
                clean_bytes = buf.getvalue()
        except Exception as exc:
            logger.warning(f"Failed to post-process Gemini image resolution: {exc}")

        b64_data = base64.b64encode(clean_bytes).decode("utf-8")

        return {
            "created": int(time.time()),
            "data": [
                {
                    "b64_json": b64_data,
                    "revised_prompt": cleaned_prompt,
                }
            ],
        }
