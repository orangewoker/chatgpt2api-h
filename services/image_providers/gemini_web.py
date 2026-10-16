"""Gemini Web direct image generation provider."""

from __future__ import annotations

import base64
import json
import random
import string
import re
import time
from io import BytesIO
from typing import Any
from urllib.parse import urljoin, urlparse

from curl_cffi import CurlMime, requests
from PIL import Image

from services.proxy_service import proxy_settings
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
    normalized_text = str(text or "")
    normalized_text = re.sub(r"\\+u002f", "/", normalized_text)
    normalized_text = re.sub(r"\\+u003d", "=", normalized_text)
    normalized_text = re.sub(r"\\+u0026", "&", normalized_text)
    normalized_text = normalized_text.replace(r"\/", "/")
    urls: list[str] = []
    for match in pattern.finditer(normalized_text):
        url = match.group(0)
        # Clean unicode escapes or trailing quotes
        cleaned = (
            url.replace(r"\/", "/")
            .replace(r"\u002f", "/")
            .replace(r"\u003d", "=")
            .replace(r"\u0026", "&")
            .rstrip('"\')')
        )
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


def _extract_gemini_full_size_context(text: str) -> tuple[str, str, str, str] | None:
    """Extract the token and conversation ids used by Gemini's full-size RPC."""
    raw_text = str(text or "")
    payloads: list[Any] = []
    try:
        start = raw_text.find("[")
        if start >= 0:
            payloads.append(json.loads(raw_text[start:]))
    except Exception:
        pass
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("["):
            line = line[line.find("[") :] if "[" in line else ""
        if not line:
            continue
        try:
            payloads.append(json.loads(line))
        except Exception:
            continue
    if not payloads:
        return None

    def walk(value: Any):
        if isinstance(value, str) and value.startswith("["):
            try:
                yield from walk(json.loads(value))
            except Exception:
                return
        elif isinstance(value, list):
            # A wrb.fr frame contains the JSON encoded RPC response at index 2.
            if len(value) > 2 and value[0] == "wrb.fr" and isinstance(value[2], str):
                try:
                    inner = json.loads(value[2])
                except Exception:
                    inner = None
                if isinstance(inner, list):
                    metadata = inner[1] if len(inner) > 1 else None
                    candidates = inner[4] if len(inner) > 4 else None
                    if isinstance(metadata, list) and len(metadata) >= 2:
                        cid, rid = str(metadata[0] or ""), str(metadata[1] or "")
                        if cid and rid and isinstance(candidates, list):
                            for candidate in candidates:
                                if not isinstance(candidate, list):
                                    continue
                                rcid = str(candidate[0] or "") if candidate else ""
                                if not rcid.startswith("rc_"):
                                    continue
                                # Old format stores [url, token] at indexes 3/5;
                                # new dict format stores them at indexes 3/4.
                                def find_image_context(value: Any):
                                    if isinstance(value, list):
                                        if len(value) > 5 and isinstance(value[3], str) and isinstance(value[5], str) and "googleusercontent.com" in value[3]:
                                            yield value[3], value[5]
                                        elif len(value) > 4 and isinstance(value[3], str) and isinstance(value[4], str) and "googleusercontent.com" in value[3]:
                                            yield value[3], value[4]
                                        for nested in value:
                                            yield from find_image_context(nested)
                                    elif isinstance(value, dict):
                                        for nested in value.values():
                                            yield from find_image_context(nested)

                                def find_image_id(value: Any) -> str | None:
                                    if isinstance(value, str) and value.startswith("http://googleusercontent.com/image_generation_content/"):
                                        return value
                                    if isinstance(value, (list, dict)):
                                        for item in (value.values() if isinstance(value, dict) else value):
                                            found = find_image_id(item)
                                            if found:
                                                return found
                                    return None

                                image_id = find_image_id(candidate)
                                if image_id:
                                    for source_url, token in find_image_context(candidate):
                                        yield source_url, token, json.dumps([cid, rid, rcid]), image_id
            for item in value:
                yield from walk(item)

    for payload in payloads:
        for result in walk(payload):
            if result:
                return result
    return None


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
        self._build_label: str | None = None
        self._session_id: str | None = None
        self._language = "en"
        self._push_id = ""

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

            match_at = re.search(r'"at":"([^"]+)"', resp.text)
            if match_at:
                self.snlm0e = match_at.group(1)

            # These values are sent by the browser on c8o8Fe full-size
            # downloads. They rotate with the Gemini web build/session.
            for pattern in (
                r'"bl":"([^"]+)"',
                r'"buildLabel":"([^"]+)"',
                r'"cfb2h":"([^"]+)"',
            ):
                match_bl = re.search(pattern, resp.text)
                if match_bl:
                    self._build_label = match_bl.group(1)
                    break
            match_sid = re.search(r'"(?:f\.sid|FdrFJe)"\s*:\s*"?(-?\d+)', resp.text)
            if match_sid:
                self._session_id = match_sid.group(1)
            match_lang = re.search(r'"hl":"([a-zA-Z-]+)"', resp.text)
            if match_lang:
                self._language = match_lang.group(1)
            match_push = re.search(r'"qKIAYe"\s*:\s*"([^\"]+)"', resp.text)
            if match_push:
                self._push_id = match_push.group(1)
            if self.snlm0e:
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

    def _fetch_full_size_url(self, token: str, prompt: str, metadata_json: str, image_index: int = 0, *, image_id: str) -> str | None:
        """Call the same c8o8Fe RPC used by Gemini's Download full size button."""
        try:
            cid, rid, rcid = json.loads(metadata_json)
            gen_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6)) * 3
            gen_id = gen_id[:16]
            inner = [
                [[None, None, None, [None, None, None, None, None, token]],
                 [image_id, image_index],
                 None, [23, prompt], None, None, None, None, None, gen_id],
                [rid, rcid, cid, None, gen_id], 1, 0, 1,
            ]
            outer = [[[
                "c8o8Fe", json.dumps(inner, separators=(",", ":"), ensure_ascii=False), None, "generic"
            ]]]
            params = {
                "rpcids": "c8o8Fe",
                "_reqid": str(random.randint(100000, 999999)),
                "rt": "c",
                "source-path": f"/app/{cid.removeprefix('c_')}",
                "hl": self._language,
            }
            if self._build_label:
                params["bl"] = self._build_label
            if self._session_id:
                params["f.sid"] = self._session_id
            response = self.session.post(
                f"{GEMINI_WEB_BASE}/_/BardChatUi/data/batchexecute",
                params=params,
                data={"at": self.snlm0e, "f.req": json.dumps(outer, separators=(",", ":"), ensure_ascii=False)},
                headers={
                    "x-goog-ext-525001261-jspb": "[1,null,null,null,null,null,null,0,[4,4]]",
                    "x-goog-ext-73010989-jspb": "[0]",
                },
                timeout=GEMINI_DOWNLOAD_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                response.close()
                return None
            body = response.text
            response.close()
            # batchexecute may return one JSON frame per line or a single
            # concatenated frame. Walk every decodable JSON fragment and look
            # for the c8o8Fe result instead of relying on a fixed line shape.
            candidates: list[Any] = []
            for line in body.splitlines():
                line = line.strip()
                if not line.startswith("["):
                    line = line[line.find("[") :] if "[" in line else ""
                if not line:
                    continue
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    continue
            if not candidates:
                try:
                    start = body.find("[")
                    if start >= 0:
                        candidates.append(json.loads(body[start:]))
                except Exception:
                    pass

            def find_url(value: Any) -> str | None:
                if isinstance(value, str) and value.startswith("["):
                    try:
                        return find_url(json.loads(value))
                    except Exception:
                        return None
                if isinstance(value, list):
                    if len(value) > 2 and value[0] == "wrb.fr" and value[1] == "c8o8Fe":
                        return find_url(value[2])
                    if value and isinstance(value[0], str) and value[0].startswith("http"):
                        return value[0]
                    for item in value:
                        result = find_url(item)
                        if result:
                            return result
                elif isinstance(value, dict):
                    for item in value.values():
                        result = find_url(item)
                        if result:
                            return result
                return None

            for candidate in candidates:
                url = find_url(candidate)
                if url:
                    return url
        except Exception as exc:
            logger.debug("Gemini full-size RPC failed: %s", type(exc).__name__)
        return None

    def _upload_images(self, images: list[str | tuple[bytes, str, str]]) -> list[list[Any]]:
        """Upload reference bytes to Gemini storage and attach IDs to the prompt."""
        if len(images) > 10:
            raise GeminiWebImageError("Gemini 最多支持 10 张参考图。", status_code=400, code="invalid_image")
        validated = []
        for index, item in enumerate(images):
            if isinstance(item, str):
                try:
                    encoded = item.split(",", 1)[1] if item.startswith("data:") else item
                    if len(encoded) > (MAX_GEMINI_IMAGE_BYTES + 2) // 3 * 4:
                        raise ValueError("size limit")
                    data = base64.b64decode(encoded, validate=True)
                    filename = f"reference-{index + 1}.png"
                except (ValueError, IndexError) as exc:
                    raise GeminiWebImageError("参考图 Base64 数据无效。", status_code=400, code="invalid_image") from exc
            else:
                data, filename, _mime = item
            if not data or len(data) > MAX_GEMINI_IMAGE_BYTES:
                raise GeminiWebImageError("参考图为空或超过 50MB。", status_code=400, code="invalid_image")
            try:
                with Image.open(BytesIO(data)) as image:
                    if image.width * image.height > 40_000_000:
                        raise ValueError("pixel limit")
                    mime = Image.MIME.get(image.format or "", "")
                    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                        raise ValueError("unsupported format")
                    image.verify()
                    extension = "jpg" if mime == "image/jpeg" else mime.split("/")[1]
                    filename = str(filename).rsplit(".", 1)[0] + "." + extension
            except Exception as exc:
                raise GeminiWebImageError("参考图必须是有效的 JPEG/PNG/WebP/GIF 图片。", status_code=400, code="invalid_image") from exc
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(filename).replace("\\", "/").rsplit("/", 1)[-1])[:120] or "reference.png"
            validated.append((data, safe_name, mime))
        if not self._push_id:
            raise GeminiWebImageError("Gemini 页面缺少文件上传标识，请刷新 Cookie 后重试。", code="image_upload_failed")
        attachments = []
        for data, filename, mime in validated:
            multipart = CurlMime()
            try:
                multipart.addpart(name="file", content_type=mime, filename=filename, data=data)
                response = self.session.post(
                    "https://content-push.googleapis.com/upload",
                    headers={"X-Tenant-Id": "bard-storage", "Push-ID": self._push_id},
                    multipart=multipart,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
                try:
                    file_id = response.text.strip()
                    if response.status_code != 200 or not file_id.startswith("/contrib_service/") or len(file_id) > 4096:
                        raise GeminiWebImageError("Gemini 参考图上传失败，请重试。", code="image_upload_failed")
                    attachments.append([[file_id], filename])
                finally:
                    response.close()
            except GeminiWebImageError:
                raise
            except Exception as exc:
                raise GeminiWebImageError("Gemini 参考图上传连接失败，请重试。", code="image_upload_failed") from exc
            finally:
                multipart.close()
        return attachments

    def _download_full_size(self, url: str) -> bytes:
        """Resolve Google's HTTP *and text-body* redirects without preview fallback.

        The /gg-dl/...=d-I?alr=yes response can be a URL, not an image. Its
        destination can return another URL before the final /rd-gg-dl/ image.
        Following HTTP 3xx alone silently missed this protocol in older versions.
        """
        parsed = urlparse(url)
        if not parsed.query and "=" not in parsed.path.rsplit("/", 1)[-1]:
            url += "=d-I?alr=yes"
        visited: set[str] = set()
        for _ in range(8):
            parsed = urlparse(url)
            if (parsed.scheme != "https" or parsed.username or parsed.password
                    or parsed.port not in (None, 443)
                    or not (re.fullmatch(r"lh\d+\.google(?:usercontent)?\.com", parsed.hostname or "")
                            or parsed.hostname == "work.fife.usercontent.google.com")
                    or not parsed.path.startswith(("/gg/", "/rd-gg/", "/gg-dl/", "/rd-gg-dl/", "/ggms/"))
                    or url in visited):
                break
            visited.add(url)
            response = self.session.get(url, timeout=GEMINI_DOWNLOAD_TIMEOUT_SECONDS, allow_redirects=False)
            try:
                if response.status_code in {301, 302, 303, 307, 308}:
                    url = urljoin(url, str(response.headers.get("location") or "").strip())
                    continue
                if response.status_code != 200:
                    break
                length = str(response.headers.get("content-length") or "")
                if length.isdigit() and int(length) > MAX_GEMINI_IMAGE_BYTES:
                    break
                content = bytes(response.content or b"")
                if len(content) > MAX_GEMINI_IMAGE_BYTES:
                    break
                # alr=yes uses a 200 text/plain body to convey the next URL.
                if len(content) < 16_384:
                    target = content.decode("utf-8", errors="replace").strip()
                    if target.startswith('"'):
                        try:
                            target = json.loads(target)
                        except ValueError:
                            target = ""
                    if isinstance(target, str) and target.startswith("https://"):
                        url = target
                        continue
                if len(content) <= 1024:
                    break
                try:
                    with Image.open(BytesIO(content)) as image:
                        if image.width * image.height > 40_000_000:
                            break
                        image.verify()
                except (OSError, ValueError, Image.DecompressionBombError):
                    break
                return content
            finally:
                response.close()
        raise GeminiWebImageError(
            "Gemini 全尺寸原图下载失败；未返回预览图，请重试。",
            code="full_size_download_failed",
        )

    def generate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        ratio: str | None = None,
        images: list[str | tuple[bytes, str, str]] | None = None,
    ) -> dict[str, Any]:
        cleaned_prompt = str(prompt or "").strip()
        if not cleaned_prompt:
            raise GeminiWebImageError("prompt is required", status_code=400, code="invalid_prompt")
        if size and size != "auto":
            match = re.fullmatch(r"(\d+)x(\d+)", size.lower())
            if not match or not all(1 <= int(v) <= 4096 for v in match.groups()):
                raise GeminiWebImageError("Gemini image dimensions must be between 1 and 4096.", status_code=400, code="invalid_size")

        at_token = self._ensure_snlm0e()
        attachments = self._upload_images(images) if images else None

        # Build prompt with ratio & generation intent
        aspect_ratio = _determine_aspect_ratio(size, ratio)
        is_zh = any('\u4e00' <= c <= '\u9fff' for c in cleaned_prompt)
        if is_zh:
            ratio_hint = f"（宽高比 {aspect_ratio}）" if aspect_ratio else ""
            # Resolution intent is only a prompt hint, not a pixel guarantee.
            # Preserve the actual pixels returned by the full-size download.
            full_prompt = f"请生成一张高质量4K画质图片：{cleaned_prompt}{ratio_hint}"
            lang = ["zh-CN"]
        else:
            ratio_hint = f" (aspect ratio {aspect_ratio})" if aspect_ratio else ""
            full_prompt = f"Please generate a high-resolution 4K quality image: {cleaned_prompt}{ratio_hint}"
            lang = ["en"]

        # Construct Google Batchexecute RPC payload
        req_payload = [
            [full_prompt, 0, None, attachments, None, None, 0],
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

        # A valid full-size context is sufficient even if the preview URL's
        # path shape changes and the legacy preview extractor misses it.
        context = _extract_gemini_full_size_context(resp_text)
        image_urls = _extract_google_image_urls(resp_text)
        if not context and not image_urls:
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

        if not context:
            raise GeminiWebImageError(
                "Gemini 未返回全尺寸图片标识；未返回预览图，请重试。",
                code="full_size_unavailable",
            )
        _source_url, token, metadata_json, image_id = context
        full_size_url = self._fetch_full_size_url(
            token, full_prompt, metadata_json, image_id=image_id,
        )
        if not full_size_url:
            raise GeminiWebImageError(
                "Gemini 全尺寸下载 RPC 未返回有效地址；未返回预览图，请重试。",
                code="full_size_unavailable",
            )
        try:
            raw_bytes = self._download_full_size(full_size_url)
        except GeminiWebImageError:
            raise
        except Exception as exc:
            raise GeminiWebImageError(
                "Gemini 全尺寸原图下载失败；未返回预览图，请重试。",
                code="full_size_download_failed",
            ) from exc

        # Gemini's full-size download is already the original web artifact.
        # Do not remove its watermark or resize/re-encode it locally; those
        # operations belong only to the GPT image provider path.
        clean_bytes = raw_bytes

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
