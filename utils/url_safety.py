from __future__ import annotations

import ipaddress
from urllib.parse import ParseResult, urlparse


class UnsafeOutboundURL(ValueError):
    pass


class ResponseBodyTooLarge(ValueError):
    pass


def validate_outbound_http_url(value: object, *, require_https: bool = False) -> ParseResult:
    """Reject obvious local/private outbound targets before an image fetch."""
    source = str(value or "").strip()
    parsed = urlparse(source)
    allowed_schemes = {"https"} if require_https else {"http", "https"}
    hostname = str(parsed.hostname or "").strip().lower()
    if (
        parsed.scheme not in allowed_schemes
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise UnsafeOutboundURL("invalid outbound image URL")
    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
        or hostname.endswith(".internal")
    ):
        raise UnsafeOutboundURL("local outbound image URL is not allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return parsed
    if not address.is_global:
        raise UnsafeOutboundURL("private outbound image URL is not allowed")
    return parsed


def read_limited_response_body(response: object, max_bytes: int) -> bytes:
    body = bytearray()
    iter_content = getattr(response, "iter_content", None)
    chunks = None
    if callable(iter_content):
        try:
            chunks = iter(iter_content(chunk_size=64 * 1024))
        except TypeError:
            chunks = None
    if chunks is None:
        chunks = iter((bytes(getattr(response, "content", b"") or b""),))
    for chunk in chunks:
        if not chunk:
            continue
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ResponseBodyTooLarge("response body exceeds limit")
    return bytes(body)
