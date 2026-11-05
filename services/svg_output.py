"""Resolve SVG from an existing picture conversation; never start a file-generation turn."""
import base64
import re
import time

from utils.helper import is_svg_bytes, ensure_ok, UpstreamHTTPError

SVG_PATH_RE = re.compile(r"(?:sandbox:)?(/(?:mnt/data|workspace)/[^\s\"'\)\]]+\.svg)", re.I)
SVG_TEXT_RE = re.compile(r"<svg\b[\s\S]*?</svg\s*>", re.I)


def resolve_svg_outputs(backend, conversation_id, file_ids, sediment_ids, message, timeout):
    def wrap(data):
        return [{"b64_json": base64.b64encode(data).decode("ascii")}]

    # A web picture turn may return SVG directly in its assistant response.
    for match in SVG_TEXT_RE.finditer(message):
        raw = match.group().encode("utf-8")
        if is_svg_bytes(raw):
            return wrap(raw)
    deadline = time.monotonic() + max(0, timeout)
    seen = set()
    file_ids, sediment_ids = list(file_ids), list(sediment_ids)
    while True:
        try:
            document = backend._get_conversation(conversation_id)
        except UpstreamHTTPError as exc:
            if exc.status_code not in {404, 409, 423, 429, 500, 502, 503, 504} or time.monotonic() >= deadline:
                raise
            time.sleep(min(2, max(0, deadline - time.monotonic())))
            continue
        for record in backend._extract_image_tool_records(document):
            file_ids.extend(item for item in record.get("file_ids", []) if item not in file_ids)
            sediment_ids.extend(item for item in record.get("sediment_ids", []) if item not in sediment_ids)
        urls = backend.resolve_conversation_image_urls(conversation_id, file_ids, sediment_ids, poll=False)
        for url in urls:
            if url in seen:
                continue
            for raw in backend.download_image_bytes([url]):
                if is_svg_bytes(raw):
                    return wrap(raw)
            seen.add(url)
        for artifact in backend._extract_editable_artifacts(document, SVG_PATH_RE):
            if artifact.author_role == "user" or not backend._looks_like_editable_primary(artifact, (".svg",), {"image/svg+xml"}, ("svg",)):
                continue
            url = backend._resolve_editable_download_url(conversation_id, artifact)
            if not url or url in seen:
                continue
            response = backend.session.get(url, timeout=60)
            ensure_ok(response, "svg_download")
            if is_svg_bytes(response.content):
                return wrap(response.content)
            seen.add(url)
        for node in (document.get("mapping") or {}).values():
            msg = node.get("message") or {}
            if (msg.get("author") or {}).get("role") != "assistant":
                continue
            for part in (msg.get("content") or {}).get("parts") or []:
                if isinstance(part, str):
                    for match in SVG_TEXT_RE.finditer(part):
                        raw = match.group().encode("utf-8")
                        if is_svg_bytes(raw):
                            return wrap(raw)
        if time.monotonic() >= deadline:
            return []  # format_image_result reports an explicit output_format_mismatch.
        time.sleep(min(2, max(0, deadline - time.monotonic())))
