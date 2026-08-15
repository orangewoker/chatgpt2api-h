from __future__ import annotations

import base64
import unittest
from io import BytesIO

from PIL import Image

from services.image_providers.agnes import (
    AGNES_IMAGE_ENDPOINT,
    AgnesImageError,
    AgnesImageProvider,
)


def png_base64() -> str:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeResponse:
    def __init__(
        self,
        payload: object = None,
        *,
        status_code: int = 200,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text
        self.content = content
        self.headers = dict(headers or {})

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def iter_content(self, chunk_size: int | None = None):
        size = max(1, int(chunk_size or len(self.content) or 1))
        for index in range(0, len(self.content), size):
            yield self.content[index:index + size]

    def close(self) -> None:
        pass


class FakeSession:
    def __init__(self, post_response: FakeResponse, get_response: FakeResponse | None = None) -> None:
        self.post_response = post_response
        self.get_response = get_response or FakeResponse(content=b"downloaded")
        self.posts: list[dict] = []
        self.gets: list[dict] = []

    def post(self, url: str, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return self.post_response

    def get(self, url: str, **kwargs):
        self.gets.append({"url": url, **kwargs})
        return self.get_response


class AgnesImageProviderTests(unittest.TestCase):
    def test_text_to_image_uses_return_base64_and_native_size_controls(self) -> None:
        payload = AgnesImageProvider.build_payload(
            prompt="draw a skyline",
            size="1920x1088",
        )

        self.assertEqual(payload["model"], "agnes-image-2.1-flash")
        self.assertEqual(payload["size"], "1K")
        self.assertEqual(payload["ratio"], "16:9")
        self.assertTrue(payload["return_base64"])
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["extra_body"], {"response_format": "b64_json"})

    def test_image_to_image_uses_extra_body_with_data_uri(self) -> None:
        payload = AgnesImageProvider.build_payload(
            prompt="make it blue",
            size="2K",
            ratio="3:2",
            images=[png_base64(), png_base64()],
        )

        self.assertEqual(payload["size"], "2K")
        self.assertEqual(payload["ratio"], "3:2")
        self.assertNotIn("return_base64", payload)
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["extra_body"]["response_format"], "b64_json")
        self.assertEqual(len(payload["extra_body"]["image"]), 2)
        self.assertTrue(all(item.startswith("data:image/png;base64,") for item in payload["extra_body"]["image"]))

    def test_generate_normalizes_base64_response(self) -> None:
        encoded = png_base64()
        session = FakeSession(FakeResponse({
            "created": 123,
            "data": [{"b64_json": encoded, "revised_prompt": None}],
        }))
        provider = AgnesImageProvider("secret-key", session=session)

        result = provider.generate(prompt="draw", size=None)

        self.assertEqual(result, {
            "created": 123,
            "data": [{"b64_json": encoded, "revised_prompt": "draw"}],
        })
        self.assertEqual(session.posts[0]["url"], AGNES_IMAGE_ENDPOINT)
        self.assertEqual(session.posts[0]["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(session.posts[0]["timeout"], 360)

    def test_generate_downloads_url_only_response(self) -> None:
        image_bytes = base64.b64decode(png_base64())
        session = FakeSession(
            FakeResponse({"created": 456, "data": [{"url": "https://images.example/result.png"}]}),
            FakeResponse(content=image_bytes, headers={"Content-Type": "image/png"}),
        )
        provider = AgnesImageProvider("secret-key", session=session)

        result = provider.generate(prompt="draw")

        self.assertEqual(
            result["data"][0]["b64_json"],
            base64.b64encode(image_bytes).decode("ascii"),
        )
        self.assertEqual(session.gets[0]["url"], "https://images.example/result.png")
        self.assertEqual(session.gets[0]["headers"]["Accept-Encoding"], "identity")
        self.assertFalse(session.gets[0]["allow_redirects"])
        self.assertEqual(session.gets[0]["accept_encoding"], "identity")
        self.assertTrue(session.gets[0]["stream"])

    def test_http_auth_error_is_structured_without_api_key(self) -> None:
        session = FakeSession(FakeResponse(
            {"error": {"message": "invalid credential", "code": "bad_auth"}},
            status_code=401,
        ))
        provider = AgnesImageProvider("do-not-leak-this-key", session=session)

        with self.assertRaises(AgnesImageError) as caught:
            provider.generate(prompt="draw")

        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(caught.exception.error_type, "authentication_error")
        self.assertEqual(caught.exception.code, "bad_auth")
        self.assertNotIn("do-not-leak-this-key", str(caught.exception))

    def test_invalid_response_is_rejected(self) -> None:
        provider = AgnesImageProvider(
            "secret-key",
            session=FakeSession(FakeResponse({"created": 1, "data": []})),
        )

        with self.assertRaises(AgnesImageError) as caught:
            provider.generate(prompt="draw")

        self.assertEqual(caught.exception.code, "no_image_generated")

        provider = AgnesImageProvider(
            "secret-key",
            session=FakeSession(FakeResponse({"data": [{"b64_json": png_base64()}] * 5})),
        )
        with self.assertRaises(AgnesImageError) as too_many:
            provider.generate(prompt="draw")
        self.assertEqual(too_many.exception.code, "invalid_upstream_response")

    def test_invalid_ratio_is_rejected_before_request(self) -> None:
        with self.assertRaises(AgnesImageError) as caught:
            AgnesImageProvider.build_payload(prompt="draw", ratio="5:4")

        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.code, "invalid_ratio")

        with self.assertRaises(AgnesImageError) as conflicting:
            AgnesImageProvider.build_payload(
                prompt="draw",
                size="1024x1024",
                ratio="16:9",
            )
        self.assertEqual(conflicting.exception.code, "invalid_ratio")

    def test_url_only_response_rejects_private_or_non_https_url(self) -> None:
        for url in ("http://images.example/result.png", "https://127.0.0.1/result.png"):
            with self.subTest(url=url):
                provider = AgnesImageProvider(
                    "secret-key",
                    session=FakeSession(FakeResponse({"data": [{"url": url}]})),
                )
                with self.assertRaises(AgnesImageError) as caught:
                    provider.generate(prompt="draw")
                self.assertEqual(caught.exception.code, "invalid_upstream_response")

    def test_rate_limit_exposes_retry_after_without_leaking_key(self) -> None:
        provider = AgnesImageProvider(
            "sensitive-key",
            session=FakeSession(FakeResponse(
                {"error": {"message": "sensitive-key is limited"}},
                status_code=429,
                headers={"Retry-After": "120"},
            )),
        )

        with self.assertRaises(AgnesImageError) as caught:
            provider.generate(prompt="draw")

        self.assertEqual(caught.exception.retry_after_seconds, 120)
        self.assertNotIn("sensitive-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
